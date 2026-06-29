"""Production inference: Kafka + Redis(Feast) + Qdrant + MLflow → Kafka/ClickHouse.

Feature contract — MUST match training.py:FEATURE_COLS and feature_computer.py keys.
Any change to feature names here must also update both of those files.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
import json
from typing import Any

import mlflow
import numpy as np
from confluent_kafka import Consumer, Producer

from shared.clients import ClickHouseClient, QdrantPatternStore, RedisFeatureStore
from shared.config import KAFKA
from shared.mlflow_utils import setup_mlflow
from shared.observability.logging import get_logger, setup_logging
from shared.observability.metrics import (
    FRAUD_INFERENCE_DURATION,
    FRAUD_PREDICTIONS_TOTAL,
    FRAUD_RATE_GAUGE,
    FRAUD_SCORE_HISTOGRAM,
    REDIS_FEATURE_FETCH_DURATION,
    REDIS_FEATURE_MISSES_TOTAL,
    start_metrics_server,
)
from shared.observability.tracing import get_tracer, setup_tracing
from shared.resilience import CircuitBreaker, CircuitOpenError, FallbackModel

V_COLS = [f"v{i}" for i in range(1, 29)]
MODEL_WEIGHT = 0.7
VECTOR_WEIGHT = 0.3
FRAUD_THRESHOLD = 0.5

# Feature contract — must exactly match training.py:FEATURE_COLS
# and the dict keys returned by StreamingFeatureComputer.compute()
FEATURE_COLS: list[str] = [
    "amount",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sum_24h",
    "amount_mean_24h",
    "amount_std_24h",
    "velocity_ratio",
    "amount_deviation",
    "hour_of_day",
    "is_night",
]

log = get_logger(__name__)
tracer = get_tracer(__name__)


class FraudInferenceService:
    """
    Real-time fraud scoring (streaming path):
      Kafka(transactions) → Redis(velocity features) + Qdrant(vector NN) + XGBoost(MLflow)
      → Kafka(predictions|alerts) + ClickHouse(audit)

    Observability wired in:
      - Prometheus metrics on every scored transaction
      - Structured JSON logs via structlog
      - OTel spans for Redis + XGBoost + Qdrant steps
      - Circuit breakers on Redis and Qdrant
    """

    def __init__(self, model_name: str = "fraud_detection_model"):
        setup_mlflow()
        self.model = mlflow.xgboost.load_model(f"models:/{model_name}@champion")
        self.model_version = f"{model_name}@champion"
        
        # Initialize Feast Feature Store Client (Pillar 16)
        import feast
        from pathlib import Path
        repo_path = Path(__file__).parent.parent.resolve() / "feature_repo"
        log.info("initializing_feast_online_feature_store", repo_path=str(repo_path))
        self.feature_store = feast.FeatureStore(repo_path=str(repo_path))
        
        self.qdrant = QdrantPatternStore()
        self.ch = ClickHouseClient()
        self.feature_cols = FEATURE_COLS
        self.fallback = FallbackModel()
        # Circuit breakers — isolate dependency failures
        self.redis_breaker = CircuitBreaker(name="redis", failure_threshold=5, timeout=30)
        self.qdrant_breaker = CircuitBreaker(name="qdrant", failure_threshold=3, timeout=60)
        self._recent_labels: list[bool] = []  # rolling window for FRAUD_RATE_GAUGE

    def _vector(self, txn: dict) -> list[float]:
        vec = [float(txn.get(c, txn.get(c.upper(), 0))) for c in V_COLS]
        vec.append(float(txn["amount"]) / 25000.0)
        vec.append(float(txn.get("time_seconds", txn.get("Time", 0))) / 172800.0)
        return vec

    def score(self, txn: dict) -> dict:
        card_id = txn.get("card_id", txn.get("transaction_id", "unknown"))
        start = time.monotonic()

        with tracer.start_as_current_span("kafka_score_transaction") as span:
            span.set_attribute("card_id", card_id)
            span.set_attribute("amount", float(txn.get("amount", 0)))

            # ── 1. Fetch online features from Feast (with circuit breaker) ──────────────
            with tracer.start_as_current_span("feast_online_feature_fetch"):
                feat_start = time.monotonic()
                try:
                    # Query Feast online store
                    response = self.redis_breaker.call(
                        self.feature_store.get_online_features,
                        features=[
                            "transaction_features:txn_count_1h",
                            "transaction_features:txn_count_24h",
                            "transaction_features:amount_sum_24h",
                            "transaction_features:amount_mean_24h",
                            "transaction_features:amount_std_24h",
                            "transaction_features:velocity_ratio",
                            "transaction_features:amount_deviation",
                            "transaction_features:hour_of_day",
                            "transaction_features:is_night",
                        ],
                        entity_rows=[{"card_id": card_id}]
                    )
                    feats_dict = response.to_dict()
                    # Strip feature view prefix dynamically (e.g. transaction_features__amount -> amount)
                    redis_feats = {k.split("__")[-1]: v[0] for k, v in feats_dict.items() if v}
                except CircuitOpenError:
                    log.warning("redis_circuit_open_using_defaults", card_id=card_id)
                    redis_feats = {}
                except Exception as exc:
                    log.warning("feast_fetch_failed_using_defaults", card_id=card_id, error=str(exc))
                    redis_feats = {}
                REDIS_FEATURE_FETCH_DURATION.observe(time.monotonic() - feat_start)

            if not redis_feats:
                REDIS_FEATURE_MISSES_TOTAL.inc()

            # ── 2. Build feature row ─────────────────────────────────────────────────
            row = {**txn, **redis_feats}
            row.setdefault("txn_count_1h", 1)
            row.setdefault("txn_count_24h", 1)
            row.setdefault("amount_sum_24h", row.get("amount", 0))
            row.setdefault("amount_mean_24h", row.get("amount", 0))
            row.setdefault("amount_std_24h", 0.0)
            row.setdefault("velocity_ratio", 1.0)
            row.setdefault("amount_deviation", 1.0)
            row.setdefault("hour_of_day", 0)
            row.setdefault("is_night", 0)

            X = np.array([[float(row.get(c, 0)) for c in self.feature_cols]])

            # ── 3. XGBoost inference ────────────────────────────────────────────
            with tracer.start_as_current_span("xgboost_predict"):
                try:
                    model_score = float(self.model.predict_proba(X)[0][1])
                except Exception as exc:
                    log.warning("model_predict_failed_using_fallback", error=str(exc))
                    model_score = self.fallback.predict_proba(row)

            # ── 4. Qdrant vector similarity score (with circuit breaker) ──────────
            with tracer.start_as_current_span("qdrant_vector_score"):
                try:
                    vector_score = self.qdrant_breaker.call(
                        self.qdrant.vector_fraud_score, self._vector(txn)
                    )
                except CircuitOpenError:
                    log.warning("qdrant_circuit_open", card_id=card_id)
                    vector_score = 0.0

            fraud_score = MODEL_WEIGHT * model_score + VECTOR_WEIGHT * vector_score
            is_fraud = fraud_score >= FRAUD_THRESHOLD
            elapsed = time.monotonic() - start

            # ── 5. Prometheus metrics ──────────────────────────────────────────
            FRAUD_INFERENCE_DURATION.labels(model_version=self.model_version).observe(elapsed)
            FRAUD_PREDICTIONS_TOTAL.labels(
                model_version=self.model_version,
                result="fraud" if is_fraud else "legit",
            ).inc()
            FRAUD_SCORE_HISTOGRAM.labels(model_version=self.model_version).observe(fraud_score)

            # Rolling fraud rate (last 1000 predictions)
            self._recent_labels.append(is_fraud)
            if len(self._recent_labels) > 1000:
                self._recent_labels.pop(0)
            FRAUD_RATE_GAUGE.labels(model_version=self.model_version).set(
                sum(self._recent_labels) / len(self._recent_labels)
            )

            # ── 6. Structured log ──────────────────────────────────────────────
            log.info(
                "transaction_scored",
                transaction_id=txn["transaction_id"],
                card_id=card_id,
                fraud_score=round(fraud_score, 4),
                fraud_label=is_fraud,
                latency_ms=round(elapsed * 1000, 2),
                redis_hit=bool(redis_feats),
                model_version=self.model_version,
            )

        return {
            "transaction_id": txn["transaction_id"],
            "card_id": card_id,
            "amount": float(txn["amount"]),
            "fraud_score": round(fraud_score, 6),
            "model_score": round(model_score, 6),
            "vector_score": round(vector_score, 6),
            "fraud_label": is_fraud,
            "model_version": self.model_version,
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }

    def _audit(self, prediction: dict) -> None:
        import pandas as pd

        row = pd.DataFrame(
            [{
                "transaction_id": prediction["transaction_id"],
                "amount": prediction["amount"],
                "fraud_score": prediction["fraud_score"],
                "vector_score": prediction["vector_score"],
                "fraud_label": int(prediction["fraud_label"]),
                "model_version": prediction["model_version"],
            }]
        )
        self.ch.insert_df("scored_transactions", row, database="fraud")

    def run(self, timeout_sec: float = 60.0) -> None:
        setup_logging(json_logs=True)
        setup_tracing(service_name="fraud-inference-consumer")
        # Expose Prometheus metrics on port 8001 (API uses 8000)
        start_metrics_server(port=8001)

        consumer = Consumer(
            {
                "bootstrap.servers": KAFKA.bootstrap_servers,
                "group.id": "fraud-inference-service",
                "auto.offset.reset": "earliest",
            }
        )
        producer = Producer({"bootstrap.servers": KAFKA.bootstrap_servers})
        consumer.subscribe([KAFKA.transactions_topic])

        log.info("inference_service_started", topic=KAFKA.transactions_topic, timeout_sec=timeout_sec)
        start = datetime.now(timezone.utc)

        try:
            while (datetime.now(timezone.utc) - start).total_seconds() < timeout_sec:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    log.warning("kafka_consumer_error", error=str(msg.error()))
                    continue

                txn = json.loads(msg.value().decode())
                pred = self.score(txn)

                topic = KAFKA.alerts_topic if pred["fraud_label"] else KAFKA.predictions_topic
                producer.produce(
                    topic,
                    key=pred["transaction_id"].encode(),
                    value=json.dumps(pred).encode(),
                )
                producer.poll(0)

                self._audit(pred)
        finally:
            producer.flush()
            consumer.close()
            log.info("inference_service_stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    FraudInferenceService().run(timeout_sec=args.timeout)
