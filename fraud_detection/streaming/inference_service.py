"""Production inference: Kafka + Redis + Qdrant + MLflow → Kafka/ClickHouse."""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import mlflow
import numpy as np
from confluent_kafka import Consumer, Producer

from shared.clients import ClickHouseClient, QdrantPatternStore, RedisFeatureStore
from shared.config import KAFKA
from shared.mlflow_utils import setup_mlflow

V_COLS = [f"v{i}" for i in range(1, 29)]
MODEL_WEIGHT = 0.7
VECTOR_WEIGHT = 0.3
FRAUD_THRESHOLD = 0.5


class FraudInferenceService:
    """
    Real-time fraud scoring (production path):
      Kafka(transactions) → Redis(Spark features) + Qdrant(vector NN) + XGBoost(MLflow)
      → Kafka(predictions) + Kafka(alerts) + ClickHouse(audit)
    """

    def __init__(self, model_name: str = "fraud_detection_model"):
        setup_mlflow()
        self.model = mlflow.xgboost.load_model(f"models:/{model_name}@champion")
        self.model_version = f"{model_name}@champion"
        self.redis = RedisFeatureStore()
        self.qdrant = QdrantPatternStore()
        self.ch = ClickHouseClient()
        self.feature_cols = V_COLS + ["amount", "time_seconds", "txn_count_1h", "amount_mean_1h"]

    def _vector(self, txn: dict) -> list[float]:
        vec = [float(txn.get(c, txn.get(c.upper(), 0))) for c in V_COLS]
        vec.append(float(txn["amount"]) / 25000.0)
        vec.append(float(txn.get("time_seconds", txn.get("Time", 0))) / 172800.0)
        return vec

    def score(self, txn: dict) -> dict:
        card_id = txn.get("card_id", txn.get("transaction_id", "unknown"))
        redis_feats = self.redis.get_features(card_id) or {}

        row = {**txn, **redis_feats}
        row.setdefault("txn_count_1h", 1)
        row.setdefault("amount_mean_1h", row.get("amount", 0))

        X = np.array([[float(row.get(c, 0)) for c in self.feature_cols]])
        model_score = float(self.model.predict_proba(X)[0][1])
        vector_score = self.qdrant.vector_fraud_score(self._vector(txn))

        fraud_score = MODEL_WEIGHT * model_score + VECTOR_WEIGHT * vector_score
        is_fraud = fraud_score >= FRAUD_THRESHOLD

        return {
            "transaction_id": txn["transaction_id"],
            "card_id": card_id,
            "amount": float(txn["amount"]),
            "fraud_score": round(fraud_score, 6),
            "model_score": round(model_score, 6),
            "vector_score": round(vector_score, 6),
            "fraud_label": is_fraud,
            "model_version": self.model_version,
            "scored_at": datetime.utcnow().isoformat(),
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
        consumer = Consumer(
            {
                "bootstrap.servers": KAFKA.bootstrap_servers,
                "group.id": "fraud-inference-service",
                "auto.offset.reset": "earliest",
            }
        )
        producer = Producer({"bootstrap.servers": KAFKA.bootstrap_servers})
        consumer.subscribe([KAFKA.transactions_topic])

        print(f"Inference service listening on {KAFKA.transactions_topic}")
        start = datetime.now()

        try:
            while (datetime.now() - start).total_seconds() < timeout_sec:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
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
                label = "FRAUD" if pred["fraud_label"] else "OK"
                print(
                    f"[{label}] {pred['transaction_id']} "
                    f"score={pred['fraud_score']:.4f} "
                    f"(model={pred['model_score']:.4f}, vector={pred['vector_score']:.4f})"
                )
        finally:
            producer.flush()
            consumer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    FraudInferenceService().run(timeout_sec=args.timeout)
