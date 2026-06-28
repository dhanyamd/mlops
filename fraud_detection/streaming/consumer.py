"""Real-time inference consumer: Kafka → Feature Store → Model → Kafka predictions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import mlflow
import numpy as np
from confluent_kafka import Consumer, Producer
from sqlalchemy import create_engine

from fraud_detection.streaming.feature_computer import StreamingFeatureComputer
from shared.config import KAFKA, POSTGRES
from shared.mlflow_utils import setup_mlflow

FEATURE_COLS = [
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
    "account_age_days",
    "avg_transaction_amount",
    "risk_score",
]

FRAUD_THRESHOLD = 0.5


class RealtimeInferencePipeline:
    """
    Real-time inference path (ML Academy Fraud Detection diagram):
    Kafka → Spark Streaming → Online Store → Inference → Kafka → Actioning Services
    """

    def __init__(self, model_name: str = "fraud_detection_model"):
        setup_mlflow()
        try:
            self.model = mlflow.xgboost.load_model(f"models:/{model_name}@champion")
            self._use_proba = True
        except Exception:
            self.model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
            self._use_proba = False
        self.feature_computer = StreamingFeatureComputer()
        self.user_profiles = self._load_profiles()
        self.model_version = f"{model_name}@champion"

    def _load_profiles(self) -> dict:
        try:
            import pandas as pd

            engine = create_engine(POSTGRES.url)
            df = pd.read_sql("SELECT * FROM fraud_detection.user_profiles", engine)
            risk_map = {"low": 0.2, "medium": 0.5, "high": 0.9}
            return {
                row["user_id"]: {
                    "account_age_days": int(row["account_age_days"]),
                    "avg_transaction_amount": float(row["avg_transaction_amount"]),
                    "risk_score": risk_map.get(row["risk_tier"], 0.5),
                }
                for _, row in df.iterrows()
            }
        except Exception:
            return {}

    def _enrich_with_profile(self, user_id: str, features: dict) -> dict:
        profile = self.user_profiles.get(
            user_id,
            {"account_age_days": 365, "avg_transaction_amount": 100.0, "risk_score": 0.5},
        )
        features.update(profile)
        return features

    def score_transaction(self, txn: dict) -> dict:
        ts = datetime.fromisoformat(txn["timestamp"]) if "timestamp" in txn else datetime.now()
        features = self.feature_computer.compute(txn["user_id"], txn["amount"], ts)
        features = self._enrich_with_profile(txn["user_id"], features)

        X = np.array([[features[c] for c in FEATURE_COLS]])
        if self._use_proba:
            fraud_score = float(self.model.predict_proba(X)[0][1])
        else:
            fraud_score = float(self.model.predict(X)[0])

        return {
            "transaction_id": txn["transaction_id"],
            "user_id": txn["user_id"],
            "amount": txn["amount"],
            "fraud_score": round(fraud_score, 6),
            "fraud_label": fraud_score >= FRAUD_THRESHOLD,
            "model_version": self.model_version,
            "scored_at": datetime.now().isoformat(),
            "features": {k: features[k] for k in FEATURE_COLS},
        }

    def run_consumer(self, timeout_sec: float = 30.0) -> None:
        consumer = Consumer(
            {
                "bootstrap.servers": KAFKA.bootstrap_servers,
                "group.id": "fraud-inference",
                "auto.offset.reset": "earliest",
            }
        )
        producer = Producer({"bootstrap.servers": KAFKA.bootstrap_servers})
        consumer.subscribe([KAFKA.transactions_topic])

        print(f"Listening on {KAFKA.transactions_topic}...")
        start = datetime.now()

        try:
            while (datetime.now() - start).total_seconds() < timeout_sec:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    print(f"Consumer error: {msg.error()}")
                    continue

                txn = json.loads(msg.value().decode())
                prediction = self.score_transaction(txn)

                producer.produce(
                    KAFKA.predictions_topic,
                    key=txn["transaction_id"].encode(),
                    value=json.dumps(prediction).encode(),
                )
                producer.poll(0)

                label = "FRAUD" if prediction["fraud_label"] else "OK"
                print(
                    f"[{label}] {prediction['transaction_id']} "
                    f"score={prediction['fraud_score']:.4f} amount=${prediction['amount']}"
                )

                self._action(prediction)
        finally:
            producer.flush()
            consumer.close()

    def _action(self, prediction: dict) -> None:
        """Downstream actioning: update DB + alert on fraud."""
        try:
            import pandas as pd

            engine = create_engine(POSTGRES.url)
            row = pd.DataFrame([{
                "transaction_id": prediction["transaction_id"],
                "user_id": prediction["user_id"],
                "amount": prediction["amount"],
                "fraud_score": prediction["fraud_score"],
                "fraud_label": prediction["fraud_label"],
                "model_version": prediction["model_version"],
            }])
            row.to_sql(
                "scored_transactions",
                engine,
                schema="fraud_detection",
                if_exists="append",
                index=False,
            )
        except Exception as exc:
            print(f"DB action skipped: {exc}")

        if prediction["fraud_label"]:
            print(f"  ALERT: Fraud detected for {prediction['user_id']} — blocking transaction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    RealtimeInferencePipeline().run_consumer(timeout_sec=args.timeout)
