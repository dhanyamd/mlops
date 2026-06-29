"""Training pipeline — reads pre-computed velocity features, logs to MLflow.

Feature alignment contract:
  Offline (training) : transaction_features.parquet  → compute_velocity_features()
  Online  (inference): Redis                         → StreamingFeatureComputer.compute()
  Both must produce identical feature names. Any change here MUST be reflected in
  inference_service.py:feature_cols and feature_computer.py return dict keys.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

from fraud_detection.data.generate_data import compute_velocity_features
from shared.clients import ClickHouseClient
from shared.config import DATA_ROOT
from shared.mlflow_utils import promote_model, setup_mlflow
from shared.observability.logging import get_logger
from shared.observability.metrics import (
    MODEL_AUC_GAUGE,
    MODEL_F1_GAUGE,
    PIPELINE_RUNS_TOTAL,
    TRAINING_DURATION,
    TRAINING_RUNS_TOTAL,
)

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature contract — must match inference_service.py:feature_cols AND
# the keys returned by StreamingFeatureComputer.compute() written to Redis.
# ---------------------------------------------------------------------------
FEATURE_COLS: list[str] = [
    "amount",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sum_24h",
    "amount_mean_24h",   # 24-hour rolling mean — NOT 1h (matches feature_computer.py)
    "amount_std_24h",
    "velocity_ratio",    # txn_count_1h / max(txn_count_24h, 1)
    "amount_deviation",  # current amount / rolling mean
    "hour_of_day",
    "is_night",
]


@dataclass
class FraudTrainingConfig:
    model_name: str = "fraud_detection_model"
    experiment_name: str = "fraud_detection"
    test_size: float = 0.2
    raw_key: str = "fraud_detection/offline/raw_transactions.parquet"


class FraudTrainingPipeline:
    def __init__(self, config: FraudTrainingConfig):
        self.cfg = config
        self.ch = ClickHouseClient()
        self.model: xgb.XGBClassifier | None = None
        self.metrics: dict = {}
        self.feature_cols = FEATURE_COLS

    def load_features(self) -> "FraudTrainingPipeline":
        """Load features using Feast point-in-time joins (Pillar 16).

        This enforces point-in-time correctness: for each training transaction,
        Feast joins historical features as they existed at event_timestamp.
        This completely eliminates future data leakage during training.
        """
        import feast
        from shared.clients import S3Client
        
        repo_path = Path(__file__).parent.parent.resolve() / "feature_repo"
        log.info("querying_feast_historical_features", repo_path=str(repo_path))
        
        # Read raw transactions from S3 Data Lake (no local parquets)
        s3 = S3Client()
        log.info("extracting_raw_transactions_from_s3", key=self.cfg.raw_key)
        txns = s3.read_df(self.cfg.raw_key)
        
        # Prepare entity dataframe with timestamps and join keys
        entity_df = txns[["user_id", "event_timestamp"]].copy()
        entity_df = entity_df.rename(columns={"user_id": "card_id"})
        entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])
        
        fs = feast.FeatureStore(repo_path=str(repo_path))
        
        feature_refs = [
            "transaction_features:txn_count_1h",
            "transaction_features:txn_count_24h",
            "transaction_features:amount_sum_24h",
            "transaction_features:amount_mean_24h",
            "transaction_features:amount_std_24h",
            "transaction_features:velocity_ratio",
            "transaction_features:amount_deviation",
            "transaction_features:hour_of_day",
            "transaction_features:is_night",
        ]
        
        # Point-in-time join
        training_data = fs.get_historical_features(
            entity_df=entity_df,
            features=feature_refs
        ).to_df()
        
        # Add labels and amount from raw txns back to the dataframe
        training_data["is_fraud"] = txns["is_fraud"].values
        training_data["amount"] = txns["amount"].values
        
        # Validate feature contract
        missing = [c for c in self.feature_cols if c not in training_data.columns]
        if missing:
            raise ValueError(
                f"[training] Feast schema missing columns: {missing}\n"
                f"Available: {list(training_data.columns)}"
            )
            
        self.df = training_data
        return self

    def split(self) -> "FraudTrainingPipeline":
        """Time-ordered split — no data leakage.

        Sort by event_timestamp so earlier events train the model and later
        events test it. Never shuffle time-series data before splitting.
        """
        df = self.df.sort_values("event_timestamp").reset_index(drop=True)
        cutoff = int(len(df) * (1 - self.cfg.test_size))
        train, test = df.iloc[:cutoff], df.iloc[cutoff:]
        self.X_train = train[self.feature_cols]
        self.y_train = train["is_fraud"]
        self.X_test = test[self.feature_cols]
        self.y_test = test["is_fraud"]
        return self

    def train(self) -> "FraudTrainingPipeline":
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=50,
            random_state=42,
            eval_metric="logloss",
        )
        self.model.fit(self.X_train, self.y_train)
        return self

    def evaluate(self) -> "FraudTrainingPipeline":
        proba = self.model.predict_proba(self.X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        report = classification_report(self.y_test, preds, output_dict=True)
        self.metrics = {
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "auc": float(roc_auc_score(self.y_test, proba)),
        }
        return self

    def register(self) -> "FraudTrainingPipeline":
        setup_mlflow()
        mlflow.set_experiment(self.cfg.experiment_name)
        with mlflow.start_run(run_name="fraud_xgb_velocity_features"):
            mlflow.log_metrics(self.metrics)
            # Log the feature contract so inference knows exactly what columns to send
            mlflow.log_param("feature_cols", ",".join(self.feature_cols))
            mlflow.log_param("n_features", len(self.feature_cols))
            mlflow.xgboost.log_model(self.model, "model", registered_model_name=self.cfg.model_name)
            result = mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model",
                self.cfg.model_name,
            )
            promote_model(self.cfg.model_name, result.version, alias="champion")
        return self

    def run(self) -> dict:
        start = time.monotonic()
        try:
            metrics = self.load_features().split().train().evaluate().register().metrics
            TRAINING_RUNS_TOTAL.labels(project="fraud_detection", status="success").inc()
            PIPELINE_RUNS_TOTAL.labels(pipeline="fraud_training", status="success").inc()
            MODEL_AUC_GAUGE.labels(
                project="fraud_detection",
                model_version=self.cfg.model_name,
            ).set(metrics.get("auc", 0))
            MODEL_F1_GAUGE.labels(
                project="fraud_detection",
                model_version=self.cfg.model_name,
            ).set(metrics.get("f1", 0))
            log.info("training_complete", **metrics)
            return metrics
        except Exception as exc:
            TRAINING_RUNS_TOTAL.labels(project="fraud_detection", status="failed").inc()
            PIPELINE_RUNS_TOTAL.labels(pipeline="fraud_training", status="failed").inc()
            log.error("training_failed", error=str(exc))
            raise
        finally:
            elapsed = time.monotonic() - start
            TRAINING_DURATION.labels(project="fraud_detection").observe(elapsed)
            log.info("training_duration", seconds=round(elapsed, 1))
