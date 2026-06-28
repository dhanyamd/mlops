"""Training pipeline — reads from ClickHouse warehouse, logs to MLflow."""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

from shared.clients import ClickHouseClient
from shared.mlflow_utils import promote_model, setup_mlflow

V_COLS = [f"v{i}" for i in range(1, 29)]


@dataclass
class FraudTrainingConfig:
    model_name: str = "fraud_detection_model"
    experiment_name: str = "fraud_detection"
    test_size: float = 0.2


class FraudTrainingPipeline:
    def __init__(self, config: FraudTrainingConfig):
        self.cfg = config
        self.ch = ClickHouseClient()
        self.model: xgb.XGBClassifier | None = None
        self.metrics: dict = {}
        self.feature_cols = V_COLS + ["amount", "time_seconds"]

    def load_from_warehouse(self) -> "FraudTrainingPipeline":
        df = self.ch.query_df(
            f"""
            SELECT transaction_id, time_seconds, amount, is_fraud,
                   {", ".join(V_COLS)}
            FROM fraud.transactions
            ORDER BY time_seconds
            """
        )
        df["txn_count_1h"] = 1  # batch baseline; Spark computes real-time version
        df["amount_mean_1h"] = df["amount"]
        self.feature_cols = V_COLS + ["amount", "time_seconds", "txn_count_1h", "amount_mean_1h"]
        self.df = df
        return self

    def split(self) -> "FraudTrainingPipeline":
        # Time-ordered split — no leakage
        cutoff = int(len(self.df) * (1 - self.cfg.test_size))
        train, test = self.df.iloc[:cutoff], self.df.iloc[cutoff:]
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
        with mlflow.start_run(run_name="fraud_xgb_clickhouse"):
            mlflow.log_metrics(self.metrics)
            mlflow.xgboost.log_model(self.model, "model", registered_model_name=self.cfg.model_name)
            result = mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model",
                self.cfg.model_name,
            )
            promote_model(self.cfg.model_name, result.version, alias="champion")
        return self

    def run(self) -> dict:
        return self.load_from_warehouse().split().train().evaluate().register().metrics
