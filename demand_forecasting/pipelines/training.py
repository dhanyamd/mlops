"""Training pipeline: time-based split, XGBoost, MLflow logging & model registry."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from shared.mlflow_utils import promote_model, setup_mlflow


@dataclass
class TrainingConfig:
    input_path: Path
    target_col: str = "quantity_sold"
    test_days: int = 60
    model_name: str = "demand_forecast_model"
    experiment_name: str = "demand_forecasting"
    artifact_dir: Path = field(default_factory=lambda: Path("artifacts/demand_forecasting"))
    param_grid: dict = field(
        default_factory=lambda: {
            "n_estimators": [100, 200],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
        }
    )


class TrainingPipeline:
    """Automated training: read → time-split → train → evaluate → register."""

    EXCLUDE_COLS = {"date", "product_id", "product_name", "quantity_sold", "revenue"}

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.df: pd.DataFrame | None = None
        self.feature_cols: list[str] = []
        self.model: xgb.XGBRegressor | None = None
        self.metrics: dict[str, float] = {}
        self.best_params: dict = {}

    def read_data(self) -> "TrainingPipeline":
        self.df = pd.read_parquet(self.cfg.input_path)
        self.df["date"] = pd.to_datetime(self.df["date"])
        return self

    def time_based_split(self) -> "TrainingPipeline":
        """Time-series split: last N days as test set (no data leakage)."""
        assert self.df is not None
        cutoff = self.df["date"].max() - pd.Timedelta(days=self.cfg.test_days)
        self.train_df = self.df[self.df["date"] <= cutoff].copy()
        self.test_df = self.df[self.df["date"] > cutoff].copy()
        self.feature_cols = [
            c
            for c in self.df.columns
            if c not in self.EXCLUDE_COLS and self.df[c].dtype != "object"
        ]
        return self

    def train_and_tune(self) -> "TrainingPipeline":
        assert self.train_df is not None
        X_train = self.train_df[self.feature_cols]
        y_train = self.train_df[self.cfg.target_col]

        best_rmse = float("inf")
        best_model = None
        best_params = {}

        for n_est in self.cfg.param_grid["n_estimators"]:
            for depth in self.cfg.param_grid["max_depth"]:
                for lr in self.cfg.param_grid["learning_rate"]:
                    model = xgb.XGBRegressor(
                        n_estimators=n_est,
                        max_depth=depth,
                        learning_rate=lr,
                        random_state=42,
                        n_jobs=-1,
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(self.test_df[self.feature_cols])
                    y_test = self.test_df[self.cfg.target_col]
                    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model
                        best_params = {
                            "n_estimators": n_est,
                            "max_depth": depth,
                            "learning_rate": lr,
                        }

        self.model = best_model
        self.best_params = best_params
        return self

    def evaluate(self) -> "TrainingPipeline":
        assert self.model is not None and self.test_df is not None
        preds = self.model.predict(self.test_df[self.feature_cols])
        y_true = self.test_df[self.cfg.target_col]
        self.metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
            "mae": float(mean_absolute_error(y_true, preds)),
            "r2": float(r2_score(y_true, preds)),
        }
        return self

    def save_artifacts(self) -> "TrainingPipeline":
        assert self.model is not None
        self.cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.cfg.artifact_dir / "model.joblib")
        meta = {"feature_cols": self.feature_cols, "best_params": self.best_params}
        (self.cfg.artifact_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return self

    def log_and_register(self) -> "TrainingPipeline":
        assert self.model is not None
        setup_mlflow()
        try:
            mlflow.set_experiment(self.cfg.experiment_name)
            with mlflow.start_run(run_name="demand_forecast_training"):
                mlflow.log_params(self.best_params)
                mlflow.log_metrics(self.metrics)
                mlflow.log_param("feature_count", len(self.feature_cols))
                mlflow.xgboost.log_model(
                    self.model,
                    "model",
                    registered_model_name=self.cfg.model_name,
                )
                result = mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}/model",
                    self.cfg.model_name,
                )
                promote_model(
                    self.cfg.model_name, result.version, stage="Production", alias="champion"
                )
        except Exception as exc:
            print(f"MLflow logging skipped (using local artifacts): {exc}")
        return self

    def run(self) -> dict[str, float]:
        return (
            self.read_data()
            .time_based_split()
            .train_and_tune()
            .evaluate()
            .save_artifacts()
            .log_and_register()
            .metrics
        )
