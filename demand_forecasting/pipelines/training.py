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
    input_key: str | Path = "demand_forecasting/features/features.parquet"
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
    input_path: Path | str | None = None

    def __post_init__(self):
        if self.input_path is not None:
            self.input_key = self.input_path


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
        key = self.cfg.input_key
        # Self-healing: if local Path or local file exists, read locally
        if isinstance(key, Path) or (isinstance(key, str) and not key.startswith("s3://") and Path(key).exists()):
            self.df = pd.read_parquet(key)
        else:
            from shared.clients import S3Client
            s3 = S3Client()
            self.df = s3.read_df(str(key))
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

    def train_and_select(self) -> "TrainingPipeline":
        assert self.train_df is not None
        assert self.test_df is not None
        from demand_forecasting.pipelines.model_selection import ModelSelector
        
        selector = ModelSelector(experiment_name=self.cfg.experiment_name)
        best_forecaster, best_metrics = selector.select_best_model(
            train_df=self.train_df,
            test_df=self.test_df,
            feature_cols=self.feature_cols,
            target_col=self.cfg.target_col,
            model_name=self.cfg.model_name
        )
        
        self.model = best_forecaster
        self.metrics = best_metrics
        self.best_params = {"model_type": best_forecaster.name}
        return self

    def save_artifacts(self) -> "TrainingPipeline":
        assert self.model is not None
        self.cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
        # For sklearn-compatible wrappers, save the inner model. Otherwise save wrapper.
        if hasattr(self.model, "model"):
            joblib.dump(self.model.model, self.cfg.artifact_dir / "model.joblib")
        else:
            joblib.dump(self.model, self.cfg.artifact_dir / "model.joblib")
            
        meta = {"feature_cols": self.feature_cols, "best_params": self.best_params}
        (self.cfg.artifact_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return self

    def run(self) -> dict[str, float]:
        import time
        from shared.observability.metrics import TRAINING_DURATION, TRAINING_RUNS_TOTAL, MODEL_AUC_GAUGE
        
        start = time.monotonic()
        try:
            self.read_data()
            self.time_based_split()
            self.train_select_result = self.train_and_select()
            self.save_artifacts()
            
            # Emit Prometheus training metrics
            TRAINING_RUNS_TOTAL.labels(project="demand_forecasting", status="success").inc()
            # Set the MAE metric as a gauge indicator of final performance
            MODEL_AUC_GAUGE.labels(project="demand_forecasting", model_version=self.cfg.model_name).set(self.metrics.get("mae", 0.0))
            
            return self.metrics
        except Exception as exc:
            TRAINING_RUNS_TOTAL.labels(project="demand_forecasting", status="failed").inc()
            raise exc
        finally:
            duration = time.monotonic() - start
            TRAINING_DURATION.labels(project="demand_forecasting").observe(duration)

