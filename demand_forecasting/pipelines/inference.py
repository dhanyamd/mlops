"""Batch inference pipeline: score forecasts and write to warehouse + monitor drift."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from shared.clients import ClickHouseClient
from shared.mlflow_utils import setup_mlflow


@dataclass
class InferenceConfig:
    features_path: Path
    model_path: Path = Path("artifacts/demand_forecasting/model.joblib")
    metadata_path: Path = Path("artifacts/demand_forecasting/metadata.json")
    model_name: str = "demand_forecast_model"
    forecast_horizon_days: int = 14
    output_table: str = "demand_forecasting.forecasts"
    use_mlflow: bool = True


class InferencePipeline:
    """Batch inference: load model, predict, postprocess, write to warehouse."""

    def __init__(self, config: InferenceConfig):
        self.cfg = config
        self.df: pd.DataFrame | None = None
        self.model = None
        self.feature_cols: list[str] = []
        self.forecasts: pd.DataFrame | None = None
        self.model_version: str = "local"

    def load_model(self) -> "InferencePipeline":
        if self.cfg.use_mlflow:
            try:
                setup_mlflow()
                model = mlflow.pyfunc.load_model(f"models:/{self.cfg.model_name}@champion")
                if model is not None:
                    self.model = model
                    self.model_version = f"mlflow-{self.cfg.model_name}-champion"
                    if self.cfg.metadata_path.exists():
                        self.feature_cols = json.loads(self.cfg.metadata_path.read_text())[
                            "feature_cols"
                        ]
                    return self
            except Exception:
                pass

        self.model = joblib.load(self.cfg.model_path)
        meta = json.loads(self.cfg.metadata_path.read_text())
        self.feature_cols = meta["feature_cols"]
        self.model_version = "local-artifact"
        return self

    def read_features(self) -> "InferencePipeline":
        self.df = pd.read_parquet(self.cfg.features_path)
        self.df["date"] = pd.to_datetime(self.df["date"])
        return self

    def predict(self) -> "InferencePipeline":
        assert self.df is not None and self.model is not None
        # Score most recent data per product for next N days (simplified batch forecast)
        latest = self.df.sort_values("date").groupby("product_id").tail(1).copy()
        X = latest[self.feature_cols]
        preds = self.model.predict(X)
        latest["predicted_demand"] = np.maximum(0, preds)
        latest["lower_bound"] = latest["predicted_demand"] * 0.85
        latest["upper_bound"] = latest["predicted_demand"] * 1.15

        forecast_rows = []
        for _, row in latest.iterrows():
            for day in range(1, self.cfg.forecast_horizon_days + 1):
                forecast_rows.append(
                    {
                        "product_id": row["product_id"],
                        "forecast_date": (date.today() + timedelta(days=day)).isoformat(),
                        "predicted_demand": round(float(row["predicted_demand"]), 2),
                        "lower_bound": round(float(row["lower_bound"]), 2),
                        "upper_bound": round(float(row["upper_bound"]), 2),
                        "model_version": self.model_version,
                    }
                )
        self.forecasts = pd.DataFrame(forecast_rows)
        return self

    def write_output(self) -> "InferencePipeline":
        assert self.forecasts is not None
        ch = ClickHouseClient()
        ch.insert_df("forecasts", self.forecasts, database="demand")
        return self

    def run(self) -> pd.DataFrame:
        self.load_model().read_features().predict().write_output()
        assert self.forecasts is not None
        return self.forecasts
