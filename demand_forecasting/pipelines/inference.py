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
    features_key: str | Path = "demand_forecasting/features/features.parquet"
    model_path: Path = Path("artifacts/demand_forecasting/model.joblib")
    metadata_path: Path = Path("artifacts/demand_forecasting/metadata.json")
    model_name: str = "demand_forecast_model"
    forecast_horizon_days: int = 14
    output_table: str = "demand_forecasting.forecasts"
    use_mlflow: bool = True
    features_path: Path | str | None = None

    def __post_init__(self):
        if self.features_path is not None:
            self.features_key = self.features_path


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
        key = self.cfg.features_key
        # Self-healing: if local Path or local file exists, read locally
        if isinstance(key, Path) or (isinstance(key, str) and not key.startswith("s3://") and Path(key).exists()):
            self.df = pd.read_parquet(key)
        else:
            from shared.clients import S3Client
            s3 = S3Client()
            self.df = s3.read_df(str(key))
        self.df["date"] = pd.to_datetime(self.df["date"])
        return self

    def predict(self) -> "InferencePipeline":
        assert self.df is not None and self.model is not None
        # Score most recent data per product for next N days (simplified batch forecast)
        latest = self.df.sort_values("date").groupby("product_id").tail(1).copy()
        try:
            # Try predicting with full dataframe (required for Prophet/PyFunc wraps needing 'date')
            preds = self.model.predict(latest)
        except Exception:
            # Fallback to feature matrix (for standard XGBoost/LightGBM models)
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

        # ── Hierarchical Reconciliation (Pillar 2) ───────────────────────────
        from demand_forecasting.pipelines.reconciliation import HierarchicalReconciler
        reconciler = HierarchicalReconciler(n_products=len(latest))
        
        # We reconcile bottom level forecasts for each day in the horizon
        for day in range(1, self.cfg.forecast_horizon_days + 1):
            day_str = (date.today() + timedelta(days=day)).isoformat()
            
            # Extract forecasts for this day
            day_preds = np.array([r["predicted_demand"] for r in forecast_rows if r["forecast_date"] == day_str])
            
            # Simulate a top-level independent forecast (e.g., aggregate model would predict total)
            simulated_top_forecast = float(np.sum(day_preds) * 0.96)  # 4% mismatch
            
            # Perform OLS reconciliation
            reconciled_total, reconciled_products = reconciler.reconcile_ols(
                top_forecast=simulated_top_forecast,
                product_forecasts=day_preds
            )
            
            # Append aggregate 'TOTAL' forecast row
            forecast_rows.append(
                {
                    "product_id": "TOTAL",
                    "forecast_date": day_str,
                    "predicted_demand": round(reconciled_total, 2),
                    "lower_bound": round(reconciled_total * 0.85, 2),
                    "upper_bound": round(reconciled_total * 1.15, 2),
                    "model_version": f"{self.model_version}-reconciled-ols",
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
