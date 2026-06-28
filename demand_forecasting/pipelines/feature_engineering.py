"""Feature engineering for demand forecasting: lag features, rolling stats, calendar features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class FeatureEngineeringConfig:
    input_path: Path
    output_path: Path
    target_col: str = "quantity_sold"
    lag_periods: tuple[int, ...] = (1, 7, 14, 28)
    rolling_windows: tuple[int, ...] = (7, 14, 28)


class FeatureEngineeringPipeline:
    """Lag features + rolling averages — standard for demand forecasting."""

    def __init__(self, config: FeatureEngineeringConfig):
        self.cfg = config
        self.df: pd.DataFrame | None = None

    def read_data(self) -> "FeatureEngineeringPipeline":
        self.df = pd.read_parquet(self.cfg.input_path)
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values(["product_id", "date"]).reset_index(drop=True)
        return self

    def add_calendar_features(self) -> "FeatureEngineeringPipeline":
        assert self.df is not None
        self.df["day_of_week"] = self.df["date"].dt.dayofweek
        self.df["month"] = self.df["date"].dt.month
        self.df["week_of_year"] = self.df["date"].dt.isocalendar().week.astype(int)
        self.df["is_weekend"] = (self.df["day_of_week"] >= 5).astype(int)
        return self

    def add_lag_features(self) -> "FeatureEngineeringPipeline":
        assert self.df is not None
        for lag in self.cfg.lag_periods:
            self.df[f"lag_{lag}"] = self.df.groupby("product_id")[self.cfg.target_col].shift(lag)
        return self

    def add_rolling_features(self) -> "FeatureEngineeringPipeline":
        assert self.df is not None
        for window in self.cfg.rolling_windows:
            grouped = self.df.groupby("product_id")[self.cfg.target_col]
            self.df[f"roll_mean_{window}"] = grouped.transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).mean()
            )
            self.df[f"roll_std_{window}"] = grouped.transform(
                lambda s: s.shift(1).rolling(window, min_periods=1).std()
            )
        return self

    def add_price_features(self) -> "FeatureEngineeringPipeline":
        assert self.df is not None
        if "unit_price" in self.df.columns and "discount_pct" in self.df.columns:
            self.df["effective_price"] = self.df["unit_price"] * (1 - self.df["discount_pct"] / 100)
            self.df["margin"] = self.df["unit_price"] - self.df["cost"]
        return self

    def finalize(self) -> "FeatureEngineeringPipeline":
        assert self.df is not None
        self.df = self.df.dropna().reset_index(drop=True)
        return self

    def save(self) -> pd.DataFrame:
        assert self.df is not None
        self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(self.cfg.output_path, index=False)
        return self.df

    def run(self) -> pd.DataFrame:
        return (
            self.read_data()
            .add_calendar_features()
            .add_lag_features()
            .add_rolling_features()
            .add_price_features()
            .finalize()
            .save()
        )
