"""Preprocessing pipeline: clean, impute, remove outliers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class PreprocessingConfig:
    input_path: Path
    output_path: Path
    merge_on: str = "date"
    nan_strategy: str = "median"
    iqr_multiplier: float = 1.5


class PreprocessingPipeline:
    """End-to-end preprocessing: read → impute → remove outliers → encode."""

    def __init__(self, config: PreprocessingConfig):
        self.cfg = config
        self.df: pd.DataFrame | None = None

    def read_data(self) -> "PreprocessingPipeline":
        self.df = pd.read_parquet(self.cfg.input_path)
        self.df["date"] = pd.to_datetime(self.df["date"])
        return self

    def preprocess_nan(self) -> "PreprocessingPipeline":
        assert self.df is not None
        numeric = self.df.select_dtypes(include="number").columns
        if self.cfg.nan_strategy == "median":
            self.df[numeric] = self.df[numeric].fillna(self.df[numeric].median())
        else:
            self.df[numeric] = self.df[numeric].fillna(self.df[numeric].mean())
        self.df["has_promotion"] = self.df["has_promotion"].fillna(False).astype(int)
        return self

    def remove_outliers(self) -> "PreprocessingPipeline":
        assert self.df is not None
        target_col = "quantity_sold"
        q1, q3 = self.df[target_col].quantile([0.25, 0.75])
        iqr = q3 - q1
        k = self.cfg.iqr_multiplier
        mask = self.df[target_col].between(q1 - k * iqr, q3 + k * iqr)
        self.df = self.df[mask].reset_index(drop=True)
        return self

    def encode_categoricals(self) -> "PreprocessingPipeline":
        assert self.df is not None
        categorical_cols = [c for c in ["category", "channel"] if c in self.df.columns]
        if categorical_cols:
            self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
        return self

    def save(self) -> pd.DataFrame:
        assert self.df is not None
        self.cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(self.cfg.output_path, index=False)
        return self.df

    def run(self) -> pd.DataFrame:
        return self.read_data().preprocess_nan().remove_outliers().encode_categoricals().save()
