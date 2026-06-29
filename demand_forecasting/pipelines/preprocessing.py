"""Preprocessing pipeline: clean, impute, remove outliers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class PreprocessingConfig:
    input_key: str | Path = "demand_forecasting/processed/unified.parquet"
    output_key: str | Path = "demand_forecasting/processed/cleaned.parquet"
    merge_on: str = "date"
    nan_strategy: str = "median"
    iqr_multiplier: float = 1.5
    input_path: Path | str | None = None
    output_path: Path | str | None = None

    def __post_init__(self):
        if self.input_path is not None:
            self.input_key = self.input_path
        if self.output_path is not None:
            self.output_key = self.output_path


class PreprocessingPipeline:
    """End-to-end preprocessing: read → impute → remove outliers → encode."""

    def __init__(self, config: PreprocessingConfig):
        self.cfg = config
        self.df: pd.DataFrame | None = None

    def read_data(self) -> "PreprocessingPipeline":
        key = self.cfg.input_key
        # Self-healing: if local Path or local file exists, read locally (for testing/dev parity)
        if isinstance(key, Path) or (isinstance(key, str) and not key.startswith("s3://") and Path(key).exists()):
            self.df = pd.read_parquet(key)
        else:
            from shared.clients import S3Client
            s3 = S3Client()
            self.df = s3.read_df(str(key))
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
        key = self.cfg.output_key
        if isinstance(key, Path) or (isinstance(key, str) and not key.startswith("s3://") and not "/" in key):
            Path(key).parent.mkdir(parents=True, exist_ok=True)
            self.df.to_parquet(key, index=False)
        else:
            from shared.clients import S3Client
            s3 = S3Client()
            s3.write_df(self.df, str(key))
        return self.df

    def run(self) -> pd.DataFrame:
        return self.read_data().preprocess_nan().remove_outliers().encode_categoricals().save()
