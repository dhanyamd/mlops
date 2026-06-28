"""Batch ETL for historical fraud data (Snowflake/ClickHouse → Feature Store offline)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from fraud_detection.data.generate_data import compute_velocity_features
from shared.config import POSTGRES


@dataclass
class BatchETLConfig:
    offline_dir: Path


class BatchETLPipeline:
    """Historical data ETL: warehouse + labels → velocity features → offline store."""

    def __init__(self, config: BatchETLConfig):
        self.cfg = config

    def extract_from_warehouse(self) -> pd.DataFrame:
        path = self.cfg.offline_dir / "raw_transactions.parquet"
        if path.exists():
            return pd.read_parquet(path)

        engine = create_engine(POSTGRES.url)
        txns = pd.read_sql(
            text(
                """
                SELECT t.transaction_id, t.user_id, t.amount, t.is_fraud,
                       NOW() as event_timestamp, NOW() as created_timestamp
                FROM fraud_detection.transaction_labels t
                """
            ),
            engine,
        )
        return txns

    def transform(self, txns: pd.DataFrame) -> pd.DataFrame:
        if "merchant_category" not in txns.columns:
            txns["merchant_category"] = "unknown"
        return compute_velocity_features(txns)

    def load(self, features: pd.DataFrame) -> Path:
        out = self.cfg.offline_dir / "transaction_features.parquet"
        features.to_parquet(out, index=False)
        return out

    def run(self) -> Path:
        txns = self.extract_from_warehouse()
        features = self.transform(txns)
        return self.load(features)
