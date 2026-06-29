"""Batch ETL for historical fraud data (Snowflake/ClickHouse → Feature Store offline).

Quality gates wired in:
  1. After extract: validate_fraud_raw() — checks nulls, ranges, uniqueness
  2. After transform: validate_fraud_features() — checks computed velocity features
  If either gate fails → DataQualityError → pipeline halts → Prometheus metric = 0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from fraud_detection.data.generate_data import compute_velocity_features
from shared.config import POSTGRES
from shared.data_quality.quality import (
    DataQualityError,
    validate_fraud_features,
    validate_fraud_raw,
)
from shared.observability.logging import get_logger

log = get_logger(__name__)


@dataclass
class BatchETLConfig:
    output_key: str = "fraud_detection/offline/transaction_features.parquet"
    raw_key: str = "fraud_detection/offline/raw_transactions.parquet"


class BatchETLPipeline:
    """Historical data ETL: warehouse + labels → quality gate → velocity features → quality gate → offline store."""

    def __init__(self, config: BatchETLConfig):
        self.cfg = config

    def extract_from_warehouse(self) -> pd.DataFrame:
        from shared.clients import S3Client
        s3 = S3Client()
        log.info("extracting_raw_transactions_from_s3", key=self.cfg.raw_key)
        return s3.read_df(self.cfg.raw_key)

    def transform(self, txns: pd.DataFrame) -> pd.DataFrame:
        if "merchant_category" not in txns.columns:
            txns["merchant_category"] = "unknown"
        df = compute_velocity_features(txns)
        return df.rename(columns={"user_id": "card_id"})

    def load(self, features: pd.DataFrame) -> str:
        from shared.clients import S3Client
        s3 = S3Client()
        log.info("uploading_features_to_s3", key=self.cfg.output_key, rows=len(features))
        s3.write_df(features, self.cfg.output_key)
        return self.cfg.output_key

    def run(self) -> str:
        # ── Extract ──────────────────────────────────────────────────────────
        txns = self.extract_from_warehouse()

        # ── Quality Gate 1: raw data validation ─────────────────────────────
        raw_result = validate_fraud_raw(txns)
        if not raw_result.passed:
            raise DataQualityError(
                f"Fraud ETL halted — raw data quality check failed:\n{raw_result.summary}"
            )

        # ── Transform ────────────────────────────────────────────────────────
        features = self.transform(txns)

        # ── Quality Gate 2: computed features validation ────────────────────
        feat_result = validate_fraud_features(features)
        if not feat_result.passed:
            raise DataQualityError(
                f"Fraud ETL halted — feature quality check failed:\n{feat_result.summary}"
            )

        # ── Load ─────────────────────────────────────────────────────────────
        return self.load(features)


