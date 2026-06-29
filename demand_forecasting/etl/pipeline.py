"""ETL: ClickHouse (warehouse) + PostgreSQL (catalog) → unified training dataset.

Quality gates wired in:
  1. After extract_sales: validate raw sales data
  2. After transform: validate merged/unified data
  If either gate fails → DataQualityError → pipeline halts
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from shared.clients import ClickHouseClient
from shared.data_quality.quality import DataQualityError, validate_demand_raw
from shared.db.models import ExternalDaily, Product, Promotion
from shared.db.session import get_session
from shared.observability.logging import get_logger

log = get_logger(__name__)


@dataclass
class ETLConfig:
    output_key: str = "demand_forecasting/processed/unified.parquet"


class ETLPipeline:
    """
    Production ETL pattern:
      ClickHouse (OLAP sales) + PostgreSQL (OLTP catalog/promotions/external)
      → unified feature-ready dataset
    """

    def __init__(self, config: ETLConfig):
        self.cfg = config
        self.ch = ClickHouseClient()
        self.unified: pd.DataFrame | None = None

    def extract_sales(self) -> "ETLPipeline":
        if not self.ch.ping():
            raise ConnectionError("ClickHouse required — run: make infra-up")
        self.sales = self.ch.query_df(
            """
            SELECT sale_date AS date, store_id, item_id, product_id,
                   quantity_sold, revenue
            FROM demand.sales
            ORDER BY product_id, sale_date
            """
        )
        self.sales["date"] = pd.to_datetime(self.sales["date"])
        return self

    def extract_catalog(self, session: Session) -> "ETLPipeline":
        from sqlalchemy import select

        self.products = pd.read_sql(select(Product), session.bind)
        self.promotions = pd.read_sql(select(Promotion), session.bind)
        self.external = pd.read_sql(select(ExternalDaily), session.bind)
        self.external = self.external.rename(columns={"sale_date": "date"})
        self.external["date"] = pd.to_datetime(self.external["date"])
        return self

    def transform(self) -> "ETLPipeline":
        df = self.sales.merge(self.products, on="product_id", how="left", suffixes=("", "_cat"))
        df = df.merge(self.external, on="date", how="left")

        promo_flags = []
        for _, row in df.iterrows():
            active = self.promotions[
                (self.promotions["product_id"] == row["product_id"])
                & (self.promotions["start_date"] <= row["date"].date())
                & (self.promotions["end_date"] >= row["date"].date())
            ]
            promo_flags.append(
                {
                    "has_promotion": len(active) > 0,
                    "discount_pct": float(active["discount_pct"].max()) if len(active) else 0.0,
                    "channel": active["channel"].iloc[0] if len(active) else "none",
                }
            )
        self.unified = pd.concat([df.reset_index(drop=True), pd.DataFrame(promo_flags)], axis=1)
        return self

    def load(self) -> pd.DataFrame:
        assert self.unified is not None
        from shared.clients import S3Client
        s3 = S3Client()
        log.info("uploading_unified_dataset_to_s3", key=self.cfg.output_key, rows=len(self.unified))
        s3.write_df(self.unified, self.cfg.output_key)
        return self.unified

    def run(self) -> pd.DataFrame:
        session = get_session()
        try:
            self.extract_sales()

            # ── Quality Gate: validate raw sales from warehouse ─────────────
            raw_result = validate_demand_raw(self.sales)
            if not raw_result.passed:
                raise DataQualityError(
                    f"Demand ETL halted — raw data quality check failed:\n{raw_result.summary}"
                )

            self.extract_catalog(session).transform()
            log.info("demand_etl_complete", rows=len(self.unified))
            return self.load()
        finally:
            session.close()
