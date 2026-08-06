"""Macro ingestion from FRED (keyless public CSV, real St. Louis Fed data).

Run inline::

    uv run python flows/ingest_macro_data.py [--series VIXCLS CPIAUCSL]

Default series come from Settings/env (INGEST_DEFAULT_MACRO_SERIES).

Lands daily/monthly macro observations (VIX, CPI, yields, fed funds,
unemployment) into Bronze ``FRED_MACRO``, quarantining anything that breaks
the contract. Idempotent: MERGE upsert on the natural key (series_id, date).
"""

from __future__ import annotations

import pandas as pd
from prefect import flow, task

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from ingest.metrics import PipelineMetrics
from ingest.providers.fred import FredProvider
from ingest.quality import validate_macro
from ingest.store import write_macro, write_quarantine

log = get_logger("flows.ingest_macro_data")


@task(name="fetch-macro", retries=2, retry_delay_seconds=10)
def fetch_macro(series_ids: list[str]) -> pd.DataFrame:
    return FredProvider().fetch_observations(series_ids)


@task(name="validate-macro")
def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    good, bad = validate_macro(df)
    log.info("validation_gate", rows=len(df), valid=len(good), invalid=len(bad))
    return good, bad


@task(name="write-bronze-macro", retries=2, retry_delay_seconds=5)
def write_bronze(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return write_macro(df, get_settings())


@task(name="quarantine-invalid")
def quarantine(df: pd.DataFrame, source: str) -> int:
    if df.empty:
        return 0
    return write_quarantine(df, source, get_settings())


@flow(name="ingest-macro-data")
def ingest_macro_data(series_ids: list[str] | None = None) -> dict[str, int]:
    """Land real FRED macro series into Bronze (defaults from Settings/env)."""
    configure_logging()
    settings = get_settings()
    ids = series_ids or csv_list(settings.ingest_default_macro_series)
    metrics = PipelineMetrics(flow="ingest-macro-data")
    with metrics.stage("fetch"):
        raw = fetch_macro(ids)
    with metrics.stage("validate", rows=len(raw)):
        good, bad = validate(raw)
    with metrics.stage("write-bronze", rows=len(good)):
        written = write_bronze(good)
    with metrics.stage("quarantine", rows=len(bad)):
        quarantined = quarantine(bad, "fred_macro")
    result = {
        "fetched": int(len(raw)),
        "written": written,
        "quarantined": quarantined,
    }
    metrics.flush()
    log.info("ingest_macro_data_complete", **result)
    return result


if __name__ == "__main__":
    import argparse

    configure_logging()
    parser = argparse.ArgumentParser(
        description="Land real FRED macro series into Bronze (defaults from Settings/env)."
    )
    parser.add_argument("--series", dest="series_ids", nargs="+", default=None)
    args = parser.parse_args()
    ingest_macro_data(series_ids=args.series_ids)
