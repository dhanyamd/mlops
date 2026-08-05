"""Fundamentals ingestion from SEC EDGAR (no API key, real data).

Run::

    uv run python flows/ingest_fundamentals.py --tickers AAPL MSFT

Set EDGAR_USER_AGENT in .env (SEC requires a descriptive User-Agent).
Default tickers come from Settings/env (INGEST_DEFAULT_TICKERS).
"""

from __future__ import annotations

import pandas as pd
from prefect import flow, task

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from ingest.providers.sec_edgar import EdgarFundamentalsProvider
from ingest.quality import validate_facts
from ingest.store import write_company_facts, write_quarantine

log = get_logger("flows.ingest_fundamentals")


@task(name="fetch-company-facts", retries=2, retry_delay_seconds=10)
def fetch_facts(tickers: list[str]) -> pd.DataFrame:
    return EdgarFundamentalsProvider().fetch_facts(tickers)


@task(name="validate-facts")
def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    good, bad = validate_facts(df)
    log.info("validation_gate", rows=len(df), valid=len(good), invalid=len(bad))
    return good, bad


@task(name="write-bronze-facts", retries=2, retry_delay_seconds=5)
def write_bronze(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return write_company_facts(df, get_settings())


@flow(name="ingest-fundamentals")
def ingest_fundamentals(tickers: list[str] | None = None) -> dict[str, int]:
    configure_logging()
    if tickers is None:
        tickers = csv_list(get_settings().ingest_default_tickers)
    raw = fetch_facts(tickers)
    good, bad = validate(raw)
    written = write_bronze(good)
    quarantined = write_quarantine(bad, "company_facts", get_settings())
    result = {
        "fetched": int(len(raw)),
        "written": written,
        "quarantined": quarantined,
    }
    log.info("ingest_fundamentals_complete", **result)
    return result


if __name__ == "__main__":
    import argparse

    configure_logging()
    parser = argparse.ArgumentParser(
        description="Land real SEC EDGAR fundamentals into Bronze (defaults from Settings/env)."
    )
    parser.add_argument("--tickers", nargs="+", default=None)
    args = parser.parse_args()
    ingest_fundamentals(tickers=args.tickers)
