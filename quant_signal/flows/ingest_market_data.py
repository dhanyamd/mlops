"""Orchestrated market-data ingestion: provider → contract gate → Bronze.

Run inline (no Prefect server needed)::

    make ingest                      # real daily bars via default provider/symbols
    uv run python flows/ingest_market_data.py --provider yahoo --days 365
    uv run python flows/ingest_market_data.py --provider binance --symbols BTCUSDT --days 3

Providers (all REAL, keyless, verified live):
    yahoo    — US equity daily OHLCV (unofficial endpoint, research-grade)
    binance  — crypto minute OHLCV (single venue)
    synthetic— OFFLINE/TEST ONLY, never a production source

As a deployment it would run on a Prefect work pool with retries configured
here via task decorators. Idempotency comes from the MERGE upsert on the
natural key (symbol, timeframe, ts) — re-running a failed batch is safe.
"""

from __future__ import annotations

import pandas as pd
from prefect import flow, task

from config.logging import configure_logging, get_logger
from config.settings import Settings, csv_list, get_settings
from ingest.providers.base import BarProvider
from ingest.providers.binance import BinanceBarProvider
from ingest.providers.synthetic import SyntheticBarProvider
from ingest.providers.yahoo import YahooBarProvider
from ingest.quality import validate_bars
from ingest.store import write_crypto_bars, write_equity_bars, write_quarantine

log = get_logger("flows.ingest_market_data")

_PROVIDERS: dict[str, type[BarProvider]] = {
    "yahoo": YahooBarProvider,
    "binance": BinanceBarProvider,
    "synthetic": SyntheticBarProvider,
}


def _build_provider(provider_name: str, settings: Settings) -> BarProvider:
    provider_cls = _PROVIDERS.get(provider_name)
    if provider_cls is None:
        raise ValueError(
            f"unknown bar provider: {provider_name!r} (choose from {sorted(_PROVIDERS)})"
        )
    if provider_name == "yahoo":
        return YahooBarProvider(cache_dir=settings.yahoo_cache_dir)
    return provider_cls()


def _default_symbols(provider_name: str, settings: Settings) -> list[str]:
    if provider_name == "binance":
        return csv_list(settings.ingest_default_crypto_symbols)
    return csv_list(settings.ingest_default_symbols)


@task(name="fetch-bars", retries=2, retry_delay_seconds=5)
def fetch_bars(provider_name: str, symbols: list[str], days: int) -> pd.DataFrame:
    provider = _build_provider(provider_name, get_settings())
    return provider.fetch_bars(symbols, days)


@task(name="validate-bars")
def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    good, bad = validate_bars(df)
    log.info(
        "validation_gate",
        rows=len(df),
        valid=len(good),
        invalid=len(bad),
    )
    return good, bad


@task(name="write-bronze", retries=2, retry_delay_seconds=5)
def write_bronze(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return write_equity_bars(df, get_settings())


@task(name="write-bronze-crypto", retries=2, retry_delay_seconds=5)
def write_bronze_crypto(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return write_crypto_bars(df, get_settings())


@task(name="quarantine-invalid")
def quarantine(df: pd.DataFrame, source: str) -> int:
    if df.empty:
        return 0
    return write_quarantine(df, source, get_settings())


@flow(name="ingest-market-data")
def ingest_market_data(
    provider_name: str | None = None,
    symbols: list[str] | None = None,
    days: int | None = None,
) -> dict[str, int]:
    """Fetch, validate, land valid bars in Bronze and invalid ones in QUARANTINE.

    Anti-pollution rule: Binance (crypto) lands in CRYPTO_BARS, equity providers
    (yahoo/synthetic) land in EQUITY_BARS — never mixed in one table.
    Every default (provider, symbols, days) comes from Settings/env, never
    hardcoded here.
    """
    configure_logging()
    settings = get_settings()
    provider_name = provider_name or settings.ingest_default_provider
    if symbols is None:
        symbols = _default_symbols(provider_name, settings)
    days = days or settings.ingest_default_days
    raw = fetch_bars(provider_name, symbols, days)
    good, bad = validate(raw)
    if provider_name == "binance":
        written = write_bronze_crypto(good)
        source = "crypto_bars"
    else:
        written = write_bronze(good)
        source = "equity_bars"
    quarantined = quarantine(bad, source)
    result = {"fetched": int(len(raw)), "written": written, "quarantined": quarantined}
    log.info("ingest_market_data_complete", provider=provider_name, **result)
    return result


if __name__ == "__main__":
    import argparse

    configure_logging()
    parser = argparse.ArgumentParser(
        description="Land real market bars into Bronze (defaults from Settings/env)."
    )
    parser.add_argument("--provider", dest="provider_name", default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()
    ingest_market_data(
        provider_name=args.provider_name,
        symbols=args.symbols,
        days=args.days,
    )
