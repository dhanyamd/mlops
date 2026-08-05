"""Bronze + QUARANTINE writers. Thin, testable wrappers over SnowflakeClient.

Every write is idempotent (upsert on the natural key) and every query carries
the configured query_tag for cost attribution in QUERY_HISTORY.

Medallion discipline: one Bronze table per ASSET CLASS — never mix equities,
crypto, macro, or fundamentals in a single raw table (that is how datasets
pollute). Cross-source joins happen only in Silver/Gold with explicit contracts.
"""

from __future__ import annotations

import pandas as pd

from config.settings import Settings, get_settings
from db.snowflake import SnowflakeClient

BAR_TABLE = "EQUITY_BARS"
CRYPTO_TABLE = "CRYPTO_BARS"
FACTS_TABLE = "COMPANY_FACTS"
MACRO_TABLE = "FRED_MACRO"

# Natural keys per table. A row is unique per (asset, timeframe, timestamp).
_BAR_KEYS = ["symbol", "timeframe", "ts"]


def _client(settings: Settings | None) -> SnowflakeClient:
    return SnowflakeClient(settings or get_settings())


def write_equity_bars(df: pd.DataFrame, settings: Settings | None = None) -> int:
    client = _client(settings)
    return client.upsert_df(
        df,
        BAR_TABLE,
        merge_keys=_BAR_KEYS,
        schema=client._settings.snowflake_schema,
    )


def write_crypto_bars(df: pd.DataFrame, settings: Settings | None = None) -> int:
    client = _client(settings)
    return client.upsert_df(
        df,
        CRYPTO_TABLE,
        merge_keys=_BAR_KEYS,
        schema=client._settings.snowflake_schema,
    )


def write_company_facts(df: pd.DataFrame, settings: Settings | None = None) -> int:
    client = _client(settings)
    return client.upsert_df(
        df,
        FACTS_TABLE,
        merge_keys=["ticker", "metric", "fiscal_year", "filed_at"],
        schema=client._settings.snowflake_schema,
    )


def write_macro(df: pd.DataFrame, settings: Settings | None = None) -> int:
    client = _client(settings)
    return client.upsert_df(
        df,
        MACRO_TABLE,
        merge_keys=["series_id", "date"],
        schema=client._settings.snowflake_schema,
    )


def write_quarantine(df: pd.DataFrame, source: str, settings: Settings | None = None) -> int:
    """Persist rows that failed the contract. Never dropped, never in Silver."""
    if df.empty:
        return 0
    client = SnowflakeClient(settings or get_settings())
    return client.insert_df(
        df,
        f"QUARANTINE_{source}",
        schema=client._settings.snowflake_quarantine_schema,
    )
