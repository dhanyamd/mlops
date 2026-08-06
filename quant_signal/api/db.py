"""Read-only data access for the dashboard API.

Every function returns JSON-serializable records from the live SILVER/GOLD
layers via ``SnowflakeClient.query_df``. Tables are referenced fully qualified
(no shared ``snowflake_schema`` mutation — that would race under concurrent
requests), and all values are bound parameters. No business values are
hardcoded here: instrument lists come from ``INGEST_DEFAULT_TICKERS``.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from config.settings import csv_list, get_settings
from db.snowflake import SnowflakeClient, _q

_client: SnowflakeClient | None = None


def _get_client() -> SnowflakeClient:
    """One client per process; connections are opened per query (cheap, auto-closed)."""
    global _client
    if _client is None:
        _client = SnowflakeClient()
    return _client


def _qualified(table: str, schema: str) -> str:
    """``"DB"."SCHEMA"."TABLE"`` from settings — never depends on session schema."""
    database = _q(get_settings().snowflake_database)
    return f"{database}.{_q(schema)}.{_q(table)}"


def _records(df: pd.DataFrame) -> list[dict]:
    """NaN/NaT → None, datetimes → ISO strings, so rows are JSON-safe."""
    out: list[dict] = []
    for row in df.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or value != value:  # None, NaN, NaT
                clean[str(key)] = None
            elif isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
                clean[str(key)] = value.isoformat()
            elif isinstance(value, (int, float, str, bool)):
                clean[str(key)] = value
            else:
                clean[str(key)] = str(value)
        out.append(clean)
    return out


def default_tickers() -> list[str]:
    """Instruments the pipeline manages — the UI dropdown source (env-driven)."""
    return csv_list(get_settings().ingest_default_tickers)


def default_metrics() -> list[str]:
    """US-GAAP metrics extracted from EDGAR — UI metric dropdown (env-driven)."""
    return csv_list(get_settings().ingest_default_metrics)


def default_macro_series() -> list[str]:
    """FRED series the pipeline tracks — UI series dropdown (env-driven)."""
    return csv_list(get_settings().ingest_default_macro_series)


def market_bars(symbol: str, days: int = 750) -> list[dict]:
    """Daily OHLCV for one symbol, oldest→newest, last ``days`` trading days."""
    client = _get_client()
    qual = _qualified("GOLD_DAILY_BARS", "GOLD")
    df = client.query_df(
        f"""
        SELECT TRADE_DATE, DAY_OPEN, DAY_HIGH, DAY_LOW, DAY_CLOSE, VOLUME
        FROM {qual}
        WHERE TIMEFRAME = '1D' AND SYMBOL = %s
        ORDER BY TRADE_DATE DESC
        LIMIT %s
        """,
        (symbol.upper(), days),
    )
    return _records(df.iloc[::-1])


def fundamentals(ticker: str, metric: str | None = None) -> list[dict]:
    """Point-in-time US-GAAP facts as filed (filed_at order is the as-known order)."""
    client = _get_client()
    qual = _qualified("SILVER_COMPANY_FACTS", "SILVER")
    sql = f"""
        SELECT TICKER, METRIC, FISCAL_YEAR, VALUE, UNIT, FILED_AT
        FROM {qual}
        WHERE TICKER = %s
    """
    params: list[Any] = [ticker.upper()]
    if metric:
        sql += " AND METRIC = %s"
        params.append(metric)
    sql += " ORDER BY METRIC, FISCAL_YEAR, FILED_AT"
    return _records(client.query_df(sql, tuple(params)))


def macro_series(series_id: str | None = None, limit: int = 500) -> list[dict]:
    """FRED macro values, oldest→newest (series ID filter optional)."""
    client = _get_client()
    qual = _qualified("SILVER_FRED_MACRO", "SILVER")
    sql = f"""
        SELECT SERIES_ID, DATE, VALUE
        FROM {qual}
    """
    params: list[Any] = []
    if series_id:
        sql += " WHERE SERIES_ID = %s"
        params.append(series_id.upper())
    sql += " ORDER BY DATE DESC LIMIT %s"
    params.append(limit)
    df = client.query_df(sql, tuple(params))
    return _records(df.iloc[::-1])


def pipeline_metrics(flow: str | None = None, limit: int = 100) -> list[dict]:
    """Latest telemetry rows, grouped by run (stages chronological, runs newest-last)."""
    client = _get_client()
    qual = _qualified("SILVER_PIPELINE_METRICS", "SILVER")
    sql = f"""
        SELECT * FROM (
            SELECT RUN_ID, FLOW, STAGE, STARTED_AT, ELAPSED_MS, N_ROWS, LOADED_AT
            FROM {qual}
    """
    params: list[Any] = []
    if flow:
        sql += " WHERE FLOW = %s"
        params.append(flow)
    sql += " ORDER BY STARTED_AT DESC, STAGE ASC LIMIT %s) ORDER BY STARTED_AT ASC, STAGE ASC"
    params.append(limit)
    return _records(client.query_df(sql, tuple(params)))
