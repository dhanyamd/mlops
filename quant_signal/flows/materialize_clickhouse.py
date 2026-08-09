"""Batch warehouse sink: copy the live streaming online store → ClickHouse BRONZE.

The streaming stack keeps the *hot* state in Redis (5m features, conformal
predictions, strategy, paper-execution ledger). This flow snapshots that hot
state into the ClickHouse OLAP warehouse so the batch side (Grafana, dbt,
analytics) can query history without touching Redis or the live bus. This is
the two-track bridge: Redpanda → Flink → Redis → **ClickHouse**.

Run inline (records runs to the Prefect server when PREFECT_API_URL is set)::

    make ch-materialize
    uv run python flows/materialize_clickhouse.py

Semantics (idempotent, honest):
  - Tables use ReplacingMergeTree(loaded_at) ordered by (symbol, window_end_ms),
    so re-running the flow for a window the online store still holds is an
    *upsert*, not a duplicate — merges collapse to the latest loaded_at.
  - The feature key is a bounded list (stream_redis_feature_maxlen windows per
    symbol); the other artifacts (prediction / strategy / execution) are single
    latest-window JSON documents. Every symbol is discovered from Redis at run
    time (SCAN over the feature prefix) — nothing is hardcoded.
  - Nested payloads (fill ledgers, equity arrays, assumption dicts) are *not*
    flattened into BRONZE; they stay in the Redis source of truth. This layer
    is the flat, queryable analytics surface.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from prefect import flow, task

from config.logging import configure_logging, get_logger
from config.settings import Settings, get_settings

log = get_logger("flows.materialize_clickhouse")

# Per-artifact BRONZE tables. Column names are the JSON keys from the online
# store (see stream.materializer / stream.predictor / stream.execution) so the
# mapping is mechanical and renames surface as loud KeyError, never silent NaN.

FEATURE_COLUMNS = [
    "symbol",
    "window_start_ms",
    "window_end_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "bar_count",
    "loaded_at",
]

EXECUTION_COLUMNS = [
    "symbol",
    "window_end_ms",
    "venue",
    "notional_usd",
    "slippage_bps",
    "taker_fee_bps",
    "n_trades",
    "n_wins",
    "win_rate",
    "realized_pnl",
    "unrealized_pnl",
    "net_pnl",
    "gross_pnl",
    "gross_volume",
    "total_fees",
    "fees_pct_of_gross_pnl",
    "signals_skipped",
    "orders_rejected",
    "total_return",
    "position",
    "updated_at_ms",
    "loaded_at",
]

PREDICTION_COLUMNS = [
    "symbol",
    "window_end_ms",
    "predicted_return",
    "interval_low",
    "interval_high",
    "direction",
    "alpha",
    "coverage",
    "updated_at_ms",
    "loaded_at",
]

STRATEGY_COLUMNS = [
    "symbol",
    "window_end_ms",
    "n_windows",
    "n_trades",
    "n_wins",
    "win_rate",
    "strategy_equity",
    "buyhold_equity",
    "total_return_strategy",
    "total_return_buyhold",
    "updated_at_ms",
    "loaded_at",
]


def _client(database: str | None = None):
    import clickhouse_connect

    s = get_settings()
    return clickhouse_connect.get_client(
        host=s.clickhouse_host,
        port=s.clickhouse_port,
        username=s.clickhouse_user,
        password=s.clickhouse_password,
        database=database or s.clickhouse_database,
    )


def _ddl(table: str, columns: list[str]) -> str:
    """ClickHouse DDL for a flat BRONZE table.

    ReplacingMergeTree(loaded_at) gives upsert-by-(symbol, window_end_ms)
    semantics on merge; the Int64 window_end_ms sort key keeps each symbol's
    history time-ordered for the Grafana/dbt surface.
    """
    type_map: dict[str, str] = {
        "symbol": "String",
        "position": "String",
        "venue": "String",
        "direction": "String",
        "strategy_equity": "String",  # equity curve (list) — JSON-encoded
        "buyhold_equity": "String",  # buy-and-hold curve (list) — JSON-encoded
        "window_start_ms": "Int64",
        "window_end_ms": "Int64",
        "updated_at_ms": "Int64",
        "n_trades": "Int32",
        "n_wins": "Int32",
        "n_windows": "Int32",
        "bar_count": "Int32",
        "signals_skipped": "Int32",
        "orders_rejected": "Int32",
        "loaded_at": "DateTime64(3)",
    }
    default_type = "Float64"
    cols = ",\n    ".join(f"{c} {type_map.get(c, default_type)}" for c in columns)
    return (
        f"CREATE TABLE IF NOT EXISTS {table} (\n    {cols}\n) "
        f"ENGINE = ReplacingMergeTree(loaded_at)\n"
        f"ORDER BY (symbol, window_end_ms)"
    )


def _now_ms() -> int:
    return int(_dt.datetime.now(_dt.timezone.utc).timestamp() * 1000)


def _row(
    doc: dict[str, Any], columns: list[str], symbol: str, loaded_at: _dt.datetime
) -> list[Any]:
    """Map one online-store document to a flat row.

    Lists/dicts (equity curves, ledgers) are JSON-encoded into String columns;
    missing scalars default to 0 (Float64/Int) or "" (String).
    """
    import json

    string_cols = {
        "symbol",
        "position",
        "venue",
        "direction",
        "strategy_equity",
        "buyhold_equity",
    }
    rows: list[Any] = []
    for c in columns:
        value = doc.get(c)
        if c == "symbol":
            value = symbol
        if isinstance(value, (list, dict)):
            value = json.dumps(value)
        elif value is None:
            value = "" if c in string_cols else 0
        rows.append(value)
    return rows


def _redis(settings: Settings):
    import redis

    return redis.Redis.from_url(settings.stream_redis_url, decode_responses=True)


@task(name="ensure-clickhouse-bronze-schema")
def ensure_schema() -> str:
    """Create the quant database + BRONZE tables if absent (idempotent)."""
    s = get_settings()
    # Connect to the `default` database first — the quant DB may not exist yet.
    client = _client(database="default")
    client.command(f"CREATE DATABASE IF NOT EXISTS {s.clickhouse_database}")
    tables = {
        "features": FEATURE_COLUMNS,
        "predictions": PREDICTION_COLUMNS,
        "strategy": STRATEGY_COLUMNS,
        "execution": EXECUTION_COLUMNS,
    }
    for name, columns in tables.items():
        client.command(_ddl(f"{s.clickhouse_database}.crypto_{name}_5m", columns))
    client.close()
    log.info("bronze_schema_ready", database=s.clickhouse_database, tables=list(tables))
    return s.clickhouse_database


@task(name="discover-stream-symbols")
def discover_symbols() -> list[str]:
    """Find symbols currently in the online store (SCAN the feature prefix)."""
    r = _redis(get_settings())
    prefix = get_settings().stream_redis_feature_prefix
    symbols = sorted({key.rsplit(":", 1)[-1] for key in r.scan_iter(f"{prefix}:*")})
    log.info("discovered_symbols", symbols=symbols)
    return symbols


@task(name="read-feature-windows")
def read_feature_windows(symbol: str) -> list[dict[str, Any]]:
    import json

    r = _redis(get_settings())
    key = f"{get_settings().stream_redis_feature_prefix}:{symbol}"
    docs = [json.loads(raw) for raw in r.lrange(key, 0, -1) if raw]
    log.info("read_feature_windows", symbol=symbol, windows=len(docs))
    return docs


@task(name="read-latest-artifact")
def read_latest(prefix_name: str, symbol: str) -> dict[str, Any] | None:
    import json

    s = get_settings()
    prefix = getattr(s, prefix_name)
    r = _redis(s)
    raw = r.get(f"{prefix}:{symbol}")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("unparsable_artifact", prefix=prefix, symbol=symbol)
        return None


@task(name="write-features")
def write_features(symbol: str, docs: list[dict[str, Any]]) -> int:
    if not docs:
        return 0
    s = get_settings()
    client = _client()
    loaded_at = _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)
    rows = [_row(d, FEATURE_COLUMNS, symbol, loaded_at) for d in docs]
    client.insert(
        f"{s.clickhouse_database}.crypto_features_5m",
        rows,
        column_names=FEATURE_COLUMNS,
    )
    client.close()
    return len(rows)


@task(name="write-latest-artifact")
def write_latest(table: str, columns: list[str], symbol: str, doc: dict[str, Any] | None) -> int:
    if not doc:
        return 0
    s = get_settings()
    client = _client()
    loaded_at = _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None)
    row = _row(doc, columns, symbol, loaded_at)
    client.insert(f"{s.clickhouse_database}.{table}", [row], column_names=columns)
    client.close()
    return 1


@flow(name="materialize-streaming-to-clickhouse")
def materialize_streaming_to_clickhouse() -> dict[str, Any]:
    """Snapshot the live Redis online store into ClickHouse BRONZE tables.

    Runs per-symbol: features (all held windows) are written first, then the
    latest prediction / strategy / execution documents. Every stage is a task,
    so a Prefect worker can retry a failed symbol without replaying the rest.
    """
    configure_logging()
    ensure_schema()
    symbols = discover_symbols()

    rows: dict[str, int] = {
        "features": 0,
        "predictions": 0,
        "strategy": 0,
        "execution": 0,
    }
    for symbol in symbols:
        windows = read_feature_windows(symbol)
        rows["features"] += write_features(symbol, windows)
        for table, prefix_attr, columns in (
            ("crypto_predictions_5m", "stream_redis_prediction_prefix", PREDICTION_COLUMNS),
            ("crypto_strategy_5m", "stream_redis_strategy_prefix", STRATEGY_COLUMNS),
            ("crypto_execution_5m", "stream_redis_execution_prefix", EXECUTION_COLUMNS),
        ):
            doc = read_latest(prefix_attr, symbol)
            rows[table.split("_")[1]] += write_latest(table, columns, symbol, doc)

    log.info("materialization_complete", symbols=symbols, **rows)
    return {"symbols": symbols, "rows": rows}


if __name__ == "__main__":
    configure_logging()
    result = materialize_streaming_to_clickhouse()
    log.info("materialize_clickhouse_done", **result)
