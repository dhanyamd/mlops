"""Backfill the Redis 1h online-store history from the Snowflake engine of record.

The live Flink 1h job starts from Kafka's ``latest-offset`` after a heal, so the
online store only accrues windows going forward and the promotion gate starts
near zero scored windows — its 100-window warm-up would take ~4 days of live
windows. Snowflake BRONZE is the engine of record for the *same* Bybit minute
bars Flink consumes, so aggregate them into the exact TUMBLE(1 HOUR) OHLCV+VWAP
windows the live job emits and seed the online store, letting the gate
progressive-validate over real history instead of waiting for it to accrue.

The aggregation mirrors the Flink SQL window contract (floor-hour bucket,
first/max/min/last OHLC, volume-weighted close, bar count) and writes the same
window dict schema the materializer stores. Only complete hours are kept (the
in-progress hour is dropped, since Flink only emits closed windows). Idempotent:
each symbol's online key is atomically rebuilt (DEL + RPUSH) with the newest
``--windows`` windows, so rerunning is safe.

Run:  uv run python -m scripts.backfill_feature_windows --dry-run
      uv run python -m scripts.backfill_feature_windows
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd
import redis

from config.logging import configure_logging, get_logger
from config.settings import Settings, csv_list, get_settings
from db.snowflake import SnowflakeClient
from stream.materializer import feature_key

logger = get_logger(__name__)

_HOUR_MS = 3_600_000

_WINDOW_COLUMNS = [
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
]


def hourly_windows(bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute bars into the Flink TUMBLE(1 HOUR) window schema.

    ``bars`` carries per-minute OHLCV with ``symbol`` and ``ts_ms`` (epoch ms,
    minute-aligned). Drops the in-progress hour (the live job emits only closed
    windows) and returns windows oldest-first with the materializer's column set.
    """
    df = bars.copy()
    if df.empty:
        return pd.DataFrame(columns=_WINDOW_COLUMNS)
    df["ts_ms"] = df["ts_ms"].astype("int64")
    df["hour"] = df["ts_ms"] // _HOUR_MS
    # A window is complete only once the next hour has started producing; the
    # in-progress hour (the highest bucket) is dropped. A single-bucket input
    # has no way to be in-progress, so it is kept.
    hours = df["hour"].unique()
    if len(hours) > 1:
        df = df[df["hour"] != hours.max()]
    if df.empty:
        return pd.DataFrame(columns=_WINDOW_COLUMNS)

    agg = df.groupby(["symbol", "hour"], sort=True, as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    agg["window_start_ms"] = agg["hour"] * _HOUR_MS
    agg["window_end_ms"] = agg["window_start_ms"] + _HOUR_MS
    df = df.assign(wv=df["close"] * df["volume"])
    vwap = df.groupby(["symbol", "hour"])["wv"].sum()
    volume = df.groupby(["symbol", "hour"])["volume"].sum()
    agg["vwap"] = (vwap / volume).astype(float).values
    agg["bar_count"] = df.groupby(["symbol", "hour"]).size().values
    agg["symbol"] = agg["symbol"].str.upper()
    return agg[_WINDOW_COLUMNS]


def fetch_bars(settings: Settings, symbols: list[str]) -> pd.DataFrame:
    """Pull the raw minute OHLCV for ``symbols`` from BRONZE CRYPTO_BARS."""
    client = SnowflakeClient(settings)
    in_list = ",".join(f"'{s}'" for s in symbols)
    sql = f"""
        SELECT symbol,
               extract(epoch_second from ts) * 1000 AS ts_ms,
               open, high, low, close, volume
        FROM "{settings.snowflake_database}"."{settings.snowflake_schema}".CRYPTO_BARS
        WHERE symbol IN ({in_list})
          AND timeframe = '1Min'
        ORDER BY symbol, ts
    """
    df = client.query_df(sql)
    df.columns = [c.lower() for c in df.columns]
    df["symbol"] = df["symbol"].str.upper()
    df["ts_ms"] = df["ts_ms"].astype("int64")
    return df


def seed_online_store(
    settings: Settings,
    windows: pd.DataFrame,
    *,
    symbols: list[str],
    maxlen: int,
    dry_run: bool,
) -> dict[str, int]:
    """Atomically rebuild each symbol's 1h online key from the newest windows.

    Returns {symbol: seeded_row_count}. The live materializer keeps appending
    new closed windows after this, so the seeded list and live flow stay
    contiguous (the newest seeded window is the latest closed hour).
    """
    r = redis.Redis.from_url(settings.stream_redis_url, decode_responses=True)
    seeded: dict[str, int] = {}
    for symbol in symbols:
        rows = windows[windows["symbol"] == symbol].sort_values("window_start_ms")
        rows = rows.tail(maxlen)
        key = feature_key(settings.stream_redis_feature_prefix, symbol)
        if dry_run:
            seeded[symbol] = len(rows)
            continue
        pipe = r.pipeline(transaction=True)
        pipe.delete(key)
        for record in rows.to_dict("records"):
            pipe.rpush(key, json.dumps(record))
        pipe.execute()
        seeded[symbol] = len(rows)
        logger.info(
            "backfill_seeded",
            symbol=symbol,
            key=key,
            rows=len(rows),
            newest_window_end_ms=int(rows["window_end_ms"].max()) if len(rows) else None,
        )
    return seeded


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        default=None,
        help="comma-separated symbols (default: ingest_default_crypto_symbols)",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=None,
        help="windows per symbol to keep (default: stream_redis_feature_maxlen)",
    )
    parser.add_argument("--dry-run", action="store_true", help="aggregate + preview, write nothing")
    args = parser.parse_args()

    settings = get_settings()
    symbols = (
        csv_list(args.symbols) if args.symbols else csv_list(settings.ingest_default_crypto_symbols)
    )
    maxlen = args.windows or settings.stream_redis_feature_maxlen

    started = time.time()
    bars = fetch_bars(settings, symbols)
    logger.info(
        "backfill_fetched_bars",
        rows=len(bars),
        symbols=sorted(bars["symbol"].unique().tolist()) if len(bars) else [],
    )
    windows = hourly_windows(bars)
    logger.info(
        "backfill_aggregated",
        rows=len(windows),
        symbols=sorted(windows["symbol"].unique().tolist()) if len(windows) else [],
    )

    per_symbol = (
        windows.groupby("symbol").size().to_dict() if len(windows) else {s: 0 for s in symbols}
    )
    for symbol, n in per_symbol.items():
        logger.info(
            "backfill_symbol_windows",
            symbol=symbol,
            hours=n,
            seeded=min(n, maxlen),
            detail="dry-run, nothing written" if args.dry_run else "",
        )

    if len(windows) == 0:
        logger.error("backfill_no_data", symbols=symbols)
        sys.exit(1)

    seeded = seed_online_store(
        settings, windows, symbols=symbols, maxlen=maxlen, dry_run=args.dry_run
    )
    logger.info(
        "backfill_complete",
        dry_run=args.dry_run,
        seeded=seeded,
        elapsed_s=round(time.time() - started, 1),
    )


if __name__ == "__main__":
    main()
