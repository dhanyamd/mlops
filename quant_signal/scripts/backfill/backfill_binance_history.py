"""Backfill multi-year Binance 1Min crypto history into BRONZE CRYPTO_BARS.

Fetches keyless Binance klines in ~90-day windows (chunked so a mid-run network
failure loses at most one window, not the whole backfill) and lands each chunk
via the idempotent MERGE upsert on (symbol, timeframe, ts) — re-running picks
up where it left off. Single venue (binance), never bybit, so the momentum
probe is scored on one exchange's tape instead of a stitched seam.

Symbols are fetched concurrently within a chunk, and the per-page throttle is
tightened for the backfill (well inside Binance's keyless weight budget).

Every chunk passes the same Pydantic contract gate as the live flow; invalid
rows go to QUARANTINE_CRYPTO_BARS, never to Bronze.

Usage::

    uv run python scripts/backfill_binance_history.py --symbols BTCUSDT,ETHUSDT --days 1095
    uv run python scripts/backfill_binance_history.py --days 1095 --dry-run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import time

import pandas as pd

from config.logging import configure_logging, get_logger
from config.settings import get_settings
from ingest.providers.binance import BinanceBarProvider
from ingest.quality import validate_bars
from ingest.store import write_crypto_bars, write_quarantine

log = get_logger("scripts.backfill_binance_history")

_CHUNK_DAYS = 90
_DAY_MS = 86_400_000
_SLEEP_BETWEEN_CHUNKS_S = 1.0
# Backfill throttle: 2 symbols x ~20 pages/s = ~40 req/s = ~80 weight/s vs the
# keyless 6000 weight/min budget. The live producer keeps the conservative
# default (0.2s) because it polls a few pages per cycle.
_BACKFILL_REQUEST_INTERVAL_S = 0.05
_MAX_WORKERS = 12
_SUBWINDOWS = 6


def _chunks(start_ms: int, end_ms: int, chunk_days: int) -> list[tuple[int, int]]:
    chunk_ms = chunk_days * _DAY_MS
    out: list[tuple[int, int]] = []
    lo = start_ms
    while lo < end_ms:
        hi = min(lo + chunk_ms, end_ms)
        out.append((lo, hi))
        lo = hi
    return out


def _subwindows(lo: int, hi: int, n: int) -> list[tuple[int, int]]:
    """Split [lo, hi) into ``n`` roughly equal windows (for concurrent fetch)."""
    span = hi - lo
    size = max(1, span // n)
    out: list[tuple[int, int]] = []
    cursor = lo
    while cursor < hi:
        nxt = min(cursor + size, hi)
        out.append((cursor, nxt))
        cursor = nxt
    return out


def _fetch_chunk(
    symbols: list[str],
    lo: int,
    hi: int,
    *,
    request_interval_s: float,
    subwindows: int,
) -> pd.DataFrame:
    """Fetch one chunk's window, split into concurrent sub-windows. The live
    producer's trailing-window call is untouched; only the backfill fans out."""
    frames: list[pd.DataFrame] = []
    tasks: list[tuple[BinanceBarProvider, str, int, int]] = []
    for symbol in symbols:
        for slo, shi in _subwindows(lo, hi, subwindows):
            tasks.append(
                (BinanceBarProvider(request_interval_s=request_interval_s), symbol, slo, shi)
            )
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tasks), _MAX_WORKERS)) as pool:
        futures = {
            pool.submit(p.fetch_bars, [sym], days=0, start_ms=slo, end_ms=shi): (p, sym, slo)
            for p, sym, slo, shi in tasks
        }
        for fut in concurrent.futures.as_completed(futures):
            try:
                frames.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - surface the failing window
                p, sym, slo = futures[fut]
                raise RuntimeError(f"backfill fetch failed for {sym}@{slo}") from exc
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def backfill_binance_history(
    *,
    symbols: list[str],
    days: int,
    chunk_days: int,
    dry_run: bool,
) -> dict[str, int]:
    settings = get_settings()
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * _DAY_MS

    fetched = 0
    valid = 0
    invalid = 0
    written = 0
    for lo, hi in _chunks(start_ms, end_ms, chunk_days):
        log.info(
            "backfill_chunk_fetch",
            start_ts=int(lo),
            end_ts=int(hi),
            span_days=round((hi - lo) / _DAY_MS, 1),
        )
        df = _fetch_chunk(
            symbols,
            lo,
            hi,
            request_interval_s=_BACKFILL_REQUEST_INTERVAL_S,
            subwindows=_SUBWINDOWS,
        )
        if df is None or df.empty:
            continue
        fetched += len(df)
        good, bad = validate_bars(df)
        valid += len(good)
        invalid += len(bad)
        if not bad.empty:
            write_quarantine(bad, "crypto_bars", settings)
        if good.empty:
            continue
        if dry_run:
            log.info("backfill_chunk_preview", rows=len(good), dry_run=True)
            continue
        written += write_crypto_bars(good, settings)
        time.sleep(_SLEEP_BETWEEN_CHUNKS_S)

    log.info(
        "backfill_binance_complete",
        symbols=symbols,
        days=days,
        fetched=fetched,
        valid=valid,
        invalid=invalid,
        written=written,
        dry_run=dry_run,
    )
    return {"fetched": fetched, "valid": valid, "invalid": invalid, "written": written}


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        default=None,
        help="comma-separated symbols (default: BTCUSDT,ETHUSDT)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1095,
        help="calendar days of history to backfill (default 1095 = 3y)",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=_CHUNK_DAYS,
        help="size of each fetch+write window (default 90)",
    )
    parser.add_argument("--dry-run", action="store_true", help="fetch + validate, write nothing")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in (args.symbols or "BTCUSDT,ETHUSDT").split(",")]
    backfill_binance_history(
        symbols=symbols,
        days=args.days,
        chunk_days=args.chunk_days,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
