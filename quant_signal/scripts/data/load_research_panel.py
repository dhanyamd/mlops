"""Land the SRP research panel in the warehouse.

WHY THIS EXISTS. The research loop reads a JSON file on local disk
(``/tmp/quant_cache/fas_broad.json``): 202 symbols of daily perp bars back to
2019, plus the 8-hourly funding rates the return construction charges. The
warehouse, meanwhile, held only 1-minute bars for two symbols and no funding at
all. So "our research runs on Snowflake" was not true, and could not be made
true by pointing a query at it -- the data simply was not there.

This script puts it there. After it runs, the panel exists as two bronze tables
and the research loader can read the warehouse instead of a file that lives on
one laptop and vanishes with /tmp.

WHY TWO TABLES RATHER THAN REUSING ``crypto_bars``. That table's contract
requires open/high/low, and the panel carries only close and volume. Fabricating
the missing three to satisfy a contract would put invented numbers in the
warehouse, which is a far worse outcome than an extra table. The panel is a
genuinely different dataset anyway: daily REST-backfilled perpetuals for
research, versus streaming minute spot bars for the live path.

IDEMPOTENT. Both loads MERGE on their natural grain, so re-running after a
partial failure converges rather than duplicating. Chunked so one network fault
costs a chunk instead of the run.

    uv run python -m scripts.load_research_panel                # full load
    uv run python -m scripts.load_research_panel --dry-run      # shapes only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from config.logging import configure_logging, get_logger
from config.settings import get_settings
from db.snowflake import SnowflakeClient

logger = get_logger(__name__)

BARS_TABLE = "CRYPTO_PANEL_BARS"
FUNDING_TABLE = "CRYPTO_FUNDING"
PROVIDER = "binance"
# Chunked so a transient stage failure costs one chunk, not a 1.2M-row load.
CHUNK_ROWS = 250_000


def _frames(cache_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten the nested cache into two long frames ready for the warehouse."""
    with open(cache_path) as fh:
        cache = json.load(fh)

    loaded_at = datetime.now(timezone.utc).replace(tzinfo=None)

    bar_rows: list[tuple] = []
    for symbol, rows in (cache.get("bars") or {}).items():
        for ts_ms, close, volume in rows:
            bar_rows.append((symbol, ts_ms, float(close), float(volume)))

    fund_rows: list[tuple] = []
    for symbol, rows in (cache.get("funding") or {}).items():
        for ts_ms, rate in rows:
            fund_rows.append((symbol, ts_ms, float(rate)))

    bars = pd.DataFrame(bar_rows, columns=["SYMBOL", "TS", "CLOSE", "VOLUME"])
    funding = pd.DataFrame(fund_rows, columns=["SYMBOL", "TS", "RATE"])

    for df in (bars, funding):
        # TIMESTAMP_NTZ in UTC: the connector binds tz-aware columns
        # inconsistently, and every timestamp in this system is already UTC.
        df["TS"] = pd.to_datetime(df["TS"], unit="ms", utc=True).dt.tz_localize(None)
        df["PROVIDER"] = PROVIDER
        df["LOADED_AT"] = loaded_at

    # The merge key must be unique or MERGE raises on a duplicate match. The
    # cache can carry a repeated bar when a backfill overlapped a prior run.
    bars = bars.drop_duplicates(subset=["SYMBOL", "TS"], keep="last")
    funding = funding.drop_duplicates(subset=["SYMBOL", "TS"], keep="last")
    return bars, funding


def _load(client: SnowflakeClient, df: pd.DataFrame, table: str, schema: str) -> int:
    """MERGE a frame into bronze in bounded chunks. Returns rows sent."""
    if df.empty:
        logger.warning("nothing to load", table=table)
        return 0
    sent = 0
    for start in range(0, len(df), CHUNK_ROWS):
        chunk = df.iloc[start : start + CHUNK_ROWS]
        client.upsert_df(
            chunk,
            table,
            merge_keys=["SYMBOL", "TS"],
            schema=schema,
            # Distinct temp table per chunk: the default is per-PID, and reusing
            # one name across sequential chunks makes a mid-run failure ambiguous.
            temp_table=f"TMP_{table}_{start // CHUNK_ROWS}",
        )
        sent += len(chunk)
        logger.info("panel chunk merged", table=table, rows=sent, total=len(df))
    return sent


def main() -> int:
    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=None, help="panel JSON (default: settings)")
    ap.add_argument("--schema", default="BRONZE")
    ap.add_argument("--dry-run", action="store_true", help="report shapes, write nothing")
    args = ap.parse_args()

    settings = get_settings()
    cache_path = args.cache or settings.srp_weekly_cache

    bars, funding = _frames(cache_path)
    print(
        f"panel {cache_path}\n"
        f"  bars    {len(bars):>9,} rows  {bars['SYMBOL'].nunique():>4} symbols  "
        f"{bars['TS'].min().date()} -> {bars['TS'].max().date()}\n"
        f"  funding {len(funding):>9,} rows  {funding['SYMBOL'].nunique():>4} symbols  "
        f"{funding['TS'].min().date()} -> {funding['TS'].max().date()}"
    )
    if args.dry_run:
        return 0

    client = SnowflakeClient(settings)
    n_bars = _load(client, bars, BARS_TABLE, args.schema)
    n_fund = _load(client, funding, FUNDING_TABLE, args.schema)
    print(f"\nloaded {n_bars:,} bars and {n_fund:,} funding rows into {args.schema}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
