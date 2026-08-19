"""Read the SRP research panel from Snowflake instead of a file on one laptop.

WHAT THIS BUYS. The JSON cache works, and the backtest has always been correct
against it, but it is a local artifact: it lives in /tmp, it is not versioned,
and "which panel produced this Sharpe" is answerable only by whoever still has
the file. Reading the warehouse makes the research input a queryable, retained,
shared object -- which is the whole reason a firm puts market data in one.

POINT-IN-TIME. ``as_of`` truncates history at a date. That is what lets a
historical run be re-run honestly: without it, re-running a 2023 study today
silently includes every bar that has landed since, and the result is not the one
the decision was made on. The cutoff is applied in SQL, so the rows never reach
pandas and cannot be filtered inconsistently by a caller who forgets.

PARITY IS STRUCTURAL. This module fetches; ``research_fas_clean.build_frames``
resamples. The file path calls the same function on the same shape, so the two
sources cannot drift apart through a resampling change made in one and not the
other. ``scripts.panel_parity`` asserts the frames are identical.

    uv run python -m scripts.warehouse_panel                    # summarise
    uv run python -m scripts.warehouse_panel --as-of 2025-01-01
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from config.logging import configure_logging, get_logger
from db.snowflake import SnowflakeClient
from scripts.research_fas_clean import build_frames

logger = get_logger(__name__)

BARS_RELATION = "QUANT.SILVER.SILVER_CRYPTO_PANEL_BARS"
FUNDING_RELATION = "QUANT.SILVER.SILVER_CRYPTO_FUNDING"


def _to_rows(df: pd.DataFrame, value_cols: list[str]) -> dict[str, list[list]]:
    """Group a long frame into ``{symbol: [[ts_ms, *values], ...]}``.

    Epoch milliseconds, because that is what the cache carries and what
    ``build_frames`` parses. Converting here rather than changing the shared
    function keeps the file path byte-identical to what it was.
    """
    if df.empty:
        return {}
    df = df.copy()
    df["TS_MS"] = (pd.to_datetime(df["TS"], utc=True).astype("int64") // 1_000_000)
    out: dict[str, list[list]] = {}
    for symbol, group in df.groupby("SYMBOL", sort=False):
        group = group.sort_values("TS_MS")
        out[str(symbol)] = group[["TS_MS", *value_cols]].to_numpy().tolist()
    return out


def fetch_panel(
    client: SnowflakeClient | None = None,
    *,
    as_of: str | None = None,
    symbols: list[str] | None = None,
) -> tuple[dict, dict]:
    """Pull raw bars and funding from the warehouse in cache shape.

    ``as_of`` is an inclusive upper bound on the bar timestamp (ISO date). No
    lower bound: the strategy's own lookback windows decide how much history
    they consume, and truncating early here would quietly shorten them.

    Two properties worth knowing before trusting a cutoff run. The universe
    shrinks with the date because coins that had not listed yet contribute no
    rows -- that is the point, and it is what keeps a historical run free of
    survivorship. And the FINAL weekly bar is partial, since the cutoff will
    usually fall mid-week; drop it if the last observation matters, as the
    resample cannot distinguish "week ended early" from "week ended".
    """
    client = client or SnowflakeClient()

    where: list[str] = []
    params: list = []
    if as_of:
        where.append("TS <= %s")
        params.append(f"{as_of} 23:59:59")
    if symbols:
        placeholders = ", ".join(["%s"] * len(symbols))
        where.append(f"SYMBOL IN ({placeholders})")
        params.extend([s.upper() for s in symbols])
    clause = f" WHERE {' AND '.join(where)}" if where else ""

    bars_df = client.query_df(
        f"SELECT SYMBOL, TS, CLOSE, VOLUME FROM {BARS_RELATION}{clause}",
        tuple(params) or None,
    )
    funding_df = client.query_df(
        f"SELECT SYMBOL, TS, RATE FROM {FUNDING_RELATION}{clause}",
        tuple(params) or None,
    )
    logger.info(
        "warehouse panel fetched",
        bars=len(bars_df),
        funding=len(funding_df),
        symbols=int(bars_df["SYMBOL"].nunique()) if not bars_df.empty else 0,
        as_of=as_of or "latest",
    )
    return (
        _to_rows(bars_df, ["CLOSE", "VOLUME"]),
        _to_rows(funding_df, ["RATE"]),
    )


def load_from_snowflake(
    week_anchor: str = "W-MON",
    *,
    as_of: str | None = None,
    symbols: list[str] | None = None,
    client: SnowflakeClient | None = None,
):
    """The warehouse-backed twin of ``research_fas_clean.load``.

    Returns the identical 5-tuple (weekly close, weekly volume, weekly funding
    accrual, daily close, daily volume), so it is a drop-in for every caller.
    """
    bars, funding = fetch_panel(client, as_of=as_of, symbols=symbols)
    return build_frames(bars, funding, week_anchor)


def main() -> int:
    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", default=None, help="ISO date; truncate history here")
    ap.add_argument("--week-anchor", default="W-MON")
    ap.add_argument("--symbols", default=None, help="comma-separated subset")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None
    cw, vw, aw, dcl, dvl = load_from_snowflake(
        args.week_anchor, as_of=args.as_of, symbols=symbols
    )
    print(
        f"warehouse panel  as_of={args.as_of or 'latest'}  anchor={args.week_anchor}\n"
        f"  weekly close   {cw.shape[0]:>5} weeks x {cw.shape[1]:>4} symbols\n"
        f"  weekly volume  {vw.shape[0]:>5} x {vw.shape[1]:>4}\n"
        f"  weekly funding {aw.shape[0]:>5} x {aw.shape[1]:>4}\n"
        f"  daily close    {dcl.shape[0]:>5} x {dcl.shape[1]:>4}\n"
        f"  span           {cw.index.min().date()} -> {cw.index.max().date()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
