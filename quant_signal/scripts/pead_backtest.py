"""PEAD (post-earnings announcement drift) event study on REAL PIT data.

For every SEC EDGAR earnings filing (as-filed, dated by ``FILED_AT`` — the day
the 10-K became public), compute a standardized unexpected earnings (SUE)
surprise and the abnormal return (CAR) over the following trading days.

Method (no-lookahead discipline):
  * Universe, metric and report dates come from the environment/SILVER tables.
  * ``filed_at`` is the event date: only filings with ``filed_at <= t`` can be
    "known" at time t. Expected earnings for a filing are the last-known value
    of the PRIOR fiscal year (seasonal random walk), taken strictly from
    filings filed before this one — so a later restatement never leaks back.
  * SUE = (actual - expected) / std(prior surprises for the same ticker),
    where the std uses only surprises computed from earlier filings.
  * CAR over [0, +h] trading days = ticker buy-and-hold return minus the
    equal-weight universe return over the same global trading dates.

Output: mean CAR per SUE quintile for each window — the classic PEAD table
(high-SUE filings should drift up, low-SUE down). Research knobs (windows,
min prior surprises, quintiles) are CLI args with documented defaults;
instruments come from ``INGEST_DEFAULT_TICKERS``, not hardcoded here.

Usage:  uv run python scripts/pead_backtest.py [--metric NetIncomeLoss]
        [--windows 1,5,20] [--min-prior 5] [--quintiles 5]
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Iterable

import pandas as pd

from config.settings import csv_list, get_settings
from db.snowflake import SnowflakeClient

_WINDOW_DEFAULT = [1, 5, 20]


def _load_facts(client: SnowflakeClient, tickers: list[str], metric: str) -> pd.DataFrame:
    client._settings.snowflake_schema = "SILVER"  # type: ignore[attr-defined]
    placeholders = ", ".join(f"'{t}'" for t in tickers)
    return client.query_df(
        f"""
        SELECT TICKER, FISCAL_YEAR, VALUE, FILED_AT
        FROM SILVER_COMPANY_FACTS
        WHERE METRIC = '{metric}'
          AND TICKER IN ({placeholders})
        ORDER BY TICKER, FILED_AT
        """
    )


def _load_prices(client: SnowflakeClient, tickers: list[str]) -> tuple[list, dict[str, dict]]:
    """Global trading calendar + {symbol: {trade_date: close}} point-in-time."""
    client._settings.snowflake_schema = "GOLD"  # type: ignore[attr-defined]
    placeholders = ", ".join(f"'{t}'" for t in tickers)
    rows = client.query_df(
        f"""
        SELECT SYMBOL, TRADE_DATE, DAY_CLOSE
        FROM GOLD_DAILY_BARS
        WHERE TIMEFRAME = '1D' AND SYMBOL IN ({placeholders})
        """
    )
    calendar = sorted(rows["TRADE_DATE"].unique().tolist())
    closes: dict[str, dict] = defaultdict(dict)
    for _, row in rows.iterrows():
        closes[row["SYMBOL"]][row["TRADE_DATE"]] = row["DAY_CLOSE"]
    return calendar, dict(closes)


def _compute_sue(facts: pd.DataFrame, min_prior: int) -> list[dict]:
    """SUE per filing from the as-filed PIT timeline (strictly no lookahead).

    Follows Bernard & Thomas (1989): expected = seasonal random walk (prior
    fiscal year's last-known value as of this filing), surprise = actual minus
    expected, SUE = surprise / sigma, where sigma is the std of the most
    recent 8 surprises (the literature uses the last 8 quarters).
    """
    events: list[dict] = []
    for ticker, group in facts.groupby("TICKER", sort=True):
        latest_value_by_fy: dict[int, float] = {}
        prior_surprises: list[float] = []
        for _, row in group.sort_values("FILED_AT").iterrows():
            fy = int(row["FISCAL_YEAR"])
            actual = float(row["VALUE"])
            expected = latest_value_by_fy.get(fy - 1)
            if expected is not None:
                surprise = actual - expected
                if len(prior_surprises) >= min_prior:
                    std = pd.Series(prior_surprises[-8:]).std(ddof=1)
                    if std and not math.isnan(std):
                        events.append(
                            {
                                "ticker": ticker,
                                "fiscal_year": fy,
                                "filed_at": row["FILED_AT"],
                                "actual": actual,
                                "expected": expected,
                                "surprise": surprise,
                                "sue": surprise / std,
                            }
                        )
                prior_surprises.append(surprise)
            latest_value_by_fy[fy] = actual
    return events


def _cares(
    events: list[dict], calendar: list, closes: dict[str, dict], windows: list[int]
) -> list[dict]:
    """Attach CAR per window; each event needs prices on both endpoints."""
    for ev in events:
        filed = ev["filed_at"]
        t0_idx = next((i for i, d in enumerate(calendar) if d >= filed), None)
        if t0_idx is None:
            ev["skip"] = "no_trading_day"
            continue
        t0 = calendar[t0_idx]
        ticker_closes = closes.get(ev["ticker"], {})
        if t0 not in ticker_closes:
            ev["skip"] = "no_price_at_event"
            continue
        base = ticker_closes[t0]
        for h in windows:
            t_h_idx = t0_idx + h
            if t_h_idx >= len(calendar):
                ev[f"car{h}"] = None
                continue
            t_h = calendar[t_h_idx]
            ticker_ret = None
            if t_h in ticker_closes:
                ticker_ret = ticker_closes[t_h] / base - 1.0
            market_rets = []
            for sym, cs in closes.items():
                if sym == ev["ticker"]:
                    continue
                if t0 in cs and t_h in cs:
                    market_rets.append(cs[t_h] / cs[t0] - 1.0)
            if ticker_ret is None or not market_rets:
                ev[f"car{h}"] = None
                continue
            market_ret = sum(market_rets) / len(market_rets)
            ev[f"car{h}"] = ticker_ret - market_ret
        ev.pop("skip", None)
    return events


def _print_table(events: list[dict], windows: list[int], quintiles: int) -> None:
    df = pd.DataFrame(events)
    print(f"\nPEAD event study — {len(df)} earnings filings, {df['ticker'].nunique()} tickers")
    if df.empty:
        return
    df["quintile"] = _prior_breakpoint_labels(events, quintiles)
    unlabeled = int(df["quintile"].isna().sum())
    if unlabeled:
        print(f"  ({unlabeled} early events excluded — no prior SUE distribution yet)")
    rows = []
    for label, g in df.dropna(subset=["quintile"]).groupby("quintile", observed=True):
        row = {"quintile": label, "n": len(g), "mean_sue": g["sue"].mean()}
        for h in windows:
            valid = g[f"car{h}"].dropna()
            mean = valid.mean() if len(valid) else float("nan")
            se = valid.std(ddof=0) / math.sqrt(len(valid)) if len(valid) > 1 else float("nan")
            row[f"car{h}"] = mean
            row[f"t{h}"] = mean / se if se and se == se else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows).set_index("quintile")
    print(out.round(4).to_string())
    q1 = out.iloc[0]
    q5 = out.iloc[-1]
    print("\nPEAD spread (highest minus lowest SUE quintile):")
    for h in windows:
        spread = q5[f"car{h}"] - q1[f"car{h}"]
        print(f"  window +{h}d: {spread:+.4f} ({100 * spread:.2f} bps)")


def _prior_breakpoint_labels(events: list[dict], quintiles: int) -> list[str | None]:
    """Assign each event a quintile from PRIOR events' SUE breakpoints.

    No lookahead: an event's label depends only on SUEs of filings that were
    already public, so the trading rule is implementable on the event date.
    """
    import bisect

    labels: list[str | None] = [None] * len(events)
    prior: list[float] = []
    for pos in sorted(range(len(events)), key=lambda i: events[i]["filed_at"]):
        cutoffs: list[float] = []
        if len(prior) >= quintiles * 3:
            cutoffs = [pd.Series(prior).quantile(k / quintiles) for k in range(1, quintiles)]
        if cutoffs:
            labels[pos] = f"Q{bisect.bisect_left(cutoffs, events[pos]['sue']) + 1}"
        prior.append(events[pos]["sue"])
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--metric", default="NetIncomeLoss", help="earnings metric from US-GAAP facts"
    )
    parser.add_argument("--windows", default="1,5,20", help="post-filing trading-day windows")
    parser.add_argument(
        "--min-prior", type=int, default=5, help="min prior surprises before SUE is computed"
    )
    parser.add_argument("--quintiles", type=int, default=5, help="SUE groups for the drift table")
    args = parser.parse_args()

    settings = get_settings()
    tickers: Iterable[str] = csv_list(settings.ingest_default_tickers)
    windows = [int(w) for w in args.windows.split(",")]

    client = SnowflakeClient()
    facts = _load_facts(client, list(tickers), args.metric)
    if facts.empty:
        raise SystemExit(f"no {args.metric} filings found — run 'make ingest-fundamentals' first")
    calendar, closes = _load_prices(client, list(tickers))

    events = _compute_sue(facts, args.min_prior)
    events = _cares(events, calendar, closes, windows)
    _print_table(events, windows, args.quintiles)


if __name__ == "__main__":
    main()
