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

import pandas as pd

from config.settings import csv_list, get_settings
from db.snowflake import SnowflakeClient

_WINDOW_DEFAULT = [1, 5, 20]


def _load_facts(client: SnowflakeClient, tickers: list[str], metric: str) -> pd.DataFrame:
    """PIT facts for the metric, bound-parameter safe (the API exposes these)."""
    if not tickers:
        return pd.DataFrame()
    qual = client._qualify("SILVER_COMPANY_FACTS", None, "SILVER")
    placeholders = ", ".join(["%s"] * len(tickers))
    return client.query_df(
        f"""
        SELECT TICKER, FISCAL_YEAR, VALUE, FILED_AT
        FROM {qual}
        WHERE METRIC = %s
          AND TICKER IN ({placeholders})
        ORDER BY TICKER, FILED_AT
        """,
        (metric, *tickers),
    )


def _load_prices(client: SnowflakeClient, tickers: list[str]) -> tuple[list, dict[str, dict]]:
    """Global trading calendar + {symbol: {trade_date: close}} point-in-time."""
    if not tickers:
        return [], {}
    qual = client._qualify("GOLD_DAILY_BARS", None, "GOLD")
    placeholders = ", ".join(["%s"] * len(tickers))
    rows = client.query_df(
        f"""
        SELECT SYMBOL, TRADE_DATE, DAY_CLOSE
        FROM {qual}
        WHERE TIMEFRAME = '1D' AND SYMBOL IN ({placeholders})
        """,
        tuple(tickers),
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


def _clean(value: float) -> float | None:
    """NaN → None so the result is JSON-serializable (the API serves this)."""
    return None if value != value else round(value, 4)


def _summarize(events: list[dict], windows: list[int], quintiles: int) -> dict:
    """Quintile drift table + spread, JSON-serializable."""
    df = pd.DataFrame(events)
    summary: dict = {
        "n_events": len(df),
        "n_tickers": int(df["ticker"].nunique()) if not df.empty else 0,
    }
    if df.empty:
        summary["quintiles"] = []
        summary["spread"] = {}
        return summary
    df["quintile"] = _prior_breakpoint_labels(events, quintiles)
    summary["unlabeled"] = int(df["quintile"].isna().sum())
    rows: list[dict] = []
    for label, g in df.dropna(subset=["quintile"]).groupby("quintile", observed=True):
        row = {"quintile": label, "n": len(g), "mean_sue": _clean(g["sue"].mean())}
        for h in windows:
            valid = g[f"car{h}"].dropna()
            mean = valid.mean() if len(valid) else float("nan")
            se = valid.std(ddof=0) / math.sqrt(len(valid)) if len(valid) > 1 else float("nan")
            row[f"car{h}"] = _clean(mean)
            row[f"t{h}"] = _clean(mean / se) if se and se == se else None
        rows.append(row)
    summary["quintiles"] = rows
    q1 = rows[0]
    q5 = rows[-1]
    summary["spread"] = {
        f"+{h}d": _clean(q5.get(f"car{h}") or 0.0 - (q1.get(f"car{h}") or 0.0)) for h in windows
    }
    return summary


def compute_pead(
    metric: str = "NetIncomeLoss",
    windows: list[int] | None = None,
    min_prior: int = 5,
    quintiles: int = 5,
    tickers: list[str] | None = None,
) -> dict:
    """Full PEAD event study against the live SILVER/GOLD data. JSON-safe.

    Shared by the CLI (``main``) and the dashboard API so both show the same
    numbers. Instruments default to ``INGEST_DEFAULT_TICKERS`` — not hardcoded.
    """
    settings = get_settings()
    tickers = tickers or csv_list(settings.ingest_default_tickers)
    windows = windows or list(_WINDOW_DEFAULT)
    client = SnowflakeClient()
    facts = _load_facts(client, list(tickers), metric)
    if facts.empty:
        return {"error": f"no {metric} filings found — run 'make ingest-fundamentals' first"}
    calendar, closes = _load_prices(client, list(tickers))
    events = _cares(_compute_sue(facts, min_prior), calendar, closes, windows)
    result = _summarize(events, windows, quintiles)
    result["metric"] = metric
    result["windows"] = windows
    return result


def _print_summary(result: dict) -> None:
    if "error" in result:
        print(result["error"])
        return
    print(
        f"\nPEAD event study — {result['n_events']} earnings filings, "
        f"{result['n_tickers']} tickers (metric: {result['metric']})"
    )
    if result.get("unlabeled"):
        print(f"  ({result['unlabeled']} early events excluded — no prior SUE distribution yet)")
    out = pd.DataFrame(result["quintiles"]).set_index("quintile")
    print(out.round(4).to_string())
    print("\nPEAD spread (highest minus lowest SUE quintile):")
    for h in result["windows"]:
        spread = result["spread"].get(f"+{h}d")
        if spread is not None:
            print(f"  window +{h}d: {spread:+.4f} ({100 * spread:+.2f}%)")


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
    windows = [int(w) for w in args.windows.split(",")]
    _print_summary(compute_pead(args.metric, windows, args.min_prior, args.quintiles))


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


if __name__ == "__main__":
    main()
