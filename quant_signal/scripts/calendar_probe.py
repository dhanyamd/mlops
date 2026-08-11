"""Intraday time-of-day seasonality probe on REAL Snowflake CRYPTO_BARS history.

Tests whether the documented crypto overnight-hold / opening-drift anomalies
actually exist in OUR data (the QuantPedia "buy 21:00 / sell 23:00 UTC" 2-hour
hold and the Quantified-Strategies overnight-vs-intraday spread) before wiring
any calendar strategy into the live harness. The papers:
  * QuantPedia (2021) — BTC/ETH overnight 21:00-23:00 UTC hold;
  * Quantified Strategies (2022) — crypto overnight vs intraday returns;
  * Shen, Urquhart & Wang (Financial Review 57(2), 2022) — time-of-day
    persistence, mechanism = liquidity provision + disposition effect.

Method (no-lookahead discipline, exactly the backfill contract):
  * Minute bars come from BRONZE CRYPTO_BARS (engine of record) and are
    aggregated with the same TUMBLE(1 HOUR) OHLCV contract the live Flink job
    emits (reuses ``backfill_feature_windows.hourly_windows``).
  * A window's realized return is attributed to its START hour-of-day (UTC):
    the return you collect by holding through that hour. Only adjacent windows
    exactly 1h apart are counted, so a Flink gap never leaks into the bucket.
  * Buckets: OVERNIGHT = start hours {21,22,23,0..8}; US_SESSION = {13..20}.
  * Each strategy's per-day hold return is netted against the 10bps taker
    round trip (λ=2 gate = 20bps band); it must clear it to be tradeable.

Run:  uv run python scripts/calendar_probe.py [--symbols BTCUSDT,ETHUSDT]
      uv run python scripts/calendar_probe.py --out docs/probe_calendar.json
"""

from __future__ import annotations

import argparse
import json
import math

import pandas as pd
from scipy import stats

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from scripts.backfill_feature_windows import fetch_bars, hourly_windows

logger = get_logger(__name__)

_HOUR_MS = 3_600_000
_DAY_MS = 86_400_000
_OVERNIGHT_HOURS = frozenset([21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8])
_US_SESSION_HOURS = frozenset(range(13, 21))
_QP_HOLD_HOURS = [21, 22]  # QuantPedia 21:00 -> 23:00 UTC
_ON_HOLD_HOURS = list(range(21, 24)) + list(range(0, 9))  # 21:00 -> 09:00 UTC
_TAKER_ROUND_TRIP = 0.001  # 10 bps, the gate's cost basis


def _hourly_returns(windows: pd.DataFrame) -> pd.DataFrame:
    """Per-start-hour returns, only across adjacent 1h windows (no gaps)."""
    df = windows.sort_values("window_start_ms").reset_index(drop=True)
    df["hour"] = (df["window_start_ms"] % _DAY_MS) // _HOUR_MS
    df["dt"] = df["window_start_ms"].diff()
    df["next_close"] = df["close"].shift(-1)
    df["next_hour"] = df["hour"].shift(-1)
    df["ret"] = df["next_close"] / df["close"] - 1.0
    adj = (df["dt"] == _HOUR_MS) & df["next_hour"].notna()
    return df.loc[adj, ["window_start_ms", "hour", "ret"]].reset_index(drop=True)


def _bucket(series: pd.DataFrame, hours: frozenset) -> pd.Series:
    return series.loc[series["hour"].isin(hours), "ret"]


def _summarize_rets(rets: pd.Series) -> dict:
    n = int(rets.size)
    mean = float(rets.mean()) if n else float("nan")
    se = float(rets.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    t = float(mean / se) if se and se == se else None
    return {
        "n": n,
        "mean_bps": 1e4 * mean,
        "t": None if t is None or not math.isfinite(t) else round(t, 2),
    }


def _day_holds(returns: pd.DataFrame, hours: list[int]) -> pd.Series:
    """Per-UTC-day compounded hold return across the given start hours."""
    days: dict[int, float] = {}
    starts: dict[int, int] = {}
    for _, row in returns[returns["hour"].isin(hours)].iterrows():
        day = row["window_start_ms"] // _DAY_MS
        days[day] = days.get(day, 1.0) * (1.0 + row["ret"])
        starts[day] = starts.get(day, 0) + 1
    full = pd.Series({d: v for d, v in days.items() if starts[d] == len(hours)})
    return full


def probe_symbol(windows: pd.DataFrame, symbol: str) -> dict:
    rets = _hourly_returns(windows)
    out: dict = {
        "symbol": symbol,
        "n_hours": int(len(windows)),
        "n_returns": int(len(rets)),
        "date_min": str(windows["window_start_ms"].min()),
        "date_max": str(windows["window_start_ms"].max()),
    }
    if rets.empty:
        out["error"] = "no adjacent-hour returns"
        return out

    out["hour_table"] = [
        {"hour": h, **_summarize_rets(_bucket(rets, frozenset([h])))} for h in range(24)
    ]

    overnight = _bucket(rets, _OVERNIGHT_HOURS)
    us = _bucket(rets, _US_SESSION_HOURS)
    out["overnight"] = _summarize_rets(overnight)
    out["us_session"] = _summarize_rets(us)
    if len(overnight) > 1 and len(us) > 1:
        welch = stats.ttest_ind(overnight, us, equal_var=False)
        out["overnight_minus_us"] = {
            "diff_bps": 1e4 * (overnight.mean() - us.mean()),
            "t": round(float(welch.statistic), 2),
            "p": round(float(welch.pvalue), 4),
        }

    qp = _day_holds(rets, _QP_HOLD_HOURS)
    out["qp_21_to_23"] = _summarize_rets(qp - 1.0)
    out["qp_21_to_23"]["gross_bps"] = 1e4 * float((qp - 1.0).mean()) if qp.size else None
    out["qp_21_to_23"]["net_taker_bps"] = (
        1e4 * (float((qp - 1.0).mean()) - _TAKER_ROUND_TRIP) if qp.size else None
    )

    on = _day_holds(rets, _ON_HOLD_HOURS)
    out["overnight_hold_21_to_09"] = _summarize_rets(on - 1.0)
    out["overnight_hold_21_to_09"]["gross_bps"] = (
        1e4 * float((on - 1.0).mean()) if on.size else None
    )
    out["overnight_hold_21_to_09"]["net_taker_bps"] = (
        1e4 * (float((on - 1.0).mean()) - _TAKER_ROUND_TRIP) if on.size else None
    )

    days = (rets["window_start_ms"] // _DAY_MS).rename("day")
    dow = pd.Series(
        pd.to_datetime(rets["window_start_ms"] // _HOUR_MS * _HOUR_MS, unit="ms")
    ).dt.dayofweek.rename("dow")
    both = pd.concat([pd.Series(rets["ret"]), days.rename("day"), dow], axis=1)
    weekend = _summarize_rets(both.loc[both["dow"] >= 5, "ret"])
    weekday = _summarize_rets(both.loc[both["dow"] < 5, "ret"])
    out["weekend_hourly"] = weekend
    out["weekday_hourly"] = weekday
    return out


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--symbols", default=None, help="comma-separated symbols (default: ingest defaults)"
    )
    parser.add_argument("--out", default=None, help="JSON output path (default: print only)")
    parser.add_argument("--dry-run", action="store_true", help="fetch + aggregate, compute nothing")
    args = parser.parse_args()

    settings = get_settings()
    symbols = (
        csv_list(args.symbols) if args.symbols else csv_list(settings.ingest_default_crypto_symbols)
    )
    bars = fetch_bars(settings, symbols)
    windows = hourly_windows(bars)
    if windows.empty:
        logger.error("calendar_probe_no_data", symbols=symbols)
        raise SystemExit(1)

    results: list[dict] = []
    for symbol in symbols:
        sym_windows = windows[windows["symbol"] == symbol.upper()]
        logger.info(
            "calendar_probe_symbol",
            symbol=symbol.upper(),
            hours=len(sym_windows),
            dry_run=args.dry_run,
        )
        if args.dry_run:
            continue
        results.append(probe_symbol(sym_windows, symbol.upper()))

    if args.dry_run:
        return

    payload = {"symbols": results, "taker_round_trip_bps": 1e4 * _TAKER_ROUND_TRIP}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("calendar_probe_written", path=args.out)

    for r in results:
        if "error" in r:
            print(f"\n{symbol}: {r['error']}")
            continue
        print(
            f"\n=== {r['symbol']} — {r['n_returns']} hourly returns "
            f"({r['date_min']} .. {r['date_max']}) ==="
        )
        hour_table = pd.DataFrame(r["hour_table"])
        print("\nMean return by start-hour (UTC), bps:")
        print(hour_table.round(2).to_string(index=False))
        o, u = r["overnight"], r["us_session"]
        print(
            f"\nOVERNIGHT (start h 21-23,0-8): {o['mean_bps']:+.1f} bps/hr  n={o['n']}  t={o['t']}"
        )
        print(
            f"US SESSION (start h 13-20):     {u['mean_bps']:+.1f} bps/hr  n={u['n']}  t={u['t']}"
        )
        if "overnight_minus_us" in r:
            d = r["overnight_minus_us"]
            print(
                "overnight − US:                  "
                f"{d['diff_bps']:+.1f} bps/hr  t={d['t']}  p={d['p']}"
            )
        qp = r["qp_21_to_23"]
        print(
            f"\nQP 21:00→23:00 hold:  {qp['gross_bps']:+.1f} bps/day gross, "
            f"{qp['net_taker_bps']:+.1f} bps/day net taker (n={qp['n']}, t={qp['t']})"
        )
        on = r["overnight_hold_21_to_09"]
        print(
            f"ON 21:00→09:00 hold:  {on['gross_bps']:+.1f} bps/day gross, "
            f"{on['net_taker_bps']:+.1f} bps/day net taker (n={on['n']}, t={on['t']})"
        )
        print(
            f"\nWeekend hourly {r['weekend_hourly']['mean_bps']:+.1f} bps "
            f"(n={r['weekend_hourly']['n']}) vs "
            f"weekday {r['weekday_hourly']['mean_bps']:+.1f} bps "
            f"(n={r['weekday_hourly']['n']})"
        )


if __name__ == "__main__":
    main()
