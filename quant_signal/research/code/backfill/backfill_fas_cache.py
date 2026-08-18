"""Backfill a multi-YEAR warm cache for the FAS/SMB/RCGO research book.

The existing cache (/tmp/quant_cache/asym_warm_start.json.binance) holds only
~55 weeks of funding, which is the binding constraint on statistical power:
at Sharpe 2.28 that is t = 2.35, below the t > 3.0 bar Harvey et al. (2016)
set for newly-proposed factors. Binance perp funding starts 2019-09-10, so
~360 weeks are available -- enough for t ~ 6 and for genuine out-of-sample and
regime testing.

Writes the SAME schema scripts/research_fas_clean.load() already reads:

    {"ts": <ms>, "bars": {SYM: [[ts_ms, close, volume], ...]},
                  "funding": {SYM: [[ts_ms, rate], ...]}}

Bars are DAILY, not hourly. load() resamples to weekly (W-MON) and daily, so
daily klines carry every quantity the book uses (weekly close, weekly volume,
daily close/volume for CGO) at ~1/24th the download. Hourly would add nothing
the factors read.

Keyless Binance futures REST (fapi.binance.com), politely paced. Resumable:
each symbol is written as it completes, so a network failure loses one symbol
rather than the run.

Usage:
    uv run python -m scripts.backfill_fas_cache
    uv run python -m scripts.backfill_fas_cache --start 2019-09-01 --out /tmp/quant_cache/fas_long.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FAPI = "https://fapi.binance.com"
PACE_S = 0.25          # keyless weight budget is generous; stay well inside it
RETRIES = 4


def _get(url: str) -> list | dict:
    last = None
    for attempt in range(RETRIES):
        try:
            time.sleep(PACE_S)
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {RETRIES} tries: {url} ({last})")


def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """All (fundingTime, fundingRate) for symbol, paginated forward."""
    out: list[list] = []
    cur = start_ms
    while cur < end_ms:
        url = f"{FAPI}/fapi/v1/fundingRate?symbol={symbol}&startTime={cur}&limit=1000"
        page = _get(url)
        if not isinstance(page, list) or not page:
            break
        for r in page:
            t = int(r["fundingTime"])
            if t <= end_ms:
                out.append([t, float(r["fundingRate"])])
        nxt = int(page[-1]["fundingTime"]) + 1
        if nxt <= cur:
            break
        cur = nxt
        if len(page) < 1000:
            break
    return out


def fetch_daily_bars(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """Daily [close_time_ms, close, quote_volume] klines, paginated forward.

    close_time is used as the bar stamp so a bar is stamped when it COMPLETED
    -- matching the live registry's window_end_ms convention and keeping the
    replay point-in-time.
    """
    out: list[list] = []
    cur = start_ms
    while cur < end_ms:
        url = (
            f"{FAPI}/fapi/v1/klines?symbol={symbol}&interval=1d"
            f"&startTime={cur}&limit=1500"
        )
        page = _get(url)
        if not isinstance(page, list) or not page:
            break
        for k in page:
            close_time = int(k[6])
            if close_time <= end_ms:
                # k[5] is BASE asset volume; k[7] is QUOTE volume. The live warm
                # cache stores BASE volume, and SMB is -z(log volume): using
                # quote volume adds log(price) to the factor (quote = base *
                # price), silently turning a size tilt into a size+price tilt
                # and corrupting the CGO turnover weights too. Must match the
                # live convention.
                out.append([close_time, float(k[4]), float(k[5])])  # close, BASE vol
        nxt = int(page[-1][6]) + 1
        if nxt <= cur:
            break
        cur = nxt
        if len(page) < 1500:
            break
    return out


def discover_universe(min_history_days: int) -> list[str]:
    """Every TRADING USDT-margined perpetual, oldest-listed first.

    Liu & Tsyvinski (J. Finance 2022) build the crypto cross-section from
    "all coins with market capitalisation above one million dollars" -- 109
    names in 2014 rising to 1,707 -- not a hand-picked list. A cross-sectional
    rank is only as informative as the cross-section it ranks, and a 30-name
    universe gives the quintile sort very little to work with.

    No hardcoded symbol list and no hardcoded liquidity cut here: this returns
    the full tradable set, and the LIQUIDITY SCREEN is applied downstream by
    research_fas_clean._liquidity_mask, which derives it from the data
    (positive volume in >=99% of weeks, finite positive closes).
    """
    info = _get(f"{FAPI}/fapi/v1/exchangeInfo")
    out = []
    for s in info.get("symbols", []):
        if (
            s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
        ):
            out.append((int(s.get("onboardDate") or 0), s["symbol"]))
    cutoff = int(time.time() * 1000) - min_history_days * 86_400_000
    return [sym for onboard, sym in sorted(out) if onboard and onboard <= cutoff]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2019-09-01", help="UTC date, YYYY-MM-DD")
    ap.add_argument("--end", default="", help="UTC date; blank = now")
    ap.add_argument("--out", default="/tmp/quant_cache/fas_long.json")
    ap.add_argument("--symbols", default="", help="CSV override")
    ap.add_argument("--all-perps", action="store_true",
                    help="discover every USDT perp instead of the 31-name live universe")
    ap.add_argument("--min-history-days", type=int, default=400,
                    help="a name must have been listed at least this long to enter the pool")
    a = ap.parse_args()

    from config.settings import csv_list, get_settings

    settings = get_settings()
    if a.symbols:
        symbols = csv_list(a.symbols)
    elif a.all_perps:
        symbols = discover_universe(a.min_history_days)
    else:
        symbols = csv_list(settings.stream_xs_universe)

    start_ms = int(datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = (
        int(datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc).timestamp() * 1000)
        if a.end
        else int(time.time() * 1000)
    )
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"backfilling {len(symbols)} symbols  {a.start} -> {a.end or 'now'}")
    print(f"target: {out_path}\n")

    bars: dict[str, list] = {}
    funding: dict[str, list] = {}
    for i, sym in enumerate(symbols, 1):
        try:
            b = fetch_daily_bars(sym, start_ms, end_ms)
            f = fetch_funding(sym, start_ms, end_ms)
        except RuntimeError as e:
            print(f"  [{i:2d}/{len(symbols)}] {sym:10} FAILED: {e}")
            continue
        bars[sym] = b
        funding[sym] = f
        span_w = (b[-1][0] - b[0][0]) / (7 * 86400_000) if len(b) > 1 else 0
        fspan_w = (f[-1][0] - f[0][0]) / (7 * 86400_000) if len(f) > 1 else 0
        first = (
            datetime.fromtimestamp(b[0][0] / 1000, timezone.utc).strftime("%Y-%m-%d")
            if b
            else "-"
        )
        print(
            f"  [{i:2d}/{len(symbols)}] {sym:10} bars={len(b):5d} ({span_w:5.0f}w from {first})"
            f"  funding={len(f):6d} ({fspan_w:5.0f}w)"
        )
        # write incrementally so a later failure never loses completed work
        out_path.write_text(
            json.dumps({"ts": int(time.time() * 1000), "bars": bars, "funding": funding})
        )

    weeks = [
        (v[-1][0] - v[0][0]) / (7 * 86400_000) for v in funding.values() if len(v) > 1
    ]
    print(f"\nwrote {out_path}")
    print(f"  symbols with funding: {len(weeks)}")
    if weeks:
        print(f"  funding span: min {min(weeks):.0f}w  median {sorted(weeks)[len(weeks)//2]:.0f}w  max {max(weeks):.0f}w")
        print("  (previous cache: 55 weeks)")


if __name__ == "__main__":
    main()
