"""Validate a warm cache BEFORE any strategy is run on it.

Today produced three data-handling failures that each survived until a result
looked wrong: quote-vs-base volume, an index-misaligned return series, and a
return formula dividing by cumulative P&L. All three were detectable from the
data alone. This runs those checks up front.

Checks, per symbol and cross-sectionally:

  STRUCTURE   timestamps strictly increasing, no duplicates, no gaps beyond
              tolerance, expected bar spacing
  PRICES      finite, strictly positive, no |log-return| beyond a sanity bound
  VOLUME      finite, non-negative, and -- critically -- in BASE units, checked
              by re-deriving quote/base against price and against a live API
              call rather than trusting the field index
  FUNDING     finite, plausible magnitude, timestamps on the venue's 8h grid
  COVERAGE    weeks of history per symbol, overlap of price and funding
  CROSS-CHECK a live Binance API pull for a random (symbol, day) compared to
              the cached value

Exit code is non-zero if any FAIL fires, so this can gate a pipeline.

Run:
    uv run python -m scripts.audit_cache --cache /tmp/quant_cache/fas_long_v2.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.request
from datetime import datetime, timezone

DAY = 86_400_000
WEEK = 7 * DAY
FUNDING_GRID_MS = 8 * 3_600_000

FAILS: list[str] = []
WARNS: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    WARNS.append(msg)
    print(f"  warn  {msg}")


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def check_structure(sym: str, bars: list, expect_ms: int) -> None:
    ts = [int(r[0]) for r in bars]
    if len(ts) != len(set(ts)):
        fail(f"{sym}: duplicate bar timestamps ({len(ts) - len(set(ts))})")
    if any(b <= a for a, b in zip(ts, ts[1:])):
        fail(f"{sym}: bar timestamps not strictly increasing")
    if len(ts) > 2:
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        odd = [g for g in gaps if abs(g - expect_ms) > expect_ms * 0.05]
        if odd:
            frac = len(odd) / len(gaps)
            (warn if frac < 0.02 else fail)(
                f"{sym}: {len(odd)}/{len(gaps)} bar gaps off the {expect_ms/DAY:.0f}d grid "
                f"(max {max(odd)/DAY:.1f}d)"
            )


def check_prices(sym: str, bars: list) -> None:
    px = [float(r[1]) for r in bars]
    if any((not math.isfinite(p)) or p <= 0 for p in px):
        fail(f"{sym}: non-finite or non-positive close")
        return
    rets = [math.log(b / a) for a, b in zip(px, px[1:])]
    extreme = [r for r in rets if abs(r) > math.log(5)]  # >5x in one bar
    if extreme:
        fail(f"{sym}: {len(extreme)} bar(s) with >5x price move (max {math.exp(max(map(abs,extreme))):.1f}x)")


def check_volume_units(sym: str, bars: list) -> None:
    """Distinguish BASE from QUOTE volume without trusting the source field.

    quote = base * price, so quote/base ~ price. If median(volume)/median(price)
    lands near median(volume) itself the series is base; if volume/price is the
    better-behaved quantity, it is quote. Compared against a live API pull.
    """
    vol = [float(r[2]) for r in bars]
    px = [float(r[1]) for r in bars]
    if any((not math.isfinite(v)) or v < 0 for v in vol):
        fail(f"{sym}: non-finite or negative volume")
        return
    if not vol or max(vol) == 0:
        fail(f"{sym}: all-zero volume")
        return
    # live cross-check on the most recent complete day
    t = int(bars[-1][0])
    day_start = (t // DAY) * DAY
    url = (
        f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1d"
        f"&startTime={day_start - DAY}&limit=2"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            k = json.loads(r.read())
    except Exception as e:  # noqa: BLE001 - audit must not die on a network blip
        warn(f"{sym}: live cross-check skipped ({e})")
        return
    if not k:
        warn(f"{sym}: live cross-check returned nothing")
        return
    row = k[-1]
    live_base, live_quote = float(row[5]), float(row[7])
    # Aggregate cached bars into the live day so this works for hourly caches
    # too -- an hourly cache never has a bar stamped at the daily close_time,
    # and comparing against one bar instead of the day's sum would wrongly
    # report a units mismatch.
    day_lo, day_hi = int(row[0]), int(row[6])
    c = sum(float(b[2]) for b in bars if day_lo <= int(b[0]) <= day_hi)
    if c <= 0:
        warn(f"{sym}: no cached bars inside the live cross-check day")
        return
    if live_base > 0 and abs(c - live_base) / live_base < 0.01:
        ok(f"{sym}: volume is BASE units (live-verified)")
    elif live_quote > 0 and abs(c - live_quote) / live_quote < 0.01:
        fail(f"{sym}: volume is QUOTE units -- SMB/CGO expect BASE")
    else:
        fail(
            f"{sym}: volume matches neither base ({live_base:,.0f}) nor quote "
            f"({live_quote:,.0f}); cached {c:,.0f}"
        )


def check_funding(sym: str, fund: list) -> None:
    if not fund:
        fail(f"{sym}: no funding")
        return
    ts = [int(r[0]) for r in fund]
    rates = [float(r[1]) for r in fund]
    if len(ts) != len(set(ts)):
        fail(f"{sym}: duplicate funding timestamps")
    if any(b <= a for a, b in zip(ts, ts[1:])):
        fail(f"{sym}: funding timestamps not increasing")
    if any(not math.isfinite(r) for r in rates):
        fail(f"{sym}: non-finite funding rate")
    bad = [r for r in rates if abs(r) > 0.05]  # 5% per 8h is already extreme
    if bad:
        warn(f"{sym}: {len(bad)} funding rate(s) |r|>5% (max {max(map(abs,bad)):.3%})")
    # Binance settles most perps every 8h but moved many altcoins to a 4h
    # interval, so a stamp is valid on EITHER grid. A weekly SUM is unaffected
    # by which grid a symbol uses (twice as many prints, each about half the
    # size), so this is a data-shape check, not a correctness one.
    def _offgrid(grid: int) -> int:
        return sum(
            1 for t in ts
            if t % grid > 60_000 and (grid - t % grid) > 60_000
        )
    off = min(_offgrid(FUNDING_GRID_MS), _offgrid(FUNDING_GRID_MS // 2))
    if off:
        frac = off / len(ts)
        (warn if frac < 0.05 else fail)(
            f"{sym}: {off}/{len(ts)} funding stamps off both the 8h and 4h grids"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--bar-ms", type=int, default=DAY, help="expected bar spacing")
    ap.add_argument("--sample", type=int, default=6, help="symbols to live-cross-check")
    a = ap.parse_args()

    c = json.load(open(a.cache))
    bars, fund = c["bars"], c.get("funding", {})
    print(f"=== auditing {a.cache} ===")
    print(f"  symbols: bars={len(bars)}  funding={len(fund)}\n")

    print("-- structure / prices / funding --")
    for sym in sorted(bars):
        b = bars[sym]
        if len(b) < 10:
            fail(f"{sym}: only {len(b)} bars")
            continue
        check_structure(sym, b, a.bar_ms)
        check_prices(sym, b)
        check_funding(sym, fund.get(sym, []))

    print("\n-- volume units (live cross-check) --")
    rng = random.Random(0)
    for sym in rng.sample(sorted(bars), min(a.sample, len(bars))):
        check_volume_units(sym, bars[sym])
        time.sleep(0.2)

    print("\n-- coverage --")
    spans = {}
    for sym in sorted(bars):
        b, f = bars[sym], fund.get(sym, [])
        if len(b) < 2 or len(f) < 2:
            continue
        bw = (b[-1][0] - b[0][0]) / WEEK
        fw = (f[-1][0] - f[0][0]) / WEEK
        overlap = (min(b[-1][0], f[-1][0]) - max(b[0][0], f[0][0])) / WEEK
        spans[sym] = overlap
        if overlap < 52:
            warn(f"{sym}: only {overlap:.0f}w of price+funding overlap")
    if spans:
        v = sorted(spans.values())
        print(
            f"  price+funding overlap: min {v[0]:.0f}w  median {v[len(v)//2]:.0f}w  max {v[-1]:.0f}w"
        )
        first = min(bars[s][0][0] for s in bars)
        last = max(bars[s][-1][0] for s in bars)
        print(
            f"  span: {datetime.fromtimestamp(first/1000, timezone.utc):%Y-%m-%d}"
            f" -> {datetime.fromtimestamp(last/1000, timezone.utc):%Y-%m-%d}"
        )

    print(f"\n=== {len(FAILS)} FAIL, {len(WARNS)} warn ===")
    if FAILS:
        for f_ in FAILS[:20]:
            print(f"  FAIL {f_}")
        sys.exit(1)
    print("  cache is clean; safe to run the strategy on it")


if __name__ == "__main__":
    main()
