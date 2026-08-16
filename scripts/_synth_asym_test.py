"""Synthetic test for the FAS_avg+SMB asym signal (no network, no redis).

Verifies the signal produces real LONG/SHORT selections (not all-FLAT) when the
BTC regime gate passes, and goes FLAT when history is too short. Drives the pure
logic (_selection / _fas_scores) directly with injected hourly history.
"""

import math
import random

from stream.asym_signal import AsymSignal, _WEEK_MS, _DAY_MS, _HOUR_MS

random.seed(0)
SYMS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
]


def build(sig, days: int, btc_uptrend: bool):
    """Seed sig._closes (hourly) + sig._funding (8h) for `days` of history."""
    start = 1_700_000_000_000  # arbitrary fixed epoch ms (Monday-ish)
    n = days * 24
    for s in SYMS:
        closes = []
        ends = []
        for i in range(n):
            e = start + i * _HOUR_MS
            drift = 0.0002 if (s == "BTCUSDT" and btc_uptrend) else 0.0
            r = drift + random.gauss(0, 0.004)
            price = 100.0 * (1.0 + sum([0.0]) + 0)  # placeholder; build below
            closes.append(price)
            ends.append(e)
        # build a coherent random walk
        px = 100.0 if s != "BTCUSDT" else 20000.0
        walk = []
        for i in range(n):
            d = 0.0003 if (s == "BTCUSDT" and btc_uptrend) else 0.0
            px = px * (1.0 + d + random.gauss(0, 0.004))
            walk.append(px)
        hist = []
        for i in range(n):
            vol = random.uniform(50.0, 500.0)
            hist.append((ends[i], walk[i], vol))
        sig._closes[s] = hist
        # funding: 8h events, magnitude loosely tied to recent return sign
        fev = []
        for j in range(days * 3):
            fe = start + j * 8 * _HOUR_MS
            # small persistent per-symbol bias so cross-section spreads
            rate = random.gauss(0, 0.0001) + 0.0002 * (1 if s in ("BTCUSDT", "ETHUSDT") else -1)
            fev.append((fe, rate))
        sig._funding[s] = fev


def counts(sel):
    c = {"LONG": 0, "SHORT": 0, "FLAT": 0}
    for s, (d, _) in sel.items():
        c[d] += 1
    return c


def main():
    # ---- Case 1: full 400d history, BTC uptrend -> regime passes ----
    sig = AsymSignal(
        None,
        prediction_prefix="x",
        universe=SYMS,
        min_symbols=8,
        regime_slow_days=364,
        horizons=[4, 8, 12, 26],
        accrual_weeks=12,
        smb_weeks=12,
    )
    build(sig, days=400, btc_uptrend=True)
    last_end = sig._closes["BTCUSDT"][-1][0]
    sel = sig._selection(last_end)
    print("CASE 1 (400d, BTC up):", counts(sel))

    # ---- Case 2: truncated 360d -> regime gate should FLAT (warm-start margin) ----
    sig2 = AsymSignal(
        None,
        prediction_prefix="x",
        universe=SYMS,
        min_symbols=8,
        regime_slow_days=364,
        horizons=[4, 8, 12, 26],
        accrual_weeks=12,
        smb_weeks=12,
    )
    build(sig2, days=360, btc_uptrend=True)
    last_end2 = sig2._closes["BTCUSDT"][-1][0]
    sel2 = sig2._selection(last_end2)
    print("CASE 2 (360d, BTC up):", counts(sel2), "<- expect FLAT (gate needs 365d)")

    # ---- Case 3: BTC downtrend -> regime FLAT ----
    sig3 = AsymSignal(
        None,
        prediction_prefix="x",
        universe=SYMS,
        min_symbols=8,
        regime_slow_days=364,
        horizons=[4, 8, 12, 26],
        accrual_weeks=12,
        smb_weeks=12,
    )
    build(sig3, days=400, btc_uptrend=False)
    last_end3 = sig3._closes["BTCUSDT"][-1][0]
    sel3 = sig3._selection(last_end3)
    print("CASE 3 (400d, BTC down):", counts(sel3), "<- expect FLAT (bear regime)")

    # sanity: confirm scores are non-degenerate in case 1
    sc = sig._fas_scores(last_end)
    print(
        "CASE 1 score spread: min=%.2f max=%.2f n=%d"
        % (min(sc.values()), max(sc.values()), len(sc))
    )


if __name__ == "__main__":
    main()
