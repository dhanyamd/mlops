"""INVENTED factors from first-principles microstructure reasoning (NOT literature combos).

INVENTION 1 — LPM: Leverage-Premium Mismatch.
  In perpetuals the funding rate is the MARKET-CLEARING PRICE OF LEVERAGE (what
  leveraged longs pay to hold). Realized vol is ACTUAL risk. In an un-crowded
  market they track: you pay more for leverage when it is riskier, so
  funding/vol is stationary. When the ratio DETACHES from its own norm — spikes
  because leverage demand outruns real risk — that is complacent crowding: the
  position is fragile and mean-reverts (SHORT). When it COLLAPSES (funding ~0 or
  negative while vol stays high) the crowd has capitulated -> washed out -> rebounds
  (LONG). This is a leverage-demand/risk MISMATCH, not funding-level (carry) or vol
  alone or a linear combo. Cross-sectionally: score = -z(LPM_dev). Keyless: Binance
  daily funding + close (vol derived). 10bps, BTC-regime gate.

INVENTION 2 — FACC: Funding Acceleration (crowdedness ONSET).
  Known funding factors use the LEVEL (carry / mean-reversion). The CHANGE in
  funding relative to its trailing mean captures leveraged crowding as it BUILDS,
  before price reflects it. Accelerating-up funding = leverage piling in = fragility
  (SHORT). score = -z(funding - trailing-mean funding).

Tested on the real 347-coin Binance daily pull. Reuses research_broad backtest/metrics
(proven). This file is research/verification only; it does NOT touch live daemons.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from research_broad import (
    avg_rank_corr,
    backtest,
    load_btc_close,
    metrics,
    orthogonalize,
    report,
    zs,
)

BROAD = "/tmp/broad_pull"
BT_CSV = "/tmp/crypto_daily_long.csv"


def load_coins_daily(min_days: int = 220):
    frames = {}
    for f in sorted(glob.glob(os.path.join(BROAD, "*.csv"))):
        sym = os.path.basename(f)[:-4]
        d = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
        d = d[["close", "qvol", "funding"]].dropna()
        if len(d) < min_days:
            continue
        d = d[~d.index.duplicated()]
        frames[sym] = d
    return frames


def weekly_panel(frames):
    close, vol, fund = {}, {}, {}
    for sym, d in frames.items():
        close[sym] = d["close"].resample("W").last()
        # realized vol from daily returns, 7d rolling, weekly-last
        r = d["close"].pct_change()
        vol[sym] = r.rolling(7).std().resample("W").last()
        fund[sym] = d["funding"].resample("W").mean()
    return pd.DataFrame(close), pd.DataFrame(vol), pd.DataFrame(fund)


def build_scores(close, vol, fund):
    # INVENTION 1: LPM = funding / realized-vol, then deviation from own trailing mean
    ratio = fund / (vol + 1e-6)
    lpm_dev = ratio - ratio.rolling(12, min_periods=4).mean()
    lpm = -zs(lpm_dev)  # short detached-high, long collapsed-low
    # INVENTION 2: FACC = funding change vs trailing mean (crowdedness onset)
    facc = -zs(fund - fund.rolling(4, min_periods=2).mean())
    # benchmarks for orthogonaliy
    carry = -zs(fund)  # classic funding carry (short high funding)
    mom2 = zs(close.pct_change(2))
    return {"lpm": lpm, "facc": facc, "carry": carry, "mom2": mom2}


def main():
    print("[invent] loading daily coins ...")
    frames = load_coins_daily()
    print(f"[invent] {len(frames)} coins with >=220 daily bars")
    if not frames:
        print("[invent] no broad_pull data; run pull_broad_universe.py first.")
        return
    close, vol, fund = weekly_panel(frames)
    scores = build_scores(close, vol, fund)
    btc = load_btc_close().resample("W").last().reindex(close.index).ffill()
    regime = (btc > btc.rolling(52, min_periods=13).mean()).astype(int)

    print("\n=== INVENTED FACTORS (weekly, 10bps, BTC-regime gate) ===")
    for key in ("lpm", "facc", "carry", "mom2"):
        report(f"{key}_LS", metrics(backtest(scores[key], close, regime=regime)))
        report(f"{key}_LS_IVW", metrics(backtest(scores[key], close, inv_vol=True, regime=regime)))

    print("\n=== ORTHOGONALITY (is LPM distinct from carry/momentum?) ===")
    print(f"  rank-corr LPM~CARRY : {avg_rank_corr(scores['lpm'], scores['carry']):.2f}")
    print(f"  rank-corr LPM~MOM2  : {avg_rank_corr(scores['lpm'], scores['mom2']):.2f}")
    print(f"  rank-corr FACC~CARRY: {avg_rank_corr(scores['facc'], scores['carry']):.2f}")
    lpm_orth = orthogonalize(scores["lpm"], [scores["carry"], scores["mom2"]])
    print("  LPM residualized on [CARRY, MOM2] (the distinct part):")
    report("lpm_orth_LS", metrics(backtest(lpm_orth, close, regime=regime)))
    report("lpm_orth_LS_IVW", metrics(backtest(lpm_orth, close, inv_vol=True, regime=regime)))


if __name__ == "__main__":
    main()
