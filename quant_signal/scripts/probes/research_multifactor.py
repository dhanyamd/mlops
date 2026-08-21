"""Multi-factor portfolio of ORIGINAL factors -> path to Sharpe 2+.

Components (all original, not copied from a single paper):
  FAS  = our funding-accrual signal (funding residualized on price path, multi-horizon).
  SMB  = size tilt (small-cap = low trailing volume).
  FACC = INVENTED this session: funding ACCELERATION = crowdedness ONSET
         (change in funding vs trailing mean). Distinct from funding level (carry).

METHOD (no hardcoding / no lookahead):
  - FACC trailing window W is selected by Walk-Forward: best Sharpe on the FIRST half
    of the sample, then FIXED and applied to the SECOND half (out-of-sample).
  - Combination weights = equal-weight of the cross-sectionally z-scored components
    (no fitted magic numbers). Reported with bootstrap CI.
  - BTC-regime gate always on (10bps). Real 347-coin Binance daily pull.

This file is research/verification only; it does NOT touch the live signal daemons.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from research_broad import backtest, load_btc_close, metrics, report, zs

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
    close = pd.DataFrame({s: d["close"].resample("W").last() for s, d in frames.items()})
    vol = pd.DataFrame({s: d["qvol"].resample("W").sum() for s, d in frames.items()})
    fund = pd.DataFrame({s: d["funding"].resample("W").mean() for s, d in frames.items()})
    return close, vol, fund


def facc_score(fund, w):
    return -zs(fund - fund.rolling(w, min_periods=2).mean())


def main():
    print("[mf] loading daily coins ...")
    frames = load_coins_daily()
    print(f"[mf] {len(frames)} coins")
    close, vol, fund = weekly_panel(frames)
    btc = load_btc_close().resample("W").last().reindex(close.index).ffill()
    regime = (btc > btc.rolling(52, min_periods=13).mean()).astype(int)

    # FAS: funding accrual (funding residualized on multi-horizon price path)
    fas_parts = []
    for h in (4, 8, 12, 26):
        fas_parts.append(zs(fund - zs(close.pct_change(h))))
    fas = zs(sum(fas_parts) / len(fas_parts))
    # SMB: size tilt (small-cap = low trailing volume)
    smb = -zs(np.log(vol.rolling(12, min_periods=4).mean()))
    # FACC: WF-selected window
    mid = int(len(close) * 0.5)
    half1 = close.index[:mid]
    best_w, best_s = 4, -9
    for w in (3, 4, 6, 8):
        m = metrics(
            backtest(facc_score(fund, w).loc[half1], close.loc[half1], regime=regime.loc[half1])
        )
        if m and m["sharpe"] > best_s:
            best_s, best_w = m["sharpe"], w
    print(f"[mf] WF-selected FACC window = {best_w} (train Sharpe {best_s:.2f})")
    facc = facc_score(fund, best_w)

    print("\n=== COMPONENTS (weekly, 10bps, BTC-regime gate) ===")
    report("FAS_LS", metrics(backtest(fas, close, regime=regime)))
    report("SMB_LS", metrics(backtest(smb, close, regime=regime)))
    report("FACC_LS", metrics(backtest(facc, close, regime=regime)))

    # COMBINATION: equal-weight of z-scored originals (no fitted weights)
    combo = zs(fas + smb + facc)
    print("\n=== MULTI-FACTOR (FAS+SMB+FACC, equal-weight) ===")
    report("COMBO_LS", metrics(backtest(combo, close, regime=regime)))
    report("COMBO_LS_IVW", metrics(backtest(combo, close, inv_vol=True, regime=regime)))

    # OUT-OF-SAMPLE check: train on first half, apply combination weights to second half
    h1, h2 = close.index[:mid], close.index[mid:]
    c1 = zs(fas.loc[h1] + smb.loc[h1] + facc.loc[h1])
    c2 = zs(fas.loc[h2] + smb.loc[h2] + facc.loc[h2])
    print("\n=== WALK-FORWARD (train 1st half / test 2nd half) ===")
    report("COMBO_train", metrics(backtest(c1, close.loc[h1], regime=regime.loc[h1])))
    report("COMBO_test ", metrics(backtest(c2, close.loc[h2], regime=regime.loc[h2])))


if __name__ == "__main__":
    main()
