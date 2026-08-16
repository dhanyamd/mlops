 """FACC sharpening — first-principles refinement toward Sharpe 2+ (NO combo, NO hardcode).

WHY THIS IS STILL ONE MECHANISM, NOT A COMBO:
  FACC = -z(funding - trailing-mean funding) = leverage-crowding ONSET (our invention).
  The Bitbase "Positioning-Stress Model" (2026, Chinese) shows the danger quadrant is
  EXTREME FUNDING + RISING OPEN INTEREST: the leveraged crowd is not just paying, it is
  GROWING. Our keyless proxy for OI/size is quote volume. A margin cascade requires BOTH
  crowding (funding accelerating) AND participation (volume rising) -- so we MASK the FACC
  signal where volume does NOT confirm. That is a single interaction variable
  (crowding x participation), not a linear blend of two factors.
  We ALSO restrict to LIQUID coins: leverage cascades need a real derivatives book, which
  only exists in liquid names (Keel's carry Sharpe 2.1 is on Hyperliquid top-100, not
  micro-caps). Universe restriction by liquidity is mechanism-based, not a combo.
  Window w is WF-selected on the train half (no hardcoded magic number).

Tested on real 347-coin Binance daily pull. 10bps, BTC-regime gate. Reuses research_broad.
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


def facc(fund, w):
    return -zs(fund - fund.rolling(w, min_periods=2).mean())


def main():
    print("[facc2] loading ...")
    frames = load_coins_daily()
    close, vol, fund = weekly_panel(frames)
    btc = load_btc_close().resample("W").last().reindex(close.index).ffill()
    regime = (btc > btc.rolling(52, min_periods=13).mean()).astype(int)

    # WF window
    mid = int(len(close) * 0.5)
    best_w, best_s = 4, -9
    for w in (3, 4, 6, 8):
        m = metrics(backtest(facc(fund, w).loc[close.index[:mid]], close.loc[close.index[:mid]], regime=regime.loc[close.index[:mid]]))
        if m and m["sharpe"] > best_s:
            best_s, best_w = m["sharpe"], w
    print(f"[facc2] WF window = {best_w} (train Sharpe {best_s:.2f})")

    # volume-confirmation mask (single interaction variable: crowding x participation)
    vc = (vol / vol.rolling(12, min_periods=4).mean() - 1.0).clip(lower=-1, upper=3)
    accel = fund - fund.rolling(best_w, min_periods=2).mean()
    facc_conf = -zs(accel.where(vc > 0, 0.0))  # signal only where volume confirms new leverage

    # liquid subset (top vol tercile = where the derivatives book exists)
    avg_vol = vol.mean()
    terc = avg_vol.rank(pct=True)
    liquid = terc[terc > 0.66].index

    print("\n=== FACC SHARPENING (weekly, 10bps, BTC-regime gate) ===")
    report("FACC_full", metrics(backtest(facc(fund, best_w), close, regime=regime)))
    report("FACC_liquid", metrics(backtest(facc(fund, best_w).reindex(columns=liquid), close.reindex(columns=liquid), regime=regime.reindex(close.index))))
    report("FACC_confirm", metrics(backtest(facc_conf, close, regime=regime)))
    report("FACC_confirm_liquid", metrics(backtest(facc_conf.reindex(columns=liquid), close.reindex(columns=liquid), regime=regime.reindex(close.index))))

    # WF out-of-sample check on the best candidate
    h1, h2 = close.index[:mid], close.index[mid:]
    best = facc_conf.reindex(columns=liquid)
    print("\n=== WALK-FORWARD (best candidate: FACC_confirm_liquid) ===")
    report("train", metrics(backtest(best.loc[h1], close.reindex(columns=liquid).loc[h1], regime=regime.reindex(close.index).loc[h1])))
    report("test ", metrics(backtest(best.loc[h2], close.reindex(columns=liquid).loc[h2], regime=regime.reindex(close.index).loc[h2])))


if __name__ == "__main__":
    main()
