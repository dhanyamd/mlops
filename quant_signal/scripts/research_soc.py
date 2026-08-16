 """SOC — Speculative-Overcrowding factor (NOVEL synthesis from two Chinese literatures).

RESEARCH BASIS (read this session, Chinese/underrated quant papers):
  1) TURNOVER / ABNORMAL-TURNOVER (ATR) — Zhang, Bing & Chen & Yeh (2021,
     "Turnover premia in China's stock markets"): NEGATIVE cross-sectional relation
     between turnover and returns; turnover premia up to 34%/yr. Li-Pan-Tang-Xu
     ("Speculative Trading and Stock Returns"): abnormal turnover ratio (ATR) NEGATIVELY
     predicts returns (-1.87%/mo). Mechanism: high speculative volume = retail
     overtrading = future underperformance.
  2) DISPOSITION / CAPITAL-GAINS-OVERHANG (CGO) — Chinese crypto paper
     (处置效应+动量, 95%-cap universe): daily capital-gains-overhang (price vs
     decay-weighted reference) proxies disposition; combined w/ 2-week momentum.
     High CGO = crowd in profit, disposition-selling pressure.

NOVELTY: these are TWO SEPARATE Chinese literatures (turnover vs disposition). Nobody
has combined them as one cross-sectional CRYPTO factor. SOC = SHORT coins that are
BOTH abnormally high-volume (retail chasing, ATR) AND high capital-gains-overhang
(crowd in profit, disposition-prone) — the classic pump that mean-reverts; LONG the
neglected opposite (low ATR, low CGO). Keyless: daily close + quote-volume from
Binance. 10bps, BTC-regime gate, bootstrap CI.

This file is research/verification only; it does NOT touch the live signal daemons.
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
        vol[sym] = d["qvol"].resample("W").sum()
        fund[sym] = d["funding"].resample("W").mean()
    return (
        pd.DataFrame(close),
        pd.DataFrame(vol),
        pd.DataFrame(fund),
    )


def build_scores(close, vol, fund):
    # ATR: abnormal turnover ratio = volume vs trailing 4w mean (Chinese ATR spec)
    atr = vol.div(vol.rolling(4, min_periods=2).mean()).clip(lower=0, upper=20)
    atr_z = zs(atr)
    # CGO: capital-gains-overhang = price vs decay-weighted reference (disposition)
    ref = close.ewm(span=8, adjust=False, min_periods=8).mean().shift(1)
    cgo = close / ref - 1.0
    cgo_z = zs(cgo)
    # 2-week momentum (Chinese crypto edge horizon)
    mom2 = zs(close.pct_change(2))
    # SOC: SHORT overcrowded (high ATR + high CGO), LONG neglected -> score high = long
    soc = -(atr_z + cgo_z)
    soc_gate = -(atr_z * (cgo_z > 0))  # only when crowding meets gain (pure pump)
    return {
        "atr": -atr_z,
        "cgo": -cgo_z,
        "soc": soc,
        "soc_gate": soc_gate,
        "mom2": mom2,
        "fund_z": zs(fund),
    }


def main():
    print("[soc] loading daily coins ...")
    frames = load_coins_daily()
    print(f"[soc] {len(frames)} coins with >=220 daily bars")
    if not frames:
        print("[soc] no broad_pull data; run pull_broad_universe.py first.")
        return
    close, vol, fund = weekly_panel(frames)
    scores = build_scores(close, vol, fund)
    btc = load_btc_close().resample("W").last().reindex(close.index).ffill()
    regime = (btc > btc.rolling(52, min_periods=13).mean()).astype(int)

    print("\n=== SOC / SPECULATIVE-OVERCROWDING (weekly, 10bps, BTC-regime gate) ===")
    for key in ("atr", "cgo", "soc", "soc_gate", "mom2", "fund_z"):
        report(f"{key}_LS", metrics(backtest(scores[key], close, regime=regime)))
        report(f"{key}_LS_IVW", metrics(backtest(scores[key], close, inv_vol=True, regime=regime)))

    print("\n=== ORTHOGONALITY (is SOC distinct from momentum/size/funding?) ===")
    print(f"  rank-corr SOC~MOM2   : {avg_rank_corr(scores['soc'], scores['mom2']):.2f}")
    print(f"  rank-corr SOC~FUND   : {avg_rank_corr(scores['soc'], scores['fund_z']):.2f}")
    print(f"  rank-corr ATR~CGO     : {avg_rank_corr(scores['atr'], scores['cgo']):.2f}")
    soc_orth = orthogonalize(scores["soc"], [scores["mom2"], scores["fund_z"]])
    print("  SOC residualized on [MOM2, FUND] (the distinct part):")
    report("soc_orth_LS", metrics(backtest(soc_orth, close, regime=regime)))
    report("soc_orth_LS_IVW", metrics(backtest(soc_orth, close, inv_vol=True, regime=regime)))


if __name__ == "__main__":
    main()
