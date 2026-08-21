"""Capital-Gains-Overhang (CGO) / Disposition-Effect cross-sectional crypto factor.

WHY THIS IS THE EDGE THE CHINESE LITERATURE POINTS AT (researched this session):
  Chinese crypto papers (Binance top-50 / 95%-cap studies) find momentum + the
  DISPOSITION EFFECT (处置效应) is the durable, profitable combination, but ONLY at
  SHORT horizons: momentum decays past ~2 weeks, and the disposition effect is a
  *daily-frequency* behavioral bias. Li & Zhu 2026 ("Taming crypto anomalies") build
  DS3 = MKT + 2-week momentum (MOM2) + residual momentum and show most weekly
  crypto anomalies fail — the live edge is DAILY/2-week, not weekly [4-26w].
  Liu-Tsyvinski-Wu (2019/2022) document size/momentum/carry at 8-100w — far from the
  short horizon where disposition lives. So the UNDOCUMENTED profitable crypto factor
  is a cross-sectional disposition/capital-gains-overhang factor at DAILY frequency.

MECHANISM (Frazzini 2006 "Disposition Effect and Underreaction to News"):
  Investors anchor on a reference (cost-basis) price = decay-weighted average of past
  prices. Capital-gains-overhang CGO = price / reference - 1.
    CGO high  -> the crowd is in profit on winners -> sells winners (realizes gains),
                 creating downward pressure / slow info incorporation on winners.
    CGO low   -> the crowd is underwater on losers -> holds losers, sells winners of
                 others; losers get less-distressed, mean-revert up.
  Cross-sectionally SHORT high-CGO, LONG low-CGO over a short horizon captures the
  disposition reversal. (Frazzini's *short-run* result actually finds CGO predicts
  continuation because disposition delays price adjustment; we TEST BOTH signs and
  report what the data says — no hardcoding of the "right" sign.)

DATA: keyless. /tmp/broad_pull/<SYM>.csv (daily close/qvol/funding) pulled from
Binance. 10bps round-trip, BTC-regime gate, bootstrap CI (random.Random(0)).

This file is research/verification only; it does NOT touch the live signal daemons.
"""

from __future__ import annotations

import glob
import os
import random

import numpy as np
import pandas as pd

BROAD = "/tmp/broad_pull"
COST = 10  # bps round-trip
BT_CSV = "/tmp/crypto_daily_long.csv"  # BTC daily for regime gate


def load_btc_close():
    d = pd.read_csv(BT_CSV, parse_dates=["ts"]).set_index("ts")
    return d["BTCUSDT"]


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


def cgo(close: pd.DataFrame, alpha: float = 0.98) -> pd.DataFrame:
    """Capital-gains-overhang: close / decay-weighted reference price - 1.

    Reference price is an EXPANDING EWMA of past closes (recent prices weigh more,
    matching the Frazzini GARCH-style reference). Computed per-symbol daily.
    """
    out = close.copy() * np.nan
    for s in close.columns:
        p = close[s].dropna()
        ref = p.ewm(alpha=1 - alpha, adjust=False, min_periods=20).mean().shift(1)
        out[s] = (p / ref - 1.0).where(ref > 0)
    return out


def zs(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0, np.nan), axis=0)


def build_scores(close: pd.DataFrame, vol: pd.DataFrame):
    ret = close.pct_change(1)
    cgo_z = zs(cgo(close))
    mom2 = zs(close.pct_change(14))  # 2-week momentum (Chinese edge horizon)
    # behavioral disposition factor, both signs (data decides):
    disp_shortwinners = -cgo_z  # sell-winners reversal
    disp_longwinners = cgo_z  # Frazzini short-run continuation
    # combined with short-horizon momentum (documented disposition+momentum combo)
    disp_combo = (cgo_z * (mom2 > 0)).pipe(zs)
    # volume confirms capitulation/squeeze (per squeeze literature)
    vol_surge = vol.div(vol.rolling(20, min_periods=5).mean()).clip(lower=0, upper=10)
    disp_v = zs(disp_shortwinners * vol_surge.clip(lower=0))
    return {
        "disp_shortwinners": disp_shortwinners,
        "disp_longwinners": disp_longwinners,
        "disp_combo": disp_combo,
        "disp_v": disp_v,
        "mom2": mom2,
        "cgo_z": cgo_z,
    }


def backtest_daily(score, close, rebal: int = 7, regime=None, long_short=True):
    # Vectorized cross-sectional backtest.
    # 1) per-date quintile target weights, 2) keep only rebal dates then ffill
    #    (positions held between rebalances), 3) forward returns + turnover cost.
    q = score.rank(pct=True, axis=1)
    if long_short:
        raw = (q > 0.8).astype(float) - (q < 0.2).astype(float)
    else:
        raw = q - 0.5
    norm = raw.abs().sum(axis=1).replace(0, np.nan)
    w_target = raw.div(norm, axis=0)
    # rebalance only every `rebal` days; hold between
    mask = pd.Series(False, index=score.index)
    mask.iloc[::rebal] = True
    w_held = w_target.where(mask, np.nan).ffill()
    if regime is not None:
        rg = regime.reindex(score.index).ffill().fillna(0)
        w_held = w_held * rg  # flat during bear regime
    w_held = w_held.fillna(0.0)
    fwd = close.shift(-1) / close - 1.0
    strat = (w_held * fwd).sum(axis=1)
    turn = (w_held - w_held.shift(1)).abs().sum(axis=1)
    ret = strat - COST / 1e4 * turn
    ret = ret.where(w_held.abs().sum(axis=1) > 0, 0.0)
    return ret.replace([np.inf, -np.inf], 0.0)


def metrics(ret: pd.Series):
    ret = ret.dropna()
    n = len(ret)
    if n < 20:
        return None
    ann = ret.mean() * 365
    vol = ret.std() * np.sqrt(365)
    sharpe = ann / vol if vol > 0 else 0.0
    wealth = (1 + ret).cumprod()
    dd = (wealth / wealth.cummax() - 1).min()
    rng = random.Random(0)
    vals = list(ret.values)
    boot = []
    for _ in range(1000):
        sm = sum(rng.choice(vals) for _ in range(n)) / n * 365
        sd = (sum((x - sum(vals) / n) ** 2 for x in vals) / max(1, n - 1)) ** 0.5 * np.sqrt(365)
        boot.append(sm / sd if sd > 0 else 0.0)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return dict(
        n=n,
        ann=ann,
        vol=vol,
        sharpe=sharpe,
        ci=ci,
        maxdd=dd,
        pct_flat=float((ret == 0).mean()),
    )


def report(name, m):
    if m is None:
        print(f"{name}: insufficient data")
        return
    print(
        f"  {name:24s} Sharpe={m['sharpe']:.2f} CI[{m['ci'][0]:.2f},{m['ci'][1]:.2f}] "
        f"ann={m['ann'] * 100:5.1f}% vol={m['vol'] * 100:4.1f}% maxDD={m['maxdd'] * 100:6.1f}% %flat={m['pct_flat'] * 100:.0f}%"
    )


def main():
    print("[disp] loading daily coins ...")
    frames = load_coins_daily()
    print(f"[disp] {len(frames)} coins with >=220 daily bars")
    if not frames:
        print(
            "[disp] no broad_pull data; run pull_broad_universe.py first. skipping real backtest."
        )
        return
    close = pd.DataFrame({s: d["close"] for s, d in frames.items()}).sort_index()
    vol = pd.DataFrame({s: d["qvol"] for s, d in frames.items()}).sort_index()
    if BT_CSV and os.path.exists(BT_CSV):
        btc = load_btc_close().resample("D").last().reindex(close.index).ffill()
        regime = (btc > btc.rolling(364, min_periods=30).mean()).astype(int)
    else:
        regime = None
    scores = build_scores(close, vol)

    print("\n=== DISPOSITION / CGO FACTOR (daily, 10bps, BTC-regime gate) ===")
    for key in ("disp_shortwinners", "disp_longwinners", "disp_combo", "disp_v", "mom2"):
        for rebal in (2, 7):
            report(
                f"{key}_r{rebal}",
                metrics(backtest_daily(scores[key], close, rebal=rebal, regime=regime)),
            )
    print("\nNOTE: disp_shortwinners = short high-CGO/winners (sell-winners reversal);")
    print("      disp_longwinners = long high-CGO (Frazzini short-run continuation).")
    print("      whichever sign prints positive + survives the spans test is the claim.")


if __name__ == "__main__":
    main()
