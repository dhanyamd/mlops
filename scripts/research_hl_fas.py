#!/usr/bin/env python3
"""Validate OUR FAS factor on Hyperliquid perp data (keyless, independent venue).

Reads /tmp/hl_pull/{COIN}_funding.json (8h funding) + {COIN}_candles.json (daily).
Builds weekly panels and computes OUR FAS factor exactly as research_fas_strategy.py,
then backtests on Hyperliquid's perps (a DIFFERENT exchange/venue from Binance
broad_pull) -> genuine out-of-sample check of our invented factor.

FAS (ours): cumulative funding accrual residualized on the price path, then faded
against recent price direction (trapped-crowd squeeze). See research_fas_strategy.py.
"""

import glob, json, os
import numpy as np
import pandas as pd

OUT = "/tmp/hl_pull"


def zs(s: pd.Series) -> pd.Series:
    r = s.rank(pct=True)
    return (r - 0.5) * 2


def zs_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda c: zs(c) if c.notna().sum() > 5 else c * 0.0)


def load_coin(coin):
    fp = os.path.join(OUT, f"{coin}_funding.json")
    cp = os.path.join(OUT, f"{coin}_candles.json")
    if not (os.path.exists(fp) and os.path.exists(cp)):
        return None
    f = json.load(open(fp))
    c = json.load(open(cp))
    fdf = pd.DataFrame(f)
    fdf["dt"] = pd.to_datetime(fdf["t"], unit="ms")
    fdf = fdf.set_index("dt")["r"].astype(float).sort_index()
    wf = fdf.resample("W").sum()  # sum of 8h funding rates per week
    cdf = pd.DataFrame(c)
    cdf["dt"] = pd.to_datetime(cdf["t"], unit="ms")
    cdf = cdf.set_index("dt").sort_index()
    wc = cdf["c"].astype(float).resample("W").last()
    return wf, wc


def backtest(score, ret, top=0.2, bot=0.2):
    pf = []
    for w in score.index:
        sc = score.loc[w].dropna()
        r = ret.loc[w].reindex(sc.index)
        sc = sc.reindex(r.dropna().index)
        r = r.reindex(sc.index)
        if len(sc) < 8:
            continue
        thr_hi = sc.quantile(1 - top)
        thr_lo = sc.quantile(bot)
        longs = sc[sc >= thr_hi].index
        shorts = sc[sc <= thr_lo].index
        if len(longs) == 0 or len(shorts) == 0:
            continue
        wk = r[longs].mean() - r[shorts].mean()
        wk = wk - 0.0010 * 2  # 10bps entry+exit (one rebalance/wk)
        pf.append(wk)
    return pd.Series(pf) if len(pf) >= 20 else None


def sharpe_ci(pf):
    pf = pf.dropna()
    if len(pf) < 10:
        return float("nan"), float("nan"), float("nan")
    mu, sd = pf.mean(), pf.std()
    sr = mu / sd * np.sqrt(52) if sd > 0 else 0.0
    rng = np.random.RandomState(0)
    N, B = len(pf), 2000
    arr = pf.values
    boot = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, N, N)
        s = arr[idx]
        boot[b] = (s.mean() / s.std() * np.sqrt(52)) if s.std() > 0 else 0.0
    return sr, float(np.percentile(boot, 5)), float(np.percentile(boot, 95))


def main():
    coins = sorted({os.path.basename(f)[0:-13] for f in glob.glob(f"{OUT}/*_funding.json")})
    print(f"== Hyperliquid OOS validation of OUR FAS factor == coins={len(coins)}")
    wfs, wcs = {}, {}
    for coin in coins:
        d = load_coin(coin)
        if d is None:
            continue
        wf, wc = d
        # require enough weekly history
        if wc.notna().sum() < 60:
            continue
        wfs[coin], wcs[coin] = wf, wc
    if not wfs:
        print("NO DATA YET — run pull_hyperliquid.py first")
        return
    WF = pd.DataFrame(wfs)
    WC = pd.DataFrame(wcs)
    WF, WC = WF.align(WC, join="inner")
    WC = WC.sort_index()
    WF = WF.reindex(WC.index)
    print(
        f"panel: {WC.shape[0]} weeks x {WC.shape[1]} perps  ({WC.index[0].date()}..{WC.index[-1].date()})"
    )

    ret = WC.pct_change()
    fwd = ret.shift(-1)

    # ---- OUR FAS ----
    fund_accrual = WF.rolling(12).sum()
    price_ret12 = WC.shift(1).div(WC.shift(13)) - 1.0
    pr_z = zs_df(price_ret12)
    fund_resid = fund_accrual.copy()
    for w in fund_accrual.index:
        y = fund_accrual.loc[w].dropna()
        if len(y) < 10:
            continue
        X = np.array(
            [
                price_ret12.loc[w].reindex(y.index).values,
                price_ret12.loc[w].reindex(y.index).abs().values,
            ]
        ).T
        m = ~np.isnan(X).any(axis=1)
        if m.sum() < 10:
            continue
        yy, XX = y.values[m], X[m]
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        fund_resid.loc[w, y.index[m]] = yy - XX @ beta
    fund_resid_z = zs_df(fund_resid)
    fas = zs_df(fund_resid_z * (-pr_z))
    r_recent = WC.shift(1).div(WC.shift(4)) - 1.0
    fas2 = zs_df(fund_resid_z * (-zs_df(r_recent)))

    print("\n=== OUR FAS on Hyperliquid (top/bot quintile L/S, 10bps) ===")
    for nm, sc in [("FAS", fas), ("FAS2", fas2)]:
        pf = backtest(sc, fwd)
        if pf is None:
            print(f"{nm}: insufficient data")
            continue
        sr, lo, hi = sharpe_ci(pf)
        print(f"{nm:5s} Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf)}")

    # benchmark: plain reversal on same universe (to show FAS is not just reversal)
    r4 = WC.shift(1).div(WC.shift(5)) - 1
    r8 = WC.shift(1).div(WC.shift(9)) - 1
    r12 = WC.shift(1).div(WC.shift(13)) - 1
    rev = -zs_df(zs_df(r4) + zs_df(r8) + zs_df(r12))
    pf = backtest(rev, fwd)
    sr, lo, hi = sharpe_ci(pf)
    print(
        f"\n[bench] REV_PLAIN Sharpe={sr:+.2f} CI[{lo:+.2f},{hi:+.2f}] nwk={len(pf) if pf is not None else 0}"
    )


if __name__ == "__main__":
    main()
