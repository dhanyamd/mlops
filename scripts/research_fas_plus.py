"""FAS+ : can we ADD a NOVEL orthogonal component to FAS to reach Sharpe 2+?

FAS (funding-accrual squeeze, OURS) = +1.55 weekly (POST-2024 +1.72). It is our only
working novel factor. This script researches whether FAS can be lifted toward 2+ by:
  (A) internal enhancement: accrual window sweep {4,8,12,26}w, and a DAILY-frequency
      FAS (more responsive funding signal, held weekly);
  (B) a NOVEL additive lens: FAS_SLOPE = the TREND of residualized funding pressure
      (accelerating vs decaying funding squeeze) -- distinct from FAS's level lens;
  (C) orthogonal combo: FAS_level + FAS_slope (both ours, both funding-derived but
      capturing different temporal structure -> hopefully additive, not redundant).

All keyless (Binance public: close, qvol, funding). 10bps costs + BTC-regime gate.
"""

import glob, os, random
import numpy as np
import pandas as pd

BROAD = "/tmp/broad_pull"
MIN_WEEKS = 150


def zs(s):
    r = s.rank(pct=True)
    return (r - 0.5) * 2


def zs_df(df):
    return df.apply(lambda c: zs(c) if c.notna().sum() > 5 else c * 0.0)


def load():
    frames = []
    for f in glob.glob(f"{BROAD}/*.csv"):
        d = pd.read_csv(f, parse_dates=["date"])
        d["sym"] = os.path.basename(f)[:-4]
        frames.append(d[["date", "sym", "close", "qvol", "funding"]])
    raw = pd.concat(frames, ignore_index=True)
    close = raw.pivot(index="date", columns="sym", values="close").sort_index()
    qvol = raw.pivot(index="date", columns="sym", values="qvol").sort_index()
    fund = raw.pivot(index="date", columns="sym", values="funding").sort_index()
    valid = close.notna().sum() >= MIN_WEEKS
    close, qvol, fund = close.loc[:, valid], qvol.loc[:, valid], fund.loc[:, valid]
    wc = close.resample("W").last()
    wq = qvol.resample("W").sum()
    wf = fund.resample("W").mean()
    return wc, wq, wf


def fas_signal(wc, wf, win):
    """FAS at accrual window `win` weeks: residualize funding accrual on price path, fade vs price dir."""
    fund_accrual = wf.rolling(win).sum()
    price_ret = wc.shift(1).div(wc.shift(win + 1)) - 1.0
    pr_z = zs_df(price_ret)
    resid = fund_accrual.copy()
    for w in fund_accrual.index:
        y = fund_accrual.loc[w].dropna()
        if len(y) < 10:
            continue
        X = np.array(
            [
                price_ret.loc[w].reindex(y.index).values,
                price_ret.loc[w].reindex(y.index).abs().values,
            ]
        ).T
        m = ~np.isnan(X).any(axis=1)
        if m.sum() < 10:
            continue
        yy, XX = y.values[m], X[m]
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        resid.loc[w, y.index[m]] = yy - XX @ beta
    return zs_df(zs_df(resid) * (-pr_z))


def backtest(score, ret, regime, dv, dv_med, size_mask, top=0.2, bot=0.2, use_regime=True):
    pf = []
    prev_w = None
    for w in score.index:
        sc = score.loc[w].dropna()
        if len(sc) < 20:
            prev_w = None
            continue
        r = ret.loc[w].reindex(sc.index)
        sc = sc.reindex(r.dropna().index)
        r = r.reindex(sc.index)
        if len(sc) < 20:
            prev_w = None
            continue
        if size_mask is not None:
            d = dv.loc[w].reindex(sc.index)
            med = dv_med.loc[w]
            if d.isna().all():
                prev_w = None
                continue
            sc = sc[d <= med] if size_mask == "small" else sc[d > med]
            r = r.reindex(sc.index)
            if len(sc) < 15:
                prev_w = None
                continue
        thr_hi = sc.quantile(1 - top)
        thr_lo = sc.quantile(bot)
        longs = sc[sc >= thr_hi].index
        shorts = sc[sc <= thr_lo].index
        if len(longs) == 0 or len(shorts) == 0:
            prev_w = None
            continue
        wk = r[longs].mean() - r[shorts].mean()
        cost = 0.0010 * 2 if prev_w is not None else 0.0010
        wk = wk - cost
        if use_regime and regime.loc[w] == 0:
            wk = 0.0
            prev_w = None
        else:
            prev_w = (set(longs), set(shorts))
        pf.append(wk)
    return pd.Series(pf) if len(pf) >= 30 else None


def sharpe_ci(pf):
    if pf is None:
        return float("nan"), float("nan"), float("nan")
    pf = pf.dropna()
    if len(pf) < 10:
        return float("nan"), float("nan"), float("nan")
    mu, sd = pf.mean(), pf.std()
    sr = mu / sd * np.sqrt(52) if sd > 0 else 0.0
    rng = random.Random(0)
    N, B = len(pf), 2000
    arr = pf.values
    boot = np.empty(B)
    for b in range(B):
        idx = [rng.randrange(N) for _ in range(N)]
        s = arr[idx]
        boot[b] = (s.mean() / s.std() * np.sqrt(52)) if s.std() > 0 else 0.0
    return sr, float(np.percentile(boot, 5)), float(np.percentile(boot, 95))


def rank_corr(a, b):
    out = []
    for w in a.index:
        x = a.loc[w].dropna()
        y = b.loc[w].reindex(x.index).dropna()
        x = x.reindex(y.index).dropna()
        if len(x) < 10 or x.std() == 0 or y.std() == 0:
            continue
        dx = x.rank() - x.rank().mean()
        dy = y.rank() - y.rank().mean()
        den = np.sqrt((dx**2).sum() * (dy**2).sum())
        if den > 0:
            out.append(float((dx * dy).sum() / den))
    return float(np.nanmean(out)) if out else float("nan")


def main():
    wc, wq, wf = load()
    wc = wc.dropna(how="all", axis=1)
    wq = wq.reindex(columns=wc.columns).replace(0, np.nan)
    wf = wf.reindex(columns=wc.columns).ffill().bfill()
    wc = wc[wc.notna().sum(axis=1) >= 200]
    wq = wq.loc[wc.index]
    wf = wf.loc[wc.index]
    wc = wc.interpolate(limit=2).ffill().bfill()
    wq = wq.interpolate(limit=2).ffill().bfill()
    wf = wf.ffill().bfill()
    ret = wc.pct_change(fill_method=None)
    fwd = ret.shift(-1)
    print(f"panel: {wc.shape[0]} weeks x {wc.shape[1]} coins")

    dv = wq
    dv_med = dv.median(axis=1)
    btc = wc["BTCUSDT"] if "BTCUSDT" in wc.columns else None
    if btc is None:
        import urllib.request, json

        k = json.loads(
            urllib.request.urlopen(
                "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=2000",
                timeout=25,
            )
            .read()
            .decode()
        )
        btc = pd.Series(
            [float(x[4]) for x in k], index=pd.to_datetime([x[6] for x in k], unit="ms")
        ).sort_index()
        btc = btc.resample("W").last().reindex(wc.index).ffill()
    regime = (btc > btc.rolling(52).mean()).astype(float)

    # (A) FAS accrual-window sweep
    print("\n=== (A) FAS accrual-window sweep (FULL, 10bps, regime) ===")
    fas_variants = {}
    for win in [4, 8, 12, 26]:
        sc = fas_signal(wc, wf, win)
        fas_variants[f"FAS{win}"] = sc
        pf = backtest(sc, fwd, regime, dv, dv_med, None)
        sr, lo, hi = sharpe_ci(pf)
        print(f"FAS win={win:2d}w  Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf)}")

    # (B) FAS_SLOPE : TREND of residualized funding (accelerating squeeze)
    # residualize funding accrual(12w) on price, then take its week-over-week DIFFERENCE
    print("\n=== (B) FAS_SLOPE (trend of residualized funding) ===")
    base = fas_signal(wc, wf, 12)
    resid = base  # already residualized & signed; trend = diff
    slope = zs_df(resid - resid.shift(4))
    fas_variants["FAS_slope"] = slope
    pf = backtest(slope, fwd, regime, dv, dv_med, None)
    sr, lo, hi = sharpe_ci(pf)
    print(f"FAS_slope  Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf)}")
    print(f"corr FAS12 vs FAS_slope = {rank_corr(base, slope):+.3f}")

    # (C) orthogonal combo: level + slope
    print("\n=== (C) FAS_level + FAS_slope (both ours) ===")
    combo = zs_df(base + slope)
    fas_variants["FAS12+slope"] = combo
    for label, mask in [("FULL", None), ("SMALL/illiquid", "small"), ("LARGE/liquid", "large")]:
        pf = backtest(combo, fwd, regime, dv, dv_med, mask)
        sr, lo, hi = sharpe_ci(pf)
        print(
            f"{label:16s} Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf) if pf is not None else 0}"
        )

    # also daily-frequency FAS (more responsive), held weekly
    print("\n=== (D) DAILY-frequency FAS held weekly (OOS responsiveness check) ===")
    frames = []
    for f in glob.glob(f"{BROAD}/*.csv"):
        d = pd.read_csv(f, parse_dates=["date"])
        d["sym"] = os.path.basename(f)[:-4]
        frames.append(d[["date", "sym", "close", "qvol", "funding"]])
    raw = pd.concat(frames, ignore_index=True)
    dc = raw.pivot(index="date", columns="sym", values="close").sort_index()
    dfu = raw.pivot(index="date", columns="sym", values="funding").sort_index()
    dc = dc.loc[:, dc.notna().sum() >= MIN_WEEKS]
    dfu = dfu.loc[:, dc.columns]
    dc = dc.interpolate(limit=2).ffill().bfill()
    dfu = dfu.interpolate(limit=2).ffill().bfill()
    facc = dfu.rolling(84).sum()  # ~12w of daily funding
    dpr = dc.shift(1).div(dc.shift(85)) - 1.0
    dprz = zs_df(dpr)
    dresid = facc.copy()
    for w in facc.index:
        y = facc.loc[w].dropna()
        if len(y) < 10:
            continue
        X = np.array(
            [dpr.loc[w].reindex(y.index).values, dpr.loc[w].reindex(y.index).abs().values]
        ).T
        m = ~np.isnan(X).any(axis=1)
        if m.sum() < 10:
            continue
        yy, XX = y.values[m], X[m]
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        dresid.loc[w, y.index[m]] = yy - XX @ beta
    fas_daily = zs_df(zs_df(dresid) * (-dprz)).resample("W").last()
    fas_daily = fas_daily.reindex(fwd.index)
    pf = backtest(fas_daily, fwd, regime, dv, dv_med, None)
    sr, lo, hi = sharpe_ci(pf)
    print(
        f"FAS_daily  Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf) if pf is not None else 0}"
    )
    print(f"corr FAS12 vs FAS_daily = {rank_corr(base, fas_daily):+.3f}")

    print("\n=== summary (FULL, 10bps, regime) ===")
    for nm, sc in fas_variants.items():
        pf = backtest(sc, fwd, regime, dv, dv_med, None)
        sr, lo, hi = sharpe_ci(pf)
        print(f"{nm:14s} Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]")


if __name__ == "__main__":
    main()
