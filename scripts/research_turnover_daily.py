"""Turnover-family factors at DAILY frequency (Chinese lit operates daily; weekly
aggregation washed out the acceleration effect). Same hidden-gem gap, correct freq.

Tests whether the Chinese acceleration/abnormal-turnover reversal transfers to crypto
when computed from DAILY volume+price (keyless), then held weekly (our production cadence).
"""

import glob, os, random
import numpy as np
import pandas as pd

BROAD = "/tmp/broad_pull"
MIN_DAYS = 400


def zs(s):
    r = s.rank(pct=True)
    return (r - 0.5) * 2


def zs_df(df):
    return df.apply(lambda c: zs(c) if c.notna().sum() > 5 else c * 0.0)


def load_daily():
    frames = []
    for f in glob.glob(f"{BROAD}/*.csv"):
        d = pd.read_csv(f, parse_dates=["date"])
        d["sym"] = os.path.basename(f)[:-4]
        frames.append(d[["date", "sym", "close", "qvol", "funding"]])
    raw = pd.concat(frames, ignore_index=True)
    close = raw.pivot(index="date", columns="sym", values="close").sort_index()
    qvol = raw.pivot(index="date", columns="sym", values="qvol").sort_index()
    fund = raw.pivot(index="date", columns="sym", values="funding").sort_index()
    valid = close.notna().sum() >= MIN_DAYS
    return close.loc[:, valid], qvol.loc[:, valid], fund.loc[:, valid]


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


def main():
    close, qvol, fund = load_daily()
    close = close.interpolate(limit=2).ffill().bfill()
    qvol = (
        qvol.reindex(columns=close.columns).replace(0, np.nan).interpolate(limit=2).ffill().bfill()
    )
    fund = fund.reindex(columns=close.columns).ffill().bfill()
    print(f"daily panel: {close.shape[0]} days x {close.shape[1]} coins")

    # daily returns / turnover
    ret_d = close.pct_change(fill_method=None)
    vol = qvol
    abn = vol.div(vol.rolling(20).median())  # abnormal turnover (daily)
    up = ret_d.clip(lower=0.0)
    accel = abn * up  # 加速换手: volume surge on UP days
    logv = np.log(vol.replace(0, np.nan))
    tov_d = logv.rolling(20).std()  # turnover volatility (daily)
    cv_d = logv.rolling(20).std() / (logv.rolling(20).mean().abs() + 1e-9)  # instability

    # weekly signals (mean over the week), held next week
    sig = {
        "TOV (turn-vol)": (-tov_d).resample("W").mean(),
        "AT (abn-turn)": (-abn).resample("W").mean(),
        "ATR (accel)": (-accel).resample("W").mean(),
        "STR (stability)": (-cv_d).resample("W").mean(),
    }
    # weekly forward return + weekly dollar volume (for size split)
    fwd = close.resample("W").last().pct_change(fill_method=None).shift(-1)
    dv = vol.resample("W").sum()
    dv_med = dv.median(axis=1)

    # BTC regime gate
    btc = close["BTCUSDT"] if "BTCUSDT" in close.columns else None
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
    regime = (btc > btc.rolling(52 * 7).mean()).astype(float)
    regime = regime.resample("W").last().reindex(fwd.index).ffill()

    # align
    common = sig["TOV (turn-vol)"].index.intersection(fwd.index)
    for k in sig:
        sig[k] = sig[k].reindex(common)
    fwd = fwd.reindex(common)
    dv = dv.reindex(common)
    dv_med = dv_med.reindex(common)
    regime = regime.reindex(common)

    print("\n=== DAILY-derived turnover factors, weekly L/S (10bps, BTC-regime) ===")
    for nm, sc in sig.items():
        for label, mask in [("FULL", None), ("SMALL/illiquid", "small"), ("LARGE/liquid", "large")]:
            pf = backtest(sc, fwd, regime, dv, dv_med, mask)
            if pf is None:
                print(f"{nm:16s} {label:16s} (insufficient)")
                continue
            sr, lo, hi = sharpe_ci(pf)
            print(f"{nm:16s} {label:16s} Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf)}")


if __name__ == "__main__":
    main()
