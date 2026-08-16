"""FAS meta-ensemble: can OUR novel funding factor family reach Sharpe 2+?

All components are OURS (funding-accrual-squeeze family), keyless (Binance public):
  FAS_w   = residualized w-week funding accrual faded vs price dir  (level lens)
  FAS_slope = trend of the residualized funding (corr 0.45 with level -> additive)
Combinations average the z-scored lenses. This finds the NOVEL ceiling (no known
factors mixed in). 10bps costs + BTC-regime gate, weekly, broad Binance panel.
"""

import glob, os, random
import numpy as np
import pandas as pd

BROAD = "/tmp/broad_pull"
MIN_WEEKS = 150


def zs(s):
    return (s.rank(pct=True) - 0.5) * 2


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
    facc = wf.rolling(win).sum()
    pr = wc.shift(1).div(wc.shift(win + 1)) - 1.0
    prz = zs_df(pr)
    resid = facc.copy()
    for w in facc.index:
        y = facc.loc[w].dropna()
        if len(y) < 10:
            continue
        X = np.array([pr.loc[w].reindex(y.index).values, pr.loc[w].reindex(y.index).abs().values]).T
        m = ~np.isnan(X).any(axis=1)
        if m.sum() < 10:
            continue
        yy, XX = y.values[m], X[m]
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        resid.loc[w, y.index[m]] = yy - XX @ beta
    return zs_df(zs_df(resid) * (-prz))


def backtest(score, ret, regime, dv, dv_med, size_mask):
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
            sc = sc[d <= med] if size_mask == "small" else sc[d > med]
            r = r.reindex(sc.index)
            if len(sc) < 15:
                prev_w = None
                continue
        thr_hi = sc.quantile(0.8)
        thr_lo = sc.quantile(0.2)
        longs = sc[sc >= thr_hi].index
        shorts = sc[sc <= thr_lo].index
        if len(longs) == 0 or len(shorts) == 0:
            prev_w = None
            continue
        wk = r[longs].mean() - r[shorts].mean()
        cost = 0.0010 * 2 if prev_w is not None else 0.0010
        wk = wk - cost
        if regime.loc[w] == 0:
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
    dv, dv_med = wq, wq.median(axis=1)
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

    # build all novel FAS lenses
    lenses = {}
    for win in [4, 8, 12, 26]:
        lenses[f"FAS{win}"] = fas_signal(wc, wf, win)
    base = lenses["FAS12"]
    slope = zs_df(base - base.shift(4))
    lenses["slope"] = slope

    def combo(keys):
        return zs_df(sum(lenses[k] for k in keys) / len(keys))

    print("\n=== NOVEL FAS-family combinations (FULL, 10bps, regime) ===")
    combos = {
        "FAS12": ["FAS12"],
        "FAS12+slope": ["FAS12", "slope"],
        "FAS[4,8,12,26]": ["FAS4", "FAS8", "FAS12", "FAS26"],
        "FAS[4,8,12,26]+slope": ["FAS4", "FAS8", "FAS12", "FAS26", "slope"],
        "FAS[8,12]+slope": ["FAS8", "FAS12", "slope"],
    }
    for nm, ks in combos.items():
        sc = combo(ks)
        for label, mask in [("FULL", None), ("SMALL", "small"), ("LARGE", "large")]:
            pf = backtest(sc, fwd, regime, dv, dv_med, mask)
            sr, lo, hi = sharpe_ci(pf)
            print(
                f"{nm:22s} {label:6s} Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf) if pf is not None else 0}"
            )

    print("\n=== sub-periods (regime ON) for best combo ===")
    best = "FAS[4,8,12,26]+slope"
    sc = combo(combos[best])
    for lab, sl in [
        ("PRE-2024", slice(None, "2023-12-31")),
        ("POST-2024", slice("2024-01-01", None)),
    ]:
        s = sc.loc[sl]
        pf = backtest(s, fwd.loc[s.index], regime, dv, dv_med, None)
        sr, lo, hi = sharpe_ci(pf)
        nwk = len(pf) if pf is not None else 0
        print(
            f"{best:22s} {lab:10s} "
            + (
                f"Sharpe={sr:+.2f} CI[{lo:+.2f},{hi:+.2f}] nwk={nwk}"
                if nwk >= 30
                else f"(insufficient {nwk})"
            )
        )


if __name__ == "__main__":
    main()
