"""LAST ROUND: apply the Chinese residualization breakthrough to crypto.

Chinese quant breakthrough (this research round): the edge is NOT raw turnover but
RESIDUALIZING the flow on price and keeping the residual. Money-flow IR 2.63->4.76
(NIR_MOD), turnover IR 2.96->3.46-3.87 (STR_dePLUS/优加) after stripping price.
Intuition: raw volume/funding is contaminated by price trend; the RESIDUAL flow
(flow NOT explained by price) is the genuine signal. FAS already does this for
funding (that's why it works, +1.55). We NEVER residualized VOLUME -> our raw
turnover factors failed.

Build VAS = VOLUME residualized on price path (crypto analog of STR_dePLUS/优加),
the 2nd "pure-flow" lens. Then combine FAS (funding-residualized) + VAS
(volume-residualized) ORTHOGONALLY (regress VAS on FAS, keep residual) -- the
Chinese fix for the "1+1<2" problem. Both lenses OURS, both keyless. Test if the
orthogonal two-flow combo reaches 2+.
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


def vas_signal(wc, wq, win):
    """VAS = VOLUME residualized on price path (crypto analog of STR_dePLUS/优加)."""
    vacc = wq.rolling(win).sum()
    pr = wc.shift(1).div(wc.shift(win + 1)) - 1.0
    prz = zs_df(pr)
    resid = vacc.copy()
    for w in vacc.index:
        y = vacc.loc[w].dropna()
        if len(y) < 10:
            continue
        X = np.array([pr.loc[w].reindex(y.index).values, pr.loc[w].reindex(y.index).abs().values]).T
        m = ~np.isnan(X).any(axis=1)
        if m.sum() < 10:
            continue
        yy, XX = y.values[m], X[m]
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        resid.loc[w, y.index[m]] = yy - XX @ beta
    # turnover-style negative IC: low residual volume -> future up (mirrors STR)
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


def ortho(target, control):
    resid = target.copy()
    for w in target.index:
        y = target.loc[w].dropna()
        x = control.loc[w].reindex(y.index)
        y = y.reindex(x.dropna().index)
        x = x.reindex(y.index)
        if len(y) < 10:
            continue
        X = np.column_stack([np.ones(len(y)), x.values])
        beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
        resid.loc[w, y.index] = y.values - X @ beta
    return resid


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

    H = [4, 8, 12, 26]

    # FAS family (funding residualized)
    fases = {f"FAS{h}": fas_signal(wc, wf, h) for h in H}
    fas_avg = zs_df(sum(fases.values()) / len(fases))

    # VAS family (volume residualized) -- the NEW lens
    vases = {f"VAS{h}": vas_signal(wc, wq, h) for h in H}
    vas_avg = zs_df(sum(vases.values()) / len(vases))

    print("\n=== FAS (funding-resid) vs VAS (volume-resid) -- our two pure-flow lenses ===")
    for nm, sc in [("FAS_avg", fas_avg), ("VAS_avg", vas_avg)]:
        for label, mask in [("FULL", None), ("SMALL", "small"), ("LARGE", "large")]:
            pf = backtest(sc, fwd, regime, dv, dv_med, mask)
            sr, lo, hi = sharpe_ci(pf)
            print(
                f"{nm:10s} {label:6s} Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf) if pf is not None else 0}"
            )

    # orthogonal combo: residualize VAS on FAS, then add
    vas_orth = ortho(vas_avg, fas_avg)
    combos = {
        "FAS+VAS (naive avg)": zs_df(fas_avg + vas_avg),
        "FAS+VAS (orthogonal)": zs_df(zs_df(fas_avg) + zs_df(vas_orth)),
    }
    print("\n=== FAS + VAS combination (both ours) ===")
    for nm, sc in combos.items():
        for label, mask in [("FULL", None), ("SMALL", "small"), ("LARGE", "large")]:
            pf = backtest(sc, fwd, regime, dv, dv_med, mask)
            sr, lo, hi = sharpe_ci(pf)
            print(
                f"{nm:22s} {label:6s} Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf) if pf is not None else 0}"
            )

    # rank corr between the two lenses
    def rc(a, b):
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
                out.append((dx * dy).sum() / den)
        return float(np.nanmean(out)) if out else float("nan")

    print(
        f"\nrank-corr FAS_avg vs VAS_avg = {rc(fas_avg, vas_avg):+.3f}  (low = orthogonal, additive)"
    )


if __name__ == "__main__":
    main()
