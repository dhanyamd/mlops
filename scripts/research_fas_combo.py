"""FAS (OUR novel anchor) + known factors: does the combo actually reach Sharpe 2+?

Answering the user's question directly: FAS is OURS (funding-accrual-squeeze). We keep it
as the anchor and add KNOWN cross-sectional factors (REV, SMB, MOM) to test if the
combined signal crosses 2+. Reports FULL and POST-2024 so the 2+ claim is unambiguous.

Keyless (Binance public), 10bps costs + BTC-regime gate, weekly, broad panel.
FAS definitions reused from research_fas_meta.py (multi-horizon + slope).
Known-factor definitions reused from research_fas_strategy.py (REV_PLAIN, MHT) + SMB size tilt.
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


def rank_corr(a, b):
    out = []
    for w in a.index:
        x = a.loc[w].dropna()
        y = b.loc[w].reindex(x.index).dropna()
        x = x.reindex(y.index).dropna()
        if len(x) < 10:
            continue
        dx = x.rank() - x.rank().mean()
        dy = y.rank() - y.rank().mean()
        den = np.sqrt((dx**2).sum() * (dy**2).sum())
        if den > 0:
            out.append(float((dx * dy).sum() / den))
    return float(np.mean(out)) if out else float("nan")


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
    dv_med = wq.median(axis=1)
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

    # ---- OUR novel anchor: FAS ----
    lenses = {}
    for win in [4, 8, 12, 26]:
        lenses[f"FAS{win}"] = fas_signal(wc, wf, win)
    base = lenses["FAS12"]
    slope = zs_df(base - base.shift(4))
    fas_avg = zs_df(sum(lenses.values()) / len(lenses))
    fas_slop = zs_df((sum(lenses.values()) / len(lenses)) + slope)

    # ---- KNOWN factors (reused from research_fas_strategy.py) ----
    r4 = wc.shift(1).div(wc.shift(5)) - 1
    r8 = wc.shift(1).div(wc.shift(9)) - 1
    r12 = wc.shift(1).div(wc.shift(13)) - 1
    REV = -zs_df(zs_df(r4) + zs_df(r8) + zs_df(r12))
    m1 = wc.shift(1).div(wc.shift(5)) - 1
    m2 = wc.shift(5).div(wc.shift(13)) - 1
    m3 = wc.shift(13).div(wc.shift(25)) - 1
    MOM = zs_df(m1) + zs_df(m2) + zs_df(m3)
    # SMB = small-cap tilt (long smallest by 12w dollar volume)
    SMB = zs_df(-np.log(dv.rolling(12).sum().shift(1) + 1))

    print("\n=== distinctness: FAS vs known ===")
    print(f"FAS_avg ~ REV = {rank_corr(fas_avg, REV):+.3f}")
    print(f"FAS_avg ~ SMB = {rank_corr(fas_avg, SMB):+.3f}")
    print(f"FAS_avg ~ MOM = {rank_corr(fas_avg, MOM):+.3f}")

    # ---- combos: FAS (novel) as anchor + known ----
    combos = {
        "FAS_avg (NOVEL only)": [fas_avg],
        "FAS_avg+slope (NOVEL only)": [fas_slop],
        "FAS_avg + REV": [fas_avg, REV],
        "FAS_avg + SMB": [fas_avg, SMB],
        "FAS_avg + REV + SMB": [fas_avg, REV, SMB],
        "FAS_avg+slope + REV + SMB": [fas_slop, REV, SMB],
        "FAS_avg + REV + SMB + MOM": [fas_avg, REV, SMB, MOM],
    }

    def combo(keys):
        return zs_df(sum(keys) / len(keys))

    print("\n=== COMBOS: FULL (regime ON) ===")
    for nm, ks in combos.items():
        sc = combo(ks)
        pf = backtest(sc, fwd, regime, dv, dv_med, None)
        sr, lo, hi = sharpe_ci(pf)
        print(
            f"{nm:34s} Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf) if pf is not None else 0}"
        )

    print("\n=== COMBOS: POST-2024 (regime ON) ===")
    for nm, ks in combos.items():
        sc = combo(ks).loc["2024-01-01":]
        pf = backtest(sc, fwd.loc["2024-01-01":], regime, dv, dv_med, None)
        sr, lo, hi = sharpe_ci(pf)
        nwk = len(pf) if pf is not None else 0
        print(
            f"{nm:34s} "
            + (
                f"Sharpe={sr:+.2f} CI[{lo:+.2f},{hi:+.2f}] nwk={nwk}"
                if nwk >= 30
                else f"(insufficient {nwk})"
            )
        )


if __name__ == "__main__":
    main()
