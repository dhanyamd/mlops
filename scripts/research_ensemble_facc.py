#!/usr/bin/env python3
"""ENSEMBLE + FACC: does adding the novel Funding-Acceleration leg lift the 3-factor
ensemble (REV_PLAIN + SMB + FAS) toward a robust 2+ Sharpe on the FULL sample?

FACC is near-orthogonal to REV (+0.06) and FAS (-0.02), so it is a genuine new
diversifying leg, not a copy of known factors. If ENSEMBLE4 > ENSEMBLE3 with stable
CI, the 2+ target is reachable from a factor set that includes a NOVEL funding leg.

Keyless, 10bps, BTC-regime gate, same harness as research_nova.py / research_facc.py.
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
    return close, qvol, fund


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
        thr_hi, thr_lo = sc.quantile(1 - top), sc.quantile(bot)
        longs, shorts = sc[sc >= thr_hi].index, sc[sc <= thr_lo].index
        if len(longs) == 0 or len(shorts) == 0:
            prev_w = None
            continue
        wk = r[longs].mean() - r[shorts].mean()
        wk = wk - (0.0010 * 2 if prev_w is not None else 0.0010)
        if use_regime and regime.loc[w] == 0:
            wk, prev_w = 0.0, None
        else:
            prev_w = (set(longs), set(shorts))
        pf.append(wk)
    return pd.Series(pf) if len(pf) >= 30 else None


def sharpe_ci(pf):
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
        s = arr[[rng.randrange(N) for _ in range(N)]]
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
    print("== ENSEMBLE + FACC (novel 4th leg) ==")
    close, qvol, fund = load()
    valid = close.notna().sum() >= MIN_WEEKS
    syms = valid[valid].index.tolist()
    close, qvol, fund = close[syms], qvol[syms], fund[syms]
    wc = close.resample("W").last()
    wq = qvol.resample("W").sum()
    wf = fund.resample("W").mean().ffill().bfill()
    ret = wc.pct_change(fill_method=None)
    fwd = ret.shift(-1)
    dv = wq.rolling(12).sum().shift(1)
    dv_med = dv.median(axis=1)

    if "BTCUSDT" in wc.columns:
        btc = wc["BTCUSDT"]
    else:
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

    # known + existing legs
    r4 = wc.shift(1).div(wc.shift(5)) - 1
    r8 = wc.shift(1).div(wc.shift(9)) - 1
    r12 = wc.shift(1).div(wc.shift(13)) - 1
    rev_plain = -zs_df(zs_df(r4) + zs_df(r8) + zs_df(r12))
    smb = -zs_df(dv)
    pr12 = wc.shift(1).div(wc.shift(13)) - 1.0
    pr_z = zs_df(pr12)
    fresid = wf.rolling(12).sum().copy()
    for w in fresid.index:
        y = fresid.loc[w].dropna()
        if len(y) < 10:
            continue
        X = np.array(
            [pr12.loc[w].reindex(y.index).values, pr12.loc[w].reindex(y.index).abs().values]
        ).T
        m = ~np.isnan(X).any(axis=1)
        if m.sum() < 10:
            continue
        yy, XX = y.values[m], X[m]
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        fresid.loc[w, y.index[m]] = yy - XX @ beta
    fas = zs_df(fresid.apply(zs) * (-pr_z))

    # novel FACC leg (W=6, volume-confirmed)
    vol_up = wq > wq.rolling(12).mean()
    facc = -zs_df(wf - wf.rolling(6).mean()).where(vol_up)

    ens3 = zs_df(rev_plain + smb + fas)
    ens4 = zs_df(rev_plain + smb + fas + facc)

    print("\n=== ENSEMBLE FULL (10bps, regime) ===")
    for nm, sc in [("ENSEMBLE3 (REV+SMB+FAS)", ens3), ("ENSEMBLE4 (+FACC)", ens4)]:
        for label, mask in [("FULL", None), ("SMALL", "small"), ("LARGE", "large")]:
            pf = backtest(sc, fwd, regime, dv, dv_med, mask)
            if pf is None:
                continue
            sr, lo, hi = sharpe_ci(pf)
            print(f"{nm:24s} {label:6s} Sharpe={sr:+.2f} CI[{lo:+.2f},{hi:+.2f}] nwk={len(pf)}")

    print("\n=== ENSEMBLE POST-2024 (regime ON) ===")
    for nm, sc in [("ENSEMBLE3", ens3), ("ENSEMBLE4", ens4)]:
        s = sc.loc["2024-01-01":]
        pf = backtest(s, fwd.loc["2024-01-01":], regime, dv, dv_med, None)
        if pf is None:
            continue
        sr, lo, hi = sharpe_ci(pf)
        print(f"{nm:10s} Sharpe={sr:+.2f} CI[{lo:+.2f},{hi:+.2f}] nwk={len(pf)}")

    print("\n=== ENSEMBLE PRE-2024 (regime OFF) ===")
    for nm, sc in [("ENSEMBLE3", ens3), ("ENSEMBLE4", ens4)]:
        s = sc.loc[:"2023-12-31"]
        pf = backtest(s, fwd.loc[:"2023-12-31"], regime, dv, dv_med, None, use_regime=False)
        if pf is None:
            continue
        sr, lo, hi = sharpe_ci(pf)
        print(f"{nm:10s} Sharpe={sr:+.2f} CI[{lo:+.2f},{hi:+.2f}] nwk={len(pf)}")

    print("\n=== rank-corr FACC vs ensemble legs (distinctness) ===")
    print(f"FACC ~ REV_PLAIN = {rank_corr(facc, rev_plain):+.3f}")
    print(f"FACC ~ SMB       = {rank_corr(facc, smb):+.3f}")
    print(f"FACC ~ FAS       = {rank_corr(facc, fas):+.3f}")


if __name__ == "__main__":
    main()
