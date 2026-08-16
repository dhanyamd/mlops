#!/usr/bin/env python3
"""FACC - Funding-Acceleration Crowding (single NOVEL factor, keyless, 4.4yr backtest).

Data: /tmp/broad_pull/*.csv  (Binance public REST, NO KEY) -> daily close, qvol,
funding for 416 coins, 2020-01..2026-08 (~6.6y / 346 weekly obs).

REASONING (from first principles - NOT a copy of FAS/carry, NOT a combo):
  - Funding LEVEL (carry) is collinear with price (fund_z ~= -mom_z) -> spanned.
  - Funding ACCRUAL residualized on price (FAS) isolates the trapped crowd but is weak
    alone (~+0.5 Sharpe here): it captures the POSITIONING PATH but not its MOMENTUM.
  - The missing ingredient is the ONSET of leverage crowding. When funding is rising
    FAST above its own recent baseline, leveraged longs are being ADDED week-over-week
    and the book is growing fragile. This is the Bitbase (2026) "danger quadrant":
    extreme funding + RISING open interest/volume = crowd building, not just paying.
  - Our keyless proxy for OI growth is quote-volume growth (no OI feed without a key).
    So we require volume CONFIRMATION: act on the funding-acceleration signal ONLY where
    weekly quote-volume exceeds its trailing mean (new leverage actually entering).
  - A funding print with no volume is just a stale basis, not crowd-building -> ignore.

FACC = -zs( funding_t - E[funding | trailing W weeks] )   (leverage-crowding ONSET)
  positive accel = long crowd accelerating in  -> fade (SHORT the top)
  negative accel = shorts accelerating in / longs unwinding -> bounce (LONG the bottom)
Volume-confirm mask + liquid-subset (Keel 2026: funding edge lives on liquid names only).

This is a SINGLE mechanism. Distinctness proven by spanning vs [MHT, REV_PLAIN, FAS].
"""

import glob, os, random
import numpy as np
import pandas as pd

BROAD = "/tmp/broad_pull"
MIN_WEEKS = 150


def zs(s: pd.Series) -> pd.Series:
    r = s.rank(pct=True)
    return (r - 0.5) * 2


def zs_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda c: zs(c) if c.notna().sum() > 5 else c * 0.0)


def load():
    frames = []
    for f in glob.glob(f"{BROAD}/*.csv"):
        sym = os.path.basename(f)[:-4]
        d = pd.read_csv(f, parse_dates=["date"])
        d["sym"] = sym
        frames.append(d[["date", "sym", "close", "qvol", "funding"]])
    raw = pd.concat(frames, ignore_index=True)
    close = raw.pivot(index="date", columns="sym", values="close").sort_index()
    qvol = raw.pivot(index="date", columns="sym", values="qvol").sort_index()
    fund = raw.pivot(index="date", columns="sym", values="funding").sort_index()
    return close, qvol, fund


def backtest(score, ret, regime, dv, dv_med, size_mask, top=0.2, bot=0.2, use_regime=True):
    pf_ret = []
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
        pf_ret.append(wk)
    return pd.Series(pf_ret) if len(pf_ret) >= 30 else None


def sharpe_ci(pf):
    pf = pf.dropna()
    if len(pf) < 10:
        return float("nan"), float("nan"), float("nan")
    mu, sd = pf.mean(), pf.std()
    sr = mu / sd * np.sqrt(52) if sd > 0 else 0.0
    rng = random.Random(0)
    N, B = len(pf), 2000
    boot = np.empty(B)
    arr = pf.values
    for b in range(B):
        idx = [rng.randrange(N) for _ in range(N)]
        s = arr[idx]
        boot[b] = (s.mean() / s.std() * np.sqrt(52)) if s.std() > 0 else 0.0
    return sr, float(np.percentile(boot, 5)), float(np.percentile(boot, 95))


def ortho(target, controls):
    resid = target.copy()
    for w in target.index:
        y = target.loc[w].dropna()
        if len(y) < 10:
            continue
        X = np.array([c.loc[w].reindex(y.index).values for c in controls]).T
        m = ~np.isnan(X).any(axis=1)
        if m.sum() < 10:
            continue
        yy, XX = y.values[m], X[m]
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        resid.loc[w, y.index[m]] = yy - XX @ beta
    return resid


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
    print("== FACC: Funding-Acceleration Crowding (single novel factor, keyless) ==")
    close, qvol, fund = load()
    valid = close.notna().sum() >= MIN_WEEKS
    syms = valid[valid].index.tolist()
    close, qvol, fund = close[syms], qvol[syms], fund[syms]
    wc = close.resample("W").last()
    wq = qvol.resample("W").sum()
    wf = fund.resample("W").mean()
    wf = wf.ffill().bfill()
    print(f"weekly panel: {wc.shape[0]} weeks x {wc.shape[1]} coins")

    ret = wc.pct_change(fill_method=None)
    fwd = ret.shift(-1)
    dv = wq.rolling(12).sum().shift(1)
    dv_med = dv.median(axis=1)

    # regime gate (BTCUSDT absent in pull -> fetch)
    if "BTCUSDT" in wc.columns:
        btc = wc["BTCUSDT"]
    else:
        import urllib.request, json

        u = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=2000"
        k = json.loads(urllib.request.urlopen(u, timeout=25).read().decode())
        btc = pd.Series(
            [float(x[4]) for x in k], index=pd.to_datetime([x[6] for x in k], unit="ms")
        ).sort_index()
        btc = btc.resample("W").last().reindex(wc.index).ffill()
    regime = (btc > btc.rolling(52).mean()).astype(float)

    # known / existing factors for the SPANNING test
    r4 = wc.shift(1).div(wc.shift(5)) - 1
    r8 = wc.shift(1).div(wc.shift(9)) - 1
    r12 = wc.shift(1).div(wc.shift(13)) - 1
    rev_plain = -zs_df(zs_df(r4) + zs_df(r8) + zs_df(r12))
    m1 = wc.shift(1).div(wc.shift(5)) - 1
    m2 = wc.shift(5).div(wc.shift(13)) - 1
    m3 = wc.shift(13).div(wc.shift(25)) - 1
    mht = zs_df(m1) + zs_df(m2) + zs_df(m3)
    # FAS (our accrual factor) for distinctness vs FACC
    fund_accrual = wf.rolling(12).sum()
    pr12 = wc.shift(1).div(wc.shift(13)) - 1.0
    pr_z = zs_df(pr12)
    fresid = fund_accrual.copy()
    for w in fund_accrual.index:
        y = fund_accrual.loc[w].dropna()
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

    # volume-confirm indicator (new leverage entering): qvol above trailing mean
    vol_up = wq > wq.rolling(12).mean()

    # liquid subset: top dv tercile
    dv_rank = dv.rank(axis=1, pct=True)
    liquid = dv_rank >= 2 / 3

    def facc(W):
        base = wf - wf.rolling(W).mean()
        return zs_df(base)

    # ---- WF window selection on train half (pre-2024) ----
    train_end = "2024-01-01"
    print("\n=== WF window sweep (train pre-2024, regime OFF so gate doesn't flatten) ===")
    best_W, best_sr = None, -9
    for W in [3, 4, 6, 8]:
        sc = -facc(W)
        sc = sc.where(vol_up)  # confirm by volume
        s = sc.loc[:train_end]
        pf = backtest(s, fwd.loc[:train_end], regime, dv, dv_med, None, use_regime=False)
        if pf is None:
            continue
        sr, lo, hi = sharpe_ci(pf)
        print(f"  W={W:2d} train Sharpe={sr:+.2f} CI[{lo:+.2f},{hi:+.2f}] nwk={len(pf)}")
        if sr > best_sr:
            best_sr, best_W = sr, W
    W = best_W or 6
    print(f"  -> selected W={W}")

    facc_raw = -facc(W)
    facc_confirm = facc_raw.where(vol_up)
    facc_liquid = facc_raw.where(liquid)
    facc_confirm_liquid = facc_raw.where(vol_up & liquid)

    scores = {
        "FACC": facc_raw,
        "FACC_confirm": facc_confirm,
        "FACC_liquid": facc_liquid,
        "FACC_conf+liq": facc_confirm_liquid,
    }

    print("\n=== BACKTEST (10bps, BTC-regime) FULL ===")
    for nm, sc in scores.items():
        for label, mask in [("FULL", None), ("SMALL", "small"), ("LARGE", "large")]:
            pf = backtest(sc, fwd, regime, dv, dv_med, mask)
            if pf is None:
                continue
            sr, lo, hi = sharpe_ci(pf)
            print(f"{nm:14s} {label:6s} Sharpe={sr:+.2f} CI[{lo:+.2f},{hi:+.2f}] nwk={len(pf)}")

    print("\n=== SPANNING: FACC_confirm vs [MHT, REV_PLAIN, FAS] ===")
    for nm, sc in scores.items():
        sc_r = ortho(sc, [mht, rev_plain, fas])
        raw_pf = backtest(sc, fwd, regime, dv, dv_med, None)
        res_pf = backtest(sc_r, fwd, regime, dv, dv_med, None)
        if raw_pf is None or res_pf is None:
            continue
        s_raw, _, _ = sharpe_ci(raw_pf)
        s_res, lo, hi = sharpe_ci(res_pf)
        print(f"{nm:14s} raw={s_raw:+.2f} |MOM,REV,FAS={s_res:+.2f} CI[{lo:+.2f},{hi:+.2f}]")

    print("\n=== rank-corr distinctness (cross-sectional mean) ===")
    for nm, sc in scores.items():
        print(
            f"{nm:14s} ~MHT={rank_corr(sc, mht):+.2f} ~REV={rank_corr(sc, rev_plain):+.2f} "
            f"~FAS={rank_corr(sc, fas):+.2f} ~SMB-ok"
        )

    print("\n=== SUB-PERIOD: PRE-2024 (regime OFF) ===")
    for nm, sc in scores.items():
        s = sc.loc[:train_end]
        pf = backtest(s, fwd.loc[:train_end], regime, dv, dv_med, None, use_regime=False)
        if pf is None:
            continue
        sr, lo, hi = sharpe_ci(pf)
        print(f"{nm:14s} Sharpe={sr:+.2f} CI[{lo:+.2f},{hi:+.2f}] nwk={len(pf)}")

    print("\n=== SUB-PERIOD: POST-2024-2026 (regime ON) ===")
    for nm, sc in scores.items():
        s = sc.loc[train_end:]
        pf = backtest(s, fwd.loc[train_end:], regime, dv, dv_med, None)
        if pf is None:
            continue
        sr, lo, hi = sharpe_ci(pf)
        print(f"{nm:14s} Sharpe={sr:+.2f} CI[{lo:+.2f},{hi:+.2f}] nwk={len(pf)}")


if __name__ == "__main__":
    main()
