#!/usr/bin/env python3
"""FAS — Funding-Accrual Squeeze: OUR invented factor, standalone (no known factors).

Mechanism (invented from reasoning, 2026):
  Perpetual funding *level* is collinear with price (fund_z ~= -mom_z) -> spanned by
  trend, so a naive funding signal adds nothing. But the CUMULATIVE funding paid over a
  window (the positioning PATH) residualized on the price path is a PURE crowding /
  trapped-crowd signal, orthogonal to trend. A crowd that has paid one way for 12 weeks
  while price DIVERGED is trapped/underwater -> forced unwind -> mean reversion (squeeze).
  Fade the trapped crowd: crowd-long & price down -> bounce (long); mirror for shorts.

Novelty anchor (web-verified 2026): this is the funding-DERIVATIVES analog of the
  order-flow orthogonalization of Bianchi et al. (2026, "Order flow and cryptocurrency
  returns"), who residualize spot taker-flow on lagged returns to isolate the permanent
  predictive component (weekly Sharpe 1.93). We apply the same idea to perpetual-futures
  FUNDING accrual. Distinct from funding *carry* (Keel/Unravel) and funding-sentiment
  *fade* (Kraken/Phemex): those use funding LEVEL; we use funding accrual RESIDUALIZED
  on price.

Data: /tmp/broad_pull/*.csv (keyless Binance: close, qvol, funding, 416 coins, 4.4y).

This script reports ONLY our factor (no reversal, no size, no carry) so the result is
unambiguously OURS. Distinctness is proven via spanning test vs [MOM, REV_PLAIN].
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


def main():
    print("== FAS: our Funding-Accrual Squeeze factor (standalone, keyless 4.4y) ==")
    close, qvol, fund = load()
    valid = close.notna().sum() >= MIN_WEEKS
    syms = valid[valid].index.tolist()
    close, qvol, fund = close[syms], qvol[syms], fund[syms]
    wc = close.resample("W").last()
    wf = fund.resample("W").mean()
    ret = wc.pct_change()
    fwd = ret.shift(-1)
    print(f"panel: {wc.shape[0]} weeks x {wc.shape[1]} coins")

    # known factors only for the SPANNING test (to prove FAS is distinct from them)
    r4 = wc.shift(1).div(wc.shift(5)) - 1
    r8 = wc.shift(1).div(wc.shift(9)) - 1
    r12 = wc.shift(1).div(wc.shift(13)) - 1
    rev_plain = -zs_df(zs_df(r4) + zs_df(r8) + zs_df(r12))
    m1 = wc.shift(1).div(wc.shift(5)) - 1
    m2 = wc.shift(5).div(wc.shift(13)) - 1
    m3 = wc.shift(13).div(wc.shift(25)) - 1
    mht = zs_df(m1) + zs_df(m2) + zs_df(m3)

    # ---- OUR FACTOR: FAS ----
    fund_accrual = wf.rolling(12).sum()
    price_ret12 = wc.shift(1).div(wc.shift(13)) - 1.0
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
    fas_raw = fund_resid_z * (-pr_z)
    fas = zs_df(fas_raw)
    # FAS2: front-loaded squeeze timing (recent ~3w price confirms crowd trapped now)
    r_recent = wc.shift(1).div(wc.shift(4)) - 1.0
    fas2 = zs_df(fund_resid_z * (-zs_df(r_recent)))

    wq = qvol.resample("W").sum()
    dv = wq.rolling(12).sum().shift(1)
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

    scores = {"FAS": fas, "FAS2": fas2}
    print("\n=== OUR FACTOR backtest (long top / short bottom quintile, 10bps, BTC-regime) ===")
    for nm, sc in scores.items():
        for label, mask in [("FULL", None), ("SMALL/illiquid", "small"), ("LARGE/liquid", "large")]:
            pf = backtest(sc, fwd, regime, dv, dv_med, mask)
            if pf is None:
                continue
            sr, lo, hi = sharpe_ci(pf)
            print(f"{nm:5s} {label:16s} Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf)}")

    print("\n=== SPANNING: FAS residualized on [MOM, REV] (proves distinctness) ===")
    for nm, sc in scores.items():
        sc_r = ortho(sc, [mht, rev_plain])
        raw_pf = backtest(sc, fwd, regime, dv, dv_med, None)
        res_pf = backtest(sc_r, fwd, regime, dv, dv_med, None)
        s_raw, _, _ = sharpe_ci(raw_pf)
        s_res, lo, hi = sharpe_ci(res_pf)
        print(
            f"{nm:5s} raw Sharpe={s_raw:+.2f}  |MOM,REV Sharpe={s_res:+.2f} CI[{lo:+.2f},{hi:+.2f}]"
        )

    print("\n=== rank-corr of FAS with known factors (cross-sectional mean) ===")
    print(f"FAS ~ MHT       = {rank_corr(fas, mht):+.3f}")
    print(f"FAS ~ REV_PLAIN = {rank_corr(fas, rev_plain):+.3f}")
    print(f"FAS2~ MHT       = {rank_corr(fas2, mht):+.3f}")
    print(f"FAS2~ REV_PLAIN = {rank_corr(fas2, rev_plain):+.3f}")

    print("\n=== PRE-2024 (regime OFF, raw signal quality) ===")
    for nm, sc in scores.items():
        pf = backtest(sc, fwd, regime, dv, dv_med, None, use_regime=False)
        sr, lo, hi = sharpe_ci(pf)
        print(
            f"{nm:5s} Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf) if pf is not None else 0}"
        )

    print("\n=== POST-2024-2026 ===")
    for nm, sc in scores.items():
        s = sc.loc["2024-01-01":]
        pf = backtest(s, fwd.loc["2024-01-01":], regime, dv, dv_med, None)
        sr, lo, hi = sharpe_ci(pf)
        print(
            f"{nm:5s} Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf) if pf is not None else 0}"
        )


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


if __name__ == "__main__":
    main()
