"""Turnover-family factors ported from Chinese quant literature to keyless crypto.

HIDDEN-GEM GAP (research synthesis):
  Chinese quant lit has a hugely profitable TURNOVER factor family that has NEVER
  been ported to crypto (A-shares are retail-dominated; crypto is even MORE so):
    * 加速换手 / Acceleration Turnover (华安证券 240316): volume-surge-on-UP-days
      -> NEXT-WEEK REVERSAL. Rank IC -10.5%, ICIR -4.29, best in SMALL caps.
    * 异常换手率 / Abnormal Turnover: current turnover / trailing median -> reversal
      (stronger in arbitrage-limited names). Mirrored in crypto by Garfinkel-Hsiao-Hu
      (2025): abnormal volume = investor DISAGREEMENT (Miller) -> lower future returns.
    * 量稳换手率 STR / Stability of Turnover (东吴证券): LOW turnover variance -> positive.
  Independently, the CRYPTO literature confirms the same family dominates:
    * Crypto Factor Zoo (2026): only 3 factors explain all crypto returns; TURNOVER
      VOLATILITY is the #1 EQUAL-WEIGHTED factor.
    * Trading Activity Variation (SSRN 4291073): variability of turnover -> negative
      returns, stronger in costlier-to-arbitrage coins.
  => GAP = port the Chinese acceleration/abnormal-turnover reversal to crypto, keyless
     (volume + price only). Crypto + small/illiquid should amplify it (both lits agree).

Data: /tmp/broad_pull (keyless Binance: close, qvol, funding, 416 coins, 4.4y), weekly.
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
    # keep coins with enough daily history, then resample to weekly (match research_cer)
    valid = close.notna().sum() >= MIN_WEEKS
    close, qvol, fund = close.loc[:, valid], qvol.loc[:, valid], fund.loc[:, valid]
    wc = close.resample("W").last()
    wq = qvol.resample("W").sum()
    wf = fund.resample("W").mean()
    return wc, wq, wf


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


def ortho(target, controls):
    resid = target.copy()
    for w in target.index:
        y = target.loc[w].dropna()
        if len(y) < 10:
            continue
        X = np.column_stack([controls[i].loc[w].reindex(y.index) for i in range(len(controls))])
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
    wq = wq.reindex(columns=wc.columns)
    wf = wf.reindex(columns=wc.columns)
    wc = wc[wc.notna().sum(axis=1) >= 200]
    wq = wq.loc[wc.index]
    wf = wf.loc[wc.index]
    wq = wq.replace(0, np.nan)
    wc = wc.interpolate(limit=2).ffill().bfill()
    wq = wq.interpolate(limit=2).ffill().bfill()
    wf = wf.ffill().bfill()

    ret = wc.pct_change(fill_method=None)
    fwd = ret.shift(-1)
    print(f"panel: {wc.shape[0]} weeks x {wc.shape[1]} coins")

    dv = wq  # weekly dollar (quote) volume = turnover proxy
    dv_med = dv.median(axis=1)

    # regime: BTC uptrend gate (crash protection)
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

    # log turnover for stability
    logv = np.log(wq.replace(0, np.nan))

    # ---- TURNOVER FAMILY (Chinese lit -> crypto) ----
    # 1) TOV: turnover VOLATILITY (std of log-volume). Crypto Factor Zoo #1 + Trading Activity Variation.
    tov = logv.rolling(12).std()
    f_tov = -zs_df(tov)  # high vol-of-volume -> low future return

    # 2) AT: ABNORMAL turnover = current / trailing median. 异常换手率 + Garfinkel disagreement.
    vol_ratio = wq.div(wq.rolling(12).median())
    f_at = -zs_df(vol_ratio)

    # 3) ATR: ACCELERATION turnover = volume surge ON UP-weeks -> reversal. 华安 加速换手 放量上涨.
    up_ret = ret.clip(lower=0.0)
    atr = vol_ratio * up_ret
    f_atr = -zs_df(atr)

    # 4) STR: STABILITY of turnover = LOW coeff of variation -> positive. 东吴 量稳.
    cv = logv.rolling(12).std() / (logv.rolling(12).mean().abs() + 1e-9)
    f_str = zs_df(-cv)

    # ---- FAS (our funding-accrual squeeze) for the both-ours ensemble ----
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
    fas = zs_df(zs_df(fund_resid) * (-pr_z))

    # known controls
    r4 = wc.shift(1).div(wc.shift(5)) - 1
    r8 = wc.shift(1).div(wc.shift(9)) - 1
    r12 = wc.shift(1).div(wc.shift(13)) - 1
    rev_plain = -zs_df(zs_df(r4) + zs_df(r8) + zs_df(r12))
    m1 = wc.shift(1).div(wc.shift(5)) - 1
    m2 = wc.shift(5).div(wc.shift(13)) - 1
    m3 = wc.shift(13).div(wc.shift(25)) - 1
    mht = zs_df(m1) + zs_df(m2) + zs_df(m3)

    scores = {
        "TOV (turnover-vol)": f_tov,
        "AT (abnormal-turn)": f_at,
        "ATR (accel-turn)": f_atr,
        "STR (stability)": f_str,
        "FAS (ours)": fas,
    }

    print("\n=== TURNOVER FAMILY backtest (top/bot quintile L/S, 10bps, BTC-regime) ===")
    for nm, sc in scores.items():
        for label, mask in [("FULL", None), ("SMALL/illiquid", "small"), ("LARGE/liquid", "large")]:
            pf = backtest(sc, fwd, regime, dv, dv_med, mask)
            if pf is None:
                continue
            sr, lo, hi = sharpe_ci(pf)
            print(f"{nm:20s} {label:16s} Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf)}")

    # best turnover factor combined with FAS (both-ours)
    best = "ATR (accel-turn)"
    combo = zs_df(scores[best] + fas)
    print(f"\n=== {best} + FAS (both-ours ensemble) ===")
    for label, mask in [("FULL", None), ("SMALL/illiquid", "small"), ("LARGE/liquid", "large")]:
        pf = backtest(combo, fwd, regime, dv, dv_med, mask)
        sr, lo, hi = sharpe_ci(pf)
        print(
            f"{label:16s} Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf) if pf is not None else 0}"
        )

    print("\n=== SPANNING vs [MOM, REV] (distinctness) ===")
    for nm, sc in scores.items():
        sc_r = ortho(sc, [mht, rev_plain])
        raw_pf = backtest(sc, fwd, regime, dv, dv_med, None)
        res_pf = backtest(sc_r, fwd, regime, dv, dv_med, None)
        s_raw, _, _ = sharpe_ci(raw_pf)
        s_res, lo, hi = sharpe_ci(res_pf)
        print(f"{nm:20s} raw={s_raw:+.2f}  |MOM,REV={s_res:+.2f} CI[{lo:+.2f},{hi:+.2f}]")

    print("\n=== rank-corr vs known factors ===")
    for nm, sc in scores.items():
        print(
            f"{nm:20s} ~ REV={rank_corr(sc, rev_plain):+.3f}  ~ MHT={rank_corr(sc, mht):+.3f}  ~ FAS={rank_corr(sc, fas):+.3f}"
        )

    print("\n=== sub-periods (regime ON) ===")
    for nm, sc in scores.items():
        for lab, sl in [
            ("PRE-2024", slice(None, "2023-12-31")),
            ("POST-2024", slice("2024-01-01", None)),
        ]:
            s = sc if sl is None else sc.loc[sl]
            pf = backtest(s, fwd.loc[s.index], regime, dv, dv_med, None)
            sr, lo, hi = sharpe_ci(pf)
            nwk = len(pf) if pf is not None else 0
            if nwk >= 30:
                print(f"{nm:20s} {lab:10s} Sharpe={sr:+.2f} CI[{lo:+.2f},{hi:+.2f}] nwk={nwk}")
            else:
                print(f"{nm:20s} {lab:10s} (insufficient weeks: {nwk})")


if __name__ == "__main__":
    main()
