#!/usr/bin/env python3
"""CER -- Cascade-Exhaustion Reversal: OUR invented factor (from cascade physics).

Mechanism (reasoned from arXiv 2607.27070 / 2608.03616):
  Liquidation cascades are SUBCRITICAL (branching ratio ~0.1-0.2) and FRONT-LOADED
  (88% of forced selling in 30min; open interest clears 25-70% at onset). Forced flow
  is therefore a ONE-OFF that EXHAUSTS: once a leverage-driven cascade happens, the
  leverage overhang is consumed and the next move is REVERSION (no diverging multiplier
  exists). Plain reversal trades every wobble; CER trades reversal ONLY when the move
  carries the exhaustion fingerprint = large move AND a volume spike (leverage/forced
  flow was clearly involved -> subcritical -> will revert). This state-gating is the
  novel part and should raise per-trade edge + cut turnover vs naive reversal.

Data: /tmp/broad_pull (keyless Binance: close, qvol, funding, 416 coins, 4.4y).
Also tests the genuinely-OURS ensemble CER+FAS (two independent lenses on leverage
exhaustion: CER = price/volume signature, FAS = funding/positioning signature).
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
        if den > 0 and x.std() > 0 and y.std() > 0:
            out.append(float((dx * dy).sum() / den))
    return float(np.mean(out)) if out else float("nan")


def main():
    print("== CER: Cascade-Exhaustion Reversal (OUR invention from cascade physics) ==")
    close, qvol, fund = load()
    valid = close.notna().sum() >= MIN_WEEKS
    syms = valid[valid].index.tolist()
    close, qvol, fund = close[syms], qvol[syms], fund[syms]
    wc = close.resample("W").last()
    wq = qvol.resample("W").sum()
    wf = fund.resample("W").mean()
    ret = wc.pct_change()
    fwd = ret.shift(-1)
    print(f"panel: {wc.shape[0]} weeks x {wc.shape[1]} coins")

    # known factors (controls only)
    r4 = wc.shift(1).div(wc.shift(5)) - 1
    r8 = wc.shift(1).div(wc.shift(9)) - 1
    r12 = wc.shift(1).div(wc.shift(13)) - 1
    rev_plain = -zs_df(zs_df(r4) + zs_df(r8) + zs_df(r12))
    m1 = wc.shift(1).div(wc.shift(5)) - 1
    m2 = wc.shift(5).div(wc.shift(13)) - 1
    m3 = wc.shift(13).div(wc.shift(25)) - 1
    mht = zs_df(m1) + zs_df(m2) + zs_df(m3)

    # ---- CER: reversal ONLY in cascade-exhaustion weeks ----
    abs_ret = ret.abs()
    abs_ret_z = zs_df(abs_ret)  # cross-sectional size of the move
    vol_ratio = wq.div(wq.rolling(12).median())  # current weekly vol vs trailing median
    large_move = abs_ret_z > 1.0  # top ~16% moves
    vol_spike = vol_ratio > 1.5  # leverage/forced flow present
    cascade_flag = large_move & vol_spike
    cer_raw = ret.apply(lambda c: c.where(cascade_flag[c.name], 0.0) * -1.0)
    # ^ fade the flagged week's sign; flat otherwise. z-score cross-sectionally.
    cer = zs_df(cer_raw)

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
    cer_fas = zs_df(cer + fas)

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

    scores = {"CER": cer, "FAS": fas, "CER+FAS (ours)": cer_fas}
    # DEBUG
    cmb = cer + fas
    print(
        "DEBUG cer_fas vs fas row-corr mean:",
        np.nanmean(
            [
                np.corrcoef(
                    cer_fas.loc[w].dropna(),
                    fas.loc[w].reindex(cer_fas.loc[w].dropna().index).dropna(),
                )[0, 1]
                for w in cer_fas.index
                if cer_fas.loc[w].notna().sum() > 10
            ]
        ),
    )
    print("DEBUG cer nonzero weeks:", int((cer.abs().max(axis=1) > 0).sum()), "of", len(cer))
    print("DEBUG cer_fas nonzero weeks:", int((cer_fas.abs().max(axis=1) > 0).sum()))
    print("\n=== OUR factors backtest (top/bot quintile L/S, 10bps, BTC-regime) ===")
    for nm, sc in scores.items():
        for label, mask in [("FULL", None), ("SMALL/illiquid", "small"), ("LARGE/liquid", "large")]:
            pf = backtest(sc, fwd, regime, dv, dv_med, mask)
            if pf is None:
                continue
            sr, lo, hi = sharpe_ci(pf)
            print(f"{nm:16s} {label:16s} Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf)}")

    print("\n=== SPANNING vs [MOM, REV] (distinctness) ===")
    for nm, sc in scores.items():
        sc_r = ortho(sc, [mht, rev_plain])
        raw_pf = backtest(sc, fwd, regime, dv, dv_med, None)
        res_pf = backtest(sc_r, fwd, regime, dv, dv_med, None)
        s_raw, _, _ = sharpe_ci(raw_pf)
        s_res, lo, hi = sharpe_ci(res_pf)
        print(f"{nm:16s} raw={s_raw:+.2f}  |MOM,REV={s_res:+.2f} CI[{lo:+.2f},{hi:+.2f}]")

    print("\n=== rank-corr vs known factors ===")
    print(f"CER  ~ REV_PLAIN = {rank_corr(cer, rev_plain):+.3f}")
    print(f"CER  ~ MHT       = {rank_corr(cer, mht):+.3f}")
    print(f"CER  ~ FAS       = {rank_corr(cer, fas):+.3f}")
    print(f"CFAS ~ REV_PLAIN = {rank_corr(cer_fas, rev_plain):+.3f}")

    print("\n=== sub-periods (regime ON) ===")
    for nm, sc in scores.items():
        for lab, sl in [
            ("PRE-2024", slice(None, "2023-12-31")),
            ("POST-2024", slice("2024-01-01", None)),
        ]:
            s = sc if sl is None else sc.loc[sl]
            pf = backtest(s, fwd.loc[s.index], regime, dv, dv_med, None)
            sr, lo, hi = sharpe_ci(pf)
            print(
                f"{nm:16s} {lab:10s} Sharpe={sr:+.2f} CI[{lo:+.2f},{hi:+.2f}] nwk={len(pf) if pf is not None else 0}"
            )


if __name__ == "__main__":
    main()
