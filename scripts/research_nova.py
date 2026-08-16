#!/usr/bin/env python3
"""NOVA - Volume-Exhaustion Reversal (novel, keyless, 4.4yr backtest).

Data: /tmp/broad_pull/*.csv  (Binance public REST, NO KEY) -> daily close, qvol,
funding for 416 coins, 2022-03-17..2026-08-12 (~4.4y).

Reasoning (from breakthrough papers):
- Mispricing/JFQA: reversal is the #1 weekly-alpha driver; concentrated in
  illiquid coins; it is a LIQUIDITY/illiquidity effect, not an information effect.
- Crypto Factor Zoo (2026): turnover-volatility / volume microstructure dominates
  the priced factor set; volume is NOT collinear with price.
- Prior inventions (ASYM/FVCS/REVX) all conditioned on FUNDING, which is
  collinear with price (fund_z ~= -mom_z) -> spanned. NOVA conditions on VOLUME
  trend instead -> orthogonal input -> genuinely distinct.

Mechanism: not every dip reverts. A price drop on COLLAPSING volume is a
mechanical/illiquidity dip (no real selling pressure) -> reverts. A drop on
RISING/spike volume is informed distribution -> continues. So fade the move ONLY
when volume is exhausted.

NOVA = zs( rev_plain * (1 + vol_exhaust) )
  rev_plain  = -zs(multi-horizon 4/8/12w return)   (plain reversal)
  vol_exhaust= -zs(short/long volume ratio - 1)      (declining volume = positive)
The volume gate does NOT use price -> residual carries independent information.
"""

import glob, os, random
import numpy as np
import pandas as pd

BROAD = "/tmp/broad_pull"
MIN_WEEKS = 150  # ~3y of weekly history to be in the panel


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


def main():
    print("== loading broad_pull (keyless Binance) ==")
    close, qvol, fund = load()
    print(
        f"coins={close.shape[1]} dates={close.shape[0]} "
        f"({close.index[0].date()}..{close.index[-1].date()})"
    )

    # keep coins with enough history
    valid = close.notna().sum() >= MIN_WEEKS
    syms = valid[valid].index.tolist()
    print(f"coins with >= {MIN_WEEKS} weekly obs: {len(syms)}")
    close = close[syms]
    qvol = qvol[syms]
    fund = fund[syms]

    # weekly resample
    wc = close.resample("W").last()
    wq = qvol.resample("W").sum()
    wf = fund.resample("W").mean()
    dret = close.pct_change()
    wvol = dret.resample("W").std()  # weekly realized vol = high-range proxy
    dv = wq.rolling(12).sum().shift(1)  # trailing annualized dollar volume
    dv_med = dv.median(axis=1)
    print(f"weekly panel: {wc.shape[0]} weeks x {wc.shape[1]} coins")

    ret = wc.pct_change()
    fwd = ret.shift(
        -1
    )  # forward return: signal formed on w close earns w->w+1 (NO same-week look-ahead)

    # --- factor scores (cross-sectional z per week) ---
    # plain reversal: multi-horizon 4/8/12w
    r4 = wc.shift(1).div(wc.shift(5)) - 1
    r8 = wc.shift(1).div(wc.shift(9)) - 1
    r12 = wc.shift(1).div(wc.shift(13)) - 1
    rev_raw = zs_df(r4) + zs_df(r8) + zs_df(r12)
    rev_plain = -zs_df(rev_raw)

    # momentum (MHT-like, multi-horizon)
    m1 = wc.shift(1).div(wc.shift(5)) - 1
    m2 = wc.shift(5).div(wc.shift(13)) - 1
    m3 = wc.shift(13).div(wc.shift(25)) - 1
    mht = zs_df(m1) + zs_df(m2) + zs_df(m3)

    # volume exhaustion: short/long volume ratio trend
    vshort = wq.rolling(4).mean()
    vlong = wq.rolling(12).mean()
    vol_ratio = vshort.div(vlong) - 1.0
    vol_exhaust = -zs_df(vol_ratio)  # declining volume -> positive

    # NOVA: reversal gated by volume exhaustion
    nova_raw = rev_plain * (1.0 + 0.5 * vol_exhaust)
    nova = zs_df(nova_raw)

    # ---- FAS: Funding-Accrual Squeeze (NOVEL, keyless funding history) ----
    # Reasoning: funding *level* is collinear with price (fund_z ~= -mom_z) -> spanned.
    # But the CUMULATIVE funding paid over a window (positioning PATH) residualized
    # on the price path is a PURE crowding signal orthogonal to trend. A crowd that
    # has paid one way for 12 weeks while price DIVERGED is trapped/underwater ->
    # forced unwind -> mean reversion (squeeze). Nobody does XS factors on
    # funding-accrual residualized on price. Uses only keyless 4.4y funding.
    fund_accrual = wf.rolling(12).sum()  # cumulative funding paid, trailing 12w
    price_ret12 = wc.shift(1).div(wc.shift(13)) - 1.0
    pr_z = zs_df(price_ret12)
    # residualize fund_accrual on price path -> pure positioning (orthogonal to trend)
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
        yy = y.values[m]
        XX = X[m]
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        fund_resid.loc[w, y.index[m]] = yy - XX @ beta
    fund_resid_z = zs_df(fund_resid)
    # fade the trapped crowd: crowd-long & price down -> bounce (long); mirror for shorts
    fas_raw = fund_resid_z * (-pr_z)
    fas = zs_df(fas_raw)

    # ---- FAS2: Funding-Accrual Squeeze, refined (front-loaded) timing ----
    # FAS1 interacts the funding residual with the 12w FORMATION return (pr_z).
    # FAS2 triggers the squeeze on RECENT price confirmation the crowd is trapped
    # NOW, motivated by cascade physics (forced flow is front-loaded, 30min-30d,
    # not the slow 12w formation). Use a ~3w recent return as the "trapped now"
    # signal: crowd-long & price just fell, or crowd-short & price just rose.
    r_recent = wc.shift(1).div(wc.shift(4)) - 1.0  # ~3w recent return
    fas2_raw = fund_resid_z * (-zs_df(r_recent))
    fas2 = zs_df(fas2_raw)

    # ---- SMB: Size / illiquidity (best OOS diversifier, keyless) ----
    # Springer 2024: SIZE (MARCAP) is the #1 crypto diversifier OOS with costs;
    # agentic paper's top OOS factors are all small-cap/liquidity-scarce. dv is
    # trailing annualized dollar-volume; zs is rank-based so log() is redundant AND
    # would hit log(0) for freshly-listed coins -> use dv directly. Long small/illiquid.
    smb = -zs_df(dv)

    # ---- MOM3: short-horizon risk-adjusted momentum (RMOM3-style candidate) ----
    # Springer 2024: RMOM3 (3-week momentum) is a GOOD diversifier OOS; the long
    # multi-horizon MHT L/S is negative here, so test a SHORT 3w window instead.
    mom3 = zs_df(wc.shift(1).div(wc.shift(4)) - 1.0)

    # ---- ENSEMBLE: Unravel's recipe -> ~3 distinct, arithmetic-averaged,
    # market-neutral factors raise Sharpe above any single one (3 orthogonal
    # portfolios ~ Sharpe 2.5). We AVERAGE the RAW z-scored distinct streams
    # (NO residualizing a component on the others -> that double-counts & kills
    # alpha, the bug in the old COMBO). Components chosen for distinct mechanisms:
    #   REV_PLAIN = short-horizon reversal (price mechanism)
    #   SMB       = size/illiquidity cross-section (liquidity mechanism)
    #   FAS       = funding-accrual squeeze (positioning mechanism, our invention)
    # MOM3 is a candidate added only if its spanning test shows distinct positive alpha.
    ensemble = zs_df(rev_plain + smb + fas)

    # BTC regime gate (use BTCUSDT if present else fetch)
    if "BTCUSDT" in wc.columns:
        btc = wc["BTCUSDT"]
    else:
        import urllib.request, json

        u = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=2000"
        k = json.loads(urllib.request.urlopen(u, timeout=25).read().decode())
        btc = pd.Series(
            [float(x[4]) for x in k], index=pd.to_datetime([x[6] for x in k], unit="ms")
        ).sort_index()
        btc = btc.resample("W").last()
        btc = btc.reindex(wc.index).ffill()
    btc_ma = btc.rolling(52).mean()
    regime = (btc > btc_ma).astype(float)  # 1 = uptrend

    scores = {
        "NOVA": nova,
        "FAS": fas,
        "FAS2": fas2,
        "SMB": smb,
        "MOM3": mom3,
        "ENSEMBLE": ensemble,
        "MHT": mht,
        "REV_PLAIN": rev_plain,
    }
    print("\n=== BACKTEST (long top quintile - short bottom, 10bps, BTC-regime gated) ===")
    for name, sc in scores.items():
        for label, mask in [("FULL", None), ("SMALL/illiquid", "small"), ("LARGE/liquid", "large")]:
            pf = backtest(sc, fwd, regime, dv, dv_med, mask)
            if pf is None:
                continue
            sr, lo, hi = sharpe_ci(pf)
            print(
                f"{name:10s} {label:16s} Sharpe={sr:+.2f}  CI[{lo:+.2f},{hi:+.2f}]  nwk={len(pf)}"
            )

    # ---- SPANNING TEST (up front, as demanded) ----
    print("\n=== SPANNING: NOVA residualized on [MHT, REV_PLAIN] ===")
    nova_r = ortho(nova, [mht, rev_plain])
    for label, mask in [("FULL", None), ("SMALL/illiquid", "small"), ("LARGE/liquid", "large")]:
        raw_pf = backtest(nova, fwd, regime, dv, dv_med, mask)
        res_pf = backtest(nova_r, fwd, regime, dv, dv_med, mask)
        if raw_pf is None or res_pf is None:
            continue
        s_raw, _, _ = sharpe_ci(raw_pf)
        s_res, lo, hi = sharpe_ci(res_pf)
        print(
            f"NOVA {label:16s} raw Sharpe={s_raw:+.2f}  |MOM,REV Sharpe={s_res:+.2f} CI[{lo:+.2f},{hi:+.2f}]"
        )

    print("\n=== SPANNING: SMB / MOM3 / FAS2 / ENSEMBLE residualized on [MHT, REV_PLAIN] ===")
    for nm, sc in [("SMB", smb), ("MOM3", mom3), ("FAS2", fas2), ("ENSEMBLE", ensemble)]:
        sc_r = ortho(sc, [mht, rev_plain])
        raw_pf = backtest(sc, fwd, regime, dv, dv_med, None)
        res_pf = backtest(sc_r, fwd, regime, dv, dv_med, None)
        if raw_pf is None or res_pf is None:
            continue
        s_raw, _, _ = sharpe_ci(raw_pf)
        s_res, lo, hi = sharpe_ci(res_pf)
        print(
            f"{nm:10s} raw Sharpe={s_raw:+.2f}  |MOM,REV Sharpe={s_res:+.2f} CI[{lo:+.2f},{hi:+.2f}]"
        )

    print("\n=== rank-corr of factors with known factors (cross-sectional mean) ===")
    print(f"NOVA ~ MHT       = {rank_corr(nova, mht):+.3f}")
    print(f"NOVA ~ REV_PLAIN = {rank_corr(nova, rev_plain):+.3f}")
    print(f"FAS  ~ MHT       = {rank_corr(fas, mht):+.3f}")
    print(f"FAS  ~ REV_PLAIN = {rank_corr(fas, rev_plain):+.3f}")
    print(f"SMB  ~ MHT       = {rank_corr(smb, mht):+.3f}")
    print(f"SMB  ~ REV_PLAIN = {rank_corr(smb, rev_plain):+.3f}")
    print(f"SMB  ~ FAS       = {rank_corr(smb, fas):+.3f}")
    print(f"FAS2 ~ MHT       = {rank_corr(fas2, mht):+.3f}")
    print(f"FAS2 ~ REV_PLAIN = {rank_corr(fas2, rev_plain):+.3f}")
    print(f"FAS2 ~ FAS       = {rank_corr(fas2, fas):+.3f}")
    print(f"MOM3 ~ REV_PLAIN = {rank_corr(mom3, rev_plain):+.3f}")
    print(f"ENS ~ REV_PLAIN  = {rank_corr(ensemble, rev_plain):+.3f}")
    print(f"MHT  ~ REV_PLAIN = {rank_corr(mht, rev_plain):+.3f}")

    # ---- ROBUSTNESS: different data slices (user asked "test on diff data") ----
    # PRE-2023 run WITHOUT the bear-regime gate to reveal raw signal quality
    # (with the gate, the 2022 bear keeps the book flat -> all zeros, which is
    # by-design crash protection, not a signal failure).
    run_block(
        scores,
        fwd,
        regime,
        dv,
        dv_med,
        rowslice=slice(None, "2023-12-31"),
        tag="SUB-PERIOD PRE 2024 (regime OFF - raw signal)",
        use_regime=False,
    )
    run_block(
        scores,
        fwd,
        regime,
        dv,
        dv_med,
        rowslice=slice("2024-01-01", None),
        tag="SUB-PERIOD POST 2024-2026",
    )
    med_dv = dv.median().sort_values(ascending=False)
    top100 = med_dv.index[:100]
    run_block(
        scores,
        fwd,
        regime,
        dv,
        dv_med,
        cols=top100,
        tag="TOP-100 LIQUID SUBSET (different universe)",
    )
    print("\n=== NOVA volume-weight sensitivity (FULL) ===")
    for wgt in [0.3, 0.5, 0.7, 1.0]:
        nv = zs_df(rev_plain * (1.0 + wgt * vol_exhaust))
        pf = backtest(nv, fwd, regime, dv, dv_med, None)
        sr, lo, hi = sharpe_ci(pf)
        print(f"  weight={wgt}: Sharpe={sr:+.2f} CI[{lo:+.2f},{hi:+.2f}]")


def backtest(score, ret, regime, dv, dv_med, size_mask, top=0.2, bot=0.2, use_regime=True):
    weeks = score.index
    pf_ret = []
    prev_w = None
    for w in weeks:
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
            if size_mask == "small":
                sc = sc[d <= med]
            else:
                sc = sc[d > med]
            r = r.reindex(sc.index)
            if len(sc) < 15:
                prev_w = None
                continue
        # long top, short bottom
        thr_hi = sc.quantile(1 - top)
        thr_lo = sc.quantile(bot)
        longs = sc[sc >= thr_hi].index
        shorts = sc[sc <= thr_lo].index
        if len(longs) == 0 or len(shorts) == 0:
            prev_w = None
            continue
        wk = r[longs].mean() - r[shorts].mean()
        cost = 0.0010 * 2 if prev_w is not None else 0.0010  # 10bps enter; +10bps exit if held
        wk = wk - cost
        if use_regime and regime.loc[w] == 0:  # flat in bear regime (optional)
            wk = 0.0
            prev_w = None
        else:
            prev_w = (set(longs), set(shorts))
        pf_ret.append(wk)
    if len(pf_ret) < 30:
        return None
    return pd.Series(pf_ret)


def sharpe_ci(pf: pd.Series):
    pf = pf.dropna()
    if len(pf) < 10:
        return float("nan"), float("nan"), float("nan")
    mu = pf.mean()
    sd = pf.std()
    sr = mu / sd * np.sqrt(52) if sd > 0 else 0.0
    rng = random.Random(0)
    N = len(pf)
    B = 2000
    boot = np.empty(B)
    arr = pf.values
    for b in range(B):
        idx = [rng.randrange(N) for _ in range(N)]
        s = arr[idx]
        m = s.mean()
        v = s.std()
        boot[b] = (m / v * np.sqrt(52)) if v > 0 else 0.0
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
        yy = y.values[m]
        XX = X[m]
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
        xr = x.rank()
        yr = y.rank()
        dx = xr - xr.mean()
        dy = yr - yr.mean()
        den = np.sqrt((dx**2).sum() * (dy**2).sum())
        if den > 0:
            out.append(float((dx * dy).sum() / den))
    return float(np.mean(out)) if out else float("nan")


def run_block(scores, fwd, regime, dv, dv_med, rowslice=None, cols=None, tag="", use_regime=True):
    print(f"\n--- {tag} ---")
    for name, sc in scores.items():
        s = sc if cols is None else sc[cols]
        if rowslice is not None:
            s = s.loc[rowslice]
        for label, mask in [("FULL", None), ("SMALL", "small"), ("LARGE", "large")]:
            pf = backtest(s, fwd, regime, dv, dv_med, mask, use_regime=use_regime)
            if pf is None:
                continue
            sr, lo, hi = sharpe_ci(pf)
            print(f"{name:10s} {label:8s} Sharpe={sr:+.2f} CI[{lo:+.2f},{hi:+.2f}] nwk={len(pf)}")


if __name__ == "__main__":
    main()
