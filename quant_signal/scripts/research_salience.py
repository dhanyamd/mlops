"""Salience-Turnover Confirmation (SAT) -- the research bet, and its parts.

WHY THIS, AND WHERE IT COMES FROM
---------------------------------
Everything this project invented from scratch (FAS, RCGO) failed on clean data.
What survived was the one BORROWED factor, size. So this file stops inventing
mechanisms and starts from published cross-sectional evidence, then makes one
novel combination that the sources imply but none of them test.

Three independent literatures converge on the same two quantities:

  1. Cosemans & Frehen (JFE 2021), "Salience Theory and Stock Prices".
     Investors overweight past returns that STAND OUT against the market that
     day. Salient upside -> overpriced -> low future return, and vice versa.
     Implemented here verbatim from the paper, equations (8), (3) and (9):

         sigma(r_is, rbar_s) = |r_is - rbar_s| / (|r_is| + |rbar_s| + theta)   (8)
         omega_is            = delta^k_is / sum_s' (delta^k_is' * pi_s')       (3)
         ST_i,t              = cov[omega_is, r_is] = E^ST[r_is] - rbar_i,t     (9)

     k_is is the salience RANK of day s (1 = most salient). Parameters are the
     BGS (2012) calibration the paper itself uses -- theta = 0.1, delta = 0.7 --
     not fitted here. Equation (9) collapses to something simple and unfittable:
     the salience-weighted mean daily return minus the equal-weighted one.
     Prediction: ST enters NEGATIVELY.

  2. Zhongtai Securities, 凸显性收益因子 STR series (行为金融研究系列).
     Reproduced Cosemans-Frehen on A-shares, found IC significantly negative and
     the strategy BEATING short-term reversal -- and, crucially, built variants
     from a TRADING perspective (STT/STT2) using turnover rather than returns.
     They treat return-salience and turnover-salience as two SEPARATE factors.

  3. Chinese turnover-volatility factor (换手率相对波动率): the standard
     deviation of trailing turnover carries a large negative IC, with the stated
     reading that names with the most unstable short-term turnover are the ones
     being hyped (炒作), and hyped names subsequently underperform.

  4. "Crypto factor zoo": an iterative GRS reduction over 36 crypto return
     predictors converges on turnover volatility + salience (+ an on-chain term
     we cannot compute), and reports both remain economically meaningful under
     moderate transaction costs. Cai & Zhao (JBF 2024) separately find the
     salience effect in crypto is an order of magnitude larger than in equities.

THE NOVEL STEP
--------------
Every source above treats salience and turnover as SEPARATE factors and adds
them. None conditions one on the other. But they measure different halves of
one mechanism: salience is what draws attention, turnover is the footprint of
that attention actually being ACTED ON.

That distinction has a testable implication. A coin whose return was salient
but whose trading never responded is a distortion that has not yet been
expressed in price -- there is nothing to revert. A coin whose return was
salient AND whose turnover exploded is one where the crowd has already
transacted; the mispricing is complete and is what reverts.

So turnover should not be ADDED to salience as a second opinion. It should
CONFIRM it -- scaling conviction in the salience bet without touching its sign,
which is set by ST alone:

    SAT_i = z( -z(ST_i) * c_i ),    c_i = 0.5 + cross-sectional rank-pct(TVOL_i)

c_i in [0.5, 1.5] modulates magnitude only. This is the bet.

A NOTE ON TURNOVER IN CRYPTO
----------------------------
Turnover is volume/float, and we have no float for perps. But TVOL here is the
coefficient of variation of volume over the window, and float is constant across
a 4-week window, so CV(turnover) = CV(volume) EXACTLY -- the float cancels. This
is an identity, not an approximation, and it is why the Chinese turnover-vol
factor ports to perps unchanged.

WINDOW
------
Cosemans & Frehen state the window should match the forecast horizon and use one
month for monthly forecasts. Crypto trades 7 days a week, so their ~21-observation
month is 28 calendar days here. That is the primary spec; --window exposes it for
the robustness check reported separately rather than searched over.

Run:
    uv run python -m scripts.research_salience --cache /tmp/quant_cache/fas_broad.json
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from scripts.research_fas_clean import _liquidity_mask, _rank_z, load, smb_scores

THETA = 0.1   # BGS (2012) calibration, as used by Cosemans & Frehen -- not fitted
DELTA = 0.7   # BGS (2012) calibration, as used by Cosemans & Frehen -- not fitted
MIN_OBS = 15  # the paper's minimum daily observations to form ST


def salience_and_turnover(dcl: pd.DataFrame, dvl: pd.DataFrame, weeks, sym, window: int):
    """ST (equation 9) and turnover volatility, point-in-time at each week end.

    Only daily observations stamped at or before the week end are used, so a
    score formed at w never sees the return it is asked to predict.
    """
    dret = dcl[sym].pct_change()
    rbar = dret.mean(axis=1, skipna=True)          # market return that day, equation (8)

    st = pd.DataFrame(index=weeks, columns=sym, dtype=float)
    tv = pd.DataFrame(index=weeks, columns=sym, dtype=float)

    for w in weeks:
        upto = dret.index[dret.index <= w]
        if len(upto) < MIN_OBS:
            continue
        days = upto[-window:]
        R = dret.loc[days].to_numpy(dtype=float)          # (S, N)
        M = rbar.loc[days].to_numpy(dtype=float)          # (S,)
        V = dvl.loc[days, sym].to_numpy(dtype=float)

        sig = np.abs(R - M[:, None]) / (np.abs(R) + np.abs(M)[:, None] + THETA)

        for j in range(R.shape[1]):
            r, s, v = R[:, j], sig[:, j], V[:, j]
            ok = np.isfinite(r) & np.isfinite(s)
            if ok.sum() < MIN_OBS:
                continue
            r_ok, s_ok = r[ok], s[ok]
            # rank 1 = MOST salient; ties broken by order, which delta makes
            # numerically irrelevant at these window lengths
            k = np.empty(len(s_ok), dtype=float)
            k[np.argsort(-s_ok, kind="stable")] = np.arange(1, len(s_ok) + 1)
            dk = DELTA**k
            omega = dk / dk.mean()                  # equation (3) with pi_s = 1/S
            st.loc[w, sym[j]] = float((omega * r_ok).mean() - r_ok.mean())   # equation (9)

            vv = v[np.isfinite(v)]
            if len(vv) >= MIN_OBS and vv.mean() > 0:
                # CV of volume == CV of turnover (float cancels over the window)
                tv.loc[w, sym[j]] = float(vv.std() / vv.mean())
    return st.astype(float), tv.astype(float)


def rank_pct(row: pd.Series) -> pd.Series:
    r = row.dropna()
    if len(r) < 2:
        return pd.Series(np.nan, index=row.index)
    return row.rank(pct=True)


def book_returns(score: pd.DataFrame, cw, sym, cost_bps=10.0, cap=1.0, top=0.20):
    """Same quintile long/short construction the rest of the project uses."""
    fwd = cw[sym].shift(-1) / cw[sym] - 1.0
    rets, ridx, pos = [], [], None
    for w in score.index[:-1]:
        if w not in fwd.index:
            continue
        s = score.loc[w].dropna()
        s = s[s != 0]
        if len(s) < 8:
            pos = None
            continue
        r = s.sort_values()
        n = max(2, int(round(top * len(r))))
        wp = pd.Series(0.0, index=sym)
        wp[list(r.index[-n:])] = 1.0 / n
        wp[list(r.index[:n])] = -1.0 / n
        f = fwd.loc[w]
        if cap is not None:
            f = f.clip(upper=cap)
        ret = float((wp * f).reindex(sym).sum(skipna=True))
        if pos is not None and cost_bps:
            ret -= cost_bps / 1e4 * float((wp - pos).abs().sum())
        rets.append(ret)
        ridx.append(w)
        pos = wp
    return pd.Series(rets, index=ridx)


def ic_of(score: pd.DataFrame, cw, sym) -> tuple[float, float]:
    """Spearman IC and its t-stat -- the Chinese standard for factor efficacy."""
    fwd = cw[sym].shift(-1) / cw[sym] - 1.0
    ics = []
    for w in score.index[:-1]:
        if w not in fwd.index:
            continue
        a, b = score.loc[w], fwd.loc[w]
        m = a.notna() & b.notna()
        if m.sum() >= 8:
            ics.append(a[m].corr(b[m], method="spearman"))
    s = pd.Series(ics).dropna()
    if len(s) < 3:
        return float("nan"), float("nan")
    return float(s.mean()), float(s.mean() / s.std() * math.sqrt(len(s)))


def report(name: str, score: pd.DataFrame, cw, sym, cost_bps: float) -> pd.Series:
    r = book_returns(score, cw, sym, cost_bps=cost_bps)
    r = r[r != 0]
    ic, ict = ic_of(score, cw, sym)
    v = r.std() * math.sqrt(52)
    sh = r.mean() * 52 / v if v > 0 else 0.0
    t = (r.mean() / r.std()) * math.sqrt(len(r)) if len(r) > 2 else float("nan")
    wl = float((1 + r).cumprod().iloc[-1])
    print(
        f"{name:<22} {len(r):>4} {r.mean()*52:>7.1%} {v:>7.1%} {sh:>7.2f} "
        f"{t:>6.2f} {wl:>9.2f} {ic:>8.4f} {ict:>7.2f}"
    )
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="/tmp/quant_cache/fas_broad.json")
    ap.add_argument("--window", type=int, default=28, help="daily observations in the ST state space")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    a = ap.parse_args()

    cw, vw, aw, dcl, dvl = load(a.cache)
    sym = _liquidity_mask(cw, vw)
    print(f"cache {a.cache}   symbols {len(sym)}   weeks {len(cw)}   window {a.window}d")
    print(f"theta={THETA} delta={DELTA} (BGS 2012 calibration, not fitted)\n")

    st, tv = salience_and_turnover(dcl, dvl, cw.index, sym, a.window)
    cov = st.notna().sum(axis=1)
    print(f"ST coverage: median {int(cov.median())}/{len(sym)} symbols per week\n")

    smb = smb_scores(vw, sym)[sym].apply(_rank_z)
    st_z = (-st).apply(_rank_z)          # equation (7): ST enters NEGATIVELY
    tv_z = (-tv).apply(_rank_z)          # turnover instability = hype = short it
    conv = tv.apply(rank_pct) + 0.5      # confirmation weight in [0.5, 1.5]
    sat = (st_z * conv).apply(_rank_z)   # THE BET: turnover confirms, never flips sign

    print(f"{'config':<22} {'wks':>4} {'ann':>7} {'vol':>7} {'Sharpe':>7} "
          f"{'t':>6} {'wealth':>9} {'IC':>8} {'ICt':>7}")
    print("-" * 90)
    out = {}
    out["SMB (baseline)"] = report("SMB (baseline)", smb, cw, sym, a.cost_bps)
    out["ST alone"] = report("ST alone", st_z, cw, sym, a.cost_bps)
    out["TurnoverVol alone"] = report("TurnoverVol alone", tv_z, cw, sym, a.cost_bps)
    out["SAT (novel)"] = report("SAT (novel)", sat, cw, sym, a.cost_bps)
    print("-" * 90)
    out["SMB+ST"] = report("SMB+ST", (smb + st_z).apply(_rank_z), cw, sym, a.cost_bps)
    out["SMB+TVol"] = report("SMB+TVol", (smb + tv_z).apply(_rank_z), cw, sym, a.cost_bps)
    out["SMB+ST+TVol"] = report("SMB+ST+TVol", (smb + st_z + tv_z).apply(_rank_z), cw, sym, a.cost_bps)
    out["SMB+SAT"] = report("SMB+SAT", (smb + sat).apply(_rank_z), cw, sym, a.cost_bps)

    print("\ncorrelation of weekly returns (is SAT distinct from its parts?)")
    print(pd.DataFrame(out).corr().round(2).to_string())


if __name__ == "__main__":
    main()
