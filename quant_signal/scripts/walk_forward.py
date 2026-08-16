"""Walk-forward validation -- the only test my searching cannot contaminate.

Across this session ~50+ configurations were tried on the full sample
(momentum windows, vol weightings, carry substitutions, accrual estimators,
regime gates, position caps, RCGO weights). That is exactly the multiple-testing
problem the Deflated Sharpe Ratio penalises, and at realistic trial counts the
best in-sample result (0.95) failed DSR.

Walk-forward fixes that by construction: the configuration is CHOSEN on a
trailing training window and then evaluated on the NEXT block, which was never
used to choose it. Concatenating those out-of-sample blocks gives a return
series that is honest regardless of how much searching happened, because at no
point does the test data influence the selection.

  train window -> pick best config by in-sample Sharpe
  test window  -> record that config's returns (OUT OF SAMPLE)
  roll forward, repeat

Reported:
  * OOS Sharpe of the walk-forward series (the number that means something)
  * in-sample Sharpe of the selected configs (expect it to be higher --
    the gap IS the overfitting)
  * how often each config gets chosen (instability = the search was noise)
  * a fixed-config benchmark for reference

Run:
    uv run python -m scripts.walk_forward --cache /tmp/quant_cache/fas_long_v2.json
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from scripts.research_fas_clean import (
    _liquidity_mask,
    _rank_z,
    fas_scores,
    load,
    smb_scores,
)


def build_candidates(cw, vw, aw, dcl, dvl, sym):
    """The configuration space searched during the session, as scores."""
    fas = fas_scores(cw, aw, sym)
    smb = smb_scores(vw, sym)
    carry = (-aw.reindex(index=cw.index, columns=sym)).apply(_rank_z)
    fas_z, smb_z = fas[sym].apply(_rank_z), smb[sym].apply(_rank_z)

    def mom(k):
        return (cw[sym] / cw[sym].shift(k) - 1.0).apply(_rank_z)

    cands = {
        "FAS+SMB": (fas[sym] + smb[sym]).apply(_rank_z),
        "SMB": smb_z,
        "CARRY+SMB": (carry + smb_z).apply(_rank_z),
    }
    for k in (2, 4, 8):
        cands[f"FAS+SMB+MOM{k}"] = ((fas[sym] + smb[sym]).apply(_rank_z) + mom(k)).apply(_rank_z)
        cands[f"SMB+MOM{k}"] = (smb_z + mom(k)).apply(_rank_z)
    return cands


def returns_for(score, cw, sym, cap, cost_bps, weeks=None):
    """Weekly return series of the quintile book for one score."""
    fwd = cw[sym].shift(-1) / cw[sym] - 1.0
    idx = score.index[:-1] if weeks is None else [w for w in weeks if w in score.index]
    rets, ridx, pos = [], [], None
    for w in idx:
        s = score.loc[w].dropna()
        s = s[s != 0]
        if len(s) < 8:
            pos = None
            continue
        r = s.sort_values()
        n = max(2, int(round(0.20 * len(r))))
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


def sharpe(r: pd.Series) -> float:
    r = r[r != 0]
    if len(r) < 3:
        return float("nan")
    v = r.std() * math.sqrt(52)
    return float(r.mean() * 52 / v) if v > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="/tmp/quant_cache/fas_long_v2.json")
    ap.add_argument("--train", type=int, default=104, help="training weeks")
    ap.add_argument("--test", type=int, default=26, help="out-of-sample weeks per step")
    ap.add_argument("--cap", type=float, default=1.0)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    a = ap.parse_args()

    cw, vw, aw, dcl, dvl = load(a.cache)
    sym = _liquidity_mask(cw, vw)
    cands = build_candidates(cw, vw, aw, dcl, dvl, sym)
    print(f"candidates: {len(cands)}   symbols: {len(sym)}   weeks: {len(cw)}")
    print(f"train={a.train}w  test={a.test}w  cap={a.cap}  cost={a.cost_bps}bps\n")

    # precompute each candidate's full return series once
    series = {k: returns_for(s, cw, sym, a.cap, a.cost_bps) for k, s in cands.items()}
    all_weeks = sorted(set().union(*[set(s.index) for s in series.values()]))

    oos, picks = [], []
    i = a.train
    while i + a.test <= len(all_weeks):
        tr = all_weeks[i - a.train : i]
        te = all_weeks[i : i + a.test]
        # choose ONLY on the training block
        scored = {k: sharpe(s.reindex(tr).dropna()) for k, s in series.items()}
        scored = {k: v for k, v in scored.items() if np.isfinite(v)}
        if not scored:
            i += a.test
            continue
        best = max(scored, key=scored.get)
        picks.append(best)
        oos.append(series[best].reindex(te).dropna())   # evaluate on UNSEEN block
        i += a.test

    if not oos:
        print("not enough data for a walk-forward")
        return
    wf = pd.concat(oos).sort_index()
    wf = wf[wf != 0]

    print("=== WALK-FORWARD (out-of-sample) ===")
    print(f"  steps            : {len(picks)}")
    print(f"  OOS weeks        : {len(wf)}")
    print(f"  OOS ann return   : {wf.mean() * 52:.1%}")
    print(f"  OOS Sharpe       : {sharpe(wf):.2f}   <-- the honest number")
    t = (wf.mean() / wf.std()) * math.sqrt(len(wf))
    print(f"  OOS t-stat       : {t:.2f}")
    wl = (1 + wf).cumprod()
    print(f"  OOS maxDD        : {(wl / wl.cummax() - 1).min():.1%}   wealth {float(wl.iloc[-1]):.2f}x")

    print("\n  config chosen per step (instability => the search was noise):")
    for k, c in pd.Series(picks).value_counts().items():
        print(f"    {k:<20} {c:>3}/{len(picks)}")

    print("\n=== reference: FIXED configs over the same OOS weeks ===")
    for k in ("FAS+SMB", "FAS+SMB+MOM2", "SMB"):
        if k in series:
            fixed = series[k].reindex(wf.index).dropna()
            print(f"  {k:<16} OOS Sharpe {sharpe(fixed):>6.2f}")


if __name__ == "__main__":
    main()
