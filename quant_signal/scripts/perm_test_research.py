"""Permutation test on the RESEARCH book: is Sharpe 1.93/2.28 better than chance?

Standard signal-permutation test (Monte Carlo permutation for strategy
significance): keep the return series in its original order, randomly permute
WHICH symbols the signal selects each week, recompute the strategy, and count
how often a random selection beats the real one. p = P(null >= observed).

Compares the RISK-ADJUSTED statistic (Sharpe), not total P&L: raw P&L in crypto
is dominated by whichever coin happened to 10x, which is why a P&L-based null
has enormous variance and almost no power.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

from scripts.research_fas_clean import _liquidity_mask, _rank_z, fas_scores, load, smb_scores
from scripts.research_fas_invent import rcgo_scores

C = "/tmp/quant_cache/asym_warm_start.json.binance"

def book_returns(cw, score, symbols, quintile, cost_bps, rng=None):
    fwd = cw[symbols].shift(-1) / cw[symbols] - 1.0
    rets, pos = [], None
    for w in score.index[:-1]:
        s = score.loc[w].dropna()
        if len(s) < 8:
            pos = None; continue
        idx = list(s.index)
        if rng is not None:
            rng.shuffle(idx)                      # permute WHICH names get picked
            s = pd.Series(s.values, index=idx)
        r = s.sort_values(); n = max(2, int(round(quintile * len(r))))
        wp = pd.Series(0.0, index=symbols)
        wp[list(r.index[-n:])] = 1.0 / n
        wp[list(r.index[:n])] = -1.0 / n
        ret = float((wp * fwd.loc[w]).reindex(symbols).sum(skipna=True))
        if pos is not None:
            ret -= cost_bps / 1e4 * float((wp - pos).abs().sum())
        rets.append(ret); pos = wp
    r = pd.Series(rets); r = r[r != 0]
    if len(r) < 3: return 0.0
    vol = r.std() * math.sqrt(52)
    return (r.mean() * 52 / vol) if vol > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--cache", default=C)
    ap.add_argument("--no-rcgo", action="store_true",
                    help="BASELINE FAS+SMB only (config that held up on 363 weeks)")
    ap.add_argument("--w-rcgo", type=float, default=0.5)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    a = ap.parse_args()

    cw, vw, aw, dcl, dvl = load(a.cache)
    sym = _liquidity_mask(cw, vw)
    fas = fas_scores(cw, aw, sym); smb = smb_scores(vw, sym)
    rc = rcgo_scores(dcl, dvl, aw, fas.index, sym)
    score = (fas[sym] + smb[sym]).apply(_rank_z)
    if not a.no_rcgo:
        score = (score + (a.w_rcgo * rc[sym]).apply(_rank_z)).apply(_rank_z)
    q = 0.20

    obs = book_returns(cw, score, sym, q, a.cost_bps)
    print(f"observed Sharpe (GH={os.environ.get('QUANT_CGO_GH','1')}): {obs:.3f}")
    print(f"running {a.n} permutations...", flush=True)
    null = []
    for i in range(a.n):
        rng = np.random.default_rng(i)
        null.append(book_returns(cw, score, sym, q, a.cost_bps, rng=rng))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{a.n}", flush=True)
    null = np.array(null)
    p = float((null >= obs).sum() + 1) / (len(null) + 1)   # never-zero p-value
    print(f"\nnull Sharpe: mean={null.mean():.3f}  sd={null.std():.3f}  "
          f"max={null.max():.3f}  p95={np.percentile(null,95):.3f}")
    print(f"observed    : {obs:.3f}")
    print(f"p-value     : {p:.4f}   ({'SIGNIFICANT at 5%' if p<0.05 else 'NOT significant at 5%'})")

if __name__ == "__main__":
    main()
