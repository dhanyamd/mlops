"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) for the FAS book.

A raw Sharpe is inflated by two things this corrects:

  1. SELECTION BIAS. We did not test one strategy, we tested many -- w in
     {0.25,0.5,0.75,1.0}, dir in {+1,-1}, CGO lookbacks {5,6,7,8,10,14,20},
     simplified vs Grinblatt-Han, carry vs carry+size, several rebalance
     cadences. Picking the best of N trials inflates the winner even if every
     trial were worthless.
  2. NON-NORMALITY + SAMPLE LENGTH. Sharpe assumes iid normal returns. Crypto
     returns are skewed and fat-tailed, and 55 weekly observations is a small
     sample; both distort the usual significance of a Sharpe.

Method (as published):

    PSR(SR*) = Z[ (SR - SR*) * sqrt(T-1) / sqrt(1 - g3*SR + (g4-1)/4 * SR^2) ]

with g3 = skew, g4 = kurtosis (non-excess) of the per-period returns, T the
number of observations, and Z the standard normal CDF.

The DSR sets SR* to the EXPECTED MAXIMUM Sharpe across N independent trials
under the null that every trial has zero true skill:

    SR* = sd(SR_trials) * [ (1-gamma) * Z^-1(1 - 1/N)
                            + gamma  * Z^-1(1 - 1/(N*e)) ]

gamma = Euler-Mascheroni (0.5772...). DSR = PSR(SR*) is then the probability
the strategy's true Sharpe exceeds what the best of N lucky trials would show.
DSR > 0.95 is the usual bar.

Note on N: Lopez de Prado stresses that trials are usually NOT independent
(ours share the same FAS+SMB core), so the effective N is smaller than the raw
count. We therefore report a RANGE of N rather than a single flattering value,
and quote the conservative end.

Run:
    uv run python -m scripts.deflated_sharpe
    uv run python -m scripts.deflated_sharpe --cache /tmp/quant_cache/fas_long.json
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
from scipy import stats

from scripts.perm_test_research import book_returns
from scripts.research_fas_clean import _liquidity_mask, _rank_z, fas_scores, load, smb_scores
from scripts.research_fas_invent import rcgo_scores

EULER = 0.577215664901532


def psr(sr: float, sr_star: float, T: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_star)."""
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0 or T < 2:
        return float("nan")
    z = (sr - sr_star) * math.sqrt(T - 1) / math.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sr(sd_trials: float, n_trials: int) -> float:
    """Expected maximum Sharpe over n independent zero-skill trials."""
    if n_trials < 2 or sd_trials <= 0:
        return 0.0
    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(sd_trials * ((1.0 - EULER) * a + EULER * b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/quant_cache/asym_warm_start.json.binance")
    ap.add_argument("--no-rcgo", action="store_true", help="BASELINE FAS+SMB only")
    ap.add_argument("--w-rcgo", type=float, default=0.5)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument(
        "--trials",
        default="10,20,40,80",
        help="candidate effective trial counts (they are not independent, so report a range)",
    )
    a = ap.parse_args()

    cw, vw, aw, dcl, dvl = load(a.cache)
    sym = _liquidity_mask(cw, vw)
    fas = fas_scores(cw, aw, sym)
    smb = smb_scores(vw, sym)
    rc = rcgo_scores(dcl, dvl, aw, fas.index, sym)
    score = (fas[sym] + smb[sym]).apply(_rank_z)
    if not a.no_rcgo:
        score = (score + (a.w_rcgo * rc[sym]).apply(_rank_z)).apply(_rank_z)

    # weekly return series of the book (same construction as the backtest)
    fwd = cw[sym].shift(-1) / cw[sym] - 1.0
    rets, pos = [], None
    for w in score.index[:-1]:
        s = score.loc[w].dropna()
        if len(s) < 8:
            pos = None
            continue
        r = s.sort_values()
        n = max(2, int(round(0.20 * len(r))))
        wp = pd.Series(0.0, index=sym)
        wp[list(r.index[-n:])] = 1.0 / n
        wp[list(r.index[:n])] = -1.0 / n
        ret = float((wp * fwd.loc[w]).reindex(sym).sum(skipna=True))
        if pos is not None:
            ret -= a.cost_bps / 1e4 * float((wp - pos).abs().sum())
        rets.append(ret)
        pos = wp
    r = pd.Series(rets)
    r = r[r != 0]

    T = len(r)
    sr_w = r.mean() / r.std()                 # per-period (weekly) Sharpe
    sr_ann = sr_w * math.sqrt(52)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))   # non-excess

    print("=== observed book ===")
    print(f"  weekly observations T : {T}")
    print(f"  weekly Sharpe         : {sr_w:.4f}")
    print(f"  annualised Sharpe     : {sr_ann:.3f}")
    print(f"  skew                  : {skew:+.3f}")
    print(f"  kurtosis (non-excess) : {kurt:.3f}")
    t_stat = sr_w * math.sqrt(T)
    print(f"  t-statistic           : {t_stat:.2f}  "
          f"({'PASS' if t_stat > 3 else 'BELOW'} Harvey et al. t>3.0)")

    # PSR against a zero benchmark: does the strategy beat SR=0 at all?
    print(f"\n  PSR(SR*=0)            : {psr(sr_w, 0.0, T, skew, kurt):.4f}")

    # Spread of trial Sharpes: estimated from the permutation null, which is the
    # honest stand-in for "how much does a zero-skill variant vary".
    print("\n=== deflating for multiple testing ===")
    null = []
    for i in range(200):
        rng = np.random.default_rng(1000 + i)
        null.append(book_returns(cw, score, sym, 0.20, a.cost_bps, rng=rng) / math.sqrt(52))
    sd_trials = float(np.std(null))
    print(f"  sd of zero-skill trial Sharpes (weekly): {sd_trials:.4f}   (200 permutations)")

    print(f"\n  {'N trials':>9} {'SR* (exp max)':>14} {'DSR':>9}  verdict")
    print("  " + "-" * 52)
    for n in [int(x) for x in a.trials.split(",")]:
        sr_star = expected_max_sr(sd_trials, n)
        d = psr(sr_w, sr_star, T, skew, kurt)
        print(f"  {n:>9} {sr_star:>14.4f} {d:>9.4f}  "
              f"{'PASS (>0.95)' if d > 0.95 else 'FAIL'}")

    print(
        "\n  Trials are NOT independent (all share the FAS+SMB core), so the true\n"
        "  effective N is below the raw count of variants swept. Quote the\n"
        "  conservative (largest-N) row."
    )


if __name__ == "__main__":
    main()
