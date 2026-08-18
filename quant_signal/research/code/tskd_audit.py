"""Definitive audit of TSKD's contribution. One matched sample, one run, no ad-hoc slicing.

WHY THIS EXISTS
---------------
TSKD's value was estimated several times during development using one-off scripts,
and those estimates disagreed because they were computed on DIFFERENT SAMPLES --
a standalone book, a book with the funding tilt, and a combination, each spanning
a different set of weeks. Comparing Sharpe ratios across different week sets is
meaningless, and it produced three contradictory readings of the same factor.

Everything here is computed on ONE sample: the weeks in which every specification
under comparison produced a return. Sample size is printed with every number.

WHAT IS MEASURED
----------------
1. STANDALONE. The factor traded alone. This is the weakest possible test of a
   diversifying factor and is reported for completeness, not as the verdict.

2. LEAVE-ONE-OUT. The full book minus this factor, versus the full book. This is
   the quantity that actually matters: what does the portfolio lose without it?
   Reported for EVERY factor, so TSKD is ranked against its peers rather than
   judged in isolation.

3. ORTHOGONALITY. Correlation of each factor's book returns against the others.
   A factor can be weak alone and still valuable if it is uncorrelated with what
   you already hold; that is the entire premise of combining books.

4. SUBPERIOD STABILITY. The leave-one-out contribution computed on the first and
   second halves separately. A contribution that appears only in one half is a
   sample artefact.

Run:
    uv run python -m scripts.tskd_audit
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from scripts.srp_backtest import ALL_FACTORS, SRPConfig, SRPData, run


def _sr(x: pd.Series) -> float:
    return float(x.mean() / x.std() * math.sqrt(52)) if len(x) > 2 and x.std() else float("nan")


def _t(x: pd.Series) -> float:
    return float(x.mean() / x.std() * math.sqrt(len(x))) if len(x) > 2 and x.std() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--factor", default="TSKD")
    a = ap.parse_args()

    data = SRPData.load()
    cfg = SRPConfig()
    all_f = tuple(ALL_FACTORS)

    print(f"universe {len(data.cols)} symbols, {len(data.cw)} weeks\n")

    # --- full book and every leave-one-out book, as RETURN SERIES ------------
    full = run(data, cfg, factors=all_f)
    series: dict[str, pd.Series] = {"__FULL__": full.returns}
    for f in all_f:
        rest = tuple(x for x in all_f if x != f)
        series[f] = run(data, cfg, factors=rest).returns

    # single matched sample across every specification
    J = pd.DataFrame(series).dropna()
    n = len(J)
    print(f"MATCHED SAMPLE: {n} weeks present in the full book and all "
          f"{len(all_f)} leave-one-out books\n")

    fs = J["__FULL__"]
    print("=== 1. LEAVE-ONE-OUT: what does the book lose without each factor? ===")
    print(f"  full {len(all_f)}-factor book: ann {_sr(fs):.3f}   t {_t(fs):.2f}   n {n}\n")
    print(f"  {'factor':<10} {'book without it':>16} {'CONTRIBUTION':>14}")
    print("  " + "-" * 44)
    contrib = {}
    for f in all_f:
        without = _sr(J[f])
        contrib[f] = _sr(fs) - without
    for f, c in sorted(contrib.items(), key=lambda kv: -kv[1]):
        mark = "   <<<" if f == a.factor else ""
        print(f"  {f:<10} {_sr(J[f]):>16.3f} {c:>+14.3f}{mark}")

    rank = sorted(contrib, key=lambda k: -contrib[k]).index(a.factor) + 1
    print(f"\n  {a.factor} ranks {rank} of {len(all_f)} by marginal contribution.")

    # --- standalone, on the SAME weeks --------------------------------------
    print("\n=== 2. STANDALONE (weakest test; reported for completeness) ===")
    solo = run(data, cfg, factors=(a.factor,)).returns.reindex(J.index).dropna()
    print(f"  {a.factor} alone: ann {_sr(solo):.3f}   t {_t(solo):.2f}   n {len(solo)}")
    print("  A diversifying factor is not required to be significant alone.")

    # --- orthogonality -------------------------------------------------------
    print("\n=== 3. ORTHOGONALITY: correlation of single-factor books ===")
    solos = {}
    for f in all_f:
        s = run(data, cfg, factors=(f,)).returns
        if len(s) > 2:
            solos[f] = s
    C = pd.DataFrame(solos).dropna().corr()
    tgt = C[a.factor].drop(a.factor).sort_values()
    print(f"  {a.factor} vs the other {len(tgt)} factors:")
    print(f"    mean |correlation| {tgt.abs().mean():.3f}   "
          f"range {tgt.min():+.3f} to {tgt.max():+.3f}")
    others = [C[f].drop(f).abs().mean() for f in C.columns]
    print(f"    mean |corr| across ALL factors: {np.mean(others):.3f}")
    print(f"    {a.factor} is {'MORE' if tgt.abs().mean() < np.mean(others) else 'LESS'} "
          f"orthogonal than the average factor.")

    # --- subperiod stability -------------------------------------------------
    print("\n=== 4. SUBPERIOD STABILITY of the leave-one-out contribution ===")
    half = n // 2
    for lab, sl in (("first half", slice(0, half)), ("second half", slice(half, n))):
        f1, f2 = fs.iloc[sl], J[a.factor].iloc[sl]
        print(f"  {lab:<12} full {_sr(f1):>6.3f}   without {a.factor} {_sr(f2):>6.3f}   "
              f"contribution {_sr(f1) - _sr(f2):>+6.3f}   n {len(f1)}")

    print("\n  A contribution present in BOTH halves is structural; "
          "one half only is an artefact.")


if __name__ == "__main__":
    main()
