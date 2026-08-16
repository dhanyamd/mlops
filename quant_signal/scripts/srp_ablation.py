"""What does the construction actually earn? -- read off the executed sweep.

The claim under test is "the way you assemble the factors matters more than
which factors you pick". Stating it as a single ratio (conventional 1.03 vs
ours 2.40) is not defensible, because those two numbers came from a TUNED
strategy and an UNTUNED baseline. Tuning only your own arm and reporting the
gap is the oldest way to manufacture a result, and it is the first thing a
referee checks.

This reads ``scripts.trial_registry`` -- populated by ``scripts.srp_sweep``,
which runs every construction over the SAME hyperparameter grid -- and reports
the gap three ways, weakest assumption first:

  1. PAIRED. For each hyperparameter cell where both constructions ran, the
     difference. This controls for hyperparameters exactly: same rank window,
     same turnover cap, same factors, only the construction differs. The mean
     paired gap is the honest "what construction earns" number, and its t-stat
     says whether it is distinguishable from zero.

  2. BEST-OF-GRID. Max Sharpe achieved by each construction. This is the fair
     version of the headline claim -- tuned versus tuned. Reported with the cell
     count per arm, because a max over more cells is mechanically higher and the
     comparison is only fair at matched n.

  3. MAIN EFFECTS. Each switch (ranking, combine, weighting) averaged over
     everything else, so an interaction in one cell cannot masquerade as a
     general result.

Run:
    uv run python -m scripts.srp_ablation
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np

from scripts.trial_registry import DEFAULT_PATH, load_trials

SRP = ("self", "books", "riskparity")
CONVENTIONAL = ("cross", "blend", "equal")
HYPER = ("rank_window", "rank_min_periods", "vol_window", "vol_min_periods",
         "funding_tilt", "turnover_cap", "top", "smooth", "factors")


def _key(cfg: dict) -> tuple:
    """Everything EXCEPT the construction -- the pairing key."""
    return tuple(
        tuple(cfg[h]) if isinstance(cfg.get(h), list) else cfg.get(h)
        for h in HYPER
    )


def _arm(cfg: dict) -> tuple[str, str, str]:
    return (cfg.get("ranking", "self"), cfg.get("combine", "books"),
            cfg.get("weighting", "riskparity"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=str(DEFAULT_PATH))
    ap.add_argument("--family", default="srp")
    a = ap.parse_args()

    trials = [t for t in load_trials(a.family, a.path)
              if math.isfinite(t.sharpe_weekly)]
    if not trials:
        raise SystemExit(f"no finite trials in {a.path}; run scripts.srp_sweep")

    ann = math.sqrt(52)
    by_arm: dict[tuple, list[float]] = defaultdict(list)
    by_cell: dict[tuple, dict[tuple, float]] = defaultdict(dict)
    for t in trials:
        arm = _arm(t.config)
        by_arm[arm].append(t.sharpe_weekly * ann)
        by_cell[_key(t.config)][arm] = t.sharpe_weekly * ann

    print(f"registry {a.path}: {len(trials)} finite trials, "
          f"{len(by_arm)} constructions, {len(by_cell)} hyperparameter cells\n")

    # -- 1. paired ---------------------------------------------------------
    print("=== 1. PAIRED: same hyperparameters, only construction differs ===")
    pairs = [(c[SRP], c[CONVENTIONAL]) for c in by_cell.values()
             if SRP in c and CONVENTIONAL in c]
    if len(pairs) < 2:
        print(f"  only {len(pairs)} paired cells so far -- sweep still filling\n")
    else:
        d = np.array([s - b for s, b in pairs])
        t_stat = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if d.std(ddof=1) else float("inf")
        print(f"  paired cells            : {len(d)}")
        print(f"  SRP mean                : {np.mean([s for s, _ in pairs]):.3f}")
        print(f"  conventional mean       : {np.mean([b for _, b in pairs]):.3f}")
        print(f"  mean gap                : {d.mean():+.3f}  (median {np.median(d):+.3f})")
        print(f"  sd of gap               : {d.std(ddof=1):.3f}")
        print(f"  t-stat of gap vs zero   : {t_stat:.2f}")
        print(f"  cells where SRP wins    : {int((d > 0).sum())}/{len(d)} "
              f"({100 * (d > 0).mean():.0f}%)\n")

    # -- 2. best of grid ---------------------------------------------------
    print("=== 2. BEST-OF-GRID: tuned vs tuned (the fair headline) ===")
    print(f"  {'construction':<32} {'n':>5} {'best':>7} {'median':>8} {'mean':>7}")
    print("  " + "-" * 64)
    for arm in sorted(by_arm, key=lambda k: -max(by_arm[k])):
        v = by_arm[arm]
        tag = "/".join(arm)
        mark = "  <- SRP" if arm == SRP else ("  <- conventional" if arm == CONVENTIONAL else "")
        print(f"  {tag:<32} {len(v):>5} {max(v):>7.3f} {np.median(v):>8.3f} "
              f"{np.mean(v):>7.3f}{mark}")
    if SRP in by_arm and CONVENTIONAL in by_arm:
        ns, nc = len(by_arm[SRP]), len(by_arm[CONVENTIONAL])
        print(f"\n  best-of-grid gap: {max(by_arm[SRP]) - max(by_arm[CONVENTIONAL]):+.3f}"
              f"   (n={ns} vs {nc}"
              f"{'; UNMATCHED -- max over more cells is mechanically higher' if ns != nc else ''})")

    # -- 3. main effects ---------------------------------------------------
    print("\n=== 3. MAIN EFFECTS: each switch, averaged over everything else ===")
    axes = {"ranking": 0, "combine": 1, "weighting": 2}
    for name, i in axes.items():
        groups: dict[str, list[float]] = defaultdict(list)
        for arm, v in by_arm.items():
            groups[arm[i]].extend(v)
        print(f"  {name}:")
        for lvl in sorted(groups, key=lambda k: -np.mean(groups[k])):
            g = groups[lvl]
            print(f"    {lvl:<14} n={len(g):>5}  mean {np.mean(g):+.3f}  "
                  f"median {np.median(g):+.3f}  best {max(g):.3f}")

    # Paired main effect for each switch: flip ONE axis, hold the other two.
    print("\n  paired (flip one switch, hold the other two fixed):")
    for name, i in axes.items():
        deltas = []
        for cell in by_cell.values():
            for arm, v in cell.items():
                alt = list(arm)
                alt[i] = "self" if name == "ranking" and arm[i] == "cross" else alt[i]
                if name == "ranking":
                    alt[i] = "self"
                    other = "cross"
                elif name == "combine":
                    alt[i] = "books"
                    other = "blend"
                else:
                    alt[i] = "riskparity"
                    other = "equal"
                if arm[i] != other:
                    continue
                good = list(arm)
                good[i] = alt[i]
                if tuple(good) in cell:
                    deltas.append(cell[tuple(good)] - v)
        if len(deltas) >= 2:
            d = np.array(deltas)
            t_ = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if d.std(ddof=1) else float("inf")
            print(f"    {name:<14} n={len(d):>5}  mean {d.mean():+.3f}  t {t_:>6.2f}  "
                  f"wins {100 * (d > 0).mean():.0f}%")
        else:
            print(f"    {name:<14} not enough paired cells yet")


if __name__ == "__main__":
    main()
