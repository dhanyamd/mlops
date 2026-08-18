"""Execute the search space and log EVERY configuration to the trial registry.

This is the file that makes the Deflated Sharpe Ratio honest. DSR needs to know
how many strategies were tried and how much their Sharpes varied; both describe
the search rather than the winner, so neither can be recovered afterwards from
the winning backtest. Previously they were supplied from memory. Here they are a
by-product of actually running the search.

WHAT IS SWEPT
-------------
Two things, over the same grid:

  1. THE STRATEGY's hyperparameters -- the knobs that were varied during
     development: rank window, vol window, funding tilt, turnover cap, quintile
     width, smoothing, and whether the ticket factors are included.

  2. THE CONSTRUCTION -- ranking x combine x weighting, all eight cells. This
     serves two purposes. It decomposes which construction choice earns the
     performance gap (the paper's ablation), and it tunes the CONVENTIONAL
     baseline over the identical grid. Without the second, "conventional scores
     1.03, ours scores 2.40" compares a tuned pipeline against an untuned one,
     which is the first thing a referee will attack and they would be right.
     Best-of-grid versus best-of-grid is the only fair comparison.

Every cell is logged, including cells that fail to produce a usable series. A
search branch that died is still a search branch; omitting it is exactly the
undercount DSR exists to catch.

RESUMABLE
---------
Configurations already present in the registry are skipped, so an interrupted
sweep resumes without double-counting (the registry deduplicates by config hash
anyway, but skipping avoids re-running ~15s of backtest per cell).

Run:
    uv run python -m scripts.srp_sweep                 # full grid
    uv run python -m scripts.srp_sweep --quick         # small grid, for a smoke test
    uv run python -m scripts.trial_registry            # inspect what was logged
    uv run python -m scripts.srp_dsr                   # deflate against it
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import time

from scripts.srp_backtest import (
    COMBINES,
    RANKINGS,
    WEIGHTINGS,
    SRPConfig,
    SRPData,
    config_dict,
    run,
)
from scripts.srp_strategy import FACTORS, TICKET_FACTORS
from scripts.trial_registry import DEFAULT_PATH, config_hash, load_trials, log_trial

FAMILY = "srp"

# The hyperparameter grid actually explored during development. Each axis is
# the set of values that were tried, not a neat range invented afterwards.
GRID = {
    "rank_window": [26, 52, 104],
    "vol_window": [26, 52],
    "funding_tilt": [0.0, 0.25, 0.5, 1.0],
    "turnover_cap": [None, 0.4, 0.6],
    "top": [0.10, 0.20, 0.30],
    "smooth": [10, 20, 30],
}
QUICK = {
    "rank_window": [52],
    "vol_window": [52],
    "funding_tilt": [0.0, 0.5],
    "turnover_cap": [None, 0.6],
    "top": [0.20],
    "smooth": [20],
}

# Factor sets tried: everything, the nine without the 5m ticket factors, and
# each single-factor book (those were run to size each factor's contribution).
FACTOR_SETS: list[tuple[str, tuple[str, ...] | None]] = [
    ("all", None),
    ("no_ticket", tuple(FACTORS)),
    ("ticket_only", tuple(TICKET_FACTORS)),
]


def _cells(grid: dict, constructions: list[tuple[str, str, str]],
           factor_sets: list) -> list[dict]:
    keys = list(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        base = dict(zip(keys, combo))
        for fname, facs in factor_sets:
            for ranking, combine, weighting in constructions:
                # A blended composite has exactly one book, so the weighting
                # axis is degenerate -- log it once rather than twice under two
                # hashes that would inflate N with a distinction that is not one.
                if combine == "blend" and weighting != "equal":
                    continue
                out.append({**base, "factor_set": fname, "factors": facs,
                            "ranking": ranking, "combine": combine,
                            "weighting": weighting})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="tiny grid, smoke test")
    ap.add_argument("--path", default=str(DEFAULT_PATH))
    ap.add_argument("--family", default=FAMILY)
    ap.add_argument("--srp-only", action="store_true",
                    help="skip the construction ablation (strategy grid only)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new cells")
    ap.add_argument("--seed", type=int, default=7,
                    help="shuffle order; a partial sweep must be an UNBIASED "
                         "subsample of the grid, not its first rows")
    a = ap.parse_args()

    grid = QUICK if a.quick else GRID
    constructions = ([("self", "books", "riskparity")] if a.srp_only
                     else [c for c in itertools.product(RANKINGS, COMBINES, WEIGHTINGS)])
    fsets = FACTOR_SETS[:1] if a.quick else FACTOR_SETS
    cells = _cells(grid, constructions, fsets)
    # An interrupted ordered sweep would leave the registry holding only the
    # low-index corner of the grid, biasing both the sd estimate and the
    # best-of-grid baseline. Shuffle with a fixed seed so any prefix is a fair
    # sample and the run is still reproducible.
    random.Random(a.seed).shuffle(cells)

    done = {t.config_hash for t in load_trials(a.family, a.path)}
    print(f"grid: {len(cells)} cells   already in registry: {len(done)}")

    t0 = time.time()
    data = SRPData.load()
    print(f"data loaded in {time.time() - t0:.1f}s "
          f"({len(data.cols)} symbols, {len(data.cw)} weeks, "
          f"ticket {'present' if data.ticket else 'ABSENT'})\n")

    n_new = n_skip = n_fail = 0
    best = (float("-inf"), None)
    for i, c in enumerate(cells, 1):
        cfg = SRPConfig(
            rank_window=c["rank_window"],
            rank_min_periods=max(4, c["rank_window"] // 2),
            vol_window=c["vol_window"],
            vol_min_periods=max(4, c["vol_window"] // 2),
            funding_tilt=c["funding_tilt"],
            turnover_cap=c["turnover_cap"],
            top=c["top"],
        )
        cd = config_dict(cfg, c["smooth"], c["factors"], True,
                         c["ranking"], c["combine"], c["weighting"])
        h = config_hash(cd)
        if h in done:
            n_skip += 1
            continue

        t = time.time()
        try:
            res = run(data, cfg, smooth=c["smooth"], factors=c["factors"],
                      costs=True, ranking=c["ranking"], combine=c["combine"],
                      weighting=c["weighting"])
            sw, nobs = res.sharpe_weekly, res.n_obs
        except Exception as exc:  # a dead branch is still a branch -- log it
            sw, nobs = float("nan"), 0
            n_fail += 1
            print(f"  [{i}/{len(cells)}] {h} RAISED {type(exc).__name__}: {exc}")

        log_trial(a.family, cd, sharpe_weekly=sw, n_obs=nobs, path=a.path,
                  note=f"{c['ranking']}/{c['combine']}/{c['weighting']}/{c['factor_set']}")
        done.add(h)
        n_new += 1
        if math.isfinite(sw) and sw > best[0]:
            best = (sw, cd)
        tag = f"{c['ranking'][:4]}/{c['combine'][:5]}/{c['weighting'][:4]}/{c['factor_set']}"
        srt = f"{sw * math.sqrt(52):6.3f}" if math.isfinite(sw) else "   n/a"
        print(f"  [{i}/{len(cells)}] {h} {tag:<28} ann {srt}  n={nobs:<4} "
              f"({time.time() - t:.1f}s)")
        if a.limit and n_new >= a.limit:
            print(f"\n  stopping at --limit {a.limit}")
            break

    print(f"\n=== swept {n_new} new, skipped {n_skip} already logged, "
          f"{n_fail} raised ===")
    if best[1]:
        print(f"  best this run: ann {best[0] * math.sqrt(52):.3f}  {best[1]}")
    print(f"  registry: {a.path}")
    print("  next:  uv run python -m scripts.srp_dsr")


if __name__ == "__main__":
    main()
