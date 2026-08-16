"""Held-out UNIVERSE test -- out-of-sample in the cross-section, not just in time.

Every other validation in this project is out-of-sample in TIME: walk-forward
picks a configuration on trailing weeks and scores it on later weeks. That does
not answer a different question a referee will ask -- were the settings tuned to
these particular coins?

NOTHING IS FITTED HERE
----------------------
This is not machine learning. No weights are learned, no gradients are taken,
nothing is regressed. The factors are defined a priori from published sources
and their signs are fixed in advance; the ranking is a rank transform. "Training"
therefore means exactly one thing: SELECTING HYPERPARAMETERS (rank window,
funding tilt, turnover cap, quintile width, smoothing). That selection is the
only channel through which the data can influence the strategy, so it is the only
thing that can be overfit -- and it is what this test holds out.

    train symbols  -> sweep the candidate grid, pick the best configuration
    test symbols   -> score THAT configuration on coins never used to choose it

The split is over SYMBOLS, and the cross-section is rebuilt inside each side:
ranks, quintile boundaries, the funding tilt and the liquidity-scaled cost model
are all cross-sectional quantities, so a 32-symbol book must re-rank those 32
against each other. Masking a 112-symbol book would leak the excluded coins into
every rank.

Several random splits are run, because a single split can be lucky. The number
that matters is the DISTRIBUTION of held-out Sharpes, not the best one.

Run:
    uv run python -m scripts.srp_holdout
    uv run python -m scripts.srp_holdout --splits 8 --train-frac 0.7
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import statistics as st

from scripts.srp_backtest import SRPConfig, SRPData, run
from scripts.trial_registry import log_trial

# Same candidate set the walk-forward uses, so the two tests are comparable.
CANDIDATES: list[dict] = [
    {"rank_window": rw, "funding_tilt": ft, "turnover_cap": tc, "top": tp}
    for rw, ft, tc, tp in itertools.product(
        [26, 52, 104], [0.0, 0.5], [None, 0.6], [0.20]
    )
]


def _cfg(d: dict) -> SRPConfig:
    rw = d["rank_window"]
    return SRPConfig(
        rank_window=rw,
        rank_min_periods=max(4, rw // 2),
        funding_tilt=d["funding_tilt"],
        turnover_cap=d["turnover_cap"],
        top=d["top"],
    )


def _label(d: dict) -> str:
    return (f"rw={d['rank_window']} ft={d['funding_tilt']} "
            f"cap={d['turnover_cap']} top={d['top']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--log-trials", action="store_true")
    a = ap.parse_args()

    full = SRPData.load()
    syms = list(full.cols)
    n_tr = int(round(a.train_frac * len(syms)))
    print(f"universe {len(syms)} symbols  ->  train {n_tr} / test {len(syms) - n_tr}")
    print(f"{len(CANDIDATES)} candidate configurations, {a.splits} random splits\n")

    rows = []
    for i in range(a.splits):
        rng = random.Random(a.seed + i)
        shuffled = syms[:]
        rng.shuffle(shuffled)
        tr_syms, te_syms = shuffled[:n_tr], shuffled[n_tr:]
        d_tr, d_te = full.subset(tr_syms), full.subset(te_syms)

        best, best_sr = None, float("-inf")
        for c in CANDIDATES:
            r = run(d_tr, _cfg(c))
            if math.isfinite(r.sharpe_ann) and r.sharpe_ann > best_sr:
                best, best_sr = c, r.sharpe_ann
        if best is None:
            print(f"  split {i + 1}: no candidate produced returns")
            continue

        # THE held-out number: the train-selected config, scored on unseen coins.
        r_te = run(d_te, _cfg(best))
        t_te = (r_te.sharpe_weekly * math.sqrt(r_te.n_obs)
                if r_te.n_obs > 2 else float("nan"))

        # MATCHED-BREADTH CONTROL. The held-out book holds far fewer symbols
        # than the training book, and Sharpe scales with breadth (IR ~ IC x
        # sqrt(N)) regardless of any overfitting. Comparing a 78-symbol training
        # result against a 34-symbol held-out result therefore differs on TWO
        # axes at once and attributes a mechanical effect to overfitting.
        #
        # So the same train-selected config is also scored on a random subset of
        # the TRAINING symbols of exactly the held-out size. Those coins were
        # seen during selection, so the only difference from the held-out book is
        # whether the config was chosen on them:
        #
        #   held-out vs control  ->  overfitting, at matched breadth
        #   train    vs control  ->  the breadth effect alone
        rng_c = random.Random(500 + i)
        ctrl_syms = tr_syms[:]
        rng_c.shuffle(ctrl_syms)
        r_ct = run(full.subset(ctrl_syms[:len(te_syms)]), _cfg(best))

        rows.append((best_sr, r_te.sharpe_ann, t_te, r_te.n_obs, best,
                     r_ct.sharpe_ann))
        print(f"  split {i + 1}:  train78 {best_sr:6.3f}   "
              f"HELD-OUT34 {r_te.sharpe_ann:6.3f} (t {t_te:5.2f})   "
              f"seen34 {r_ct.sharpe_ann:6.3f}   [{_label(best)}]")
        if a.log_trials:
            log_trial("srp_holdout", {**best, "split": i, "scope": "test"},
                      sharpe_weekly=r_te.sharpe_weekly, n_obs=r_te.n_obs,
                      note=f"held-out universe split {i}")

    if not rows:
        raise SystemExit("no split produced a result")

    tr = [x[0] for x in rows]
    te = [x[1] for x in rows]
    ts = [x[2] for x in rows if math.isfinite(x[2])]
    ct = [x[5] for x in rows]
    print(f"\n{'':16}{'train78':>9} {'held-out34':>11} {'seen34':>9}")
    print(f"  {'mean':<14}{st.mean(tr):>9.3f} {st.mean(te):>11.3f} {st.mean(ct):>9.3f}")
    print(f"  {'median':<14}{st.median(tr):>9.3f} {st.median(te):>11.3f} {st.median(ct):>9.3f}")
    print(f"  {'min':<14}{min(tr):>9.3f} {min(te):>11.3f} {min(ct):>9.3f}")
    print(f"  {'max':<14}{max(tr):>9.3f} {max(te):>11.3f} {max(ct):>9.3f}")
    if len(te) > 1:
        print(f"  {'sd':<14}{st.stdev(tr):>9.3f} {st.stdev(te):>11.3f} {st.stdev(ct):>9.3f}")

    print(f"\n  held-out splits profitable : {sum(1 for x in te if x > 0)}/{len(te)}")
    if ts:
        print(f"  held-out t > 3.0           : {sum(1 for x in ts if x > 3.0)}/{len(ts)}")

    # The naive ratio conflates two effects and OVERSTATES overfitting.
    print("\n  decomposition of the drop:")
    print(f"    naive retention (held-out / train78) : "
          f"{100.0 * st.mean(te) / st.mean(tr):.0f}%   <- conflates breadth + overfitting")
    print(f"    breadth alone   (seen34   / train78) : "
          f"{100.0 * st.mean(ct) / st.mean(tr):.0f}%   <- mechanical, IR ~ IC x sqrt(N)")
    print(f"    OVERFITTING     (held-out / seen34)  : "
          f"{100.0 * st.mean(te) / st.mean(ct):.0f}%   <- the number that matters")
    print("\n  Report the matched-breadth figure. The naive ratio charges the "
          "strategy\n  for holding fewer symbols, which is not a research defect.")


if __name__ == "__main__":
    main()
