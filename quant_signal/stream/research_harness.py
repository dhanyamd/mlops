"""Autonomous research harness: offline parameter sweep over the predictor.

Maps to the "Autonomous AI Researcher" pillar of the live-trading firm's JD:
take a research seed (the predictor + its config grid from Settings), run the
exact progressive-validation replay (``evaluate_predictor``) over every
configuration, rank the trials, and deflate the winner for the whole search —
the multiple-testing correction the literature demands:

  - Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting, and Non-Normality", JPM 40(5)
    94–107, 2014: a Sharpe selected as the best of N trials is the maximum
    of N draws; the maximum of pure noise is large (qbx-research: K=50
    typical trials → spurious max Sharpe ~0.4–0.8).
  - The winner is judged by ``passes_gate(..., n_trials=total_trials)`` —
    the SAME gate the live predictor must clear, now charged honestly for the
    search size (Harvey, Liu & Zhu 2016: "most claimed research findings in
    financial economics are likely false" unless trials are disclosed).
  - Plateau > peak (AlgoXpert, arXiv:2603.09219): an isolated optimum is a
    "cliff"; a winner whose edge is spread across time is robust. Reported
    as the top-K leaderboard + the winner's block-IC coefficient of
    variation (CV > 0.2 → unstable, reject — offline-pixel WFV guide).
  - The sweep axis is λ — the cost-aware filter multiple of Bysik &
    Ślepaczuk (arXiv:2606.00060, eq. 5); the entry hurdle is derived as
    direction_threshold = λ × round-trip taker cost, never hardcoded.

No LLM, no I/O: pure functions over the same window dicts the live predictor
consumes, so tests are hermetic (FakeKV/synthetic windows).
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from scipy import stats

from config.settings import Settings
from stream.predictive_eval import evaluate_predictor, passes_gate

_ELOG = 2.718281828459045


def build_candidates(
    *,
    lambda_values: Sequence[float],
    taker_cost_values: Sequence[float],
    feature_modes: Sequence[str],
    method: str = "grid",
    budget: int = 0,
    seed: int = 42,
) -> list[dict]:
    """Deterministic list of parameter sets to evaluate.

    Grid = full Cartesian product (exhaustive for small spaces). Random =
    seeded draws over the same space, de-duplicated by construction and
    capped at ``budget`` — the coverage-per-sample argument for sampling large
    spaces (qbx-research); the seed is pinned so a sweep is byte-identical
    across reruns (reproducibility is a first-class property of research
    infrastructure — Hasan Javed backtest-harness case study).

    Each candidate carries ``lambda`` and the derived entry hurdle
    ``direction_threshold = λ × taker_cost`` (Bysik & Ślepaczuk eq. 5: trade
    only when |r̂| > λ·c), so the sweep axis is interpretable as the cost-
    awareness multiple, not a raw threshold.
    """
    axes: dict[str, list[Any]] = {
        "lambda": [float(x) for x in lambda_values],
        "taker_cost": [float(x) for x in taker_cost_values],
        "feature_mode": list(feature_modes),
    }
    combos = list(itertools.product(axes["lambda"], axes["taker_cost"], axes["feature_mode"]))
    if method == "random":
        rng = random.Random(seed)
        if 0 < budget < len(combos):
            combos = rng.sample(combos, budget)
    return [
        {
            "lambda": lam,
            "taker_cost": cost,
            "feature_mode": mode,
            "direction_threshold": lam * cost,
        }
        for lam, cost, mode in combos
    ]


def run_sweep(
    windows: Sequence[Mapping],
    *,
    candidates: Sequence[Mapping],
    cross_windows: Mapping[str, Sequence[Mapping]] | None = None,
    alpha: float = 0.1,
    gamma: float = 0.005,
    residual_window: int = 200,
    n_blocks: int = 4,
) -> list[dict]:
    """Evaluate every candidate over the SAME replay loop the live gate uses.

    One replay per candidate keeps the multiple-testing charge meaningful:
    the winner is the best of ``len(candidates)`` trials on the same data, and
    every trial is recorded so the search's full extent is disclosed — never
    just the winner (qbx-research / Bailey & López de Prado). Feature mode
    "cross" feeds the lagged cross-coin returns; "single" uses own-symbol
    features only.
    """
    trials: list[dict] = []
    for params in candidates:
        use_cross = params.get("feature_mode") == "cross" and cross_windows is not None
        report = evaluate_predictor(
            windows,
            alpha=alpha,
            gamma=gamma,
            residual_window=residual_window,
            taker_cost=float(params["taker_cost"]),
            n_blocks=n_blocks,
            direction_threshold=float(params["direction_threshold"]),
            cross_windows=cross_windows if use_cross else None,
        )
        trials.append({"params": dict(params), "report": report})
    return trials


def rank_trials(
    trials: Sequence[dict],
    *,
    objective: str,
    min_windows: int,
    min_excess_return: float,
) -> list[dict]:
    """Order trials by the objective; constraint violators sort to the back.

    Hard constraints (enough scored windows; excess return clearing the
    trial's own cost assumption; a computable objective) are enforced before
    ranking — the qbx-research pattern of pushing constraint violators behind
    eligible trials so the leaderboard is honest. Returns shallow copies with
    ``rank`` and ``eligible`` attached; inputs are never mutated.
    """
    buckets: list[tuple[bool, dict]] = []
    for trial in trials:
        report = trial["report"]
        ok = report.get("n_scored", 0) >= min_windows
        ok = ok and (report.get("excess_return") or 0.0) >= min_excess_return
        ok = ok and report.get(objective) is not None
        buckets.append((ok, trial))
    buckets.sort(
        key=lambda b: float(b[1]["report"].get(objective) or -math.inf),
        reverse=True,
    )
    eligible = [t for ok, t in buckets if ok]
    violators = [t for ok, t in buckets if not ok]
    eligible_flags = {id(t): True for t in eligible}
    ordered: list[dict] = []
    for rank, trial in enumerate(eligible + violators, start=1):
        entry = dict(trial)
        entry["rank"] = rank
        entry["eligible"] = id(trial) in eligible_flags
        ordered.append(entry)
    return ordered


def expected_max_sharpe(trial_sharpes: Sequence[float]) -> float:
    """The spurious best Sharpe a skill-less search would produce.

    Under the null that every trial's true Sharpe is zero, the best of N
    independently estimated Sharpes is ~ E[max Z₁..Z_N] · sqrt(V[SR_k]), where
    V[SR_k] is the observed variance across trials (Bailey & López de Prado,
    JPM 40(5), 2014). E[max of N iid standard normals] ≈ (1−γ)·Φ⁻¹(1−1/N) +
    γ·Φ⁻¹(1−1/(N·e)) (Gumbel — the same approximation the gate's DSR uses).
    This is the honest benchmark the winner must beat to claim an edge: the
    "maximum of pure noise".
    """
    n = len(trial_sharpes)
    if n < 2:
        return 0.0
    variance = statistics.pvariance(trial_sharpes)
    if variance <= 0:
        return 0.0
    euler = 0.5772156649015329
    e_max = (1.0 - euler) * stats.norm.ppf(1 - 1.0 / n) + euler * stats.norm.ppf(
        1 - 1.0 / (n * _ELOG)
    )
    # scipy returns np.float64; coerce to a plain float so downstream
    # comparisons produce Python bools, not numpy scalars.
    return float(e_max * math.sqrt(variance))


def _block_ic_cv(report: dict) -> float | None:
    """Coefficient of variation of the per-block ICs.

    A real edge is spread across time, not one lucky block. CV > 0.2 flags an
    unstable winner (offline-pixel WFV guide); None when there aren't enough
    distinct blocks to tell.
    """
    ics = [
        float(b["ic"]) for b in report.get("blocks", []) if math.isfinite(b.get("ic", float("nan")))
    ]
    if len(ics) < 2:
        return None
    mean = statistics.fmean(ics)
    if mean <= 0:
        return None
    return statistics.pstdev(ics) / mean


def _strip(trial: dict) -> dict:
    """Wire-safe trial copy: drop the raw per-window returns (internal to DSR)."""
    report = {k: v for k, v in trial["report"].items() if k != "_strat_rets"}
    return {
        "params": dict(trial["params"]),
        "rank": trial.get("rank"),
        "eligible": trial.get("eligible"),
        "report": report,
    }


def summarize_sweep(trials: Sequence[dict], *, settings: Settings, n_trials: int) -> dict:
    """Full sweep report: winner verdict (DSR charged for ALL trials), expected
    max under noise, stability, and the top-K leaderboard.

    The winner's verdict reuses the live promotion gate with
    ``n_trials = total trials run`` — the multiple-testing disclosure that
    makes the search's inflation visible (Harvey / Bailey & López de Prado).
    All thresholds come from Settings, so nothing is hardcoded here.
    """
    ranked = rank_trials(
        trials,
        objective=settings.research_objective,
        min_windows=settings.stream_gate_min_windows,
        min_excess_return=settings.research_min_excess_return,
    )
    winner = next((t for t in ranked if t.get("eligible")), None)

    if winner is None:
        return {
            "n_trials": n_trials,
            "winner": None,
            "expected_max_sharpe_noise": None,
            "winner_clears_noise_floor": None,
            "trials_eligible": 0,
            "trials_total": len(ranked),
            "leaderboard": [_strip(t) for t in ranked[: settings.research_top_k]],
        }

    passes, failures = passes_gate(winner["report"], settings, n_trials=n_trials)
    sharpes = [
        float(t["report"].get(settings.research_objective))
        for t in ranked
        if t["report"].get(settings.research_objective) is not None
    ]
    emax = expected_max_sharpe(sharpes) if sharpes else 0.0
    winner_sharpe = float(winner["report"].get(settings.research_objective) or -math.inf)
    ic_cv = _block_ic_cv(winner["report"])

    return {
        "n_trials": n_trials,
        "winner": {
            **_strip(winner),
            "passes": passes,
            "failures": failures,
            "deflated_sharpe": winner["report"].get("deflated_sharpe"),
            "block_ic_cv": round(ic_cv, 4) if ic_cv is not None else None,
            "block_ic_cv_ok": ic_cv is not None and ic_cv <= settings.research_max_block_ic_cv,
        },
        "expected_max_sharpe_noise": round(emax, 4) if emax > 0 else None,
        "winner_clears_noise_floor": bool(winner_sharpe > emax),
        "trials_eligible": sum(1 for t in ranked if t.get("eligible")),
        "trials_total": len(ranked),
        "leaderboard": [_strip(t) for t in ranked[: settings.research_top_k]],
    }
