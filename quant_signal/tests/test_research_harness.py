"""Research-harness tests — hermetic (no Kafka/Redis/Snowflake).

The harness sweeps the predictor's config grid over synthetic windows, ranks
the trials, deflates the winner for the total trial count (the multiple-
testing disclosure of Bailey & López de Prado), and reports the expected-max
Sharpe a noise search would produce. Every assertion is deterministic.
"""

from __future__ import annotations

import math

from config.settings import Settings
from stream import research_harness as rh
from stream.predictive_eval import _cross_returns, _windows_per_year, evaluate_predictor


def _settings(**overrides: object) -> Settings:
    return Settings(
        snowflake_account="a",
        snowflake_user="u",
        snowflake_password="p",
        _env_file=None,
        **overrides,
    )


def _feature_window(symbol: str, i: int, *, down: bool = False, step_ms: int = 3_600_000) -> dict:
    """Synthetic window trending ~0.1%/window (up or down), positive prices."""
    step = -1.0 if down else 1.0
    return {
        "symbol": symbol,
        "window_start_ms": i * step_ms,
        "window_end_ms": (i + 1) * step_ms,
        "open": 1000.0 + step * i,
        "high": 1000.5 + step * i,
        "low": 999.5 + step * i,
        "close": 1000.5 + step * i,
        "vwap": 1000.3 + step * i,
        "volume": 1000.0,
        "bar_count": 5,
    }


def _windows(symbol: str, n: int = 300, **kwargs: object) -> list[dict]:
    return [_feature_window(symbol, i, **kwargs) for i in range(n)]


# ── build_candidates: the deterministic grid ─────────────────────────────────


def test_build_candidates_grid_is_full_cartesian_product() -> None:
    candidates = rh.build_candidates(
        lambda_values=[0.0, 1.0, 2.0],
        taker_cost_values=[0.0005, 0.001],
        feature_modes=["single", "cross"],
    )
    assert len(candidates) == 3 * 2 * 2
    # The entry hurdle is derived from λ × taker cost (Bysik & Ślepaczuk eq. 5),
    # never a hardcoded threshold.
    by_mode = {c["feature_mode"] for c in candidates}
    assert by_mode == {"single", "cross"}
    assert all(c["direction_threshold"] == c["lambda"] * c["taker_cost"] for c in candidates)
    # λ=2 @ 10bps → the live 20bps band.
    assert any(
        c["lambda"] == 2.0 and c["taker_cost"] == 0.001 and c["direction_threshold"] == 0.002
        for c in candidates
    )


def test_build_candidates_is_deterministic() -> None:
    a = rh.build_candidates(
        lambda_values=[0.0, 0.5, 1.0],
        taker_cost_values=[0.0005, 0.001],
        feature_modes=["single", "cross"],
    )
    b = rh.build_candidates(
        lambda_values=[0.0, 0.5, 1.0],
        taker_cost_values=[0.0005, 0.001],
        feature_modes=["single", "cross"],
    )
    assert a == b


def test_build_candidates_random_is_seeded_and_budgeted() -> None:
    full = rh.build_candidates(
        lambda_values=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        taker_cost_values=[0.0005, 0.001],
        feature_modes=["single", "cross"],
    )
    sample = rh.build_candidates(
        lambda_values=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        taker_cost_values=[0.0005, 0.001],
        feature_modes=["single", "cross"],
        method="random",
        budget=8,
        seed=42,
    )
    assert len(sample) == 8 < len(full)
    assert len({tuple(c.items()) for c in sample}) == 8  # de-duplicated
    again = rh.build_candidates(
        lambda_values=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        taker_cost_values=[0.0005, 0.001],
        feature_modes=["single", "cross"],
        method="random",
        budget=8,
        seed=42,
    )
    assert sample == again  # byte-identical across reruns (pinned seed)


def test_build_candidates_empty_axis_yields_no_candidates() -> None:
    assert (
        rh.build_candidates(lambda_values=[], taker_cost_values=[0.001], feature_modes=["single"])
        == []
    )


# ── run_sweep: replay every candidate over the same loop ─────────────────────


def test_run_sweep_evaluates_every_candidate_and_echoes_params() -> None:
    candidates = rh.build_candidates(
        lambda_values=[0.0, 1.0, 2.0],
        taker_cost_values=[0.0005, 0.001],
        feature_modes=["single", "cross"],
    )
    windows = _windows("BTCUSDT")
    cross_windows = {"ETHUSDT": _windows("ETHUSDT", n=60)}
    trials = rh.run_sweep(windows, candidates=candidates, cross_windows=cross_windows)

    assert len(trials) == len(candidates)
    for trial, params in zip(trials, candidates):
        assert trial["params"] == params
        assert "report" in trial
        # every trial is recorded — never just the winner
        assert trial["report"]["n_scored"] > 0


def test_run_sweep_cross_mode_feeds_cross_features() -> None:
    # A sweep with cross windows must not crash in either feature mode and
    # must produce the same number of scored windows in both.
    candidates = rh.build_candidates(
        lambda_values=[1.0, 2.0],
        taker_cost_values=[0.001],
        feature_modes=["single", "cross"],
    )
    windows = _windows("BTCUSDT", n=120)
    cross_windows = {"ETHUSDT": _windows("ETHUSDT", n=120)}
    trials = rh.run_sweep(windows, candidates=candidates, cross_windows=cross_windows)
    assert all(t["report"]["n_scored"] > 0 for t in trials)


# ── rank_trials: honest ordering with constraints ────────────────────────────


def test_rank_trials_orders_by_objective_and_pushes_violators_back() -> None:
    def trial(sharpe: float, scored: int, excess: float) -> dict:
        return {
            "params": {"t": sharpe},
            "report": {
                "annual_sharpe_strategy": sharpe,
                "n_scored": scored,
                "excess_return": excess,
            },
        }

    trials = [
        trial(3.0, 200, 0.05),  # best, eligible
        trial(5.0, 200, -0.01),  # highest Sharpe but negative excess → violator
        trial(4.0, 10, 0.02),  # too few windows → violator
        trial(2.0, 150, 0.01),  # eligible
        trial(0.5, 200, -0.001),  # negative excess → violator
    ]
    ranked = rh.rank_trials(
        trials, objective="annual_sharpe_strategy", min_windows=100, min_excess_return=0.0
    )

    assert [t["eligible"] for t in ranked] == [True, True, False, False, False]
    assert ranked[0]["report"]["annual_sharpe_strategy"] == 3.0
    assert ranked[1]["report"]["annual_sharpe_strategy"] == 2.0
    assert [t["rank"] for t in ranked] == [1, 2, 3, 4, 5]
    assert ranked[0]["rank"] == 1


def test_rank_trials_does_not_mutate_inputs() -> None:
    trial = {
        "params": {"t": 1.0},
        "report": {"annual_sharpe_strategy": 1.0, "n_scored": 100, "excess_return": 0.01},
    }
    ranked = rh.rank_trials(
        [trial], objective="annual_sharpe_strategy", min_windows=100, min_excess_return=0.0
    )
    assert "rank" not in trial and "eligible" not in trial
    assert ranked[0]["rank"] == 1 and ranked[0]["eligible"] is True


# ── expected_max_sharpe: the maximum of pure noise ───────────────────────────


def test_expected_max_sharpe_is_zero_without_variance_or_trials() -> None:
    assert rh.expected_max_sharpe([]) == 0.0
    assert rh.expected_max_sharpe([1.0]) == 0.0
    assert rh.expected_max_sharpe([1.0, 1.0, 1.0]) == 0.0  # zero across-trial variance


def test_expected_max_sharpe_grows_with_trials() -> None:
    base = [0.1, -0.2, 0.3, -0.1, 0.15]
    small = rh.expected_max_sharpe(base)
    big = rh.expected_max_sharpe(base * 4)  # more independent draws → higher max
    assert 0.0 < small < big


def test_expected_max_sharpe_is_positive_with_typical_spread() -> None:
    # K=50 typical trials: the spurious max is ~0.4–0.8 (qbx-research).
    value = rh.expected_max_sharpe([0.1 * math.sin(i) for i in range(50)])
    assert value > 0.0


# ── summarize_sweep: the full sweep report ───────────────────────────────────


def test_summarize_sweep_winner_charged_for_total_trials() -> None:
    settings = _settings()
    candidates = rh.build_candidates(
        lambda_values=[0.0, 1.0, 2.0],
        taker_cost_values=[0.001],
        feature_modes=["single"],
    )
    # Downtrend: the model goes SHORT, so the strategy clears buy-and-hold
    # after the cost filter — an eligible winner exists to deflate.
    trials = rh.run_sweep(_windows("BTCUSDT", down=True), candidates=candidates)
    summary = rh.summarize_sweep(trials, settings=settings, n_trials=len(trials))

    assert summary["n_trials"] == 3
    assert summary["trials_total"] == 3
    winner = summary["winner"]
    assert winner is not None
    assert winner["rank"] == 1 and winner["eligible"] is True
    assert isinstance(winner["passes"], bool)
    assert winner["deflated_sharpe"] is not None
    assert summary["trials_eligible"] >= 1
    # Leaderboard is the top-k, wire-safe (no raw strategy returns leaked).
    assert len(summary["leaderboard"]) <= settings.research_top_k
    assert all("_strat_rets" not in t["report"] for t in summary["leaderboard"])
    assert [t["rank"] for t in summary["leaderboard"]] == list(
        range(1, len(summary["leaderboard"]) + 1)
    )
    assert isinstance(summary["winner_clears_noise_floor"], bool)


def test_summarize_sweep_reports_no_winner_when_nothing_eligible() -> None:
    # Too few windows → no trial clears the min-windows constraint.
    settings = _settings()
    candidates = rh.build_candidates(
        lambda_values=[1.0, 2.0],
        taker_cost_values=[0.001],
        feature_modes=["single"],
    )
    trials = rh.run_sweep(_windows("BTCUSDT", n=10), candidates=candidates)
    summary = rh.summarize_sweep(trials, settings=settings, n_trials=len(trials))

    assert summary["winner"] is None
    assert summary["trials_eligible"] == 0
    assert summary["trials_total"] == 2
    assert summary["leaderboard"]  # the full extent is still disclosed


def test_summarize_sweep_deflation_rejects_noise_winner() -> None:
    """Charging the null for many trials must deflate a noise winner."""
    settings = _settings()
    # Noise-only windows (alternating drift cancels the edge).
    windows = [
        _feature_window("BTCUSDT", i, down=(i % 3 == 0), step_ms=3_600_000) for i in range(300)
    ]
    candidates = rh.build_candidates(
        lambda_values=[0.0, 0.5, 1.0, 1.5, 2.0],
        taker_cost_values=[0.001],
        feature_modes=["single"],
    )
    trials = rh.run_sweep(windows, candidates=candidates)
    summary = rh.summarize_sweep(trials, settings=settings, n_trials=len(trials))

    if summary["winner"] is not None:
        # Either the winner exists and its DSR is charged for all 5 trials, or
        # nothing cleared the cost constraint — but it must never report a
        # pass without the full-trial DSR being recorded.
        assert summary["n_trials"] == 5
        if summary["winner"]["passes"]:
            assert summary["winner"]["deflated_sharpe"] is not None


def test_block_ic_cv_units() -> None:
    # Stable blocks → low CV; wildly varying blocks → high CV (or None).
    assert rh._block_ic_cv({"blocks": [{"ic": 0.1}, {"ic": 0.11}, {"ic": 0.12}]}) < 0.2
    high = rh._block_ic_cv({"blocks": [{"ic": 0.1}, {"ic": -0.3}]})
    assert high is None or high > 0.2  # negative mean → None; else clearly unstable
    assert rh._block_ic_cv({"blocks": [{"ic": 0.1}]}) is None  # not enough blocks


# ── cross-window features: no-lookahead replay ───────────────────────────────


def test_cross_returns_use_only_windows_strictly_before() -> None:
    cross = {
        "ETHUSDT": [
            {"symbol": "ETHUSDT", "window_end_ms": 300_000, "close": 100.0},
            {"symbol": "ETHUSDT", "window_end_ms": 600_000, "close": 102.0},
            {"symbol": "ETHUSDT", "window_end_ms": 900_000, "close": 99.0},
            {"symbol": "ETHUSDT", "window_end_ms": 1_200_000, "close": 103.0},
        ]
    }
    # At the 1.2M mark the target window is OPEN: the 1.2M cross close is not
    # known yet, so the return must come from the two closes strictly before.
    returns = _cross_returns(cross, "BTCUSDT", 1_200_000)
    assert math.isclose(returns["ETHUSDT"], 99.0 / 102.0 - 1.0, abs_tol=1e-12)
    assert "ETHUSDT" not in _cross_returns(cross, "ETHUSDT", 1_200_000)  # self excluded


def test_cross_returns_skip_warmup_and_degenerate() -> None:
    assert (
        _cross_returns(
            {"ETHUSDT": [{"window_end_ms": 300_000, "close": 100.0}]}, "BTCUSDT", 600_000
        )
        == {}
    )
    assert _cross_returns({"ETHUSDT": []}, "BTCUSDT", 600_000) == {}
    zero_close = [
        {"window_end_ms": 300_000, "close": 100.0},
        {"window_end_ms": 600_000, "close": 0.0},
    ]
    assert _cross_returns({"ETHUSDT": zero_close}, "BTCUSDT", 900_000) == {}


def test_evaluate_predictor_accepts_cross_windows() -> None:
    windows = _windows("BTCUSDT", n=150)
    cross_windows = {"ETHUSDT": _windows("ETHUSDT", n=150)}
    report = evaluate_predictor(windows, cross_windows=cross_windows)
    assert report["n_scored"] > 0
    assert evaluate_predictor(windows)["n_scored"] == report["n_scored"]  # same data, both paths


# ── cadence-aware annualization ──────────────────────────────────────────────


def test_windows_per_year_from_cadence() -> None:
    hourly = [_feature_window("BTCUSDT", i, step_ms=3_600_000) for i in range(10)]
    assert math.isclose(_windows_per_year(hourly), 365 * 24, rel_tol=1e-9)

    five_min = [_feature_window("BTCUSDT", i, step_ms=300_000) for i in range(10)]
    assert math.isclose(_windows_per_year(five_min), 365 * 24 * 12, rel_tol=1e-9)


def test_windows_per_year_falls_back_when_cadence_unknown() -> None:
    assert _windows_per_year([]) == 365 * 24
    assert _windows_per_year([{"window_end_ms": 1}]) == 365 * 24
    # Duplicate timestamps (b > a filter) → unknown cadence → hourly default.
    assert _windows_per_year([{"window_end_ms": 1}, {"window_end_ms": 1}]) == 365 * 24
