"""Promotion-gate tests — hermetic (no Kafka/Redis/Snowflake).

The gate is the predicate the live predictor must clear before it may trade:
progressive validation over synthetic feature windows, cost-adjusted P&L,
conformal coverage, block stability, and a Deflated Sharpe charged for the
number of trials. Every assertion is deterministic.
"""

from __future__ import annotations

import math

from config.settings import Settings
from stream import predictive_eval
from stream.predictive_eval import evaluate_predictor, passes_gate


def _settings(**overrides: object) -> Settings:
    return Settings(
        snowflake_account="a",
        snowflake_user="u",
        snowflake_password="p",
        _env_file=None,
        **overrides,
    )


def _feature_window(symbol: str, i: int, *, down: bool = False) -> dict:
    """Synthetic 5m window trending ~0.1%/window (up or down), positive prices."""
    step = -1.0 if down else 1.0
    return {
        "symbol": symbol,
        "window_start_ms": i * 300_000,
        "window_end_ms": (i + 1) * 300_000,
        "open": 1000.0 + step * i,
        "high": 1000.5 + step * i,
        "low": 999.5 + step * i,
        "close": 1000.5 + step * i,
        "vwap": 1000.3 + step * i,
        "volume": 1000.0,
        "bar_count": 5,
    }


def _closes(windows: list[dict]) -> list[float]:
    return [float(w["close"]) for w in windows]


def _realized(closes: list[float]) -> list[float]:
    return [c1 / c0 - 1.0 for c0, c1 in zip(closes, closes[1:])]


# ── evaluate_predictor: the progressive-validation replay ───────────────────


def test_evaluate_predictor_scores_an_uptrend() -> None:
    windows = [_feature_window("BTCUSDT", i) for i in range(300)]
    report = evaluate_predictor(windows)

    assert report["n_scored"] == len(windows) - 1  # first window only predicts
    assert report["mae"] > 0.0
    assert report["skill_vs_zero"] is not None and report["skill_vs_zero"] > 0.5
    assert report["direction_accuracy"] > 0.85
    assert isinstance(report["ic"], float)
    assert report["coverage"] is not None and 0.0 <= report["coverage"] <= 1.0
    assert report["total_return_buyhold"] > 0.0
    assert report["annual_sharpe_strategy"] is not None
    assert report["_strat_rets"]  # raw per-window strategy returns for DSR
    assert len(report["blocks"]) >= 1
    assert report["passes"] is None  # not gated until passes_gate() runs


def test_evaluate_predictor_scores_a_downtrend_and_beats_buyhold() -> None:
    """On a clean downtrend the model goes SHORT and clears buy-and-hold."""
    windows = [_feature_window("ETHUSDT", i, down=True) for i in range(300)]
    report = evaluate_predictor(windows)

    assert report["direction_accuracy"] > 0.85
    assert report["total_return_buyhold"] < 0.0
    assert report["total_return_strategy"] > 0.0
    assert report["excess_return"] > 0.0


def test_taker_cost_charged_per_flip_not_per_window(monkeypatch) -> None:
    """A position that never flips pays exactly one taker cost, not one per window."""
    monkeypatch.setattr(predictive_eval, "_direction", lambda y_hat, threshold: "LONG")
    windows = [_feature_window("BTCUSDT", i) for i in range(40)]
    taker = 0.001
    report = evaluate_predictor(windows, taker_cost=taker)

    realized = _realized(_closes(windows))
    assert len(report["_strat_rets"]) == len(realized)
    # Always-LONG from FLAT = one open (flip), then holding: sum = Σr - taker.
    assert math.isclose(sum(report["_strat_rets"]), sum(realized) - taker, abs_tol=1e-12)


def test_evaluate_predictor_needs_small_minimum_and_skips_malformed() -> None:
    report = evaluate_predictor([_feature_window("BTCUSDT", 0)])
    assert report == {"n_windows": 1, "n_scored": 0}
    assert evaluate_predictor([{"symbol": "BTCUSDT"}]) == {"n_windows": 0}


def test_evaluate_predictor_feature_modes_all_score_the_same_windows() -> None:
    """Every feature-mode ablation scores the same windows (features never
    reject a window) and produces a well-formed report — the honest ablation
    chain for the harness."""
    windows = [_feature_window("BTCUSDT", i) for i in range(150)]
    cross = {"ETHUSDT": [_feature_window("ETHUSDT", i) for i in range(150)]}
    n_scored = set()
    for mode in ("single", "history", "vol", "cross"):
        report = evaluate_predictor(windows, cross_windows=cross, feature_mode=mode)
        n_scored.add(report["n_scored"])
        assert report["n_scored"] > 0
        assert isinstance(report["skill_vs_zero"], float)
        assert report["_strat_rets"]
    assert len(n_scored) == 1  # identical coverage across all variants


# ── passes_gate: the promotion predicate ────────────────────────────────────


def _passing_report() -> dict:
    # Modest positive edge with real variance so the DSR is well-conditioned.
    strat_rets = [0.001 + 0.01 * math.sin(i) for i in range(400)]
    return {
        "n_windows": 400,
        "n_scored": 400,
        "mae": 0.001,
        "skill_vs_zero": 0.6,
        "skill_vs_persistence": 0.5,
        "ic": 0.15,
        "direction_accuracy": 0.58,
        "coverage": 0.89,
        "nominal_alpha": 0.1,
        "excess_return": 0.02,
        "_strat_rets": strat_rets,
    }


def test_gate_passes_a_skillful_report_and_records_verdict() -> None:
    report = _passing_report()
    passed, failures = passes_gate(report, _settings(), n_trials=1)

    assert passed is True, failures
    assert report["passes"] is True  # verdict recorded on the report
    assert report["deflated_sharpe"] is not None
    assert report["deflated_sharpe"] > _settings().stream_gate_min_dsr


def test_gate_rejects_each_failed_condition() -> None:
    report = {
        "n_scored": 50,  # warm-up
        "skill_vs_zero": -0.1,
        "skill_vs_persistence": -0.2,
        "ic": -0.05,
        "direction_accuracy": 0.40,
        "coverage": 0.50,
        "nominal_alpha": 0.1,
        "excess_return": -0.01,
        "_strat_rets": [0.001 + 0.01 * math.sin(i) for i in range(400)],
    }
    passed, failures = passes_gate(report, _settings(), n_trials=1)

    assert passed is False
    assert report["passes"] is False
    text = "\n".join(failures)
    assert "n_scored 50 < min 100" in text
    assert "no skill vs zero" in text
    assert "no skill vs persistence" in text
    assert "ic" in text and "<= min" in text
    assert "direction_accuracy" in text
    assert "coverage" in text
    assert "excess_return" in text


def test_gate_rejects_noise_with_large_trial_count() -> None:
    """Charging the null for many trials must deflate a lucky (zero-edge) Sharpe."""
    report = _passing_report()
    report["_strat_rets"] = [0.01 * math.sin(i) for i in range(400)]  # zero mean
    passed, failures = passes_gate(report, _settings(), n_trials=10_000)

    assert passed is False
    assert any("deflated_sharpe" in f for f in failures)


def test_gate_fails_warmup_without_enough_scored_windows() -> None:
    report = _passing_report()
    report["n_scored"] = 50
    passed, failures = passes_gate(report, _settings(), n_trials=1)

    assert passed is False
    assert any("warm-up" in f for f in failures)


def test_gate_requires_strategy_returns_for_dsr() -> None:
    """A report without raw strategy returns cannot be promoted (DSR unverifiable)."""
    report = _passing_report()
    del report["_strat_rets"]
    passed, failures = passes_gate(report, _settings(), n_trials=1)

    assert passed is False
    assert report["deflated_sharpe"] is None
    assert any("deflated_sharpe not computable" in f for f in failures)


def test_gate_honors_custom_thresholds() -> None:
    report = _passing_report()
    report["skill_vs_zero"] = 0.01
    # A permissive skill floor lets a thin-edge model through.
    passed, failures = passes_gate(report, _settings(stream_gate_min_skill=-1.0), n_trials=1)
    assert passed is True, failures


# ── helpers ─────────────────────────────────────────────────────────────────


def test_mase_skill_units() -> None:
    assert predictive_eval._mase_skill(1.0, 2.0) == 0.5  # half the baseline error
    assert predictive_eval._mase_skill(2.0, 1.0) == -1.0  # twice the baseline
    assert predictive_eval._mase_skill(1.0, 0.0) is None  # degenerate baseline


def test_dsr_helper_edge_cases() -> None:
    assert predictive_eval._dsr([], 10) is None
    assert predictive_eval._dsr([0.01, 0.02], 10) is None  # too few periods
    assert predictive_eval._dsr([0.01] * 100, 1) is None  # zero variance → ill-posed
