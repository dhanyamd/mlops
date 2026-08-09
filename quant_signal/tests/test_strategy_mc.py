"""Strategy-validation Monte Carlo tests — hermetic (no Kafka/Redis/Snowflake).

QuantPad-style: bootstrap the realized per-period returns into 10k simulated
futures and score them against pass/fail rules. By default the rules are
risk-scaled to the strategy's own realized terminal volatility (a fixed
6%/8% contract is structurally unreachable for a 5m signal and would pin the
gauge at 0/100/0); explicit contracts are still honored. Also covers the
analytic two-barrier benchmark and the pass-rate confidence interval.
"""

from __future__ import annotations

import pytest

from stream.strategy_mc import (
    StrategyMonteCarlo,
    effective_rules,
    strategy_returns_from_equity,
    two_barrier_pass_probability,
)

# A steadily profitable strategy (deterministic, small positive drift).
_PROFITABLE = [1.0 + 0.001 * i + 0.0005 * (i % 2) for i in range(80)]

# A losing strategy (steady decay).
_LOSING = [1.0 - 0.002 * i for i in range(80)]

# A sawtooth with big drawdowns (bad under the DD rule).
_SINUOUS = [1.0 + 0.02 * (1 if i % 2 else -1) * (1 + i / 40) for i in range(80)]


def test_returns_derived_from_equity() -> None:
    import numpy as np

    returns = strategy_returns_from_equity([1.0, 1.05, 1.10])
    np.testing.assert_allclose(returns, [0.05, 1.10 / 1.05 - 1.0])
    assert strategy_returns_from_equity([1.0]).size == 0


def test_validation_is_deterministic_with_seed() -> None:
    a = StrategyMonteCarlo(n_sims=10_000, seed=42).validate(_PROFITABLE)
    b = StrategyMonteCarlo(n_sims=10_000, seed=42).validate(_PROFITABLE)
    assert a is not None and b is not None
    assert a["pass_probability"] == b["pass_probability"]
    assert a["equity_fan"] == b["equity_fan"]
    assert a["terminal_histogram"] == b["terminal_histogram"]


def test_profitable_strategy_passes_more_than_losing() -> None:
    good = StrategyMonteCarlo(n_sims=10_000, seed=1).validate(_PROFITABLE)
    bad = StrategyMonteCarlo(n_sims=10_000, seed=1).validate(_LOSING)
    assert good is not None and bad is not None
    assert good["pass_probability"] > 0.9
    assert bad["pass_probability"] < 0.1
    assert good["expected_terminal"] > 1.0
    assert bad["expected_terminal"] < 1.0


def test_drawdown_rule_bites_on_sinuous_strategy() -> None:
    strict = StrategyMonteCarlo(n_sims=10_000, seed=1, max_drawdown=0.03).validate(_SINUOUS)
    loose = StrategyMonteCarlo(n_sims=10_000, seed=1, max_drawdown=0.5).validate(_SINUOUS)
    assert strict is not None and loose is not None
    assert strict["pass_probability"] <= loose["pass_probability"]
    assert strict["median_max_drawdown"] > 0.03  # typical path breaches the strict rule


def test_tighter_drawdown_rule_raises_bust_rate() -> None:
    """The red (busted) verdict must grow as the trailing-DD rule tightens —
    this is exactly what the what-if stress preview leans on."""
    strict = StrategyMonteCarlo(n_sims=10_000, seed=1, max_drawdown=0.03).validate(_SINUOUS)
    loose = StrategyMonteCarlo(n_sims=10_000, seed=1, max_drawdown=0.5).validate(_SINUOUS)
    assert strict is not None and loose is not None
    assert strict["bust_rate"] > loose["bust_rate"]
    assert strict["bust_rate"] > 0.5  # most futures breach a 3% trailing stop


def test_validation_payload_shapes() -> None:
    result = StrategyMonteCarlo(n_sims=5_000, seed=3).validate(_PROFITABLE)
    assert result is not None
    assert result["n_periods"] == len(_PROFITABLE) - 1
    assert result["n_sims"] == 5_000
    for pct in ["10", "25", "50", "75", "90"]:
        assert len(result["equity_fan"][pct]) == len(_PROFITABLE)  # steps incl. start
    assert result["equity_fan"]["50"][0] == 1.0  # every future starts at 1.0
    hist = result["terminal_histogram"]
    assert len(hist["counts"]) == len(hist["edges"]) - 1
    assert sum(hist["counts"]) == 5_000
    dd_hist = result["drawdown_histogram"]
    assert len(dd_hist["counts"]) == len(dd_hist["edges"]) - 1
    assert 0.0 <= result["pass_probability"] <= 1.0
    assert result["expected_return"] == round(result["expected_terminal"] - 1.0, 6)


def test_sample_paths_carry_outcome_and_stats() -> None:
    result = StrategyMonteCarlo(n_sims=5_000, seed=3).validate(_PROFITABLE)
    assert result is not None
    sp = result["sample_paths"]
    assert len(sp) == 100  # capped at _SAMPLE_PATHS
    for p in sp:
        assert "equity" in p and "outcome" in p
        assert "terminal_return" in p and "max_drawdown" in p
        assert p["outcome"] in ("passed", "busted", "neutral")
        assert len(p["equity"]) == len(_PROFITABLE)  # steps incl. start


def test_validation_needs_enough_history() -> None:
    assert StrategyMonteCarlo(n_sims=100).validate([1.0, 1.001, 1.002]) is None


def test_effective_rules_explicit_contract_passthrough() -> None:
    """QuantPad/FTMO fixed contracts are honored verbatim."""
    import numpy as np

    rules = effective_rules(
        np.asarray(_PROFITABLE[1:]) / np.asarray(_PROFITABLE[:-1]) - 1.0,
        target=0.06,
        max_drawdown=0.08,
        target_sigma=1.0,
        max_drawdown_sigma=1.5,
    )
    assert rules["rules"] == "explicit"
    assert rules["target"] == 0.06
    assert rules["max_drawdown"] == 0.08


def test_effective_rules_risk_scales_to_terminal_volatility() -> None:
    """Default mode: target = 1·σ_T, max drawdown = 1.5·σ_T (σ_T = σ·√n).

    Risk-scaling is the honest default for a 5m signal: the target:drawdown
    ratio, not absolute percentages, decides how hard a challenge is
    (OneTradeJournal/CrossTrade/PropFlux)."""
    import math

    import numpy as np

    returns = np.asarray([0.001, -0.0005, 0.002, -0.001, 0.0015, 0.0008, -0.0004, 0.0011])
    sigma = float(np.std(returns))
    sigma_terminal = sigma * math.sqrt(returns.size)
    rules = effective_rules(
        returns,
        target=None,
        max_drawdown=None,
        target_sigma=1.0,
        max_drawdown_sigma=1.5,
    )
    assert rules["rules"] == "risk-scaled"
    assert rules["sigma_terminal"] == pytest.approx(sigma_terminal, abs=1e-6)
    assert rules["target"] == pytest.approx(1.0 * sigma_terminal, abs=1e-6)
    assert rules["max_drawdown"] == pytest.approx(1.5 * sigma_terminal, abs=1e-6)
    assert rules["edge_bps"] == pytest.approx(float(np.mean(returns)) * 1e4, abs=1e-3)


def test_effective_rules_degenerate_on_zero_variance() -> None:
    """No realized variance, no explicit contract -> nothing to score against."""
    import numpy as np

    rules = effective_rules(
        np.zeros(40),
        target=None,
        max_drawdown=None,
        target_sigma=1.0,
        max_drawdown_sigma=1.5,
    )
    assert rules["rules"] == "degenerate"
    assert rules["target"] is None and rules["max_drawdown"] is None

    empty = effective_rules(
        np.array([], dtype=float),
        target=None,
        max_drawdown=None,
        target_sigma=1.0,
        max_drawdown_sigma=1.5,
    )
    assert empty["rules"] == "none"


def test_two_barrier_no_drift_ratio() -> None:
    """μ=0 gambler's ruin: p = b/(a+b) — pure geometry (MIT 18.642 / STAT 3106)."""
    p = two_barrier_pass_probability(0.0, sigma=0.02, target=0.05, max_drawdown=0.10)
    assert p == pytest.approx(0.10 / 0.15, abs=1e-4)


def test_two_barrier_drift_moves_probability() -> None:
    """Positive edge beats the geometric ratio; negative edge loses to it."""
    base = two_barrier_pass_probability(0.0, sigma=0.02, target=0.05, max_drawdown=0.10)
    up = two_barrier_pass_probability(3.0, sigma=0.02, target=0.05, max_drawdown=0.10)
    down = two_barrier_pass_probability(-3.0, sigma=0.02, target=0.05, max_drawdown=0.10)
    assert base is not None and up is not None and down is not None
    assert up > base > down


def test_two_barrier_handles_extreme_negative_drift_without_overflow() -> None:
    """The naive e^{γ(a+b)} ratio overflows for large γ; the stable form must
    return a finite probability in [0, 1]."""
    p = two_barrier_pass_probability(-20.0, sigma=0.0005, target=0.002, max_drawdown=0.003)
    assert p is not None and 0.0 <= p <= 1.0


def test_two_barrier_undefined_cases() -> None:
    assert two_barrier_pass_probability(1.0, sigma=0.0, target=0.05, max_drawdown=0.10) is None
    assert two_barrier_pass_probability(1.0, sigma=0.02, target=0.0, max_drawdown=0.0) is None


def test_validation_emits_confidence_interval_analytic_and_rules() -> None:
    """The honest '62% (52–71%)' convention: CI brackets the bootstrap pass
    rate, the analytic two-barrier benchmark is a valid probability, and the
    rules dict names whether the contract was explicit or risk-scaled."""
    result = StrategyMonteCarlo(n_sims=10_000, seed=7).validate(_PROFITABLE)
    assert result is not None
    assert result["rules"] in ("explicit", "risk-scaled")
    assert (
        0.0 <= result["pass_ci_low"] <= result["pass_probability"] <= result["pass_ci_high"] <= 1.0
    )
    analytic = result["analytic_pass_probability"]
    assert analytic is None or 0.0 <= analytic <= 1.0
    assert result["edge_bps"] > 0.0
    assert result["sigma_terminal"] > 0.0
    assert result["max_drawdown_rule"] > 0.0
    assert result["target"] > 0.0
