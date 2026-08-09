"""Strategy-validation Monte Carlo tests — hermetic (no Kafka/Redis/Snowflake).

QuantPad-style: bootstrap the realized per-period returns into 10k simulated
futures and score them against prop-firm-style rules. Covers determinism,
the pass-probability logic, distribution shapes, and the equity→returns
derivation.
"""

from __future__ import annotations

from stream.strategy_mc import StrategyMonteCarlo, strategy_returns_from_equity

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
