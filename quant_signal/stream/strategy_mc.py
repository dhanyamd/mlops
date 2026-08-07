"""Strategy validation Monte Carlo (QuantPad-style pass probability).

Bootstraps the strategy's *realized* per-period returns into ``n_sims``
whole-strategy futures — one vectorized 10k-path matrix, no Python loop — and
evaluates each future against prop-firm-style rules:

  pass   <=>  reached the profit target (``target``) WITHOUT breaching the
              max-drawdown rule
  busted <=>  breached max drawdown from peak before reaching the target
  neutral<=>  survived, but never reached the target

The fraction of passing futures is the pass probability QuantPad headlines
("know your pass probability before you pay for an evaluation"). We also emit
the equity fan (percentile bands), outcome-colored sample paths for the
terminal visualization, terminal-equity and max-drawdown distributions, and
tail expectations.

Pure math lives in ``StrategyMonteCarlo`` for hermetic tests; the API drives it
from the live ``strategy:crypto:5m:<SYMBOL>`` equity curve.
"""

from __future__ import annotations

import numpy as np

_FAN_PERCENTILES = [10, 25, 50, 75, 90]

# Histogram resolution (display, not a model knob).
_HIST_BINS = 24

# Sample paths emitted for the fan visualization (thin colored lines).
_SAMPLE_PATHS = 100


def strategy_returns_from_equity(equity: list[float]) -> np.ndarray:
    """Per-period realized returns from a cumulative equity curve (starts 1.0)."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2 or not np.all(np.isfinite(eq)):
        return np.array([], dtype=float)
    return eq[1:] / eq[:-1] - 1.0


class StrategyMonteCarlo:
    """Bootstrap-resample the strategy's returns into simulated futures.

    Each simulated future reorders the realized returns (sampling with
    replacement), compounds them into an equity path, and is scored against the
    prop-firm rules — the same "thousands of Monte Carlo simulations" QuantPad
    runs, here as one vectorized numpy pass.
    """

    def __init__(
        self,
        *,
        n_sims: int = 10_000,
        max_drawdown: float = 0.08,
        target: float | None = None,
        seed: int | None = None,
    ) -> None:
        self._n_sims = n_sims
        self._max_drawdown = max_drawdown
        self._target = target
        self._seed = seed

    def _classify(
        self, steps: np.ndarray, max_dd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        never_broke = max_dd < self._max_drawdown
        if self._target is not None:
            hit_target = np.max(steps, axis=1) >= (1.0 + self._target)
        else:
            hit_target = steps[:, -1] >= 1.0
        passed = never_broke & hit_target
        busted = ~never_broke
        neutral = never_broke & ~hit_target
        return passed, busted, neutral

    def validate(self, equity: list[float]) -> dict | None:
        """Full validation payload from an equity curve, or None if too short."""
        returns = strategy_returns_from_equity(equity)
        if len(returns) < 3:
            return None
        rng = np.random.default_rng(self._seed)
        n = len(returns)
        n_sims = self._n_sims
        idx = rng.integers(0, n, size=(n_sims, n))
        sim_returns = returns[idx]  # (n_sims, periods)

        steps = np.concatenate(
            [np.ones((n_sims, 1)), np.cumprod(1.0 + sim_returns, axis=1)], axis=1
        )
        peak = np.maximum.accumulate(steps, axis=1)
        dd = (peak - steps) / peak
        max_dd = np.max(dd, axis=1)
        terminal = steps[:, -1]

        passed, busted, neutral = self._classify(steps, max_dd)
        pass_prob = float(np.mean(passed))
        bust_rate = float(np.mean(busted))
        neutral_rate = float(np.mean(neutral))
        mean_terminal = float(np.mean(terminal))

        fan = {
            str(q): [round(float(v), 6) for v in np.percentile(steps, q, axis=0)]
            for q in _FAN_PERCENTILES
        }

        t_counts, t_edges = np.histogram(terminal, bins=_HIST_BINS)
        passed_bin, _ = np.histogram(terminal[passed], bins=t_edges)
        busted_bin, _ = np.histogram(terminal[busted], bins=t_edges)
        neutral_bin = t_counts - passed_bin - busted_bin

        d_counts, d_edges = np.histogram(max_dd, bins=_HIST_BINS)

        n_samples = min(_SAMPLE_PATHS, n_sims)
        sample_idx = np.linspace(0, n_sims - 1, n_samples, dtype=int)
        sample_paths = [
            {
                "equity": [round(float(v), 6) for v in steps[i]],
                "outcome": "passed" if passed[i] else ("busted" if busted[i] else "neutral"),
                "terminal_return": round(float(terminal[i] - 1.0), 6),
                "max_drawdown": round(float(max_dd[i]), 6),
            }
            for i in sample_idx
        ]

        return {
            "n_periods": n,
            "n_sims": n_sims,
            "max_drawdown_rule": self._max_drawdown,
            "target": self._target,
            "pass_probability": round(pass_prob, 4),
            "bust_rate": round(bust_rate, 4),
            "neutral_rate": round(neutral_rate, 4),
            "expected_terminal": round(mean_terminal, 6),
            "expected_return": round(mean_terminal - 1.0, 6),
            "median_terminal": round(float(np.median(terminal)), 6),
            "best10_terminal": round(float(np.percentile(terminal, 90)), 6),
            "worst10_terminal": round(float(np.percentile(terminal, 10)), 6),
            "avg_max_drawdown": round(float(np.mean(max_dd)), 6),
            "median_max_drawdown": round(float(np.median(max_dd)), 6),
            "p95_max_drawdown": round(float(np.percentile(max_dd, 95)), 6),
            "equity_fan": fan,
            "median_path": fan["50"],
            "sample_paths": sample_paths,
            "terminal_histogram": {
                "counts": [int(c) for c in t_counts],
                "passed": [int(c) for c in passed_bin],
                "busted": [int(c) for c in busted_bin],
                "neutral": [int(c) for c in neutral_bin],
                "edges": [round(float(e), 6) for e in t_edges],
            },
            "drawdown_histogram": {
                "counts": [int(c) for c in d_counts],
                "edges": [round(float(e), 6) for e in d_edges],
            },
        }
