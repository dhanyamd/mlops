"""Strategy validation Monte Carlo (QuantPad-style pass probability).

Bootstraps the strategy's *realized* per-period returns into ``n_sims``
whole-strategy futures — one vectorized 10k-path matrix, no Python loop — and
evaluates each future against pass/fail rules:

  pass   <=>  reached the profit target (``target``) WITHOUT breaching the
              max-drawdown rule
  busted <=>  breached max drawdown from peak before reaching the target
  neutral<=>  survived, but never reached the target (the "ran out of time"
              verdict — CrossTrade's naming)

Rules are either an explicit fixed contract (QuantPad/FTMO-style, e.g. 6%/8%)
or **risk-scaled to the strategy's own realized terminal volatility**
(``target_sigma``/``max_drawdown_sigma`` multiples of σ_T = per-window σ·√n).
Risk-scaling is the honest default for a 5m signal whose per-window returns are
a few bps: a fixed human-scaled 6%/8% contract is structurally unreachable, so
the pass probability pins at 0%/100%/0% and never moves. The prop-firm
literature is explicit that the *ratio* of target to drawdown, not the absolute
percentages, decides how hard a challenge is (OneTradeJournal 2026; CrossTrade;
PropFlux) — scaling the rules to realized risk restores the gauge to a
well-posed, moving question about the strategy's realized edge.

We also emit the **analytic two-barrier benchmark** — P(hit +target before
−max-DD) for a drifted Brownian motion (gambler's ruin, Doob optional stopping;
MIT 18.642 / Columbia STAT 3106) — plus an 80% confidence interval on the pass
rate ("a '62% (52–71%)' result is honest; a bare '62.3%' would be false
precision" — PropFlux methodology). Bootstrap (finite horizon) ≤ analytic
(no horizon cap), with the gap being the neutral/timeout mass.

Pure math lives in ``StrategyMonteCarlo`` for hermetic tests; the API drives it
from the live ``strategy:crypto:5m:<SYMBOL>`` equity curve.
"""

from __future__ import annotations

import math

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


def effective_rules(
    returns: np.ndarray,
    *,
    target: float | None,
    max_drawdown: float | None,
    target_sigma: float,
    max_drawdown_sigma: float,
) -> dict:
    """Resolve the pass/fail rules actually applied to a bootstrap horizon.

    An explicit contract (``target``/``max_drawdown``) is used verbatim — the
    QuantPad/FTMO fixed-contract mode. Otherwise the rules are risk-scaled to
    the strategy's own realized terminal volatility σ_T = per-window σ·√n, so
    the pass probability is a well-posed question about the strategy's edge
    relative to its risk (target:drawdown ratio, not absolute percentages —
    OneTradeJournal/CrossTrade/PropFlux).
    """
    if returns.size == 0:
        return {
            "target": target,
            "max_drawdown": max_drawdown,
            "sigma": 0.0,
            "sigma_terminal": 0.0,
            "edge_bps": 0.0,
            "rules": "none",
        }
    mu = float(np.mean(returns))
    sigma = float(np.std(returns))
    n = returns.size
    sigma_terminal = sigma * math.sqrt(n)
    edge_bps = mu * 10000.0
    if sigma_terminal <= 0.0:
        # Zero realized variance: nothing to score against in risk-scaled mode.
        if target is not None and max_drawdown is not None:
            return {
                "target": target,
                "max_drawdown": max_drawdown,
                "sigma": round(sigma, 8),
                "sigma_terminal": round(sigma_terminal, 6),
                "edge_bps": round(edge_bps, 3),
                "rules": "explicit",
            }
        return {
            "target": None,
            "max_drawdown": None,
            "sigma": round(sigma, 8),
            "sigma_terminal": round(sigma_terminal, 6),
            "edge_bps": round(edge_bps, 3),
            "rules": "degenerate",
        }
    target_eff = target_sigma * sigma_terminal if target is None else target
    dd_eff = max_drawdown_sigma * sigma_terminal if max_drawdown is None else max_drawdown
    return {
        "target": round(target_eff, 6),
        "max_drawdown": round(dd_eff, 6),
        "sigma": round(sigma, 8),
        "sigma_terminal": round(sigma_terminal, 6),
        "edge_bps": round(edge_bps, 3),
        "rules": "explicit" if (target is not None and max_drawdown is not None) else "risk-scaled",
    }


def two_barrier_pass_probability(
    edge_bps: float,
    sigma: float,
    target: float,
    max_drawdown: float,
) -> float | None:
    """P(hit +target before −max_drawdown) for a drifted Brownian motion.

    Gambler's ruin for X_t = mu·t + sigma·B_t (Doob optional stopping):

        p = (1 − exp(−2·mu·b/σ²)) / (1 − exp(−2·mu·(a+b)/σ²)),   mu ≠ 0
        p = b / (a + b),                                          mu = 0

    with a = target (up barrier) and b = max_drawdown (down barrier) in
    equity-fraction units and mu = edge_bps/10⁴ per window. This is the
    *no-horizon-cap* diffusion limit; the bootstrap pass rate is the
    finite-horizon analogue, so bootstrap ≤ analytic, with the gap being the
    neutral (timeout) mass. Returns None when not well-defined.
    """
    if sigma <= 0.0 or target is None or max_drawdown is None:
        return None
    if target + max_drawdown <= 0.0:
        return None
    a, b = target, max_drawdown
    s2 = sigma * sigma
    two_mu = 2.0 * edge_bps / 10000.0
    if abs(two_mu) < 1e-12:
        p = b / (a + b)
    elif two_mu > 0.0:
        p = (1.0 - math.exp(-two_mu * b / s2)) / (1.0 - math.exp(-two_mu * (a + b) / s2))
    else:
        # Negative drift: p = (e^{γb}−1)/(e^{γ(a+b)}−1) with γ=−2μ/σ². The naive
        # ratio overflows for large γ, so factor out e^{−γ(a+b)}:
        #   p = e^{−γa}·(1−e^{−γb})/(1−e^{−γ(a+b)}),  all terms bounded in (0,1].
        gamma = -two_mu / s2
        p = math.exp(-gamma * a) * (1.0 - math.exp(-gamma * b)) / (1.0 - math.exp(-gamma * (a + b)))
    return round(min(max(p, 0.0), 1.0), 4)


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
        max_drawdown: float | None = None,
        target: float | None = None,
        target_sigma: float = 1.0,
        max_drawdown_sigma: float = 1.5,
        seed: int | None = None,
    ) -> None:
        self._n_sims = n_sims
        self._max_drawdown = max_drawdown
        self._target = target
        self._target_sigma = target_sigma
        self._max_drawdown_sigma = max_drawdown_sigma
        self._seed = seed

    def _classify(
        self,
        steps: np.ndarray,
        max_dd: np.ndarray,
        *,
        target: float | None,
        max_drawdown: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        never_broke = max_dd < max_drawdown
        if target is not None:
            hit_target = np.max(steps, axis=1) >= (1.0 + target)
        else:
            hit_target = steps[:, -1] >= 1.0
        passed = never_broke & hit_target
        busted = ~never_broke
        neutral = never_broke & ~hit_target
        return passed, busted, neutral

    def geometry_sweep(
        self,
        returns: np.ndarray,
        n_periods: int,
        target: float,
        max_drawdown: float,
        sweep_rows: int = 7,
        sweep_cols: int = 7,
        n_sims: int = 1_000,
    ) -> dict:
        """Pass-probability heat grid across R:R configurations.

        Holds the strategy's *expected daily return* constant while sweeping the
        win-rate × R:R shape (research-backed insight from PropSim/QuantPad: path-
        dependent trailing-DD rules make geometry matter more than edge):

        - Each cell holds the daily EV constant: wr × avg_win + (1-wr) × avg_loss
          is constant across the grid, so we only move probability mass around.
        - Returns a heat grid of pass probabilities plus the win-rate/R:R for each
          cell, so the UI can show exactly where a trader's geometry sits.

        ``returns`` is the per-period realized returns (for calibrating avg size).
        """
        if len(returns) < 2:
            return {"grid": [], "wr_axis": [], "rr_axis": [], "ev": 0.0}

        base_ev = float(np.mean(returns))
        loss_mask = returns <= 0
        avg_loss = abs(float(np.mean(returns[loss_mask]))) if loss_mask.any() else 0.01
        sigma = float(np.std(returns))

        rr_values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        wr_values = np.linspace(0.40, 0.80, sweep_cols)
        loss_unit = avg_loss if avg_loss > 0 else 0.01

        rng = np.random.default_rng(self._seed)
        n_trades = n_periods

        full_grid: list[list[float]] = []
        for rr in rr_values:
            row: list[float] = []
            for wr in wr_values:
                win_size = rr * loss_unit
                denom = rr * wr - (1.0 - wr)
                if denom <= 0:
                    row.append(0.0)
                    continue
                loss_size = base_ev / denom
                if loss_size <= 0:
                    row.append(0.0)
                    continue
                draws = rng.random((n_sims, n_trades))
                wins = draws < wr
                path_returns = np.where(wins, win_size, -loss_size)
                steps = np.concatenate(
                    [np.ones((n_sims, 1)), np.cumprod(1.0 + path_returns, axis=1)], axis=1
                )
                peak = np.maximum.accumulate(steps, axis=1)
                dd = (peak - steps) / peak
                max_dd_sim = np.max(dd, axis=1)
                hit_target = steps[:, -1] >= (1.0 + target)
                passed = (max_dd_sim < max_drawdown) & hit_target
                row.append(round(float(np.mean(passed)), 4))
            full_grid.append(row)

        edge_sweep = self._edge_sweep(
            sigma,
            base_ev,
            n_periods=n_trades,
            target=target,
            max_drawdown=max_drawdown,
            n_sims=n_sims,
        )

        return {
            "grid": full_grid,
            "wr_axis": [round(float(w), 3) for w in wr_values],
            "rr_axis": [round(float(r), 3) for r in rr_values],
            "ev": round(base_ev, 6),
            "sigma": round(sigma, 6),
            "edge_sweep": edge_sweep,
            "seed": self._seed,
        }

    def _edge_sweep(
        self,
        sigma: float,
        base_ev: float,
        *,
        n_periods: int,
        target: float,
        max_drawdown: float,
        n_sims: int,
    ) -> dict:
        """Pass probability vs per-period edge (prop-ev style sweep).

        Holds the strategy's realized volatility constant and sweeps the edge
        from strongly negative to strongly positive, so you can see where the
        challenge flips from 'no geometry passes' to 'passable' — the honest
        answer for a currently-negative-EV strategy, which otherwise renders as
        an all-zero heat grid.

        Also interpolates the *breakeven edge* — the per-period edge (bps) at
        which pass probability crosses 50% — the single number a PM reads first:
        'how much edge does my current volatility and drawdown rule demand?'
        """
        edges_bps = np.linspace(-12, 12, 25)
        passes: list[float] = []
        rng = np.random.default_rng(self._seed)
        for eb in edges_bps:
            edge = eb / 10000
            per = rng.normal(edge, sigma, size=(n_sims, n_periods))
            steps = np.concatenate([np.ones((n_sims, 1)), np.cumprod(1.0 + per, axis=1)], axis=1)
            peak = np.maximum.accumulate(steps, axis=1)
            dd = (peak - steps) / peak
            max_dd_sim = np.max(dd, axis=1)
            hit_target = steps[:, -1] >= (1.0 + target)
            passed = (max_dd_sim < max_drawdown) & hit_target
            passes.append(round(float(np.mean(passed)), 4))

        break_even: float | None = None
        for i in range(1, len(passes)):
            if passes[i - 1] < 0.5 <= passes[i]:
                x0, x1 = edges_bps[i - 1], edges_bps[i]
                p0, p1 = passes[i - 1], passes[i]
                frac = (0.5 - p0) / (p1 - p0) if p1 != p0 else 0.0
                break_even = round(float(x0 + frac * (x1 - x0)), 3)
                break
        if break_even is None:
            break_even = round(float(edges_bps[0] if passes[-1] < 0.5 else edges_bps[-1]), 3)

        return {
            "edges_bps": [round(float(e), 2) for e in edges_bps],
            "pass": passes,
            "current_edge_bps": round(base_ev * 10000, 3),
            "breakeven_edge_bps": break_even,
            "n_periods": n_periods,
            "target": target,
            "max_drawdown": max_drawdown,
            "seed": self._seed,
        }

    def validate(self, equity: list[float]) -> dict | None:
        """Full validation payload from an equity curve, or None if too short."""
        returns = strategy_returns_from_equity(equity)
        if len(returns) < 3:
            return None
        rules = effective_rules(
            returns,
            target=self._target,
            max_drawdown=self._max_drawdown,
            target_sigma=self._target_sigma,
            max_drawdown_sigma=self._max_drawdown_sigma,
        )
        if rules["target"] is None or rules["max_drawdown"] is None:
            return None  # degenerate realized volatility: nothing to score against
        target, max_drawdown = rules["target"], rules["max_drawdown"]
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

        passed, busted, neutral = self._classify(
            steps, max_dd, target=target, max_drawdown=max_drawdown
        )
        pass_prob = float(np.mean(passed))
        bust_rate = float(np.mean(busted))
        neutral_rate = float(np.mean(neutral))
        mean_terminal = float(np.mean(terminal))

        # Honest uncertainty on the pass rate: 80% binomial interval
        # (se = sqrt(p(1-p)/n_sims) — PropFlux's "62% (52–71%)" convention).
        se = math.sqrt(max(pass_prob * (1.0 - pass_prob), 0.0) / n_sims)
        z80 = 1.2816
        pass_ci_low = max(0.0, pass_prob - z80 * se)
        pass_ci_high = min(1.0, pass_prob + z80 * se)

        # Analytic two-barrier benchmark (drifted Brownian motion, no horizon
        # cap): the diffusion limit the bootstrap is the finite-horizon of.
        analytic = two_barrier_pass_probability(
            rules["edge_bps"], rules["sigma"], target, max_drawdown
        )

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
            "max_drawdown_rule": round(max_drawdown, 6),
            "target": round(target, 6),
            "rules": rules["rules"],
            "sigma_terminal": rules["sigma_terminal"],
            "edge_bps": rules["edge_bps"],
            "pass_probability": round(pass_prob, 4),
            "pass_ci_low": round(pass_ci_low, 4),
            "pass_ci_high": round(pass_ci_high, 4),
            "analytic_pass_probability": analytic,
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
