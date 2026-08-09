"""Forecast calibration monitor — "does the model mean what it says?"

Replays the MC engine's *exact* 1-step predictive distribution (Student-t
log-returns with EWMA volatility — the t+EWMA GBM of ``MonteCarloEngine``),
point-in-time, and scores every forecast against the close that actually
materialized. Because the engine is deterministic given (trailing closes,
window_end_ms), the monitor recomputes what the model would have predicted for
every historical window using the same code path as the live consumer (the
Kappa/replay principle) — no lookahead, offline/online parity by construction.
Scoring with the SAME t(df) the engine simulates with is what makes the
monitor honest: a Normal band scored with Normal CDFs under-covers fat tails
forever (Horváth & Šopov 2016), so the monitor and the model must share the
predictive family.

Scoring stack (research-backed):

  coverage          empirical 10–90 band hit rate vs its true nominal level
  PIT               probability integral transform; i.i.d. U(0,1) under
                    correct calibration (Diebold, Gunther & Tay 1998;
                    Gneiting, Balabdaoui & Raftery 2007)
  Kupiec POF        unconditional-coverage LR test (~chi2_1)
  Christoffersen CC conditional-coverage LR test (independence + coverage,
                    ~chi2_2) — conformal intervals are VaR forecasts, so the
                    canonical VaR backtests apply (Retzlaff 2025)
  e-process         anytime-valid evidence accumulation (Arnold, Henzi &
                    Ziegel 2021): product of Beta-mixture density ratios is a
                    test martingale under calibration; Ville's inequality
                    bounds the false-alarm probability under *continuous*
                    monitoring, so the panel flags the exact window where
                    calibration breaks with no fixed-sample-size assumption
  MCB / KS          scalar miscalibration of the PIT distribution vs U(0,1)
                    (Wasserstein-1, Wessel et al. 2026)

Only 1-step-ahead forecasts feed the formal tests: the 12-step fan forecasts
overlap (the close at t+1 is also horizon-12 of the forecast at t-11), so
their PITs are dependent; the 1-step PITs are i.i.d. under calibration, which
is what makes the tests valid. The multi-step fan overlay is served as a
visual diagnostic (central-bank fan-chart practice), honestly labeled.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import stats
from scipy.special import logsumexp

from stream.simulation import MonteCarloEngine

# Beta density alternatives pooled into the anytime-valid e-value: symmetric
# underdispersion/overdispersion shapes plus mild skews. Each component is a
# density, so each mixture ratio has expectation 1 under the uniform null.
_EVALUE_COMPONENTS = [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (0.7, 1.3), (1.3, 0.7)]

_PIT_BINS = 10
_SIGNIFICANCE = 0.05


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _coverage_tests(records: Sequence[Mapping], nominal: float) -> dict | None:
    """Kupiec POF + Christoffersen conditional-coverage LR tests on hit series."""
    n = len(records)
    if n == 0:
        return None
    hits = [1 if r.get("hit") else 0 for r in records]
    x = sum(hits)
    pof = _kupiec(n, x, nominal)
    if n < 2:
        cc = None
    else:
        cc = _christoffersen(hits, pof["lr"])
    return {
        "n": n,
        "hits": x,
        "coverage": round(x / n, 4),
        "pof": pof,
        "cc": cc,
    }


def _kupiec(n: int, x: int, p: float) -> dict:
    """Unconditional-coverage (POF) LR test: L(p0) vs L(p_hat), ~chi2_1."""
    p_hat = x / n
    if p_hat == 0.0:
        lr = -2.0 * (n * math.log(1.0 - p) - 0.0)
    elif p_hat == 1.0:
        lr = -2.0 * (x * math.log(p) - 0.0)
    else:
        lr = -2.0 * (
            (n - x) * math.log(1.0 - p)
            + x * math.log(p)
            - ((n - x) * math.log(1.0 - p_hat) + x * math.log(p_hat))
        )
    return {
        "lr": round(max(lr, 0.0), 4),
        "p": round(float(stats.chi2.sf(max(lr, 0.0), 1)), 4),
        "reject": lr > 0.0 and float(stats.chi2.sf(lr, 1)) < _SIGNIFICANCE,
    }


def _christoffersen(hits: Sequence[int], pof_lr: float) -> dict:
    """Conditional-coverage test: independence LR + unconditional LR, ~chi2_2."""
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(hits, hits[1:]):
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    m = n00 + n01 + n10 + n11
    if m == 0:
        return None
    pi = (n01 + n11) / m
    den0, den1 = n00 + n01, n10 + n11
    pi0 = n01 / den0 if den0 else 0.5
    pi1 = n11 / den1 if den1 else 0.5
    eps = 1e-12
    ln_restricted = (n00 + n10) * math.log(max(1 - pi, eps)) + (n01 + n11) * math.log(max(pi, eps))
    ln_full = (
        n00 * math.log(max(1 - pi0, eps))
        + n01 * math.log(max(pi0, eps))
        + n10 * math.log(max(1 - pi1, eps))
        + n11 * math.log(max(pi1, eps))
    )
    lr_ind = max(-2.0 * (ln_restricted - ln_full), 0.0)
    lr_cc = pof_lr + lr_ind
    return {
        "lr": round(lr_cc, 4),
        "p": round(float(stats.chi2.sf(lr_cc, 2)), 4),
        "reject": lr_cc > 0.0 and float(stats.chi2.sf(lr_cc, 2)) < _SIGNIFICANCE,
    }


def _pit_stats(pits: Sequence[float]) -> dict:
    """PIT histogram (binomial CI under the U(0,1) null) + MCB + KS score."""
    n = len(pits)
    counts, edges = np.histogram(pits, bins=_PIT_BINS, range=(0.0, 1.0))
    expected = n / _PIT_BINS
    se = math.sqrt(n * (1.0 / _PIT_BINS) * (1.0 - 1.0 / _PIT_BINS))
    s = np.sort(np.asarray(pits, dtype=float))
    positions = np.arange(1, n + 1) / n
    mcb = float(np.mean(np.abs(s - positions)))
    d_plus = float(np.max(positions - s)) if n else 0.0
    d_minus = float(np.max(s - (np.arange(n) / n))) if n else 0.0
    ks = max(d_plus, d_minus, 0.0)
    return {
        "n": n,
        "counts": [int(c) for c in counts],
        "edges": [round(float(e), 3) for e in edges],
        "expected": round(expected, 2),
        "ci_lo": round(max(expected - 1.96 * se, 0.0), 2),
        "ci_hi": round(expected + 1.96 * se, 2),
        "mcb": round(mcb, 4),
        "ks": round(ks, 4),
    }


def _pit_e_values(pits: Sequence[float], alpha: float) -> dict:
    """Anytime-valid e-process over the PITs (test martingale under calibration).

    Each observation contributes a Beta-mixture density ratio; the running
    product is an e-process, so by Ville's inequality the alarm level 1/alpha
    is valid under continuous monitoring.
    """
    k = len(_EVALUE_COMPONENTS)
    logE = 0.0
    process: list[float] = []
    for z in pits:
        zz = min(max(float(z), 1e-9), 1.0 - 1e-9)
        logpdfs = [stats.beta.logpdf(zz, a, b) for a, b in _EVALUE_COMPONENTS]
        e_t = math.exp(logsumexp(logpdfs) - math.log(k))
        logE += math.log(max(e_t, 1e-300))
        process.append(round(math.exp(logE), 6))
    e_value = process[-1] if process else 0.0
    return {
        "e_value": round(e_value, 4),
        "anytime_p": round(min(1.0, 1.0 / e_value) if e_value else 1.0, 6),
        "alarm": e_value >= 1.0 / alpha,
        "alpha": alpha,
        "threshold": round(1.0 / alpha, 2),
        "process": process,
    }


def reality_report(
    features: Sequence[Mapping],
    engine: MonteCarloEngine,
    *,
    nominal_coverage: float = 0.8,
    alpha: float = 0.005,
) -> dict | None:
    """Replay 1-step forecasts point-in-time and score them against reality.

    ``features`` are the online-store feature windows (chronological).
    Returns None when there is not enough history to calibrate + score at
    least one forecast.
    """
    pairs: list[tuple[int, float]] = []
    for w in features:
        close = w.get("close")
        ts = w.get("window_end_ms")
        if isinstance(close, (int, float)) and close == close and ts is not None:
            pairs.append((int(ts), float(close)))
    pairs.sort(key=lambda p: p[0])
    closes = [c for _, c in pairs]
    stamps = [t for t, _ in pairs]
    if len(closes) < engine._vol_windows + 2:
        return None

    records: list[dict] = []
    for i in range(engine._vol_windows, len(closes) - 1):
        dist = engine.calibrate_dist(closes[: i + 1])
        if dist is None:
            continue
        mu, sigma, nu = dist
        s0 = closes[i]
        # 1-step log-return predictive distribution: std-t(nu) centered at
        # (mu - sigma^2/2). The 10–90 band edges use the SAME t(df) the engine
        # simulates with — Normal quantiles here would under-cover the engine's
        # fat tails (Horváth & Šopov 2016), breaking offline/online parity.
        log_mean = mu - sigma * sigma / 2.0
        z10 = stats.t.ppf(0.10, nu)
        lo = s0 * math.exp(log_mean + z10 * sigma)
        hi = s0 * math.exp(log_mean - z10 * sigma)
        realized = closes[i + 1]
        pit = float(stats.t.cdf((math.log(realized / s0) - log_mean) / sigma, nu))
        records.append(
            {
                "window_end_ms": stamps[i],
                "base": s0,
                "lo": lo,
                "hi": hi,
                "realized": realized,
                "pit": pit,
                "hit": lo <= realized <= hi,
            }
        )
    if not records:
        return None

    pits = [r["pit"] for r in records]
    coverage = _coverage_tests(records, nominal_coverage)
    pit_stats = _pit_stats(pits)
    evalue = _pit_e_values(pits, alpha)

    # The fan that would have been drawn `horizon_steps` windows ago, with the
    # realized path overlaid — the BoE-style visual diagnostic. Multi-step
    # horizons overlap, so this is presentation, not a formal test.
    i_issue = len(closes) - 1 - engine._horizon_steps
    fan = None
    realized_path: list[dict] | None = None
    if i_issue >= engine._vol_windows:
        payload = engine.forecast(closes[: i_issue + 1], window_end_ms=stamps[i_issue])
        if payload is not None:
            fan = {
                "window_end_ms": stamps[i_issue],
                "base_price": payload["base_price"],
                "sigma": payload["sigma"],
                "percentiles": payload["percentiles"],
            }
            pts: list[dict] = []
            for s in range(engine._horizon_steps + 1):
                j = i_issue + s
                if j >= len(closes):
                    break
                lo, hi = payload["percentiles"]["10"][s], payload["percentiles"]["90"][s]
                pts.append(
                    {
                        "step": s,
                        "close": closes[j],
                        "window_end_ms": stamps[j],
                        "in_band": s == 0 or lo <= closes[j] <= hi,
                    }
                )
            realized_path = pts

    return {
        "symbol": None,
        "window_end_ms": stamps[-1],
        "updated_at": _now_iso(),
        "horizon_steps": engine._horizon_steps,
        "nominal_coverage": nominal_coverage,
        "n_windows": len(records),
        "coverage": coverage,
        "pit": pit_stats,
        "evalue": evalue,
        "fan": fan,
        "realized": realized_path,
    }
