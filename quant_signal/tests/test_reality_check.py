"""Hermetic tests for the forecast calibration monitor (stream/reality_check.py).

These test the *statistics* against controlled synthetic data: a correctly
specified GBM should look well calibrated, a fat-tailed process should trip the
coverage / PIT / e-value monitors, and the hand-built test tables should
resolve to exactly the right verdicts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stream.reality_check import (
    _christoffersen,
    _coverage_tests,
    _kupiec,
    _pit_e_values,
    _pit_stats,
    reality_report,
)
from stream.simulation import MonteCarloEngine


def _engine(*, vol_windows: int = 40, drift: bool = False) -> MonteCarloEngine:
    return MonteCarloEngine(n_paths=10_000, horizon_steps=12, vol_windows=vol_windows, drift=drift)


def _features(closes: list[float], start_ms: int = 1_700_000_000_000) -> list[dict]:
    return [
        {"window_end_ms": start_ms + i * 300_000, "close": float(c)} for i, c in enumerate(closes)
    ]


def _gbm_closes(n: int = 260, seed: int = 7, sigma: float = 0.0006) -> list[float]:
    rng = np.random.default_rng(seed)
    return [60000.0 * math.exp(sum(rng.normal(0.0, sigma) for _ in range(i + 1))) for i in range(n)]


def _t_closes(n: int = 260, seed: int = 7, sigma: float = 0.0006) -> list[float]:
    """Fat-tailed (t_4) random walk with the same scale — normal bands under-cover."""
    rng = np.random.default_rng(seed)
    scale = sigma / math.sqrt(2.0)  # t_4 has variance 2
    return [
        60000.0 * math.exp(sum(scale * rng.standard_t(4) for _ in range(i + 1))) for i in range(n)
    ]


def _regime_closes(
    n: int = 260,
    seed: int = 13,
    hi: float = 0.004,
    lo: float = 0.0001,
    break_i: int = 110,
) -> list[float]:
    """A 40× volatility regime break (high -> low) mid-sample. EWMA λ=0.94
    decays the high-vol memory with a ~11-window half-life, so the monitor
    keeps drawing wide bands for a while into the calm regime -> over-coverage
    and PITs piled near 0.5. This is the mis-specification the monitor must
    flag. Parameter-swept (seeds 10-19) so the KS/MCB trip signals clear the
    research-critical values for every seed."""
    rng = np.random.default_rng(seed)
    sigs = [hi] * break_i + [lo] * (n - break_i)
    rets = [rng.normal(0.0, s) for s in sigs]
    return [60000.0 * math.exp(sum(rets[: i + 1])) for i in range(n)]


# ── coverage tests ───────────────────────────────────────────────────────────


def test_kupiec_calibrated_hits_do_not_reject() -> None:
    n, x, p = 100, 80, 0.8
    res = _kupiec(n, x, p)
    assert res["lr"] == pytest.approx(0.0, abs=0.05)
    assert res["p"] >= 0.05
    assert res["reject"] is False


def test_kupiec_all_hits_rejects() -> None:
    res = _kupiec(100, 100, 0.8)
    assert res["reject"] is True
    assert res["p"] < 0.05


def test_kupiec_no_hits_rejects() -> None:
    res = _kupiec(100, 0, 0.8)
    assert res["reject"] is True


def test_christoffersen_flat_sequence_rejects_dependence() -> None:
    n = 100
    hits = [1 if i % 2 == 0 else 0 for i in range(n)]  # perfectly alternating
    pof = _kupiec(n, sum(hits), 0.5)
    assert pof["p"] >= 0.05  # unconditional coverage fine at 0.5
    cc = _christoffersen(hits, pof["lr"])
    assert cc is not None
    assert cc["p"] < 0.01  # but the dependence is flagged
    assert cc["reject"] is True


def test_christoffersen_too_short_returns_none() -> None:
    assert _christoffersen([1], 0.0) is None


def test_coverage_tests_payload_shape() -> None:
    records = [{"hit": i < 80} for i in range(100)]
    res = _coverage_tests(records, 0.8)
    assert res is not None
    assert res["n"] == 100 and res["hits"] == 80 and res["coverage"] == 0.8
    assert set(res["pof"]) == {"lr", "p", "reject"}
    assert set(res["cc"]) == {"lr", "p", "reject"}


# ── PIT diagnostics ──────────────────────────────────────────────────────────


def test_pit_stats_uniform_is_flat() -> None:
    pits = [(i + 0.5) / 100 for i in range(100)]
    res = _pit_stats(pits)
    assert res["n"] == 100
    assert all(res["expected"] - 2.5 <= c <= res["expected"] + 2.5 for c in res["counts"])
    assert res["mcb"] < 0.05
    assert res["ks"] < 0.05


def test_pit_stats_concentrated_shows_spike() -> None:
    res = _pit_stats([0.5] * 100)
    assert res["counts"][5] == 100  # the middle bin absorbs everything
    assert res["mcb"] > 0.2
    assert res["ks"] > 0.4


# ── anytime-valid e-process ──────────────────────────────────────────────────


def test_pit_e_values_uniform_does_not_alarm() -> None:
    pits = [(i + 0.5) / 200 for i in range(200)]
    res = _pit_e_values(pits, alpha=0.005)
    assert res["alarm"] is False
    assert res["e_value"] < 10.0


def test_pit_e_values_peaked_alarms() -> None:
    res = _pit_e_values([0.5] * 120, alpha=0.005)
    assert res["alarm"] is True
    assert res["e_value"] >= res["threshold"]


def test_pit_e_values_process_is_cumulative() -> None:
    """The process is a running product: each step is the previous e-value
    times the next mixture density ratio, so a constant peak PIT makes it grow
    monotonically and the final value equals the reported e-value."""
    res = _pit_e_values([0.5] * 6, alpha=0.005)
    assert len(res["process"]) == 6
    assert res["process"][0] > 0.0
    assert all(res["process"][i] < res["process"][i + 1] for i in range(5))
    assert res["process"][-1] == pytest.approx(res["e_value"], rel=0.01)


# ── end-to-end replay ────────────────────────────────────────────────────────


def test_reality_report_correctly_specified_gbm_is_calibrated() -> None:
    report = reality_report(_features(_gbm_closes(seed=11)), _engine())
    assert report is not None
    cov = report["coverage"]
    assert cov["n"] > 150
    assert 0.68 <= cov["coverage"] <= 0.92  # ~Binomial(n, 0.8) noise band
    assert cov["pof"]["p"] >= 0.01
    assert report["pit"]["mcb"] < 0.08
    assert report["evalue"]["alarm"] is False
    assert report["evalue"]["e_value"] < 100.0
    assert len(report["fan"]["percentiles"]["50"]) == report["horizon_steps"] + 1
    assert len(report["realized"]) == report["horizon_steps"] + 1


def test_reality_report_t_engine_absorbs_fat_tails() -> None:
    """The t+EWMA engine is the Horváth–Šopov (2016) / Bollerslev (1987) fix:
    fat-tailed (t_4) data is scored with the SAME t(df) family the engine
    simulates with, so the monitor stays calibrated instead of under-covering
    forever like a Normal-scored model would."""
    report = reality_report(_features(_t_closes(seed=23)), _engine())
    assert report is not None
    cov = report["coverage"]
    assert 0.72 <= cov["coverage"] <= 0.88  # ~Binomial(n, 0.8) band
    assert report["pit"]["mcb"] < 0.06
    assert report["evalue"]["alarm"] is False


def test_reality_report_vol_regime_break_trips_monitor() -> None:
    """A structural vol break that EWMA (λ=0.94) cannot track fast enough
    inflates the bands in the low-vol regime: coverage and the PIT
    miscalibration scores (MCB/KS, Wasserstein-1 and Kolmogorov–Smirnov vs
    U(0,1)) move beyond their research-derived calibrated noise bounds:
      - one-sample KS 5% critical value  c(0.05)/sqrt(n)  with c(0.05)=1.3581
        (Miller 1956; Stephens finite-n correction) -- the band edges must
        violate it;
      - MCB null mean E[sqrt(n)*W1(Fhat,U)] -> E[int_0^1 |B(t)|dt] = 0.3133
        (Brownian bridge; del Barrio, Giné & Matrán 1999; Ramdas et al. 2017),
        so the raw MCB floor is its null mean, which the break must exceed.
    The anytime-valid e-value is NOT asserted here: as a test martingale it can
    lose evidence during long calibrated stretches (Ramdas, Grünwald, Vovk &
    Shafer 2023), so a single finite sample's crossing of 1/α is
    seed-dependent -- the stable trip signals are the coverage/PIT statistics."""
    report = reality_report(_features(_regime_closes(seed=13)), _engine())
    assert report is not None
    n_pit = report["pit"]["n"]
    assert report["coverage"]["coverage"] >= 0.79  # bands stay wide post-break
    assert report["pit"]["ks"] >= 1.3581 / math.sqrt(n_pit)  # 5% KS critical
    assert report["pit"]["mcb"] > 0.3133 / math.sqrt(n_pit)  # MCB null mean


def test_reality_report_insufficient_history_returns_none() -> None:
    assert reality_report(_features([100.0, 101.0, 102.0]), _engine()) is None


def test_reality_report_zero_variance_returns_none() -> None:
    assert reality_report(_features([100.0] * 80), _engine()) is None


def test_reality_report_uses_1step_only_and_labels_overlay() -> None:
    """Formal tests are scored on 1-step-ahead forecasts only (the overlap-free
    horizon), and the multi-step fan is served as a labeled visual overlay."""
    engine = _engine()
    closes = _gbm_closes(seed=3)
    report = reality_report(_features(closes), engine)
    assert report is not None
    # Every scored record is a 1-step forecast: warm-up consumes vol_windows
    # closes, and the last scored forecast predicts the final close.
    assert report["n_windows"] == len(closes) - 1 - engine._vol_windows
    # The fan overlay starts at the fan-issue window's actual close (step 0 is
    # the base price, always in band by construction), then shows the realized
    # path inside the issued bands.
    assert report["fan"]["base_price"] == pytest.approx(report["realized"][0]["close"], abs=1e-6)
    assert report["realized"][0]["in_band"] is True
    assert len(report["realized"]) == report["horizon_steps"] + 1
    # The realized overlay's last step is the current close (the newest window).
    assert report["realized"][-1]["close"] == closes[-1]
