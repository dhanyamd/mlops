"""Prediction layer tests — hermetic (no Kafka, no Redis, no Snowflake).

Covers the three M3.5 pieces that matter for correctness:
  1. Adaptive Conformal Inference (Gibbs & Candès 2021) — long-run coverage
     tracks 1 - alpha on stationary data, and the level adapts.
  2. The GBM Monte Carlo engine — determinism, percentile ordering, and
     driftless terminal-value expectation E[S_T] = S_0.
  3. The online predictor — learns a synthetic trend and lands a well-formed
     prediction in the online store.
"""

from __future__ import annotations

import math
import random

import pytest

from stream.kv import FakeKV
from stream.predictor import (
    _STRESS_SHOCK_THRESHOLD,
    ConformalInterval,
    OnlinePredictor,
    _direction,
    _features,
    _own_history_features,
    prediction_key,
    strategy_key,
)
from stream.simulation import MonteCarloEngine, SimulationConsumer, simulation_key

# ── ACI conformal intervals ─────────────────────────────────────────────────


def test_conformal_coverage_tracks_target_on_stationary_data() -> None:
    conf = ConformalInterval(alpha=0.1, gamma=0.005, residual_window=200)
    y_hat = 0.0
    interval = (y_hat, y_hat)
    rng = random.Random(42)
    for _ in range(2000):
        y = rng.gauss(0.0, 1.0)
        interval = conf.predict(y_hat)
        conf.update(y, y_hat, interval)

    coverage = conf.coverage()
    assert coverage is not None
    assert 0.86 <= coverage <= 0.94  # long-run ~= 1 - alpha = 0.9


def test_conformal_alpha_t_stays_bounded() -> None:
    conf = ConformalInterval(alpha=0.1, gamma=0.005)
    rng = random.Random(7)
    for _ in range(500):
        y = rng.gauss(0.0, 1.0)
        interval = conf.predict(0.0)
        conf.update(y, 0.0, interval)
    assert 0.01 <= conf.alpha_t <= 0.99


def test_conformal_interval_narrower_when_errors_shrink() -> None:
    tight = ConformalInterval(alpha=0.1)
    wide = ConformalInterval(alpha=0.1)
    for _ in range(200):
        tight.update(0.5, 0.0, (0.0, 0.0))
        wide.update(5.0, 0.0, (0.0, 0.0))
    _, wide_high = wide.predict(0.0)
    _, tight_high = tight.predict(0.0)
    assert tight_high < wide_high


# ── GBM Monte Carlo engine ──────────────────────────────────────────────────


def _rising_closes(n: int = 50) -> list[float]:
    """Deterministic, positive price series with non-zero volatility."""
    import math

    return [100.0 * math.exp(0.001 * i + 0.01 * math.sin(i)) for i in range(n)]


def test_mc_engine_is_deterministic_with_seed() -> None:
    closes = _rising_closes()
    a = MonteCarloEngine(n_paths=500, horizon_steps=6, seed=123).forecast(closes)
    b = MonteCarloEngine(n_paths=500, horizon_steps=6, seed=123).forecast(closes)
    assert a is not None and b is not None
    assert a["percentiles"] == b["percentiles"]
    assert a["median_path"] == b["median_path"]


def test_mc_percentile_bands_monotonic_and_shaped() -> None:
    closes = _rising_closes()
    forecast = MonteCarloEngine(n_paths=1000, horizon_steps=12, seed=9).forecast(closes)
    assert forecast is not None
    steps = forecast["horizon_steps"] + 1
    for pct in ["10", "25", "50", "75", "90"]:
        assert len(forecast["percentiles"][pct]) == steps
    p10 = forecast["percentiles"]["10"]
    p50 = forecast["percentiles"]["50"]
    p90 = forecast["percentiles"]["90"]
    for t in range(steps):
        assert p10[t] <= p50[t] <= p90[t]
    assert forecast["median_path"][0] == forecast["base_price"]


def test_mc_driftless_terminal_expectation_matches_theory() -> None:
    """With mu = 0, GBM has E[S_T] = S_0 (martingale in level)."""
    closes = _rising_closes()
    engine = MonteCarloEngine(n_paths=8000, horizon_steps=12, drift=False, seed=3)
    calibrated = engine.calibrate(closes)
    assert calibrated is not None
    mu, sigma = calibrated
    assert mu == 0.0
    paths = engine.paths(closes[-1], mu, sigma)
    mean_terminal = float(paths[-1].mean())
    assert 0.98 * closes[-1] <= mean_terminal <= 1.02 * closes[-1]


def test_mc_risk_metrics_present() -> None:
    closes = _rising_closes()
    forecast = MonteCarloEngine(n_paths=2000, horizon_steps=12, seed=11).forecast(closes)
    assert forecast is not None
    assert forecast["var95"] <= 0.0  # driftless: 5th-percentile return is negative
    assert forecast["es95"] <= forecast["var95"]
    assert 0.0 <= forecast["prob_up"] <= 1.0
    assert forecast["sigma_annualized"] > 0.0
    hist = forecast["returns_histogram"]
    assert len(hist["counts"]) == len(hist["edges"]) - 1  # bins vs boundaries
    assert sum(hist["counts"]) == forecast["n_paths"]  # every path lands in a bin
    surface = forecast["surface_grid"]
    assert surface["steps"] == forecast["horizon_steps"]
    assert len(surface["counts"]) == surface["steps"]  # one row per forward step
    assert all(len(row) == len(hist["counts"]) for row in surface["counts"])
    # Every step's density uses the full path ensemble, so rows sum to n_paths.
    assert all(sum(row) == forecast["n_paths"] for row in surface["counts"])


def test_mc_sigma_scale_widens_risk_and_keeps_surface_invariant() -> None:
    """The what-if stress knob (σ×scale) widens risk metrics and the surface's
    fixed σ-based bands, while every step's density row still sums to n_paths."""
    closes = _rising_closes()
    base = MonteCarloEngine(n_paths=4000, horizon_steps=12, seed=5).forecast(closes)
    stressed = MonteCarloEngine(n_paths=4000, horizon_steps=12, seed=5).forecast(
        closes, sigma_scale=4.0, scenario={"name": "stress"}
    )
    assert base is not None and stressed is not None
    assert stressed["var95"] < base["var95"]
    assert stressed["es95"] < base["es95"]
    assert stressed["scenario"] == {"name": "stress"}
    assert base["scenario"] is None
    assert stressed["surface_grid"]["edges"][-1] > base["surface_grid"]["edges"][-1]
    for row in stressed["surface_grid"]["counts"]:
        assert sum(row) == stressed["n_paths"]


def test_mc_engine_needs_enough_history() -> None:
    assert MonteCarloEngine(vol_windows=40).calibrate([100.0]) is None


def test_mc_sobol_rqmc_sampler_drifts_up_to_coin_flip() -> None:
    """Scrambled Sobol QMC is a low-discrepancy point set: for a driftless
    martingale the terminal P(up) must sit essentially at 0.5 — far tighter
    than the O(N^-1/2) error band of crude pseudo-random MC would allow
    (Sobol 1967; Bratley & Fox 1988; Owen 1997/2003 scrambled nets). The
    scramble is seeded per engine, so the draw is reproducible."""
    closes = _rising_closes()
    engine = MonteCarloEngine(n_paths=16_384, horizon_steps=12, drift=False, seed=3)
    forecast = engine.forecast(closes)
    assert forecast is not None
    assert forecast["sampler"] == "sobol-rqmc"
    # QMC drives the sampling error of P(up) to ~1/N; 2σ of a binomial at
    # N=16k is ~0.0078, so 1% is a generous bound on the |error|.
    assert abs(forecast["prob_up"] - 0.5) < 0.01


def test_mc_crude_sampler_still_works_and_is_labeled() -> None:
    """The crude pseudo-random path remains available (A/B fallback) and is
    labeled honestly in the payload."""
    closes = _rising_closes()
    engine = MonteCarloEngine(n_paths=2000, horizon_steps=12, drift=False, seed=3, sampler="crude")
    forecast = engine.forecast(closes)
    assert forecast is not None
    assert forecast["sampler"] == "crude"
    assert 0.0 <= forecast["prob_up"] <= 1.0


def test_mc_edge_block_metrics_well_formed() -> None:
    """The decision layer over the simulated distribution is well-formed and
    honest: probabilities are proper, Kelly is capped (firms run fractional
    Kelly — Thorp 2006; MacLean, Thorp & Ziemba 2010), and a driftless engine
    (edge ≈ the Jensen drift term −σ²/2·T, tiny on 5m horizons) stays FLAT."""
    closes = _rising_closes()
    forecast = MonteCarloEngine(n_paths=4000, horizon_steps=12, seed=7).forecast(closes)
    assert forecast is not None
    e = forecast["edge"]
    assert set(e) == {
        "expected_return",
        "expected_log_return",
        "edge_bps",
        "edge_per_risk",
        "prob_up",
        "odds_up",
        "odds_down",
        "odds_ratio",
        "kelly_fraction",
        "half_kelly",
        "position",
    }
    assert 0.0 <= e["prob_up"] <= 1.0
    assert 0.0 <= e["odds_up"] <= 1.0
    assert 0.0 <= e["odds_down"] <= 1.0
    assert e["half_kelly"] == pytest.approx(e["kelly_fraction"] / 2.0, abs=0.0001)  # 4-dp rounding
    assert abs(e["kelly_fraction"]) <= 0.25 + 1e-9  # capped full-Kelly
    if e["odds_down"] > 0:
        assert e["odds_ratio"] is not None and e["odds_ratio"] > 0.0
    # Driftless (default) engine: the only edge is the tiny −σ²/2·T Jensen term,
    # far below the edge_min_sigma floor -> the honest call is FLAT.
    assert e["position"] == "FLAT"
    assert abs(e["edge_per_risk"]) < 0.05


def test_mc_edge_block_drift_gives_directional_position() -> None:
    """With the MLE drift on, a rising series produces a positive simulated
    edge and the decision layer leans LONG (clearing the edge_min_sigma floor)."""
    closes = _rising_closes()
    forecast = MonteCarloEngine(n_paths=4000, horizon_steps=12, drift=True, seed=7).forecast(closes)
    assert forecast is not None
    e = forecast["edge"]
    assert e["edge_bps"] > 0.0
    assert e["edge_per_risk"] > 0.05  # clears the LONG floor
    assert e["position"] == "LONG"
    assert e["prob_up"] > 0.5


# ── Online predictor + consumer ─────────────────────────────────────────────


def _feature_window(symbol: str, i: int) -> dict:
    """Synthetic 5m window trending upward by ~0.5% per window."""
    return {
        "symbol": symbol,
        "window_start_ms": i * 300_000,
        "window_end_ms": (i + 1) * 300_000,
        "open": 100.0 + i,
        "high": 101.0 + i,
        "low": 99.0 + i,
        "close": 100.5 + i,
        "vwap": 100.6 + i,
        "volume": 1000.0,
        "bar_count": 5,
    }


def test_predictor_learns_trend_and_writes_prediction() -> None:
    kv = FakeKV()
    predictor = OnlinePredictor(kv, prediction_prefix="prediction:crypto:5m")
    for i in range(120):
        predictor.handle(_feature_window("btcusdt", i))

    stored = kv.get_json(prediction_key("prediction:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["symbol"] == "BTCUSDT"
    assert stored["interval_low"] <= stored["interval_high"]
    assert stored["predicted_return"] > 0.0  # learned the uptrend
    assert stored["direction"] == "LONG"
    assert "coverage" in stored and "alpha" in stored


def test_direction_threshold_filters_noise() -> None:
    """Predictions below the trade threshold must stay FLAT (no fee bleed),
    and the same lean must trade once it clears the threshold."""
    assert _direction(0.0005, 0.001) == "FLAT"
    assert _direction(-0.0005, 0.001) == "FLAT"
    assert _direction(0.00005, 0.0001) == "FLAT"
    assert _direction(0.00015, 0.0001) == "LONG"  # legacy 1e-4 behavior
    assert _direction(0.0011, 0.001) == "LONG"
    assert _direction(-0.0011, 0.001) == "SHORT"


def test_predictor_ignores_malformed_messages() -> None:
    kv = FakeKV()
    predictor = OnlinePredictor(kv, prediction_prefix="prediction:crypto:5m")
    predictor.handle({"symbol": "BTCUSDT"})  # no numeric close
    predictor.handle({"close": 100.0})  # no symbol
    assert kv.get_json(prediction_key("prediction:crypto:5m", "BTCUSDT")) is None


def test_features_reject_implausible_ratio_windows() -> None:
    """Corrupt bars must never become features. Both real-world failure modes
    are caught: an ``open``/``close`` pair off by orders of magnitude
    (ret_in_window=650) and a ``close`` shifted so the open→close move is
    still implausibly large (ret_in_window=0.6, far above the 50% 1h bound)."""
    corrupt_price_units = {
        "symbol": "BTCUSDT",
        "open": 100.0,
        "high": 65211.0,
        "low": 99.0,
        "close": 65100.0,
        "vwap": 541.0,
        "volume": 1000.0,
        "bar_count": 5,
    }
    corrupt_small_close = {
        "symbol": "BTCUSDT",
        "open": 100.0,
        "high": 121.5,
        "low": 99.0,
        "close": 160.0,
        "vwap": 111.9,
        "volume": 10800.0,
        "bar_count": 9,
    }
    assert _features(corrupt_price_units) is None
    assert _features(corrupt_small_close) is None
    # A plausible window is still accepted.
    sane = {
        "symbol": "BTCUSDT",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "vwap": 100.6,
        "volume": 1000.0,
        "bar_count": 5,
    }
    assert _features(sane) is not None


def test_predictor_cross_coin_features_use_only_lagged_leader_windows() -> None:
    """The seesaw/lead-lag feature set: a follower's model sees the leaders'
    returns from windows that closed BEFORE it — never the current window."""
    kv = FakeKV()
    predictor = OnlinePredictor(
        kv,
        prediction_prefix="prediction:crypto:5m",
        cross_symbols=["BTCUSDT", "ETHUSDT"],
    )
    base = 1_786_378_500_000
    # Leaders first: BTC 100 -> 99 -> 98 (down), ETH 100 -> 101 -> 102 (up).
    # The follower's model at t uses the leaders' close-to-close returns over
    # their most recent completed window (t-1), never the current window.
    for i, (btc_close, eth_close) in enumerate([(100.0, 100.0), (99.0, 101.0)]):
        predictor.handle(
            {
                "symbol": "BTCUSDT",
                "window_end_ms": base - 600_000 + i * 300_000,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": btc_close,
                "vwap": 100.0,
                "volume": 1000.0,
                "bar_count": 5,
            }
        )
        predictor.handle(
            {
                "symbol": "ETHUSDT",
                "window_end_ms": base - 600_000 + i * 300_000,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": eth_close,
                "vwap": 100.0,
                "volume": 1000.0,
                "bar_count": 5,
            }
        )
    predictor.handle(
        {
            "symbol": "ETHUSDT",
            "window_end_ms": base - 300_000,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 101.0,
            "vwap": 100.0,
            "volume": 1000.0,
            "bar_count": 5,
        }
    )
    # Follower window at t: sees BTC's previous-bar return (100 -> 99) and
    # ETH's (100 -> 101), but NOT this window's own close as a feature source.
    predictor.handle(
        {
            "symbol": "SOLUSDT",
            "window_end_ms": base,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "vwap": 100.6,
            "volume": 1000.0,
            "bar_count": 5,
        }
    )
    sol = predictor._state("SOLUSDT")
    assert sol.last_features is not None
    assert sol.last_features["lag_btcusdt_ret"] == pytest.approx(-0.01)
    assert sol.last_features["lag_ethusdt_ret"] == pytest.approx(0.01)
    # A window with only the follower (no cross history yet) still predicts:
    # features degrade gracefully, the model is never blocked.
    predictor2 = OnlinePredictor(
        kv, prediction_prefix="prediction:crypto:5m", cross_symbols=["BTCUSDT"]
    )
    predictor2.handle(
        {
            "symbol": "SOLUSDT",
            "window_end_ms": base,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "vwap": 100.6,
            "volume": 1000.0,
            "bar_count": 5,
        }
    )
    sol2 = predictor2._state("SOLUSDT")
    assert "lag_btcusdt_ret" not in (sol2.last_features or {})
    # The leader's own model never uses its own lag as a feature.
    predictor.handle(
        {
            "symbol": "BTCUSDT",
            "window_end_ms": base,
            "open": 99.0,
            "high": 100.0,
            "low": 98.0,
            "close": 99.5,
            "vwap": 99.5,
            "volume": 1000.0,
            "bar_count": 5,
        }
    )
    btc = predictor._state("BTCUSDT")
    assert "lag_btcusdt_ret" not in (btc.last_features or {})
    assert "lag_ethusdt_ret" in (btc.last_features or {})


def test_predictor_not_poisoned_by_corrupt_feature_window() -> None:
    """A corrupt window must be dropped without corrupting the online model:
    a sane trend learned afterwards still predicts a sane magnitude."""
    kv = FakeKV()
    predictor = OnlinePredictor(
        kv,
        prediction_prefix="prediction:crypto:5m",
        strategy_prefix="strategy:crypto:5m",
    )
    corrupt = {
        "symbol": "BTCUSDT",
        "window_end_ms": 100_000,
        "open": 100.0,
        "high": 65211.0,
        "low": 99.0,
        "close": 65100.0,
        "vwap": 541.0,
        "volume": 1000.0,
        "bar_count": 5,
    }
    predictor.handle(corrupt)  # must be rejected before touching the model
    assert kv.get_json(prediction_key("prediction:crypto:5m", "BTCUSDT")) is None
    for i in range(80):
        predictor.handle(_feature_window("btcusdt", i))
    stored = kv.get_json(prediction_key("prediction:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert abs(stored["predicted_return"]) < 0.1  # sane magnitude, not 1e9
    assert -0.1 <= stored["interval_low"] <= stored["interval_high"] <= 0.1


def test_predictor_skips_learning_on_implausible_realized() -> None:
    """A corrupt close that implies a >10% 5m move must not be learned: it
    would otherwise explode the target scaler and compound equity to 1e33."""
    kv = FakeKV()
    predictor = OnlinePredictor(
        kv,
        prediction_prefix="prediction:crypto:5m",
        strategy_prefix="strategy:crypto:5m",
    )
    windows = [
        # window 0: baseline close 100
        {
            "symbol": "BTCUSDT",
            "window_end_ms": 0,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "vwap": 100.0,
            "volume": 1000.0,
            "bar_count": 5,
        },
        # window 1: corrupt close 120.8 (open 65000) -> ret_in_window=-0.998 rejected
        {
            "symbol": "BTCUSDT",
            "window_end_ms": 300_000,
            "open": 65000.0,
            "high": 65001.0,
            "low": 64999.0,
            "close": 120.8,
            "vwap": 65000.0,
            "volume": 1000.0,
            "bar_count": 5,
        },
        # window 2: sane close 65100, but realized vs window 0 = +650 -> skipped
        {
            "symbol": "BTCUSDT",
            "window_end_ms": 600_000,
            "open": 65002.0,
            "high": 65003.0,
            "low": 65001.0,
            "close": 65100.0,
            "vwap": 65002.0,
            "volume": 1000.0,
            "bar_count": 5,
        },
        # window 3: realized vs 65100 is sane -> learned
        {
            "symbol": "BTCUSDT",
            "window_end_ms": 900_000,
            "open": 65102.0,
            "high": 65103.0,
            "low": 65101.0,
            "close": 65150.0,
            "vwap": 65102.0,
            "volume": 1000.0,
            "bar_count": 5,
        },
    ]
    for w in windows:
        predictor.handle(w)
    strat = kv.get_json(strategy_key("strategy:crypto:5m", "BTCUSDT"))
    assert strat is not None
    assert strat["n_windows"] == 1  # only the single sane matured period recorded
    assert all(0.9 <= e <= 1.1 for e in strat["strategy_equity"])
    stored = kv.get_json(prediction_key("prediction:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert abs(stored["predicted_return"]) < 0.1


def test_predictor_tracks_strategy_equity_vs_buyhold() -> None:
    """Realized directions compound a strategy curve alongside buy-and-hold."""
    kv = FakeKV()
    predictor = OnlinePredictor(
        kv,
        prediction_prefix="prediction:crypto:5m",
        strategy_prefix="strategy:crypto:5m",
    )
    for i in range(60):
        predictor.handle(_feature_window("btcusdt", i))

    stored = kv.get_json(strategy_key("strategy:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["symbol"] == "BTCUSDT"
    assert stored["window_end_ms"] == 60 * 300_000  # event-time provenance tag
    assert stored["n_windows"] == 59  # 60 windows → 59 matured periods
    assert len(stored["strategy_equity"]) == len(stored["buyhold_equity"]) == 60
    # Each price step is +0.5 absolute on a ~100 base → ~+0.5% per window.
    assert stored["buyhold_equity"][-1] > 1.0
    assert stored["total_return_buyhold"] > 0.0
    assert stored["win_rate"] is None or 0.0 <= stored["win_rate"] <= 1.0
    assert stored["strategy_equity"][0] == stored["buyhold_equity"][0] == 1.0


def test_predictor_equity_curve_capped_at_maxlen() -> None:
    kv = FakeKV()
    predictor = OnlinePredictor(
        kv,
        prediction_prefix="prediction:crypto:5m",
        strategy_prefix="strategy:crypto:5m",
        strategy_maxlen=20,
    )
    for i in range(60):
        predictor.handle(_feature_window("btcusdt", i))

    stored = kv.get_json(strategy_key("strategy:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert len(stored["strategy_equity"]) == 20  # trimmed to the cap
    assert stored["n_windows"] == 59  # counter keeps the true total


def test_simulation_consumer_lands_forecast_from_feature_stream() -> None:
    kv = FakeKV()
    consumer = SimulationConsumer(
        kv,
        simulation_prefix="simulation:crypto:5m",
        n_paths=500,
        horizon_steps=6,
        seed=5,
    )
    for i in range(50):
        consumer.handle(_feature_window("ethusdt", i))

    stored = kv.get_json(simulation_key("simulation:crypto:5m", "ETHUSDT"))
    assert stored is not None
    assert stored["symbol"] == "ETHUSDT"
    assert stored["base_price"] > 0.0
    assert len(stored["percentiles"]["50"]) == stored["horizon_steps"] + 1
    assert stored["window_end_ms"] == 50 * 300_000  # event-time provenance tag
    assert "updated_at" in stored

    # QuantPad-style "all paths (N)" fan chart: subsampled raw paths shipped
    # for display, percentiles always computed from every path.
    assert "sample_paths" in stored
    assert len(stored["sample_paths"]) == 200  # default _SAMPLE_PATHS cap
    assert len(stored["sample_paths"][0]) == stored["horizon_steps"] + 1
    assert len(stored["sample_paths"][-1]) == stored["horizon_steps"] + 1


def test_warm_start_replays_stored_feature_history() -> None:
    """Warm-starting from the online store must calibrate immediately."""
    kv = FakeKV()
    history = [_feature_window("BTCUSDT", i) for i in range(30)]
    for window in history:
        kv.push_json("feature:crypto:5m:BTCUSDT", window, maxlen=200)

    predictor = OnlinePredictor(kv, prediction_prefix="prediction:crypto:5m")
    predictor.warm_start(kv.list_json("feature:crypto:5m:BTCUSDT"))
    stored = kv.get_json(prediction_key("prediction:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["symbol"] == "BTCUSDT"
    assert stored["predicted_return"] > 0.0
    assert stored["interval_low"] <= stored["interval_high"]
    assert stored["coverage"] is not None

    sim_consumer = SimulationConsumer(
        kv,
        simulation_prefix="simulation:crypto:5m",
        n_paths=500,
        horizon_steps=6,
        seed=7,
    )
    sim_consumer.warm_start(kv.list_json("feature:crypto:5m:BTCUSDT"))
    simulation = kv.get_json(simulation_key("simulation:crypto:5m", "BTCUSDT"))
    assert simulation is not None
    assert simulation["base_price"] == history[-1]["close"]


# ── Own-symbol history features (HAR-RV multi-scale vol + vol-shock) ────────


def test_own_history_features_multi_scale_vol_and_shock() -> None:
    closes = [100.0 * math.exp(0.001 * i + 0.01 * math.sin(i)) for i in range(30)]
    feat = _own_history_features(closes)
    assert "lag_ret" in feat and "rv_1h" in feat
    assert feat["lag_ret"] == pytest.approx(closes[-1] / closes[-2] - 1.0)
    assert feat["rv_1h"] == pytest.approx(abs(closes[-1] / closes[-2] - 1.0))
    assert feat["rv_4h"] > 0.0
    assert feat["rv_24h"] > 0.0
    assert 0.0 <= feat["vol_shock"] <= 10.0  # clipped against scaler poisoning


def test_own_history_features_warmup_graceful() -> None:
    assert _own_history_features([]) == {}
    assert _own_history_features([100.0]) == {}
    f2 = _own_history_features([100.0, 101.0])
    assert set(f2) == {"lag_ret", "rv_1h"}
    f4 = _own_history_features([100.0, 101.0, 102.0, 103.0, 104.0])
    assert "rv_4h" in f4 and "rv_24h" not in f4 and "vol_shock" not in f4
    f25 = _own_history_features([100.0 + i for i in range(26)])
    assert "rv_24h" in f25 and "vol_shock" in f25


def test_own_history_features_ignores_corrupt_bar() -> None:
    """A corrupt close must not leak into the multi-scale vol or the shock."""
    closes = [100.0] * 26 + [100000.0, 101000.0]
    feat = _own_history_features(closes)
    assert feat["lag_ret"] == pytest.approx(0.01)  # last sane return
    # If the 999x (100 -> 100000) return leaked in, rv_24h ~ sqrt(999^2/24) ~ 204.
    assert feat["rv_24h"] < 0.01


def test_own_history_features_prefix_stable_no_lookahead() -> None:
    """Features for a window depend only on closes strictly before it: a longer
    history changes nothing once the trailing 24h window is covered."""
    full = [100.0 * math.exp(0.002 * i + 0.01 * math.sin(i)) for i in range(40)]
    assert _own_history_features(full) == _own_history_features(full[-25:])


def test_predictor_feeds_own_history_after_warmup() -> None:
    kv = FakeKV()
    predictor = OnlinePredictor(kv, prediction_prefix="prediction:crypto:1h")
    predictor.handle(_feature_window("btcusdt", 0))
    assert "lag_ret" not in (predictor._state("BTCUSDT").last_features or {})
    for i in range(1, 30):
        predictor.handle(_feature_window("btcusdt", i))
    feat = predictor._state("BTCUSDT").last_features
    assert feat is not None
    assert "lag_ret" in feat and "rv_4h" in feat and "rv_24h" in feat
    assert feat["lag_ret"] == pytest.approx((100.5 + 28) / (100.5 + 27) - 1.0)


# ── Cross-vol spillover features + stress regime aggregates ─────────────────


def _vol_transmitter_windows(symbol: str, base: int, n: int = 30) -> list[dict]:
    """Flat ~0.1% baseline, then a 5% shock hour — a tail shock vs its day."""
    windows: list[dict] = []
    for i in range(n):
        if i == n - 1:
            close = 100.0 * 1.05
        else:
            close = 100.0 * (1.0 + 0.001 * math.sin(i))
        windows.append(
            {
                "symbol": symbol,
                "window_start_ms": base - (n + 1 - i) * 300_000,
                "window_end_ms": base - (n - i) * 300_000,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": close,
                "vwap": 100.0,
                "volume": 1000.0,
                "bar_count": 5,
            }
        )
    return windows


def test_predictor_feeds_cross_vol_and_stress_aggregates() -> None:
    kv = FakeKV()
    predictor = OnlinePredictor(
        kv,
        prediction_prefix="prediction:crypto:1h",
        vol_symbols=["SOLUSDT"],
    )
    base = 1_786_378_500_000
    for w in _vol_transmitter_windows("SOLUSDT", base):
        predictor.handle(w)
    predictor.handle(
        {
            "symbol": "BTCUSDT",
            "window_end_ms": base,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "vwap": 100.6,
            "volume": 1000.0,
            "bar_count": 5,
        }
    )
    feat = predictor._state("BTCUSDT").last_features
    assert feat is not None
    assert feat["lag_solusdt_rv24h"] > 0.0
    assert feat["lag_solusdt_vol_shock"] >= _STRESS_SHOCK_THRESHOLD
    assert feat["stress_max"] >= feat["lag_solusdt_vol_shock"]
    assert feat["stress_count"] >= 1.0
