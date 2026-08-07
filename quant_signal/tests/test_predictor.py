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

import random

from stream.kv import FakeKV
from stream.predictor import ConformalInterval, OnlinePredictor, prediction_key, strategy_key
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


def test_mc_engine_needs_enough_history() -> None:
    assert MonteCarloEngine(vol_windows=40).calibrate([100.0]) is None


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


def test_predictor_ignores_malformed_messages() -> None:
    kv = FakeKV()
    predictor = OnlinePredictor(kv, prediction_prefix="prediction:crypto:5m")
    predictor.handle({"symbol": "BTCUSDT"})  # no numeric close
    predictor.handle({"close": 100.0})  # no symbol
    assert kv.get_json(prediction_key("prediction:crypto:5m", "BTCUSDT")) is None


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
