"""Tests for the live SCX cross-sectional signal.

Hermetic: synthetic hourly closes, no network/Snowflake. Verifies the signal
ranks the universe correctly (top quintile LONG, bottom SHORT, mid FLAT), tags
predictions with the matching ``window_end_ms``, fails CLOSED (all FLAT) when the
regime gate lacks history, and -- the SCX novelty -- drops the SHORT book for a
long-only book when the trailing short-book realized vol is in its stressed
quantile (and keeps shorts when the short side is calm).
"""

from __future__ import annotations

import math

from stream.kv import FakeKV
from stream.scx_signal import ScxSignal, prediction_key

_HOUR_MS = 3_600_000


def _build_signal(universe, **kw):
    kv = FakeKV()
    signal = ScxSignal(
        kv=kv,
        prediction_prefix="prediction:crypto:1h",
        universe=universe,
        lookback_h=14,
        quintile=0.2,
        min_symbols=4,
        rebalance_h=1,
        regime=True,
        regime_fast_days=2,
        regime_slow_days=3,
        shorts=True,
        cond_short=True,
        short_vol_l=4,
        stress_q=0.60,
        market_symbol="BTCUSDT",
        max_history=1500,
        **kw,
    )
    return signal, kv


def _feed(signal: ScxSignal, universe, drifts, hours, start_hour=0, regime_up=True):
    """Feed ``hours`` of 1h windows for every symbol with a per-symbol drift.

    ``regime_up`` drives the BTC trend so the 90/200-day UP-UP gate passes (True)
    or fails (False) without needing 200 real days of history.
    """
    for h in range(start_hour, start_hour + hours):
        end = (h + 1) * _HOUR_MS
        for s, drift in zip(universe, drifts):
            # BTC sets the regime: monotone up (pass) or down (fail).
            d = drift if s != "BTCUSDT" else (0.01 if regime_up else -0.01)
            close = 100.0 * (1.0 + d) ** h
            signal.handle({"symbol": s, "window_end_ms": end, "close": close, "volume": 1000.0})


def test_scx_selection_long_short_flat():
    # 10 names: 2 strong uptrends, 2 strong downtrends, 6 flat. BTC is market (up).
    universe = [
        "BTCUSDT",
        "WIN1USDT",
        "WIN2USDT",
        "LOS1USDT",
        "LOS2USDT",
        "FLAT1USDT",
        "FLAT2USDT",
        "FLAT3USDT",
        "FLAT4USDT",
        "FLAT5USDT",
    ]
    drifts = [0.001, 0.01, 0.01, -0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0]
    signal, kv = _build_signal(universe)
    _feed(signal, universe, drifts, hours=100)
    end = 100 * _HOUR_MS

    win1 = kv.get_json(prediction_key("prediction:crypto:1h", "WIN1USDT"))
    win2 = kv.get_json(prediction_key("prediction:crypto:1h", "WIN2USDT"))
    los1 = kv.get_json(prediction_key("prediction:crypto:1h", "LOS1USDT"))
    los2 = kv.get_json(prediction_key("prediction:crypto:1h", "LOS2USDT"))
    flat1 = kv.get_json(prediction_key("prediction:crypto:1h", "FLAT1USDT"))

    assert win1["direction"] == "LONG", win1
    assert win2["direction"] == "LONG", win2
    assert los1["direction"] == "SHORT", los1
    assert los2["direction"] == "SHORT", los2
    assert flat1["direction"] == "FLAT", flat1
    # Every prediction is tagged with the bar it was made on (executor match key).
    assert win1["window_end_ms"] == end
    assert los1["window_end_ms"] == end
    # Top/bottom quintile signals are large vs the 20 bps execution entry band.
    assert abs(win1["predicted_return"]) > 0.01
    assert abs(los1["predicted_return"]) > 0.01
    assert win1["signal"] == "scx"


def test_scx_regime_fails_closed_without_history():
    # regime=True but BTC downtrend -> UP-UP gate fails -> whole book FLAT.
    universe = ["BTCUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"]
    drifts = [0.01, 0.01, 0.01, 0.01, 0.01]
    signal, kv = _build_signal(universe)
    _feed(signal, universe, drifts, hours=30, regime_up=False)
    for s in universe:
        pred = kv.get_json(prediction_key("prediction:crypto:1h", s))
        assert pred["direction"] == "FLAT", (s, pred)


def test_scx_conditional_short_drops_shorts_when_stressed():
    # Seed a stressed short-book history: the latest rolling std is well above
    # the 60th-percentile threshold -> shorts must be dropped (long-only).
    universe = ["BTCUSDT", "W1USDT", "W2USDT", "L1USDT", "L2USDT", "F1USDT", "F2USDT"]
    drifts = [0.001, 0.01, 0.01, -0.01, -0.01, 0.0, 0.0]
    signal, kv = _build_signal(universe)
    # Build history so the regime gate passes and momentum is known.
    _feed(signal, universe, drifts, hours=100)
    # Calm weeks, then a violent short-book spike at the end -> last rolling std
    # >> 60th pct of the rolling-4w std series -> stress.
    signal._sb_hist = [
        0.01,
        0.02,
        -0.01,
        0.015,
        0.005,
        -0.02,
        0.03,
        -0.01,
        0.02,
        0.01,
        -0.015,
        -0.90,
    ]
    out = signal._selection(100 * _HOUR_MS)
    # Short book is stressed -> shorts dropped, longs (winners) remain.
    assert out["L1USDT"][0] == "FLAT", out["L1USDT"]
    assert out["W1USDT"][0] == "LONG", out["W1USDT"]


def test_scx_conditional_short_keeps_shorts_when_calm():
    # Calm, effectively-zero-vol short-book history -> shorts stay on.
    universe = ["BTCUSDT", "W1USDT", "W2USDT", "L1USDT", "L2USDT", "F1USDT", "F2USDT"]
    drifts = [0.001, 0.01, 0.01, -0.01, -0.01, 0.0, 0.0]
    signal, kv = _build_signal(universe)
    _feed(signal, universe, drifts, hours=100)
    # Identical calm returns -> no tail to speak of -> not stressed.
    signal._sb_hist = [0.01] * 12
    out = signal._selection(100 * _HOUR_MS)
    assert out["L1USDT"][0] == "SHORT", out["L1USDT"]
