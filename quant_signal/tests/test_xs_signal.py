"""Tests for the live xs_rel14 cross-sectional momentum signal.

Hermetic: synthetic hourly closes, no network/Snowflake. Verifies the signal
ranks the universe correctly (top quintile LONG, bottom SHORT, mid FLAT), tags
predictions with the matching ``window_end_ms``, and fails CLOSED (all FLAT)
when the regime gate lacks history.
"""

from __future__ import annotations

import math

from stream.kv import FakeKV
from stream.xs_signal import XsSignal, prediction_key

_HOUR_MS = 3_600_000


def _build_signal(universe, **kw):
    kv = FakeKV()
    signal = XsSignal(
        kv=kv,
        prediction_prefix="prediction:crypto:1h",
        universe=universe,
        lookback_h=14,
        quintile=0.2,
        min_symbols=4,
        volume_frac=0.5,
        volume_median_bars=5,
        rebalance_h=1,
        vol_scale=True,
        crash=None,
        market_symbol="BTCUSDT",
        max_history=1500,
        **kw,
    )
    return signal, kv


def _feed(signal: XsSignal, universe, drifts, hours, start_hour=0):
    """Feed ``hours`` of 1h windows for every symbol with a per-symbol drift."""
    for h in range(start_hour, start_hour + hours):
        end = (h + 1) * _HOUR_MS
        for s, drift in zip(universe, drifts):
            close = 100.0 * (1.0 + drift) ** h
            signal.handle({"symbol": s, "window_end_ms": end, "close": close, "volume": 1000.0})


def test_xs_selection_long_short_flat():
    # 10 names: 2 strong uptrends, 2 strong downtrends, 6 flat. BTC is market.
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
    signal, kv = _build_signal(universe, regime=False)
    # Feed 25 hours so lookback_h=14 is satisfied; rebalance_h=1 rebalances each hour.
    _feed(signal, universe, drifts, hours=25)
    end = 25 * _HOUR_MS

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


def test_xs_regime_fails_closed_without_history():
    # regime=True but only 30h of BTC history (< 2*672+2) -> whole book FLAT.
    universe = ["BTCUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"]
    drifts = [0.01, 0.01, 0.01, 0.01, 0.01]
    signal, kv = _build_signal(universe, regime=True)
    _feed(signal, universe, drifts, hours=30)
    for s in universe:
        pred = kv.get_json(prediction_key("prediction:crypto:1h", s))
        assert pred["direction"] == "FLAT", (s, pred)


def test_xs_volume_gate_excludes_thin_symbols():
    # A symbol trading below 0.5x its own trailing median volume is excluded.
    universe = [
        "BTCUSDT",
        "HOTUSDT",
        "WINUSDT",
        "LOSUSDT",
        "M1USDT",
        "M2USDT",
        "M3USDT",
        "M4USDT",
        "M5USDT",
        "M6USDT",
    ]
    # HOTUSDT trends up but its volume collapses below the gate in the last bar.
    drifts = [0.001, 0.01, 0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    signal, kv = _build_signal(universe, regime=False)
    for h in range(25):
        end = (h + 1) * _HOUR_MS
        for s, drift in zip(universe, drifts):
            # Thin bar at h=23: the close used at the rebalance for W=25H is the
            # prior completed hour (side="left"), so the gate sees the collapse.
            vol = 1000.0 if s != "HOTUSDT" else (10.0 if h == 23 else 1000.0)
            close = 100.0 * (1.0 + drift) ** h
            signal.handle({"symbol": s, "window_end_ms": end, "close": close, "volume": vol})
    hot = kv.get_json(prediction_key("prediction:crypto:1h", "HOTUSDT"))
    # With only 1 thin bar it still has a 28-ish median of 1000 -> gate fails -> FLAT.
    assert hot["direction"] == "FLAT", hot
