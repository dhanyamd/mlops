"""Paper-execution engine tests — hermetic (no Kafka, no Redis).

Covers the fill model's correctness: no-lookahead (fills use the NEXT bar's
close, never the signal bar), slippage + taker fees on both legs, LONG/SHORT/
FLAT state transitions, deterministic replay, the trade cap, mark-to-market of
the open position, and malformed-message handling.
"""

from __future__ import annotations

from stream.execution import PaperExecutionSimulator, execution_key
from stream.kv import FakeKV
from stream.predictor import prediction_key

_WINDOW_MS = 300_000


def _window(symbol: str, i: int, close: float) -> dict:
    """Feature window ``i`` with an explicit close price."""
    return {
        "symbol": symbol,
        "window_start_ms": i * _WINDOW_MS,
        "window_end_ms": (i + 1) * _WINDOW_MS,
        "open": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "vwap": close,
        "volume": 1000.0,
        "bar_count": 5,
    }


def _seed_prediction(kv: FakeKV, symbol: str, window_end_ms: int, direction: str) -> None:
    kv.set_json(
        prediction_key("prediction:crypto:5m", symbol),
        {
            "symbol": symbol,
            "window_end_ms": window_end_ms,
            "predicted_return": 0.0,
            "interval_low": -0.01,
            "interval_high": 0.01,
            "direction": direction,
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
    )


def _simulator(kv: FakeKV, **kwargs) -> PaperExecutionSimulator:
    return PaperExecutionSimulator(
        kv,
        execution_prefix="execution:crypto:5m",
        prediction_prefix="prediction:crypto:5m",
        **kwargs,
    )


def test_no_lookahead_fill_uses_next_bar_close() -> None:
    """A signal at window 0 is filled at window 1's close, not window 0's."""
    kv = FakeKV()
    sim = _simulator(kv)
    _seed_prediction(kv, "BTCUSDT", 1 * _WINDOW_MS, "LONG")

    sim.handle(_window("BTCUSDT", 0, 100.0))  # signal bar — no fill yet
    sim.handle(_window("BTCUSDT", 1, 110.0))  # fill bar

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    position = stored["position"]
    assert position is not None
    assert position["side"] == "LONG"
    assert position["entry_price"] == round(110.0 * 1.0002, 6)  # close[1] + slippage
    assert stored["n_trades"] == 0  # nothing closed yet


def test_entry_skipped_until_next_bar_after_signal() -> None:
    """No position on the signal bar itself — fills are one bar later."""
    kv = FakeKV()
    sim = _simulator(kv)
    _seed_prediction(kv, "BTCUSDT", 1 * _WINDOW_MS, "LONG")

    sim.handle(_window("BTCUSDT", 0, 100.0))
    sim.handle(_window("BTCUSDT", 1, 110.0))
    sim.handle(_window("BTCUSDT", 2, 121.0))  # exits at close[2], short pays slip

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["n_trades"] == 1
    fill = stored["fills"][0]
    assert fill["side"] == "LONG"
    assert fill["entry_price"] == round(110.0 * 1.0002, 6)
    assert fill["exit_price"] == round(121.0 * 0.9998, 6)  # sell receives less


def test_fees_charged_on_entry_and_exit() -> None:
    kv = FakeKV()
    sim = _simulator(kv, notional_usd=1000.0, slippage_bps=0.0, taker_fee_bps=10.0)
    _seed_prediction(kv, "BTCUSDT", 1 * _WINDOW_MS, "LONG")

    sim.handle(_window("BTCUSDT", 0, 100.0))
    sim.handle(_window("BTCUSDT", 1, 100.0))
    sim.handle(_window("BTCUSDT", 2, 100.0))  # flat round trip → pure fee loss

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    fill = stored["fills"][0]
    assert fill["fees"] == round(1000.0 * 0.001 * 2, 4)  # taker fee both sides
    assert fill["gross_pnl"] == 0.0
    assert fill["net_pnl"] == round(-2.0, 4)
    assert stored["realized_pnl"] == round(-2.0, 2)
    assert stored["total_fees"] == round(2.0, 2)


def test_long_profits_from_rise_and_short_profits_from_fall() -> None:
    kv = FakeKV()
    sim = _simulator(kv, slippage_bps=0.0, taker_fee_bps=0.0)
    _seed_prediction(kv, "BTCUSDT", 1 * _WINDOW_MS, "LONG")
    sim.handle(_window("BTCUSDT", 0, 100.0))
    sim.handle(_window("BTCUSDT", 1, 100.0))
    sim.handle(_window("BTCUSDT", 2, 110.0))  # +10% → LONG wins
    long_payload = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert long_payload is not None
    assert long_payload["n_trades"] == 1
    assert long_payload["n_wins"] == 1
    assert long_payload["win_rate"] == 1.0
    assert round(long_payload["realized_pnl"], 2) == round(1000.0 * 0.1, 2)

    # SHORT: same path, position flipped — the +10% rise now loses.
    kv2 = FakeKV()
    sim2 = _simulator(kv2, slippage_bps=0.0, taker_fee_bps=0.0)
    _seed_prediction(kv2, "BTCUSDT", 1 * _WINDOW_MS, "SHORT")
    sim2.handle(_window("BTCUSDT", 0, 100.0))
    sim2.handle(_window("BTCUSDT", 1, 100.0))
    sim2.handle(_window("BTCUSDT", 2, 110.0))
    short_payload = kv2.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert short_payload is not None
    assert short_payload["n_wins"] == 0
    assert round(short_payload["realized_pnl"], 2) == round(-1000.0 * 0.1, 2)


def test_flat_signal_closes_book_without_reopening() -> None:
    """A FLAT signal closes the open position and leaves the book flat."""
    kv = FakeKV()
    sim = _simulator(kv, slippage_bps=0.0, taker_fee_bps=0.0)
    _seed_prediction(kv, "BTCUSDT", 1 * _WINDOW_MS, "LONG")
    sim.handle(_window("BTCUSDT", 0, 100.0))
    sim.handle(_window("BTCUSDT", 1, 100.0))  # fill LONG
    _seed_prediction(kv, "BTCUSDT", 2 * _WINDOW_MS, "FLAT")
    sim.handle(_window("BTCUSDT", 2, 100.0))  # exit; FLAT signal → no re-entry
    sim.handle(_window("BTCUSDT", 3, 100.0))
    sim.handle(_window("BTCUSDT", 4, 100.0))

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["n_trades"] == 1
    assert stored["position"] is None  # closed, and FLAT never reopened


def test_deterministic_replay() -> None:
    """Same windows + same store → identical payloads (no hidden randomness)."""
    closes = [100.0, 101.0, 102.0, 101.5, 103.0]

    def run() -> dict:
        kv = FakeKV()
        sim = _simulator(kv, slippage_bps=2.0, taker_fee_bps=10.0)
        _seed_prediction(kv, "ETHUSDT", 1 * _WINDOW_MS, "LONG")
        _seed_prediction(kv, "ETHUSDT", 2 * _WINDOW_MS, "SHORT")
        for i, close in enumerate(closes):
            sim.handle(_window("ETHUSDT", i, close))
        return dict(kv.get_json(execution_key("execution:crypto:5m", "ETHUSDT")))

    a = run()
    b = run()
    a.pop("updated_at")
    b.pop("updated_at")
    assert a == b
    assert a["fills"] == b["fills"]
    assert a["equity"] == b["equity"]


def test_malformed_messages_are_ignored() -> None:
    kv = FakeKV()
    sim = _simulator(kv)
    sim.handle({"symbol": "BTCUSDT"})  # no close
    sim.handle({"close": 100.0})  # no symbol
    sim.handle({"symbol": "BTCUSDT", "close": 100.0})  # no window_end_ms
    assert kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT")) is None


def test_missing_signal_skips_entry() -> None:
    """No prediction for the signal bar → the book stays flat (no stale fills)."""
    kv = FakeKV()
    sim = _simulator(kv)
    _seed_prediction(kv, "BTCUSDT", 3 * _WINDOW_MS, "LONG")  # store lags behind bar 1
    sim.handle(_window("BTCUSDT", 0, 100.0))
    sim.handle(_window("BTCUSDT", 1, 100.0))  # signal for bar 0 missing → no entry
    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["position"] is None
    assert stored["n_trades"] == 0


def test_trade_cap_halts_new_entries_but_still_closes() -> None:
    kv = FakeKV()
    sim = _simulator(kv, max_trades=2, slippage_bps=0.0, taker_fee_bps=0.0)
    _seed_prediction(kv, "BTCUSDT", 1 * _WINDOW_MS, "LONG")
    sim.handle(_window("BTCUSDT", 0, 100.0))
    sim.handle(_window("BTCUSDT", 1, 100.0))  # fill #1 LONG @100
    _seed_prediction(kv, "BTCUSDT", 2 * _WINDOW_MS, "SHORT")
    sim.handle(_window("BTCUSDT", 2, 105.0))  # close #1 (+), fill #2 SHORT @105
    _seed_prediction(kv, "BTCUSDT", 3 * _WINDOW_MS, "LONG")
    sim.handle(_window("BTCUSDT", 3, 110.0))  # close #2 (−) → cap reached
    sim.handle(_window("BTCUSDT", 4, 110.0))  # fresh signal but capped → no fill

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["n_trades"] == 2
    assert stored["position"] is None  # second position was closed, no new risk
    assert stored["fills"][0]["side"] == "SHORT"  # the capped book closed the last leg


def test_open_position_marked_to_market_each_window() -> None:
    kv = FakeKV()
    sim = _simulator(kv, slippage_bps=0.0, taker_fee_bps=0.0)
    _seed_prediction(kv, "BTCUSDT", 1 * _WINDOW_MS, "LONG")
    sim.handle(_window("BTCUSDT", 0, 100.0))
    sim.handle(_window("BTCUSDT", 1, 100.0))  # enter at 100

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["position"] is not None
    assert stored["position"]["mark_price"] == 100.0
    assert stored["position"]["unrealized_pnl"] == 0.0
