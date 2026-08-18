"""SRP weekly execution: the book reaches the venue on the strategy's own clock.

Hermetic — FakeKV, no Redis, no venue, no caches.

The regression these guard against is specific. The hourly engine refuses a
signal whose ``window_end_ms`` does not match the bar, and refuses to open
unless a forecast magnitude clears the cost band. SRP has neither an hourly
window nor a forecast, so before this path existed it was silently ignored:
30 symbols processed, 0 positions opened, no error anywhere.
"""

from __future__ import annotations

from stream.execution import PaperExecutionSimulator, execution_key
from stream.kv import FakeKV
from stream.srp_execution import EXECUTION_PREFIX, WEEK_MS, rebalance
from stream.srp_publisher import BOOK_KEY, WEIGHTS_PREFIX, build_payload, publish

UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
WEEK_END = 1_786_924_800_000
CLOSES = {"BTCUSDT": 64_000.0, "ETHUSDT": 1_900.0, "SOLUSDT": 140.0}


def _sim(kv: FakeKV) -> PaperExecutionSimulator:
    return PaperExecutionSimulator(
        kv,
        execution_prefix=EXECUTION_PREFIX,
        prediction_prefix=WEIGHTS_PREFIX,
        notional_usd=1000.0,
        window_ms=WEEK_MS,
        venue=None,
        cost_filter_lambda=0.0,
        signal_max_stale_windows=1,
    )


def _seed(kv: FakeKV, *, scored: bool = True) -> None:
    """Publish a book plus the live marks the engine prices against."""
    import pandas as pd

    weights = pd.Series({"BTCUSDT": -0.4, "ETHUSDT": 0.4}) if scored else None
    directions = {"BTCUSDT": "SHORT", "ETHUSDT": "LONG", "SOLUSDT": "FLAT"}
    per_symbol, book = build_payload(
        directions if scored else {s: "FLAT" for s in UNIVERSE},
        weights,
        universe=UNIVERSE,
        stamp_ms=WEEK_END + 3_600_000,
        window_end_ms=WEEK_END,
        reason=None if scored else "positioning frames missing",
    )
    publish(kv, per_symbol, book)
    for sym, close in CLOSES.items():
        kv.set_json(f"live:crypto:{sym}", {"symbol": sym, "close": close})


def test_weekly_book_opens_positions_matching_the_published_sides() -> None:
    kv = FakeKV()
    _seed(kv)
    out = rebalance(_sim(kv), kv, UNIVERSE)
    assert out["window_end_ms"] == WEEK_END

    btc = kv.get_json(execution_key(EXECUTION_PREFIX, "BTCUSDT"))
    eth = kv.get_json(execution_key(EXECUTION_PREFIX, "ETHUSDT"))
    assert btc["position"] is not None, "short signal did not open a position"
    assert btc["position"]["side"] == "SHORT"
    assert eth["position"]["side"] == "LONG"

    # FLAT must not open anything, in either direction.
    sol = kv.get_json(execution_key(EXECUTION_PREFIX, "SOLUSDT"))
    assert sol is None or sol.get("position") is None


def test_entry_survives_having_no_forecast() -> None:
    """A weight-driven signal carries no ``predicted_return``.

    With ``cost_filter_lambda=0`` the forecast band is disabled, so the absence
    of a forecast must not block the entry. Requiring a number here is what
    silently refused every SRP signal.
    """
    kv = FakeKV()
    _seed(kv)
    record = kv.get_json(f"{WEIGHTS_PREFIX}:BTCUSDT")
    assert "predicted_return" not in record

    rebalance(_sim(kv), kv, UNIVERSE)
    assert kv.get_json(execution_key(EXECUTION_PREFIX, "BTCUSDT"))["position"] is not None


def test_unscored_book_is_held_not_traded() -> None:
    """A FLAT book means "could not score", not "go flat".

    Liquidating a book because positioning data was briefly missing would turn a
    data outage into a realised loss.
    """
    kv = FakeKV()
    _seed(kv, scored=False)
    out = rebalance(_sim(kv), kv, UNIVERSE)
    assert out["acted"] == 0
    assert "unscored" in out["reason"]
    for sym in UNIVERSE:
        led = kv.get_json(execution_key(EXECUTION_PREFIX, sym))
        assert led is None or led.get("position") is None


def test_missing_book_is_a_no_op() -> None:
    kv = FakeKV()
    out = rebalance(_sim(kv), kv, UNIVERSE)
    assert out == {"acted": 0, "reason": "no book"}


def test_book_without_a_weekly_window_is_refused() -> None:
    """No ``window_end_ms`` means the clock is unknown; trading blind is worse
    than holding."""
    kv = FakeKV()
    _seed(kv)
    book = kv.get_json(BOOK_KEY)
    book["window_end_ms"] = None
    kv.set_json(BOOK_KEY, book)
    out = rebalance(_sim(kv), kv, UNIVERSE)
    assert out["acted"] == 0
    assert out["reason"] == "no window_end_ms"
