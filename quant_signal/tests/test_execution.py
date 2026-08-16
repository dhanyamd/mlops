"""Paper-execution engine tests — hermetic (no Kafka, no Redis).

Covers the fill model's correctness: no-lookahead (fills use the NEXT bar's
close, never the signal bar), slippage + taker fees on both legs, LONG/SHORT/
FLAT state transitions, deterministic replay, the trade cap, mark-to-market of
the open position, malformed-message handling — and the cost-aware filter:
entries clear the λ·c band, positions HOLD through weak signals, and flips
require |r̂| past the 2·λ·c band (Bysik & Ślepaczuk 2026, eq. 5).

The fill-model tests below run with ``hold_until_decay=False`` (the benchmark
cadence) so they isolate the fill math; the cost-filter tests at the end use
the default ``hold_until_decay=True``.
"""

from __future__ import annotations

import pytest

from stream import bybit_demo
from stream.bybit_demo import BybitDemoVenue
from stream.execution import PaperExecutionSimulator, _build_venue, execution_key
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


def _seed_prediction(
    kv: FakeKV,
    symbol: str,
    window_end_ms: int,
    direction: str,
    predicted_return: float = 0.0,
) -> None:
    kv.set_json(
        prediction_key("prediction:crypto:5m", symbol),
        {
            "symbol": symbol,
            "window_end_ms": window_end_ms,
            "predicted_return": predicted_return,
            "interval_low": -0.01,
            "interval_high": 0.01,
            "direction": direction,
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
    )


def _simulator(kv: FakeKV, **kwargs) -> PaperExecutionSimulator:
    # Legacy cadence for the fill-model tests (they isolate the fill math, not
    # the position-management policy). The cost-filter tests opt back in with
    # hold_until_decay=True explicitly. window_ms must match the test bars
    # (the engine's module default is the 1h trading clock).
    kwargs.setdefault("hold_until_decay", False)
    kwargs.setdefault("window_ms", _WINDOW_MS)
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


# ── execution venue (Bybit Demo) ───────────────────────────────────────────


class _FakeVenue:
    """Stands in for ``BybitDemoVenue``: returns fixed fills, no network."""

    def __init__(self, open_fill: dict | None = None, close_fill: dict | None = None) -> None:
        self.open_fill = open_fill or {"fill_price": 110.5, "qty": 9.05, "fees": 0.31}
        self.close_fill = close_fill or {"fill_price": 121.4, "qty": 9.05, "fees": 0.42}
        self.opened: list[tuple[str, str]] = []
        self.closed: list[tuple[str, str]] = []

    def open_market(self, symbol: str, side: str, notional_usd: float) -> dict | None:
        self.opened.append((symbol, side))
        return dict(self.open_fill)

    def close_market(self, symbol: str, side: str, qty: float) -> dict | None:
        self.closed.append((symbol, side))
        return dict(self.close_fill)


def test_paper_venue_is_labeled_paper() -> None:
    kv = FakeKV()
    sim = _simulator(kv)
    _seed_prediction(kv, "BTCUSDT", 1 * _WINDOW_MS, "LONG")
    sim.handle(_window("BTCUSDT", 0, 100.0))
    sim.handle(_window("BTCUSDT", 1, 110.0))
    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["venue"] == "paper"
    assert stored["orders_rejected"] == 0


def test_demo_venue_records_actual_fill_prices_and_fees() -> None:
    """With a venue, entry/exit prices, qty and fees come from the venue's
    order response verbatim — not the fixed-bps paper model."""
    kv = FakeKV()
    venue = _FakeVenue()
    sim = _simulator(kv, venue=venue)
    _seed_prediction(kv, "BTCUSDT", 1 * _WINDOW_MS, "LONG")

    sim.handle(_window("BTCUSDT", 0, 100.0))  # signal bar — no fill
    sim.handle(_window("BTCUSDT", 1, 110.0))  # venue market order @110.5
    sim.handle(_window("BTCUSDT", 2, 121.0))  # venue reduce-only close @121.4

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["venue"] == "bybit-demo"
    assert stored["position"] is None
    assert venue.opened == [("BTCUSDT", "LONG")]
    assert venue.closed == [("BTCUSDT", "LONG")]
    fill = stored["fills"][0]
    assert fill["entry_price"] == round(110.5, 6)  # venue avgPrice
    assert fill["exit_price"] == round(121.4, 6)  # venue avgPrice
    assert fill["qty"] == round(9.05, 8)  # venue cumExecQty
    assert fill["fees"] == round(0.31 + 0.42, 4)  # venue cumExecFee both legs
    assert fill["gross_pnl"] == round(9.05 * (121.4 - 110.5), 4)
    assert stored["assumptions"]["venue"] == "bybit-demo"


def test_demo_venue_rejected_entry_is_skipped_and_counted() -> None:
    """A failed venue order is never faked with a paper fill — the entry is
    skipped and counted under orders_rejected."""
    kv = FakeKV()

    class _RejectingVenue:
        def open_market(self, symbol: str, side: str, notional_usd: float) -> None:
            return None

        def close_market(self, symbol: str, side: str, qty: float) -> None:
            return None

    sim = _simulator(kv, venue=_RejectingVenue())
    _seed_prediction(kv, "BTCUSDT", 1 * _WINDOW_MS, "LONG")
    sim.handle(_window("BTCUSDT", 0, 100.0))
    sim.handle(_window("BTCUSDT", 1, 110.0))

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["position"] is None
    assert stored["orders_rejected"] == 1
    assert stored["n_trades"] == 0


def test_demo_close_unfilled_keeps_position_and_retries() -> None:
    """An unfilled close leaves the position open; the next window retries the
    close (the venue reports the actual exit price) and bars_held reflects the
    extra window."""
    kv = FakeKV()

    class _UnfilledCloseVenue:
        def __init__(self) -> None:
            self.close_calls = 0

        def open_market(self, symbol: str, side: str, notional_usd: float) -> dict:
            return {"fill_price": 110.0, "qty": 9.0, "fees": 0.2}

        def close_market(self, symbol: str, side: str, qty: float) -> dict | None:
            self.close_calls += 1
            if self.close_calls == 1:
                return None
            return {"fill_price": 120.0, "qty": 9.0, "fees": 0.3}

    sim = _simulator(kv, venue=_UnfilledCloseVenue())
    _seed_prediction(kv, "BTCUSDT", 1 * _WINDOW_MS, "LONG")
    sim.handle(_window("BTCUSDT", 0, 100.0))
    sim.handle(_window("BTCUSDT", 1, 110.0))  # entry @110.0 (venue)
    sim.handle(_window("BTCUSDT", 2, 120.0))  # close unfilled → position stays

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["position"] is not None  # still open
    assert stored["orders_rejected"] == 1
    assert stored["n_trades"] == 0

    sim.handle(_window("BTCUSDT", 3, 125.0))  # close fills @120.0
    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["position"] is None
    assert stored["n_trades"] == 1
    fill = stored["fills"][0]
    assert fill["entry_price"] == round(110.0, 6)
    assert fill["exit_price"] == round(120.0, 6)
    assert fill["bars_held"] == 2  # entry window 1 → closed window 3


def test_execution_venue_settings_default_to_paper() -> None:
    """Out of the box the venue is paper and the demo keys are unset — the
    honest, no-key default. Built without the local ``.env`` so a developer's
    live demo credentials in that file can't leak into the assertion."""
    from config.settings import Settings

    settings = Settings(
        _env_file=None,
        snowflake_account="test",
        snowflake_user="test",
        snowflake_password="test",
    )
    assert settings.stream_execution_venue == "paper"
    assert settings.bybit_demo_api_key is None
    assert settings.bybit_demo_api_secret is None


def test_bybit_demo_credential_aliases_are_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """API_KEY_BYBIT / API_SECRET_BYBIT (the names some earlier Bybit bots
    export) count as demo credentials exactly like BYBIT_DEMO_API_KEY/SECRET."""
    from config.settings import get_settings

    monkeypatch.delenv("BYBIT_DEMO_API_KEY", raising=False)
    monkeypatch.delenv("BYBIT_DEMO_API_SECRET", raising=False)
    monkeypatch.setenv("API_KEY_BYBIT", "alias-key")
    monkeypatch.setenv("API_SECRET_BYBIT", "alias-secret")
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert s.has_bybit_demo_credentials is True
        assert s.demo_api_key == "alias-key"
        assert s.demo_api_secret == "alias-secret"
    finally:
        get_settings.cache_clear()


class _StubSettings:
    """Minimal settings stub — only the fields ``_build_venue`` reads."""

    def __init__(
        self,
        *,
        has_creds: bool = True,
        venue_setting: str = "paper",
        api_key: str = "k",
        api_secret: str = "s",
        recv_window_ms: int = 5000,
    ) -> None:
        self.has_bybit_demo_credentials = has_creds
        self.stream_execution_venue = venue_setting
        self.demo_api_key = api_key
        self.demo_api_secret = api_secret
        self.bybit_demo_recv_window_ms = recv_window_ms
        self.stream_execution_maker_first = True
        self.stream_execution_maker_first_exit = False


def test_build_venue_returns_none_without_creds_and_paper_setting() -> None:
    assert _build_venue(_StubSettings(has_creds=False)) is None


def test_build_venue_warns_and_returns_none_when_demo_expected_but_no_creds(
    capsys,
) -> None:
    venue = _build_venue(_StubSettings(has_creds=False, venue_setting="bybit-demo"))
    assert venue is None
    assert "no demo keys found" in capsys.readouterr().out


def test_build_venue_returns_venue_when_demo_balance_ok() -> None:
    class _OkVenue:
        def __init__(self, *args, **kwargs) -> None:
            self.balance = lambda: 50_000.0

    venue = _build_venue(_StubSettings(venue_setting="bybit-demo"), venue_cls=_OkVenue)
    assert isinstance(venue, _OkVenue)
    assert venue.balance() == 50_000.0


def test_build_venue_falls_back_to_paper_when_balance_is_none(capsys) -> None:
    """ErrCode 10003 (key/domain mismatch) surfaces as balance()==None —
    the consumer must degrade to paper, never run a dead demo venue."""

    class _RejectedKeyVenue:
        def __init__(self, *args, **kwargs) -> None:
            self.balance = lambda: None

    venue = _build_venue(_StubSettings(venue_setting="bybit-demo"), venue_cls=_RejectedKeyVenue)
    assert venue is None
    assert "falling back to paper fills" in capsys.readouterr().out


def test_build_venue_falls_back_to_paper_when_construction_raises(capsys) -> None:
    class _ExplodingVenue:
        def __init__(self, *args, **kwargs) -> None:
            raise ConnectionError("api-demo.bybit.com unreachable")

    venue = _build_venue(_StubSettings(venue_setting="bybit-demo"), venue_cls=_ExplodingVenue)
    assert venue is None
    assert "could not be initialized" in capsys.readouterr().out


def test_build_venue_explicit_paper_setting_skips_demo_even_with_creds() -> None:
    """STREAM_EXECUTION_VENUE=paper must be a real, load-bearing opt-out —
    not cosmetic — even when valid demo credentials are present."""

    class _ShouldNeverBeConstructed:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("paper mode must not construct the demo venue")

    venue = _build_venue(
        _StubSettings(venue_setting="paper", has_creds=True),
        venue_cls=_ShouldNeverBeConstructed,
    )
    assert venue is None


# ── maker-first fills (post-only limit → market fallback) ──────────────────


class _FakeHTTP:
    """Records place_order calls and lets tests script the fill status."""

    def __init__(self, status: str = "New") -> None:
        self.orders: list[dict] = []
        self.cancelled: list[str] = []
        self.status = status
        self.book = {"retCode": 0, "result": {"b": [["100.0", "1"]], "a": [["102.0", "1"]]}}

    def get_orderbook(self, **kw) -> dict:
        return self.book

    def place_order(self, **kw) -> dict:
        self.orders.append(kw)
        return {"retCode": 0, "result": {"orderId": f"o{len(self.orders)}"}}

    def get_order_history(self, **kw) -> dict:
        if self.status == "Filled":
            return {
                "retCode": 0,
                "result": {
                    "list": [
                        {
                            "orderStatus": "Filled",
                            "avgPrice": "100.5",
                            "cumExecQty": "0.01",
                            "cumFeeDetail": {"USDT": "0.02"},
                        }
                    ]
                },
            }
        return {"retCode": 0, "result": {"list": [{"orderStatus": self.status}]}}

    def cancel_order(self, **kw) -> dict:
        self.cancelled.append(str(kw.get("orderId") or ""))
        return {"retCode": 0, "result": {}}


def _fast_venue(http: _FakeHTTP, maker_first: bool = True) -> BybitDemoVenue:
    """A BybitDemoVenue with its network bound to ``http``, fast polls."""
    venue = BybitDemoVenue.__new__(BybitDemoVenue)
    venue._http = http  # type: ignore[attr-defined]
    venue._maker_first = maker_first  # type: ignore[attr-defined]
    # Mirror the constructor: the exit leg inherits the entry setting unless a
    # test overrides it explicitly.
    venue._maker_first_exit = maker_first  # type: ignore[attr-defined]
    venue._qty_for_notional = lambda symbol, notional_usd: 0.01  # type: ignore[method-assign]
    monkey = pytest.MonkeyPatch()
    monkey.setattr(bybit_demo, "_FILL_POLL_INTERVAL_S", 0.0)
    return venue


def test_maker_first_places_postonly_limit_at_bid_then_market_fallback() -> None:
    """LONG entry: a resting post-only buy at the bid first; when it never
    fills the order is cancelled and a market order takes over."""
    http = _FakeHTTP(status="New")
    venue = _fast_venue(http)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(bybit_demo, "_MAKER_FILL_POLLS", 2)

    fill = venue.open_market("BTCUSDT", "LONG", 1000)

    assert fill is None  # nothing filled, market fallback also not filled here
    limit, market = http.orders[0], http.orders[1]
    assert limit["orderType"] == "Limit"
    assert limit["timeInForce"] == "PostOnly"
    assert limit["price"] == "100.0"  # the bid, not the ask — maker pricing
    assert limit["reduceOnly"] is False
    assert market["orderType"] == "Market"
    assert len(http.cancelled) == 1  # the unfilled limit was cancelled


def test_maker_first_returns_maker_fill_without_market_fallback() -> None:
    """A limit that fills on the book returns the maker fill (lower fee) and
    never degrades to a market order."""
    http = _FakeHTTP(status="Filled")
    venue = _fast_venue(http)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(bybit_demo, "_MAKER_FILL_POLLS", 2)

    fill = venue.open_market("BTCUSDT", "LONG", 1000)

    assert fill is not None
    assert fill["fill_price"] == 100.5
    assert fill["fees"] == pytest.approx(0.02, abs=1e-9)
    assert [o["orderType"] for o in http.orders] == ["Limit"]  # no fallback
    assert http.cancelled == []


def test_maker_first_close_uses_reduce_only_postonly() -> None:
    """LONG exit sells at the bid as a reduce-only post-only limit."""
    http = _FakeHTTP(status="Filled")
    venue = _fast_venue(http)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(bybit_demo, "_MAKER_FILL_POLLS", 2)

    venue.close_market("BTCUSDT", "LONG", 0.01)

    assert http.orders[0]["orderType"] == "Limit"
    assert http.orders[0]["timeInForce"] == "PostOnly"
    assert http.orders[0]["price"] == "100.0"
    assert http.orders[0]["reduceOnly"] is True


def test_maker_first_disabled_goes_straight_to_market() -> None:
    """maker_first=False preserves the old pure-market behavior."""
    http = _FakeHTTP(status="Filled")
    venue = _fast_venue(http, maker_first=False)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(bybit_demo, "_MAKER_FILL_POLLS", 2)

    venue.open_market("BTCUSDT", "LONG", 1000)

    assert [o["orderType"] for o in http.orders] == ["Market"]


# ── cost-aware filter (Bysik & Ślepaczuk 2026, eq. 5) ───────────────────────

# Defaults used here: taker_fee_bps=10 → c=0.001; λ=2 → entry band 0.002
# (20 bps), flip band 0.004 (40 bps). ``hold_until_decay`` defaults True.


def _cost_simulator(kv: FakeKV, **kwargs) -> PaperExecutionSimulator:
    # window_ms must match the test bars (the engine's module default is the
    # 1h trading clock).
    kwargs.setdefault("window_ms", _WINDOW_MS)
    return PaperExecutionSimulator(
        kv,
        execution_prefix="execution:crypto:5m",
        prediction_prefix="prediction:crypto:5m",
        **kwargs,
    )


def test_cost_filter_blocks_entries_below_lambda_times_cost() -> None:
    """|r̂| = 10 bps < λ·c = 20 bps → the LONG signal is too weak to enter."""
    kv = FakeKV()
    sim = _cost_simulator(kv)
    _seed_prediction(kv, "BTCUSDT", 2 * _WINDOW_MS, "LONG", predicted_return=0.001)

    sim.handle(_window("BTCUSDT", 0, 100.0))
    sim.handle(_window("BTCUSDT", 1, 100.0))  # bar 1 closes with the weak forecast

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["position"] is None
    assert stored["n_trades"] == 0


def test_cost_filter_enters_when_forecast_clears_the_band() -> None:
    """|r̂| = 30 bps > 20 bps → the LONG signal enters at the signal bar close."""
    kv = FakeKV()
    sim = _cost_simulator(kv)
    _seed_prediction(kv, "BTCUSDT", 2 * _WINDOW_MS, "LONG", predicted_return=0.003)

    sim.handle(_window("BTCUSDT", 0, 100.0))
    sim.handle(_window("BTCUSDT", 1, 100.0))

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["position"]["side"] == "LONG"
    assert stored["entry_threshold_bps"] == 20.0
    assert stored["flip_threshold_bps"] == 40.0


def test_hold_until_decay_keeps_position_through_weak_signal() -> None:
    """A strong entry is NOT closed when the next signal decays below the
    entry band — the no-trade region of Gârleanu & Pedersen (2013) / banding
    (Novy-Marx & Velikov 2016)."""
    kv = FakeKV()
    sim = _cost_simulator(kv)
    _seed_prediction(kv, "BTCUSDT", 2 * _WINDOW_MS, "LONG", predicted_return=0.003)
    sim.handle(_window("BTCUSDT", 0, 100.0))
    sim.handle(_window("BTCUSDT", 1, 100.0))  # enter LONG
    _seed_prediction(kv, "BTCUSDT", 3 * _WINDOW_MS, "LONG", predicted_return=0.001)  # decay
    sim.handle(_window("BTCUSDT", 2, 100.0))

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["position"]["side"] == "LONG"  # held
    assert stored["n_trades"] == 0


def test_hold_until_decay_flips_on_strong_reversal() -> None:
    """A reversal past the 2·λ·c band closes the leg and opens the opposite
    side in the same bar (turnover 2)."""
    kv = FakeKV()
    sim = _cost_simulator(kv)
    _seed_prediction(kv, "BTCUSDT", 2 * _WINDOW_MS, "LONG", predicted_return=0.003)
    sim.handle(_window("BTCUSDT", 0, 100.0))
    sim.handle(_window("BTCUSDT", 1, 100.0))  # enter LONG
    _seed_prediction(kv, "BTCUSDT", 3 * _WINDOW_MS, "SHORT", predicted_return=-0.005)
    sim.handle(_window("BTCUSDT", 2, 100.0))  # |−0.005| > 0.004 → flip

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["n_trades"] == 1
    assert stored["position"]["side"] == "SHORT"
    assert stored["fills"][0]["side"] == "LONG"


def test_hold_until_decay_ignores_weak_reversal() -> None:
    """|r̂| = 30 bps < 40 bps flip band → a weak SHORT signal does NOT flip the
    LONG position (paying 2× turnover for no edge)."""
    kv = FakeKV()
    sim = _cost_simulator(kv)
    _seed_prediction(kv, "BTCUSDT", 2 * _WINDOW_MS, "LONG", predicted_return=0.003)
    sim.handle(_window("BTCUSDT", 0, 100.0))
    sim.handle(_window("BTCUSDT", 1, 100.0))  # enter LONG
    _seed_prediction(kv, "BTCUSDT", 3 * _WINDOW_MS, "SHORT", predicted_return=-0.003)
    sim.handle(_window("BTCUSDT", 2, 100.0))

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["position"]["side"] == "LONG"  # held
    assert stored["n_trades"] == 0


def test_unfilled_close_is_not_recorded_as_a_trade_while_position_stays_open() -> None:
    """A reduce-only exit that never fills must NOT be reported as a fill.

    Regression: the position-based ground-truth fallback was written for
    ENTRIES (position qty > 0 ⇒ filled) but also ran for EXITS, where a
    still-open position means the close FAILED. That inversion recorded
    failed closes as completed round trips with fabricated P&L while the
    position remained open on the exchange. Bybit v5 docs: confirm an exit by
    verifying via /v5/position/list that the position was closed or reduced.
    """
    http = _FakeHTTP(status="New")  # order never reaches Filled
    venue = _fast_venue(http, maker_first=False)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(bybit_demo, "_FILL_POLLS", 2)
    # The venue still reports an OPEN position — i.e. the close did not go through.
    venue.position = lambda symbol: {  # type: ignore[method-assign]
        "side": "LONG",
        "qty": 0.01,
        "entry_price": 100.0,
        "unrealized_pnl": 22.60,
    }

    assert venue.close_market("BTCUSDT", "LONG", 0.01) is None


def test_close_confirmed_by_flat_position_reports_fill_with_real_fees() -> None:
    """The mirror case: position went flat ⇒ the exit really did execute.

    Fees must come from the order-history record, never from a position's
    unrealised P&L (a different quantity entirely).
    """
    http = _FakeHTTP(status="Filled")
    venue = _fast_venue(http, maker_first=False)
    venue.position = lambda symbol: None  # type: ignore[method-assign]  # flat ⇒ closed

    fill = venue.close_market("BTCUSDT", "LONG", 0.01)

    assert fill is not None
    assert fill["fill_price"] == 100.5
    assert fill["fees"] == 0.02  # from cumFeeDetail, not unrealized_pnl


def test_exit_leg_skips_the_maker_poll_when_maker_first_exit_is_off() -> None:
    """Exits go straight to market; entries keep maker-first.

    Measured on the demo venue: 7/7 reduce-only post-only limits went
    unfilled and every fill paid 5.5 bps (Bybit's taker rate), so the 15s
    maker poll bought nothing while serialising dead wait across the book.
    Entries are unaffected -- there is no evidence the entry leg fails.
    """
    http = _FakeHTTP(status="Filled")
    venue = _fast_venue(http, maker_first=True)
    venue._maker_first_exit = False  # type: ignore[attr-defined]
    venue.position = lambda symbol: None  # type: ignore[method-assign]  # flat ⇒ closed

    venue.close_market("BTCUSDT", "LONG", 0.01)

    assert [o["orderType"] for o in http.orders] == ["Market"]  # no resting limit
    assert http.cancelled == []  # nothing to cancel

    # ...while the ENTRY leg still tries maker first.
    http2 = _FakeHTTP(status="Filled")
    venue2 = _fast_venue(http2, maker_first=True)
    venue2._maker_first_exit = False  # type: ignore[attr-defined]
    venue2.open_market("BTCUSDT", "LONG", 1000)
    assert http2.orders[0]["orderType"] == "Limit"
    assert http2.orders[0]["timeInForce"] == "PostOnly"


def test_stale_but_past_forecast_is_tradeable_when_exact_window_misses() -> None:
    """Predictor/engine offset drift must not starve the book.

    The two daemons consume the same feature topic in separate consumer
    groups, so their windows drift (a restart of either is enough). Exact
    window matching then failed on BOTH candidates, the signal read as None,
    and the entry was skipped -- measured live as 7 of 12 selected symbols
    never opening a position. A forecast stamped in the PAST, within the
    staleness bound, is still tradeable.
    """
    kv = FakeKV()
    sim = _simulator(kv, hold_until_decay=True, slippage_bps=0.0, taker_fee_bps=0.0)
    # Stamped at 3W: the bar being processed ends at 4W and prev_end is 2W, so
    # it matches NEITHER exact candidate, yet is only one window stale.
    _seed_prediction(kv, "BTCUSDT", 3 * _WINDOW_MS, "LONG", predicted_return=0.5)

    sim.handle(_window("BTCUSDT", 1, 100.0))  # seeds prev_end = 2W
    sim.handle(_window("BTCUSDT", 3, 100.0))  # window_end = 4W

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["position"] is not None, "stale-but-past forecast must still open"
    assert stored["position"]["side"] == "LONG"


def test_future_stamped_forecast_is_never_traded() -> None:
    """A forecast stamped AFTER this bar is look-ahead and must be rejected."""
    kv = FakeKV()
    sim = _simulator(kv, hold_until_decay=True, slippage_bps=0.0, taker_fee_bps=0.0)
    _seed_prediction(kv, "BTCUSDT", 9 * _WINDOW_MS, "LONG", predicted_return=0.5)

    sim.handle(_window("BTCUSDT", 1, 100.0))
    sim.handle(_window("BTCUSDT", 2, 100.0))

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["position"] is None, "future-stamped forecast is look-ahead"


def test_forecast_expiry_rolls_an_unchanged_position_instead_of_round_tripping() -> None:
    """Expiry means re-evaluate, not "must trade".

    Closing and immediately re-opening the SAME side pays a full round trip to
    end up with identical exposure -- pure cost. The research book charges only
    |w_new - w_old|, so an unchanged weight is free there; the live book must
    match that or it bleeds fees every rebalance on every held name.
    """
    kv = FakeKV()
    sim = _simulator(
        kv, hold_until_decay=True, max_hold_h=1, slippage_bps=0.0, taker_fee_bps=10.0
    )
    for i in range(1, 8):
        _seed_prediction(kv, "BTCUSDT", i * _WINDOW_MS, "LONG", predicted_return=0.5)
        sim.handle(_window("BTCUSDT", i, 100.0))

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["position"] is not None, "an agreeing forecast must keep the position"
    assert stored["n_trades"] == 0, "unchanged signal must not round-trip at expiry"
    assert stored["total_fees"] == 0.0


def test_forecast_expiry_still_closes_when_the_signal_flips() -> None:
    """The roll must not become a way to hold a position past a real reversal."""
    kv = FakeKV()
    sim = _simulator(
        kv, hold_until_decay=True, max_hold_h=1, slippage_bps=0.0, taker_fee_bps=0.0
    )
    _seed_prediction(kv, "BTCUSDT", 1 * _WINDOW_MS, "LONG", predicted_return=0.5)
    sim.handle(_window("BTCUSDT", 1, 100.0))
    _seed_prediction(kv, "BTCUSDT", 2 * _WINDOW_MS, "LONG", predicted_return=0.5)
    sim.handle(_window("BTCUSDT", 2, 100.0))
    _seed_prediction(kv, "BTCUSDT", 3 * _WINDOW_MS, "SHORT", predicted_return=-0.5)
    sim.handle(_window("BTCUSDT", 3, 100.0))

    stored = kv.get_json(execution_key("execution:crypto:5m", "BTCUSDT"))
    assert stored is not None
    assert stored["n_trades"] == 1, "a flipped signal at expiry must still close"
