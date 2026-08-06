"""Live market stream tests — hermetic (no Binance, no Snowflake).

Covers the ring-buffer hub (ingest/dedupe/snapshot/broadcast) and the poller
(wire a fake provider + fake persist into ``MarketStream._poll_once``), plus
the two API surface endpoints (REST snapshot + WebSocket) with a fake stream
injected via ``api.main.start_stream``.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.stream as stream_mod
from api.main import app
from api.stream import MarketHub, MarketStream

_BAR_COLUMNS = [
    "symbol",
    "ts",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "provider",
    "loaded_at",
]


def _bar_df(symbols: list[str], n_minutes: int = 3) -> pd.DataFrame:
    """Synthetic minute-bar frame in the exact provider contract."""
    base = pd.Timestamp("2026-08-06 12:00:00")
    rows = []
    for sym in symbols:
        for i in range(n_minutes):
            rows.append(
                {
                    "symbol": sym,
                    "ts": base + pd.to_timedelta(i, unit="min"),
                    "timeframe": "1Min",
                    "open": 100.0 + i,
                    "high": 101.0 + i,
                    "low": 99.0 + i,
                    "close": 100.5 + i,
                    "volume": 1.25,
                    "provider": "binance",
                    "loaded_at": base,
                }
            )
    return pd.DataFrame(rows, columns=_BAR_COLUMNS)


# ── MarketHub: ring buffer ──────────────────────────────────────────────────


def test_hub_ingest_returns_deltas_and_builds_snapshot() -> None:
    hub = MarketHub(["BTCUSDT"], history_minutes=10)
    deltas = hub.ingest(stream_mod._df_to_bars(_bar_df(["BTCUSDT"], n_minutes=3)))

    assert [d["ts"] for d in deltas] == [hub.snapshot("BTCUSDT")[i]["ts"] for i in range(3)]
    assert [b["symbol"] for b in deltas] == ["BTCUSDT"] * 3
    assert [b["volume"] for b in deltas] == [1.25] * 3  # fractional volume survives


def test_hub_dedupes_same_timestamp_in_place() -> None:
    hub = MarketHub(["BTCUSDT"], history_minutes=10)
    first = _bar_df(["BTCUSDT"], n_minutes=1)
    hub.ingest(stream_mod._df_to_bars(first))

    # The in-progress minute updates: same ts, new close.
    second = first.copy()
    second.loc[0, "close"] = 200.0
    deltas = hub.ingest(stream_mod._df_to_bars(second))

    assert len(deltas) == 1
    snapshot = hub.snapshot("BTCUSDT")
    assert len(snapshot) == 1  # replaced, not appended
    assert snapshot[0]["close"] == 200.0


def test_hub_snapshot_bounded_by_history() -> None:
    hub = MarketHub(["BTCUSDT"], history_minutes=3)
    hub.ingest(stream_mod._df_to_bars(_bar_df(["BTCUSDT"], n_minutes=5)))
    assert len(hub.snapshot("BTCUSDT")) == 3  # oldest 2 evicted


def test_hub_broadcast_fans_out_to_subscribers() -> None:
    hub = MarketHub(["BTCUSDT"])
    queue: asyncio.Queue[list[dict]] = asyncio.Queue()
    hub.subscribe(queue)
    try:
        hub.broadcast([{"symbol": "BTCUSDT", "ts": "x", "close": 1.0}])
        assert queue.get_nowait() == [{"symbol": "BTCUSDT", "ts": "x", "close": 1.0}]
    finally:
        hub.unsubscribe(queue)


# ── MarketStream poller (fake provider + persist) ───────────────────────────


def test_poller_polls_provider_persists_and_broadcasts(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[int] = []
    persisted: list[pd.DataFrame] = []
    hub = MarketHub(["BTCUSDT"], history_minutes=10)
    queue: asyncio.Queue[list[dict]] = asyncio.Queue()
    hub.subscribe(queue)

    def fake_provider(symbols: list[str], minutes: int) -> pd.DataFrame:
        fetched.append(minutes)
        return _bar_df(symbols, n_minutes=2)

    def fake_persist(df: pd.DataFrame) -> int:
        persisted.append(df)
        return len(df)

    stream = MarketStream(
        symbols=["BTCUSDT"],
        poll_seconds=15,
        history_minutes=10,
        provider=fake_provider,  # type: ignore[arg-type]
        persist=fake_persist,  # type: ignore[arg-type]
        hub=hub,
    )
    try:
        deltas = stream._poll_once(minutes=10)
    finally:
        hub.unsubscribe(queue)

    assert fetched == [10]  # first poll seeds full history
    assert len(deltas) == 2
    assert len(hub.snapshot("BTCUSDT")) == 2
    assert len(persisted) == 1
    assert len(persisted[0]) == 2
    assert not queue.empty()  # broadcast reached the subscriber


def test_poller_persist_failure_does_not_break_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Snowflake outage must degrade to a warning, not kill the poller."""
    hub = MarketHub(["BTCUSDT"])

    def boom(df: pd.DataFrame) -> int:  # type: ignore[no-untyped-def]
        raise RuntimeError("snowflake down")

    stream = MarketStream(
        symbols=["BTCUSDT"],
        provider=lambda s, m: _bar_df(s, n_minutes=1),  # type: ignore[arg-type]
        persist=boom,  # type: ignore[arg-type]
        hub=hub,
    )
    deltas = stream._poll_once(minutes=5)
    assert len(deltas) == 1  # in-memory stream still works
    assert len(hub.snapshot("BTCUSDT")) == 1


def test_start_stream_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = stream_mod.get_settings()
    monkeypatch.setattr(settings, "stream_enabled", False)
    assert stream_mod.start_stream() is None


def test_start_stream_returns_stream_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = stream_mod.get_settings()
    monkeypatch.setattr(settings, "stream_enabled", True)
    # start() launches a thread against the real provider — avoid that by
    # exercising start_stream only up to construction via a stubbed start.
    monkeypatch.setattr(stream_mod.MarketStream, "start", lambda self, loop=None: None)
    stream = stream_mod.start_stream()
    assert stream is not None
    assert stream.hub is not None


# ── API surface ─────────────────────────────────────────────────────────────


class _FakeStream:
    def __init__(self, hub: MarketHub) -> None:
        self.hub = hub

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        return None

    def stop(self) -> None:
        return None


@pytest.fixture
def _fake_stream(monkeypatch: pytest.MonkeyPatch) -> MarketHub:
    hub = MarketHub(["BTCUSDT"], history_minutes=10)
    hub.ingest(stream_mod._df_to_bars(_bar_df(["BTCUSDT"], n_minutes=3)))
    monkeypatch.setattr("api.main.start_stream", lambda: _FakeStream(hub))
    return hub


def test_market_live_endpoint_returns_hub_snapshot(_fake_stream: MarketHub) -> None:
    with TestClient(app) as client:
        resp = client.get("/api/market/live/btcusdt")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "BTCUSDT"
    assert body["enabled"] is True
    assert body["count"] == 3


def test_market_live_endpoint_disabled_without_stream() -> None:
    with TestClient(app) as client:
        resp = client.get("/api/market/live/btcusdt")
    body = resp.json()
    assert body["enabled"] is False
    assert body["count"] == 0


def test_ws_market_snapshot_then_delta(_fake_stream: MarketHub) -> None:
    with TestClient(app) as client:
        with client.websocket_connect("/ws/market?symbol=BTCUSDT") as ws:
            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot"
            assert snapshot["symbol"] == "BTCUSDT"
            assert len(snapshot["bars"]) == 3

            # Push a new bar through the hub → subscriber should see the delta.
            new_bar = {
                "symbol": "BTCUSDT",
                "ts": "2026-08-06T12:03:00",
                "open": 103.0,
                "high": 104.0,
                "low": 102.0,
                "close": 103.5,
                "volume": 2.0,
            }
            _fake_stream.ingest([new_bar])
            _fake_stream.broadcast([new_bar])
            delta = ws.receive_json()
            assert delta["type"] == "bar"
            assert delta["symbol"] == "BTCUSDT"
            assert delta["bar"]["close"] == 103.5


def test_ws_market_ignores_other_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    """A subscriber to BTCUSDT must only see BTCUSDT deltas."""
    hub = MarketHub(["BTCUSDT", "ETHUSDT"], history_minutes=10)
    monkeypatch.setattr("api.main.start_stream", lambda: _FakeStream(hub))
    with TestClient(app) as client:
        with client.websocket_connect("/ws/market?symbol=BTCUSDT") as ws:
            assert ws.receive_json()["type"] == "snapshot"

            # ETHUSDT delta arrives first — must NOT be forwarded to this client.
            eth_bar = {
                "symbol": "ETHUSDT",
                "ts": "2026-08-06T12:04:00",
                "open": 3000.0,
                "high": 3001.0,
                "low": 2999.0,
                "close": 3000.5,
                "volume": 5.0,
            }
            hub.ingest([eth_bar])
            hub.broadcast([eth_bar])
            btc_bar = {
                "symbol": "BTCUSDT",
                "ts": "2026-08-06T12:05:00",
                "open": 104.0,
                "high": 105.0,
                "low": 103.0,
                "close": 104.5,
                "volume": 1.0,
            }
            hub.ingest([btc_bar])
            hub.broadcast([btc_bar])
            matched = ws.receive_json()
            assert matched["type"] == "bar"
            assert matched["symbol"] == "BTCUSDT"
            assert matched["bar"]["close"] == 104.5
