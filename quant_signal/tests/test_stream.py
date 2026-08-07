"""Live market stream tests — hermetic (no Binance, no Kafka, no Snowflake).

Covers the ring-buffer hub (ingest/dedupe/snapshot/broadcast) and the Kafka
consumer (``MarketStream`` fed by ``FakeBus``), plus the API surface endpoints
(REST snapshot, REST features, WebSocket) with a fake stream injected via
``api.main.start_stream``.
"""

from __future__ import annotations

import asyncio
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.stream as stream_mod
from api.main import app
from api.stream import MarketHub, MarketStream
from stream.bars import df_to_bars
from stream.bus import FakeBus
from stream.kv import FakeKV

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
    deltas = hub.ingest(df_to_bars(_bar_df(["BTCUSDT"], n_minutes=3)))

    assert [d["ts"] for d in deltas] == [hub.snapshot("BTCUSDT")[i]["ts"] for i in range(3)]
    assert [b["symbol"] for b in deltas] == ["BTCUSDT"] * 3
    assert [b["volume"] for b in deltas] == [1.25] * 3  # fractional volume survives


def test_hub_dedupes_same_timestamp_in_place() -> None:
    hub = MarketHub(["BTCUSDT"], history_minutes=10)
    first = _bar_df(["BTCUSDT"], n_minutes=1)
    hub.ingest(df_to_bars(first))

    # The in-progress minute updates: same ts, new close.
    second = first.copy()
    second.loc[0, "close"] = 200.0
    deltas = hub.ingest(df_to_bars(second))

    assert len(deltas) == 1
    snapshot = hub.snapshot("BTCUSDT")
    assert len(snapshot) == 1  # replaced, not appended
    assert snapshot[0]["close"] == 200.0


def test_hub_snapshot_bounded_by_history() -> None:
    hub = MarketHub(["BTCUSDT"], history_minutes=3)
    hub.ingest(df_to_bars(_bar_df(["BTCUSDT"], n_minutes=5)))
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


# ── MarketStream Kafka consumer (FakeBus → hub) ─────────────────────────────


def test_stream_consumes_bus_bars_into_hub_and_broadcasts() -> None:
    hub = MarketHub(["BTCUSDT"], history_minutes=10)
    queue: asyncio.Queue[list[dict]] = asyncio.Queue()
    hub.subscribe(queue)
    bus = FakeBus()
    stream = MarketStream(
        symbols=["BTCUSDT"],
        history_minutes=10,
        bus=bus,
        raw_topic="crypto.bars.raw",
        hub=hub,
    )
    try:
        stream.start()
        for payload in df_to_bars(_bar_df(["BTCUSDT"], n_minutes=2)):
            bus.publish("crypto.bars.raw", payload["symbol"], payload)
        deadline = time.monotonic() + 5.0
        while len(hub.snapshot("BTCUSDT")) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        snapshot = hub.snapshot("BTCUSDT")
        assert len(snapshot) == 2
        assert snapshot[0]["ts"] == "2026-08-06T12:00:00"  # ISO ts preserved for hub dedupe
        assert snapshot[0]["close"] == 100.5
        assert not queue.empty()  # fan-out reached the subscriber
        assert queue.get_nowait()[0]["symbol"] == "BTCUSDT"
    finally:
        stream.stop()
        hub.unsubscribe(queue)


def test_stream_stop_terminates_consumer_thread() -> None:
    stream = MarketStream(
        symbols=["BTCUSDT"],
        bus=FakeBus(),
        raw_topic="crypto.bars.raw",
    )
    stream.start()
    stream.stop()
    assert stream._thread is None


def test_start_stream_returns_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = stream_mod.get_settings()
    monkeypatch.setattr(settings, "stream_enabled", False)
    assert stream_mod.start_stream() is None


def test_start_stream_returns_stream_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = stream_mod.get_settings()
    monkeypatch.setattr(settings, "stream_enabled", True)
    # start() would launch a consumer thread against the real bus — avoid that by
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
    hub.ingest(df_to_bars(_bar_df(["BTCUSDT"], n_minutes=3)))
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


def test_market_features_endpoint_reads_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """The features endpoint serves Flink features from the Redis online store."""
    monkeypatch.setattr("api.main.RedisKV", lambda url: FakeKV())
    with TestClient(app) as client:
        kv = client.app.state.kv
        assert kv is not None
        kv.push_json(
            "feature:crypto:5m:BTCUSDT", {"symbol": "BTCUSDT", "window_start_ms": 100}, maxlen=10
        )
        kv.push_json(
            "feature:crypto:5m:BTCUSDT", {"symbol": "BTCUSDT", "window_start_ms": 200}, maxlen=10
        )
        resp = client.get("/api/market/features/btcusdt?limit=5")
    body = resp.json()
    assert body["symbol"] == "BTCUSDT"
    assert body["enabled"] is True
    assert body["count"] == 2
    assert body["features"][0]["window_start_ms"] == 200  # newest window first


def test_market_features_endpoint_disabled_without_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = stream_mod.get_settings()
    monkeypatch.setattr(settings, "stream_enabled", False)
    with TestClient(app) as client:
        resp = client.get("/api/market/features/btcusdt")
    body = resp.json()
    assert body["enabled"] is False
    assert body["count"] == 0


def test_market_predict_endpoint_reads_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """The predict endpoint serves the conformal prediction from the online store."""
    monkeypatch.setattr("api.main.RedisKV", lambda url: FakeKV())
    with TestClient(app) as client:
        kv = client.app.state.kv
        assert kv is not None
        kv.set_json(
            "prediction:crypto:5m:BTCUSDT",
            {"symbol": "BTCUSDT", "predicted_return": 0.001, "direction": "LONG"},
        )
        resp = client.get("/api/market/predict/btcusdt")
    body = resp.json()
    assert body["symbol"] == "BTCUSDT"
    assert body["enabled"] is True
    assert body["prediction"]["direction"] == "LONG"
    assert body["prediction"]["predicted_return"] == 0.001


def test_market_predict_endpoint_disabled_without_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = stream_mod.get_settings()
    monkeypatch.setattr(settings, "stream_enabled", False)
    with TestClient(app) as client:
        resp = client.get("/api/market/predict/btcusdt")
    assert resp.json()["enabled"] is False


def test_market_simulation_endpoint_reads_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """The simulation endpoint serves the MC fan chart from the online store."""
    monkeypatch.setattr("api.main.RedisKV", lambda url: FakeKV())
    with TestClient(app) as client:
        kv = client.app.state.kv
        assert kv is not None
        kv.set_json(
            "simulation:crypto:5m:BTCUSDT",
            {"symbol": "BTCUSDT", "base_price": 100.0, "var95": -0.02},
        )
        resp = client.get("/api/market/simulation/btcusdt")
    body = resp.json()
    assert body["symbol"] == "BTCUSDT"
    assert body["enabled"] is True
    assert body["simulation"]["var95"] == -0.02


def test_market_simulation_endpoint_disabled_without_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = stream_mod.get_settings()
    monkeypatch.setattr(settings, "stream_enabled", False)
    with TestClient(app) as client:
        resp = client.get("/api/market/simulation/btcusdt")
    assert resp.json()["enabled"] is False


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
