"""Online store materializer tests — hermetic (FakeBus → FakeKV)."""

from __future__ import annotations

import threading

from stream.bus import FakeBus
from stream.kv import FakeKV
from stream.materializer import OnlineStoreMaterializer, feature_key, live_key


def _materializer(bus: FakeBus, kv: FakeKV, *, maxlen: int = 3) -> OnlineStoreMaterializer:
    return OnlineStoreMaterializer(
        bus,
        kv,
        raw_topic="crypto.bars.raw",
        features_topic="crypto.features.5m",
        live_prefix="live:crypto",
        feature_prefix="feature:crypto:5m",
        feature_maxlen=maxlen,
    )


def test_key_helpers_uppercase_symbols() -> None:
    assert live_key("live:crypto", "btcusdt") == "live:crypto:BTCUSDT"
    assert feature_key("feature:crypto:5m", "btcusdt") == "feature:crypto:5m:BTCUSDT"


def test_handle_writes_live_bar_from_raw_topic() -> None:
    bus = FakeBus()
    kv = FakeKV()
    _materializer(bus, kv).handle("crypto.bars.raw", {"symbol": "btcusdt", "ts": 1})
    assert kv.get_json("live:crypto:BTCUSDT") == {"symbol": "btcusdt", "ts": 1}


def test_handle_writes_bounded_feature_history() -> None:
    bus = FakeBus()
    kv = FakeKV()
    materializer = _materializer(bus, kv, maxlen=3)
    for ts in range(5):
        materializer.handle("crypto.features.5m", {"symbol": "BTCUSDT", "window_start_ms": ts})

    rows = kv.list_json("feature:crypto:5m:BTCUSDT")
    assert [r["window_start_ms"] for r in rows] == [2, 3, 4]  # RPUSH+LTRIM bound, oldest evicted


def test_handle_ignores_messages_without_symbol() -> None:
    bus = FakeBus()
    kv = FakeKV()
    materializer = _materializer(bus, kv)
    materializer.handle("crypto.bars.raw", {"ts": 1})
    materializer.handle("crypto.features.5m", {"window_start_ms": 1})
    assert kv.get_json("live:crypto:BTCUSDT") is None
    assert kv.list_json("feature:crypto:5m:BTCUSDT") == []


def test_run_forever_dispatches_both_topics() -> None:
    bus = FakeBus()
    kv = FakeKV()
    bus.publish("crypto.bars.raw", "BTCUSDT", {"symbol": "BTCUSDT", "ts": 10})
    bus.publish("crypto.features.5m", "BTCUSDT", {"symbol": "BTCUSDT", "window_start_ms": 20})

    stop = threading.Event()

    def _stop_after() -> None:
        deadline = threading.Event()
        while not deadline.wait(0.1):
            if kv.get_json("live:crypto:BTCUSDT") and kv.list_json("feature:crypto:5m:BTCUSDT"):
                stop.set()
                return

    watcher = threading.Thread(target=_stop_after)
    watcher.start()
    _materializer(bus, kv).run_forever(stop=stop)
    watcher.join(timeout=5.0)

    assert kv.get_json("live:crypto:BTCUSDT")["ts"] == 10
    assert kv.list_json("feature:crypto:5m:BTCUSDT")[0]["window_start_ms"] == 20


def test_handle_routes_5m_feature_topic_to_5m_prefix() -> None:
    bus = FakeBus()
    kv = FakeKV()
    materializer = OnlineStoreMaterializer(
        bus,
        kv,
        raw_topic="crypto.bars.raw",
        features_topic="crypto.features.1h",
        live_prefix="live:crypto",
        feature_prefix="feature:crypto:1h",
        feature_maxlen=3,
        features_topic_5m="crypto.features.5m",
        feature_prefix_5m="feature:crypto:5m",
    )

    materializer.handle("crypto.features.1h", {"symbol": "BTCUSDT", "window_start_ms": 30})
    materializer.handle("crypto.features.5m", {"symbol": "BTCUSDT", "window_start_ms": 31})

    assert kv.list_json("feature:crypto:1h:BTCUSDT")[0]["window_start_ms"] == 30
    assert kv.list_json("feature:crypto:5m:BTCUSDT")[0]["window_start_ms"] == 31


def test_handle_ignores_5m_topic_when_dual_mode_disabled() -> None:
    bus = FakeBus()
    kv = FakeKV()
    materializer = OnlineStoreMaterializer(
        bus,
        kv,
        raw_topic="crypto.bars.raw",
        features_topic="crypto.features.1h",
        live_prefix="live:crypto",
        feature_prefix="feature:crypto:1h",
        feature_maxlen=3,
    )

    materializer.handle("crypto.features.5m", {"symbol": "BTCUSDT", "window_start_ms": 31})

    assert kv.list_json("feature:crypto:1h:BTCUSDT") == []
    assert kv.list_json("feature:crypto:5m:BTCUSDT") == []
