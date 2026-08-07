"""Message bus tests — FakeBus semantics, no broker required."""

from __future__ import annotations

import threading

from stream.bus import FakeBus, _deserialize, _serialize


def test_serialize_roundtrip_json() -> None:
    raw = _serialize({"symbol": "BTCUSDT", "close": 100.5, "nan": None})
    assert _deserialize(raw) == {"symbol": "BTCUSDT", "close": 100.5, "nan": None}


def test_publish_consume_roundtrip_keeps_order_and_key() -> None:
    bus = FakeBus()
    for i in range(5):
        bus.publish("crypto.bars.raw", "BTCUSDT", {"ts": i, "close": 100 + i})

    got: list[tuple[str, dict]] = []
    stop = threading.Event()
    for topic, msg in bus.iter_consume("crypto.bars.raw", "test-group", stop=stop):
        got.append((topic, msg))
        if len(got) == 5:
            stop.set()

    assert [g[0] for g in got] == ["crypto.bars.raw"] * 5
    assert [g[1]["ts"] for g in got] == [0, 1, 2, 3, 4]
    assert [g[1]["close"] for g in got] == [100, 101, 102, 103, 104]


def test_consume_filters_to_subscribed_topics() -> None:
    bus = FakeBus()
    bus.publish("crypto.bars.raw", "BTCUSDT", {"ts": 1})
    bus.publish("crypto.features.5m", "BTCUSDT", {"window_start_ms": 2})
    bus.publish("unrelated", "x", {"n": 1})

    got: list[str] = []
    stop = threading.Event()
    for topic, _msg in bus.iter_consume(["crypto.bars.raw", "crypto.features.5m"], "g", stop=stop):
        got.append(topic)
        if len(got) == 2:
            stop.set()

    assert sorted(got) == ["crypto.bars.raw", "crypto.features.5m"]


def test_consume_respects_stop_event_without_messages() -> None:
    bus = FakeBus()
    stop = threading.Event()
    stop.set()
    assert list(bus.iter_consume("empty.topic", "g", stop=stop)) == []


def test_seed_makes_messages_available_to_consumers() -> None:
    bus = FakeBus()
    bus.seed("crypto.bars.raw", [{"ts": 1}, {"ts": 2}])
    got = []
    stop = threading.Event()
    for _topic, msg in bus.iter_consume("crypto.bars.raw", "g", stop=stop):
        got.append(msg)
        if len(got) == 2:
            stop.set()
    assert [m["ts"] for m in got] == [1, 2]
