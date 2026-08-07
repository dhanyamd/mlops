"""Binance producer tests — hermetic (FakeBus, fake provider/persist)."""

from __future__ import annotations

import threading
import time

import pandas as pd

from stream.bus import FakeBus
from stream.producer import BinanceProducer

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


def _bar_df(symbols: list[str], n_minutes: int = 2) -> pd.DataFrame:
    base = pd.Timestamp("2026-08-06 12:00:00")
    rows = [
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
        for sym in symbols
        for i in range(n_minutes)
    ]
    return pd.DataFrame(rows, columns=_BAR_COLUMNS)


def _producer(bus: FakeBus, *, persist=None, provider=None, **kwargs) -> BinanceProducer:
    return BinanceProducer(
        ["BTCUSDT"],
        bus=bus,
        topic="crypto.bars.raw",
        provider=provider or (lambda s, days=0, minutes=0: _bar_df(s, n_minutes=2)),
        persist=persist or (lambda df: len(df)),
        **kwargs,
    )


def test_producer_publishes_json_payloads_to_bus() -> None:
    bus = FakeBus()
    producer = _producer(bus)
    n = producer._poll_once(minutes=10)

    assert n == 2
    msgs = bus.drain("crypto.bars.raw")
    assert len(msgs) == 2
    first = msgs[0]
    # Flink event time + hub display key both present.
    assert first["ts"] == int(pd.Timestamp("2026-08-06 12:00:00").value // 1_000_000)
    assert first["ts_iso"] == "2026-08-06T12:00:00"
    assert first["symbol"] == "BTCUSDT"
    assert first["volume"] == 1.25  # fractional volume survives JSON
    assert set(first) >= {"symbol", "ts", "ts_iso", "open", "high", "low", "close", "volume"}


def test_producer_persists_frame_after_publish() -> None:
    bus = FakeBus()
    persisted: list[pd.DataFrame] = []

    def fake_persist(df: pd.DataFrame) -> int:
        persisted.append(df)
        return len(df)

    _producer(bus, persist=fake_persist)._poll_once(minutes=10)
    assert len(persisted) == 1
    assert len(persisted[0]) == 2


def test_producer_persist_failure_degrades_gracefully() -> None:
    """A Snowflake outage must not prevent publishing to Kafka."""

    def boom(df: pd.DataFrame) -> int:  # type: ignore[no-untyped-def]
        raise RuntimeError("snowflake down")

    bus = FakeBus()
    n = _producer(bus, persist=boom)._poll_once(minutes=5)
    assert n == 2
    assert len(bus.drain("crypto.bars.raw")) == 2


def test_producer_empty_frame_publishes_nothing() -> None:
    bus = FakeBus()
    producer = _producer(bus, provider=lambda s, days=0, minutes=0: pd.DataFrame())
    assert producer._poll_once(minutes=5) == 0
    assert bus.drain("crypto.bars.raw") == []


def test_producer_run_forever_loops_until_stop() -> None:
    bus = FakeBus()
    polls: list[int] = []

    def provider(symbols: list[str], days: int = 0, minutes: int = 0) -> pd.DataFrame:
        polls.append(minutes)
        return _bar_df(symbols, n_minutes=1)

    producer = _producer(bus, provider=provider, poll_seconds=1)
    stop = threading.Event()
    thread = threading.Thread(target=producer.run_forever, args=(stop,))
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while len(polls) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(polls) >= 2
    finally:
        stop.set()
        thread.join(timeout=5.0)
    assert not thread.is_alive()
