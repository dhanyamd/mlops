"""Live market stream: Kafka bus → ring buffer → WebSocket fan-out.

The API no longer owns ingestion. A consumer thread subscribes to the raw bar
topic on the Kafka bus, normalizes each message for the hub, and broadcasts
deltas to every ``/ws/market`` subscriber. The hub's ring buffer gives new
subscribers an instant snapshot without touching Kafka, Redis, or Snowflake.

The hub is intentionally thin and lock-protected: the consumer runs on a worker
thread while subscribers live on the asyncio event loop, so every shared
collection is guarded and cross-thread calls are marshalled through
``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import Any

from config.settings import csv_list, get_settings
from stream.bus import KafkaBus, MessageBus


def _hub_bar(msg: dict[str, Any]) -> dict[str, Any]:
    """Kafka bar payload → the compact shape the hub dedupes on (ISO ts)."""
    return {
        "symbol": str(msg["symbol"]).upper(),
        "ts": msg.get("ts_iso") or msg.get("ts"),
        "open": msg.get("open"),
        "high": msg.get("high"),
        "low": msg.get("low"),
        "close": msg.get("close"),
        "volume": msg.get("volume"),
    }


class MarketHub:
    """In-memory ring buffer of minute bars + WebSocket subscriber fan-out.

    Thread-safe: the consumer thread mutates buffers and broadcasts; subscriber
    queues are owned by the event loop thread. Locks make either side safe.
    """

    def __init__(self, symbols: list[str], history_minutes: int = 180) -> None:
        self._symbols = [s.upper() for s in symbols]
        self._history_minutes = history_minutes
        self._buffers: dict[str, deque[dict[str, Any]]] = {
            sym: deque(maxlen=history_minutes) for sym in self._symbols
        }
        self._subscribers: set[asyncio.Queue[list[dict[str, Any]]]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once from the FastAPI lifespan (event-loop thread)."""
        self._loop = loop

    # ── data (consumer thread) ───────────────────────────────────────────────

    def ingest(self, bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Append bars to ring buffers; return the ones that are new/updated.

        Idempotent per ``(symbol, ts)``: a repeat bar with the same timestamp
        replaces the previous in-place (the in-progress minute keeps updating),
        so only true deltas get broadcast.
        """
        new_bars: list[dict[str, Any]] = []
        with self._lock:
            for bar in bars:
                sym = bar["symbol"]
                buf = self._buffers.setdefault(sym, deque(maxlen=self._history_minutes))
                ts = bar["ts"]
                replaced = False
                for i, existing in enumerate(buf):
                    if existing["ts"] == ts:
                        buf[i] = bar
                        replaced = True
                        break
                if not replaced:
                    buf.append(bar)
                new_bars.append(bar)
        return new_bars

    def snapshot(self, symbol: str) -> list[dict[str, Any]]:
        """Ring-buffer contents for one symbol (oldest → newest), JSON-safe."""
        with self._lock:
            return list(self._buffers.get(symbol.upper(), deque()))

    # ── subscribers (event-loop thread) ─────────────────────────────────────

    def subscribe(self, queue: asyncio.Queue[list[dict[str, Any]]]) -> None:
        with self._lock:
            self._subscribers.add(queue)

    def unsubscribe(self, queue: asyncio.Queue[list[dict[str, Any]]]) -> None:
        with self._lock:
            self._subscribers.discard(queue)

    def broadcast(self, bars: list[dict[str, Any]]) -> None:
        """Schedule a fan-out on the event loop (safe to call from any thread)."""
        if not bars:
            return
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._fanout, bars)
        else:
            self._fanout(bars)

    def _fanout(self, bars: list[dict[str, Any]]) -> None:
        with self._lock:
            targets = list(self._subscribers)
        for queue in targets:
            try:
                queue.put_nowait(bars)
            except asyncio.QueueFull:
                # Slow consumer: drop the batch; the snapshot on reconnect wins.
                pass


class MarketStream:
    """Owns the Kafka consumer thread + hub. Started/stopped by the lifespan.

    Uses one stable consumer group with offset commits disabled, so every API
    restart resumes at the tail and streams only bars arriving after it starts.
    A per-process group id would give the same behaviour but leave an orphaned
    group holding offsets on the broker after every restart; fourteen had
    accumulated before this was fixed.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        *,
        history_minutes: int = 180,
        bus: MessageBus | None = None,
        raw_topic: str | None = None,
        hub: MarketHub | None = None,
    ) -> None:
        settings = get_settings()
        self._symbols = [
            s.upper() for s in (symbols or csv_list(settings.ingest_default_crypto_symbols))
        ]
        self._history_minutes = history_minutes
        self._bus = bus or KafkaBus(settings.stream_kafka_bootstrap_servers)
        self._raw_topic = raw_topic or settings.stream_kafka_topic_raw
        self._group_id = "api-live"
        self.hub = hub or MarketHub(self._symbols, history_minutes)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Attach the hub to the event loop and launch the consumer thread."""
        if self._thread is not None:
            return
        if loop is not None:
            self.hub.attach_loop(loop)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="market-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the consumer to exit and join the thread (bounded wait)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        for _topic, msg in self._bus.iter_consume(
            self._raw_topic, self._group_id, stop=self._stop, auto_commit=False
        ):
            try:
                deltas = self.hub.ingest([_hub_bar(msg)])
            except Exception:  # noqa: BLE001 - a stream must never kill the API
                import logging

                logging.getLogger(__name__).exception("market stream message rejected")
                continue
            if deltas:
                self.hub.broadcast(deltas)


def start_stream() -> MarketStream | None:
    """Build the stream if enabled by settings; else return None."""
    settings = get_settings()
    if not settings.stream_enabled:
        return None
    return MarketStream(
        history_minutes=settings.stream_history_minutes,
        bus=KafkaBus(settings.stream_kafka_bootstrap_servers),
        raw_topic=settings.stream_kafka_topic_raw,
    )


def stop_stream(stream: MarketStream | None) -> None:
    if stream is not None:
        stream.stop()
