"""Live market stream: Binance minute bars → ring buffer → WebSocket fan-out.

A background poller thread in the API process fetches recent Binance minute
bars every ``STREAM_POLL_SECONDS``, persists them to BRONZE.CRYPTO_BARS
(best-effort, Snowflake outages degrade gracefully), and broadcasts deltas to
every ``/ws/market`` subscriber. The ring buffer per symbol gives new
subscribers an instant snapshot without hitting Snowflake.

The hub is intentionally thin and lock-protected: the poller runs on a worker
thread while subscribers live on the asyncio event loop, so every shared
collection is guarded and cross-thread calls are marshalled through
``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

import pandas as pd

from config.settings import csv_list, get_settings
from ingest.providers.binance import BinanceBarProvider
from ingest.store import write_crypto_bars

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


def _df_to_bars(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame rows → JSON-safe bar dicts (datetime → ISO, NaN → None)."""
    bars: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        ts = row.get("ts")
        if ts is None or ts != ts:  # None / NaT
            continue
        bars.append(
            {
                "symbol": str(row["symbol"]).upper(),
                "ts": ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts),
                "open": _num(row.get("open")),
                "high": _num(row.get("high")),
                "low": _num(row.get("low")),
                "close": _num(row.get("close")),
                "volume": _num(row.get("volume")),
            }
        )
    return bars


def _num(value: Any) -> float | None:
    if value is None or value != value:  # None / NaN
        return None
    return float(value)


class MarketHub:
    """In-memory ring buffer of minute bars + WebSocket subscriber fan-out.

    Thread-safe: the poller thread mutates buffers and broadcasts; subscriber
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

    # ── data (poller thread) ────────────────────────────────────────────────

    def ingest(self, bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Append bars to ring buffers; return the ones that are new/updated.

        Idempotent per ``(symbol, ts)``: a repeat bar with the same timestamp
        replaces the previous in-place (the in-progress minute keeps updating),
        so every poll only returns true deltas to broadcast.
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
        """Schedule a fan-out on the event loop (safe to call from poller thread)."""
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
    """Owns the poller thread + hub. Started/stopped by the FastAPI lifespan."""

    def __init__(
        self,
        symbols: list[str] | None = None,
        *,
        poll_seconds: int = 15,
        history_minutes: int = 180,
        provider: Callable[[list[str], int], pd.DataFrame] | None = None,
        persist: Callable[[pd.DataFrame], int] | None = None,
        hub: MarketHub | None = None,
    ) -> None:
        settings = get_settings()
        self._symbols = [
            s.upper() for s in (symbols or csv_list(settings.ingest_default_crypto_symbols))
        ]
        self._poll_seconds = poll_seconds
        self._history_minutes = history_minutes
        self._provider = provider or BinanceBarProvider().fetch_bars
        self._persist = persist or write_crypto_bars
        self.hub = hub or MarketHub(self._symbols, history_minutes)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Attach the hub to the event loop and launch the poller thread."""
        if self._thread is not None:
            return
        if loop is not None:
            self.hub.attach_loop(loop)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="market-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the poller to exit and join the thread (bounded wait)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        # First poll seeds the full history so snapshots are instantly rich;
        # later polls only fetch the last few minutes (weight-cheap).
        minutes = self._history_minutes
        while not self._stop.is_set():
            try:
                self._poll_once(minutes=minutes)
            except Exception:  # noqa: BLE001 - a stream must never kill the API
                import logging

                logging.getLogger(__name__).exception("market stream poll failed")
            minutes = max(3, int(self._poll_seconds / 60) + 1)
            self._stop.wait(self._poll_seconds)

    def _poll_once(self, minutes: int) -> list[dict[str, Any]]:
        df = self._provider(self._symbols, minutes)
        if df is None or df.empty:
            return []
        bars = _df_to_bars(df)
        deltas = self.hub.ingest(bars)
        if deltas:
            self._persist_best_effort(df)
            self.hub.broadcast(deltas)
        return deltas

    def _persist_best_effort(self, df: pd.DataFrame) -> None:
        """Upsert to BRONZE.CRYPTO_BARS; a Snowflake outage must not break the stream."""
        try:
            self._persist(df)
        except Exception:  # noqa: BLE001 - best-effort persistence
            import logging

            logging.getLogger(__name__).warning(
                "market stream persist failed (best-effort)", exc_info=True
            )


def start_stream() -> MarketStream | None:
    """Build + start the stream if enabled by settings; else return None."""
    settings = get_settings()
    if not settings.stream_enabled:
        return None
    stream = MarketStream(
        poll_seconds=settings.stream_poll_seconds,
        history_minutes=settings.stream_history_minutes,
    )
    # Only the real provider is constructed here; tests build MarketStream with
    # injected fakes and call start() themselves.
    stream.start()
    return stream


def stop_stream(stream: MarketStream | None) -> None:
    if stream is not None:
        stream.stop()
