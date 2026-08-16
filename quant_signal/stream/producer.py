"""Ingestion producer: venue minute bars → Kafka bus.

Runs as a standalone process (``uv run python -m stream.producer``) — the API
no longer owns ingestion. Each poll fetches the trailing window from the venue
provider (selected by name via ``STREAM_VENUE``, default ``binance``),
publishes every bar to the raw topic (keyed by symbol → per-symbol ordering),
flushes for durability, and persists best-effort to BRONZE.CRYPTO_BARS (a
Snowflake outage degrades to a warning, never breaks the stream).

Run with ``make stream-producer``; the Kafka bus is KafkaBus unless a
``MessageBus`` fake is injected (hermetic tests).
"""

from __future__ import annotations

import threading
import time

import pandas as pd

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from ingest.store import write_crypto_bars
from stream.bars import df_to_bars
from stream.bus import KafkaBus, MessageBus
from stream.kv import KVStore, RedisKV

logger = get_logger(__name__)

# Producer-health keys in the Redis online store (read by the watchdog + dashboard).
_META_LAST_POLL = "meta:producer:crypto.bars.raw:last_poll_ts"
_META_LAST_BAR = "meta:producer:crypto.bars.raw:last_bar_max_ts"
_META_SKIP_PREFIX = "meta:producer:skipped:"


class BinanceProducer:
    """Poll the venue provider and publish JSON bars to the ingestion bus.

    Venue-agnostic: the provider is built by name from the registry (see
    ``build_bar_provider``), so this class never talks to an exchange directly.
    """

    def __init__(
        self,
        symbols: list[str],
        *,
        bus: MessageBus,
        topic: str,
        provider,
        poll_seconds: int = 15,
        poll_timeout_seconds: float | None = None,
        history_minutes: int = 180,
        persist=None,
        kv: KVStore | None = None,
        dlq_topic: str | None = None,
    ) -> None:
        self._symbols = [s.upper() for s in symbols]
        self._bus = bus
        self._topic = topic
        self._poll_seconds = poll_seconds
        self._poll_timeout = (
            poll_timeout_seconds
            if poll_timeout_seconds is not None
            else max(30.0, poll_seconds * 3)
        )
        self._history_minutes = history_minutes
        self._provider = provider
        self._persist = persist or write_crypto_bars
        self._kv = kv
        self._dlq_topic = dlq_topic or f"{topic}.dlq"
        self._stop = threading.Event()

    def run_forever(self, stop: threading.Event | None = None) -> None:
        """Poll in a loop; first cycle seeds full history, later ones windowed."""
        stop = stop or self._stop
        minutes = self._history_minutes
        while not stop.is_set():
            self._poll_with_deadline(minutes)
            minutes = max(3, int(self._poll_seconds / 60) + 1)
            stop.wait(self._poll_seconds)

    def _poll_with_deadline(self, minutes: int) -> int:
        """Run one poll cycle under a hard deadline.

        The venue fetch has a per-request timeout, but a wedged DNS/connect can
        outlive it and freeze the whole stream (observed: a 20-minute stall that
        froze Flink event time and every downstream panel). The poll body runs
        on a daemon worker thread and is abandoned if it exceeds the deadline; a
        fresh bus is installed so the next cycle can't inherit wedged state.
        """
        holder: dict[str, int] = {}

        def worker() -> None:
            try:
                holder["result"] = self._poll_once(minutes)
            except Exception:  # noqa: BLE001 - a producer must never die silently
                logger.exception("venue producer poll failed")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(self._poll_timeout)
        if thread.is_alive():
            logger.critical(
                "poll exceeded %.1fs deadline; abandoning wedged cycle", self._poll_timeout
            )
            self._recreate_bus()
            return -1
        return holder.get("result", 0)

    def _recreate_bus(self) -> None:
        recreate = getattr(self._bus, "recreate", None)
        if callable(recreate):
            self._bus = recreate()

    def _poll_once(self, minutes: int) -> int:
        # ``fetch_bars(symbols, days, minutes=None)``: pass minutes as a keyword
        # so a positional ``minutes`` doesn't silently bind to ``days`` (which
        # would fetch *days* of paged history and stall the first poll).
        # ``on_skip`` lets the venue report a dead/stale symbol (e.g. frozen
        # klines) without wedging the entire poll cycle.
        df = self._provider(self._symbols, days=0, minutes=minutes, on_skip=self._on_skip)
        if df is None or df.empty:
            return 0
        bars = df_to_bars(df)
        if not bars:
            return 0
        for bar in bars:
            self._bus.publish(self._topic, bar["symbol"], bar)
        self._bus.flush()
        self._persist_async(df)
        self._record_health(bars)
        return len(bars)

    def _persist_async(self, df: pd.DataFrame) -> None:
        try:
            t = threading.Thread(target=self._persist_best_effort, args=(df,), daemon=True)
            t.start()
        except Exception:
            logger.debug("persist thread spawn failed")

    def _persist_best_effort(self, df: pd.DataFrame) -> None:
        try:
            self._persist(df)
        except Exception:
            logger.debug("snowflake persist skipped (offline/best-effort)")

    def _record_health(self, bars: list[dict]) -> None:
        """Publish producer liveness to Redis so the watchdog/dashboard can see
        the feed is alive even when the strategy book is FLAT (bear regime)."""
        if self._kv is None:
            return
        now_ms = int(time.time() * 1000)
        max_ts = max((int(b.get("ts", 0)) for b in bars), default=0)
        self._kv.set_json(_META_LAST_POLL, {"ts": now_ms, "symbols": len(bars)})
        if max_ts:
            self._kv.set_json(
                _META_LAST_BAR,
                {"ts": max_ts, "age_s": (now_ms - max_ts) // 1000},
            )

    def _on_skip(self, symbol: str, reason: str) -> None:
        """Record a skipped symbol (dead/stale venue data) to the DLQ + Redis.

        Keeps a poison/dead symbol out of the stream but fully observable: the
        record lands on the dead-letter topic and a bounded Redis counter so the
        watchdog can alert on a sustained skip — the exact gap that hid the
        earlier producer wedge (a frozen ZECUSDT 1m feed that wedged every poll).
        """
        sym = symbol.upper()
        if self._bus is not None:
            self._bus.publish(
                self._dlq_topic,
                sym,
                {"symbol": sym, "reason": reason, "ts": int(time.time() * 1000)},
            )
            self._bus.flush()
        if self._kv is not None:
            key = _META_SKIP_PREFIX + sym
            prev = self._kv.get_json(key) or {}
            self._kv.set_json(
                key,
                {
                    "symbol": sym,
                    "reason": reason,
                    "last_ts": int(time.time() * 1000),
                    "count": int(prev.get("count", 0)) + 1,
                },
            )
        logger.warning("skipped symbol %s: %s", sym, reason)


def main() -> None:
    configure_logging()
    settings = get_settings()
    symbols = csv_list(settings.ingest_default_crypto_symbols)
    if not symbols:
        logger.error("no crypto symbols configured; set INGEST_DEFAULT_CRYPTO_SYMBOLS")
        return
    from ingest.providers.registry import build_bar_provider

    venue = settings.stream_venue
    provider = build_bar_provider(venue, settings)
    bus = KafkaBus(settings.stream_kafka_bootstrap_servers)
    kv = RedisKV(settings.stream_redis_url) if settings.stream_redis_url else None
    producer = BinanceProducer(
        symbols,
        bus=bus,
        topic=settings.stream_kafka_topic_raw,
        provider=provider.fetch_bars,
        poll_seconds=settings.stream_poll_seconds,
        poll_timeout_seconds=settings.stream_poll_timeout_seconds,
        history_minutes=settings.stream_history_minutes,
        kv=kv,
        dlq_topic=f"{settings.stream_kafka_topic_raw}.dlq",
    )
    logger.info(
        "producer publishing %s (%s) → %s (poll %ss)",
        ",".join(symbols),
        venue,
        settings.stream_kafka_topic_raw,
        settings.stream_poll_seconds,
    )
    try:
        producer.run_forever()
    except KeyboardInterrupt:
        logger.info("producer stopped")


if __name__ == "__main__":
    main()
