"""Ingestion producer: Binance minute bars → Kafka bus.

Runs as a standalone process (``uv run python -m stream.producer``) — the API
no longer owns ingestion. Each poll fetches the trailing window from Binance,
publishes every bar to the raw topic (keyed by symbol → per-symbol ordering),
flushes for durability, and persists best-effort to BRONZE.CRYPTO_BARS (a
Snowflake outage degrades to a warning, never breaks the stream).

Run with ``make stream-producer``; the Kafka bus is KafkaBus unless a
``MessageBus`` fake is injected (hermetic tests).
"""

from __future__ import annotations

import threading

import pandas as pd

from config.logging import configure_logging, get_logger
from config.settings import csv_list, get_settings
from ingest.providers.binance import BinanceBarProvider
from ingest.store import write_crypto_bars
from stream.bars import df_to_bars
from stream.bus import KafkaBus, MessageBus

logger = get_logger(__name__)


class BinanceProducer:
    """Poll Binance and publish JSON bars to the ingestion bus."""

    def __init__(
        self,
        symbols: list[str],
        *,
        bus: MessageBus,
        topic: str,
        poll_seconds: int = 15,
        history_minutes: int = 180,
        provider=None,
        persist=None,
    ) -> None:
        self._symbols = [s.upper() for s in symbols]
        self._bus = bus
        self._topic = topic
        self._poll_seconds = poll_seconds
        self._history_minutes = history_minutes
        self._provider = provider or BinanceBarProvider().fetch_bars
        self._persist = persist or write_crypto_bars
        self._stop = threading.Event()

    def run_forever(self, stop: threading.Event | None = None) -> None:
        """Poll in a loop; first cycle seeds full history, later ones windowed."""
        stop = stop or self._stop
        minutes = self._history_minutes
        while not stop.is_set():
            try:
                self._poll_once(minutes)
            except Exception:  # noqa: BLE001 - a producer must never die silently
                logger.exception("binance producer poll failed")
            minutes = max(3, int(self._poll_seconds / 60) + 1)
            stop.wait(self._poll_seconds)

    def _poll_once(self, minutes: int) -> int:
        # ``fetch_bars(symbols, days, minutes=None)``: pass minutes as a keyword
        # so a positional ``minutes`` doesn't silently bind to ``days`` (which
        # would fetch *days* of paged history and stall the first poll).
        df = self._provider(self._symbols, days=0, minutes=minutes)
        if df is None or df.empty:
            return 0
        bars = df_to_bars(df)
        for bar in bars:
            self._bus.publish(self._topic, bar["symbol"], bar)
        self._bus.flush()
        self._persist_best_effort(df)
        return len(bars)

    def _persist_best_effort(self, df: pd.DataFrame) -> None:
        try:
            self._persist(df)
        except Exception:  # noqa: BLE001 - best-effort persistence
            logger.warning("crypto persist failed (best-effort)", exc_info=True)


def main() -> None:
    configure_logging()
    settings = get_settings()
    symbols = csv_list(settings.ingest_default_crypto_symbols)
    if not symbols:
        logger.error("no crypto symbols configured; set INGEST_DEFAULT_CRYPTO_SYMBOLS")
        return
    bus = KafkaBus(settings.stream_kafka_bootstrap_servers)
    producer = BinanceProducer(
        symbols,
        bus=bus,
        topic=settings.stream_kafka_topic_raw,
        poll_seconds=settings.stream_poll_seconds,
        history_minutes=settings.stream_history_minutes,
    )
    logger.info(
        "binance producer publishing %s → %s (poll %ss)",
        ",".join(symbols),
        settings.stream_kafka_topic_raw,
        settings.stream_poll_seconds,
    )
    try:
        producer.run_forever()
    except KeyboardInterrupt:
        logger.info("producer stopped")


if __name__ == "__main__":
    main()
