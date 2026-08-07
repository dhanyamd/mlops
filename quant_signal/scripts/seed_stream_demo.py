"""Publish synthetic minute bars to the raw Kafka topic for a live demo.

Development helper for the streaming stack (M3): seeds ``crypto.bars.raw``
with plausible OHLCV bars in the *previous and current* 5-minute buckets, so
the Flink TUMBLE window fires within a couple of minutes instead of waiting
for live Binance data. Points at the bus from ``config/settings.py``.

Run with: ``uv run python -m scripts.seed_stream_demo``
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from config.logging import configure_logging
from config.settings import get_settings
from stream.bars import df_to_bars
from stream.bus import KafkaBus


def _bucket_start(stamp: pd.Timestamp) -> pd.Timestamp:
    """Start of the 5-minute bucket containing ``stamp``."""
    return stamp.floor("5min")


def _synthetic_bars(symbol: str, seed: float, minutes: int = 3) -> pd.DataFrame:
    """A rising OHLCV series over ``minutes`` bars for a symbol."""
    rows = []
    for i in range(minutes):
        base = seed + 10 * i
        rows.append(
            {
                "symbol": symbol,
                "open": base,
                "high": base + 1.5,
                "low": base - 1.0,
                "close": base + 0.8,
                "volume": 1000 + 200 * i,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    configure_logging()
    settings = get_settings()

    now = pd.Timestamp(datetime.now(UTC))
    now_min = now.floor("min")
    buckets = [_bucket_start(now_min) - timedelta(minutes=5), _bucket_start(now_min)]

    symbols = [s.strip() for s in settings.ingest_default_crypto_symbols.split(",") if s.strip()]
    frames: list[pd.DataFrame] = []
    for idx, symbol in enumerate(symbols):
        seed = 100.0 + 40 * idx
        for bucket in buckets:
            bars = _synthetic_bars(symbol, seed, minutes=3)
            bars["ts"] = bucket + pd.to_timedelta(bars.index, unit="m")
            bars["timeframe"] = "1Min"
            bars["provider"] = "demo"
            frames.append(bars)

    payloads = [bar for frame in frames for bar in df_to_bars(frame)]
    bus = KafkaBus(settings.stream_kafka_bootstrap_servers)
    for bar in payloads:
        bus.publish(settings.stream_kafka_topic_raw, bar["symbol"], bar)
    bus.flush()
    print(f"published {len(payloads)} synthetic bars to {settings.stream_kafka_topic_raw}")


if __name__ == "__main__":
    main()
