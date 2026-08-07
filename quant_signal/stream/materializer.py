"""Online store materializer: Kafka topics → Redis (the online feature store).

Consumes the raw 1m bars and the Flink-computed 5m window features, then
materializes them into Redis so the dashboard reads are sub-500ms and never
touch Kafka or Snowflake:
  live:crypto:<SYMBOL>      latest 1m bar (SET)
  feature:crypto:5m:<SYMBOL> bounded history of window features (RPUSH+LTRIM)

Run with ``make stream-materializer``. Pure logic lives in ``handle`` so tests
drive it directly with ``FakeBus``/``FakeKV``.
"""

from __future__ import annotations

import threading

from config.logging import configure_logging, get_logger
from config.settings import get_settings
from stream.bus import KafkaBus, MessageBus
from stream.kv import KVStore, RedisKV

logger = get_logger(__name__)


def live_key(prefix: str, symbol: str) -> str:
    return f"{prefix}:{symbol.upper()}"


def feature_key(prefix: str, symbol: str) -> str:
    return f"{prefix}:{symbol.upper()}"


class OnlineStoreMaterializer:
    """Write-side of the Redis online store, fed by the Kafka bus."""

    def __init__(
        self,
        bus: MessageBus,
        kv: KVStore,
        *,
        raw_topic: str,
        features_topic: str,
        live_prefix: str,
        feature_prefix: str,
        feature_maxlen: int = 200,
        group_id: str = "redis-materializer",
    ) -> None:
        self._bus = bus
        self._kv = kv
        self._raw_topic = raw_topic
        self._features_topic = features_topic
        self._live_prefix = live_prefix
        self._feature_prefix = feature_prefix
        self._feature_maxlen = feature_maxlen
        self._group_id = group_id

    def handle(self, topic: str, msg: dict) -> None:
        symbol = str(msg.get("symbol") or "").upper()
        if not symbol:
            return
        if topic == self._raw_topic:
            self._kv.set_json(live_key(self._live_prefix, symbol), msg)
        elif topic == self._features_topic:
            self._kv.push_json(
                feature_key(self._feature_prefix, symbol),
                msg,
                maxlen=self._feature_maxlen,
            )

    def run_forever(self, stop: threading.Event | None = None) -> None:
        for topic, msg in self._bus.iter_consume(
            [self._raw_topic, self._features_topic], self._group_id, stop=stop
        ):
            self.handle(topic, msg)


def main() -> None:
    configure_logging()
    settings = get_settings()
    bus = KafkaBus(settings.stream_kafka_bootstrap_servers)
    kv = RedisKV(settings.stream_redis_url)
    materializer = OnlineStoreMaterializer(
        bus,
        kv,
        raw_topic=settings.stream_kafka_topic_raw,
        features_topic=settings.stream_kafka_topic_features,
        live_prefix=settings.stream_redis_live_prefix,
        feature_prefix=settings.stream_redis_feature_prefix,
        feature_maxlen=settings.stream_redis_feature_maxlen,
    )
    logger.info(
        "materializer consuming %s+%s → %s",
        settings.stream_kafka_topic_raw,
        settings.stream_kafka_topic_features,
        settings.stream_redis_url,
    )
    try:
        materializer.run_forever()
    except KeyboardInterrupt:
        logger.info("materializer stopped")


if __name__ == "__main__":
    main()
