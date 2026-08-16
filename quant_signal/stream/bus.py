"""Message bus abstraction over Kafka, with an in-memory fake for tests.

The rest of the streaming layer depends only on ``MessageBus`` (JSON payloads
keyed by symbol), so it is fully testable without a broker: tests inject
``FakeBus``. ``KafkaBus`` wraps confluent-kafka's Producer/Consumer; importing
it never opens a connection, so tests stay hermetic.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer

logger = logging.getLogger(__name__)

# librdkafka keeps dead TCP sockets undetected by default (keepalive off) on
# macOS, so a half-open connection to redpanda silently wedges the consumer's
# fetch for minutes. Explicit socket.timeout.ms (≥ session.timeout.ms 45s +1s,
# per librdkafka#2363/#1915) + keepalive bound the hang, and the deadlines below
# abandon+recreate a consumer that still stalls.
_CONSUMER_SOCKET_TIMEOUT_MS = 60_000
_CONSUMER_POLL_DEADLINE_SECONDS = 30.0
# Above the feature publish cadence (1h for crypto.features.1h): an idle spell
# shorter than this is NORMAL (one bar per hour), not a wedged connection. Set to
# 2h so the hourly book is not force-recreated (resetting to `latest` and missing
# the single hourly publish) between bars — only a genuine poll-deadline wedge
# triggers recreation now.
_CONSUMER_MAX_IDLE_SECONDS = 7200.0


def _serialize(value: Mapping) -> bytes:
    return json.dumps(dict(value), separators=(",", ":")).encode("utf-8")


def _deserialize(payload: bytes | None) -> dict:
    if not payload:
        return {}
    try:
        return json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("dropping malformed message payload")
        return {}


class MessageBus(Protocol):
    """Publish JSON events keyed by symbol; consume them back in order."""

    def publish(self, topic: str, key: str, value: Mapping) -> None: ...

    def flush(self, timeout: float = 10.0) -> None:
        """Block until buffered messages are delivered (call at cycle end)."""

    def iter_consume(
        self,
        topics: str | Sequence[str],
        group_id: str,
        stop: threading.Event | None = None,
    ) -> Iterator[tuple[str, dict]]:
        """Yield ``(topic, message)`` pairs forever; blocks when idle.

        ``stop`` is a cooperative exit: the loop wakes regularly (≤1s) and ends
        when the event is set, so consumer threads shut down cleanly.
        """
        ...


class KafkaBus:
    """confluent-kafka backed bus. JSON on the wire, key = symbol."""

    def __init__(self, bootstrap_servers: str, *, client_id: str = "quant-signal") -> None:
        self._bootstrap_servers = bootstrap_servers
        self._client_id = client_id
        self._producer: Producer | None = None

    def _new_producer(self) -> Producer:
        return Producer(
            {
                "bootstrap.servers": self._bootstrap_servers,
                "client.id": self._client_id,
                "linger.ms": 50,
                "acks": "all",
                "retries": 5,
                "socket.keepalive.enable": True,
                "socket.timeout.ms": _CONSUMER_SOCKET_TIMEOUT_MS,
            }
        )

    def publish(self, topic: str, key: str, value: Mapping) -> None:
        if self._producer is None:
            self._producer = self._new_producer()
        self._producer.produce(topic, key=key.encode("utf-8"), value=_serialize(value))
        self._producer.poll(0)

    def flush(self, timeout: float = 10.0) -> None:
        """Block until buffered messages are delivered (call at cycle end)."""
        if self._producer is not None:
            self._producer.flush(timeout)

    def recreate(self) -> "KafkaBus":
        """Fresh client state — called after a wedged poll cycle is abandoned.

        confluent-kafka's Producer is not safe for concurrent use, so the old
        (possibly still-blocked) producer must not be reused by the next poll.
        """
        return KafkaBus(self._bootstrap_servers, client_id=self._client_id)

    def _new_consumer(self, group_id: str) -> Consumer:
        return Consumer(
            {
                "bootstrap.servers": self._bootstrap_servers,
                "group.id": group_id,
                "client.id": f"{self._client_id}-{group_id}",
                "auto.offset.reset": "latest",
                "enable.auto.commit": True,
                "auto.commit.interval.ms": 2000,
                "socket.keepalive.enable": True,
                "socket.timeout.ms": _CONSUMER_SOCKET_TIMEOUT_MS,
                # A handler that touches a slow external venue (e.g. Bybit Demo
                # fill confirmation) must be allowed to exceed the default 5min
                # poll window without the consumer leaving the group. The venue
                # HTTP calls are time-bounded, so this is a safety net, not a
                # license to block.
                "max.poll.interval.ms": 600000,
            }
        )

    @staticmethod
    def _bounded_poll(
        consumer: Consumer, timeout: float, deadline: float
    ) -> tuple[Any, str | Exception | None]:
        """Poll on a daemon thread so a wedged socket cannot stall the loop.

        Returns ``(msg, None)`` normally, ``(None, exc)`` when the poll raises,
        or ``(None, "WEDGED")`` when the poll outlives ``deadline`` — the
        connection is abandoned and the caller recreates the consumer.
        """
        result: list[Any] = []

        def _run() -> None:
            try:
                result.append(consumer.poll(timeout=timeout))
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller
                result.append(exc)

        thread = threading.Thread(target=_run, name="kafka-poll", daemon=True)
        thread.start()
        thread.join(timeout=deadline)
        if thread.is_alive():
            return None, "WEDGED"
        value = result[0] if result else None
        if isinstance(value, Exception):
            return None, value
        return value, None

    def iter_consume(
        self,
        topics: str | Sequence[str],
        group_id: str,
        stop: threading.Event | None = None,
    ) -> Iterator[tuple[str, dict]]:
        topics_list = [topics] if isinstance(topics, str) else list(topics)
        while True:
            if stop is not None and stop.is_set():
                return
            consumer = self._new_consumer(group_id)
            consumer.subscribe(topics_list)
            wedged = False
            last_message = time.monotonic()
            try:
                while True:
                    if stop is not None and stop.is_set():
                        return
                    idle = time.monotonic() - last_message
                    if idle > _CONSUMER_MAX_IDLE_SECONDS:
                        logger.critical(
                            "consumer %s idle %.0fs (no message); recreating to clear a "
                            "wedged connection",
                            group_id,
                            idle,
                        )
                        wedged = True
                        break
                    msg, err = self._bounded_poll(
                        consumer, timeout=1.0, deadline=_CONSUMER_POLL_DEADLINE_SECONDS
                    )
                    if err == "WEDGED":
                        logger.critical(
                            "consumer %s poll exceeded %.0fs deadline; abandoning wedged consumer",
                            group_id,
                            _CONSUMER_POLL_DEADLINE_SECONDS,
                        )
                        wedged = True
                        break
                    if isinstance(err, Exception):
                        logger.warning("consumer %s poll raised %r; recreating", group_id, err)
                        wedged = True
                        break
                    if msg is None:
                        continue
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            continue
                        raise KafkaException(msg.error())
                    last_message = time.monotonic()
                    yield msg.topic(), _deserialize(msg.value())
            finally:
                if not wedged:
                    consumer.close()


class FakeBus:
    """In-memory bus: per-topic FIFO queues, condition-var wakeups.

    Also lets tests seed messages so a consumer thread has data to read.
    """

    def __init__(self, bootstrap_servers: str = "in-memory", *, client_id: str = "test") -> None:
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self._queues: dict[str, list[tuple[str, dict]]] = {}
        self._condition = threading.Condition()

    def publish(self, topic: str, key: str, value: Mapping) -> None:
        with self._condition:
            self._queues.setdefault(topic, []).append((key, dict(value)))
            self._condition.notify_all()

    def flush(self, timeout: float = 10.0) -> None:
        return None

    def iter_consume(
        self,
        topics: str | Sequence[str],
        group_id: str,
        stop: threading.Event | None = None,
    ) -> Iterator[tuple[str, dict]]:
        wanted = set([topics] if isinstance(topics, str) else list(topics))
        with self._condition:
            seen: dict[str, int] = {}
        while True:
            if stop is not None and stop.is_set():
                return
            with self._condition:
                for topic in wanted:
                    queue = self._queues.get(topic)
                    idx = seen.get(topic, 0)
                    if queue and idx < len(queue):
                        seen[topic] = idx + 1
                        key, value = queue[idx]
                        break
                else:
                    self._condition.wait(timeout=0.2)
                    continue
            yield topic, dict(value)

    def seed(self, topic: str, messages: list[Mapping], *, key: str = "seed") -> None:
        for message in messages:
            self.publish(topic, key, message)

    def drain(self, topic: str) -> list[dict]:
        """All messages published to a topic so far (test helper)."""
        with self._condition:
            return [dict(value) for _key, value in self._queues.get(topic, [])]
