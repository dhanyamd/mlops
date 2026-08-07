"""Message bus abstraction over Kafka, with an in-memory fake for tests.

The rest of the streaming layer depends only on ``MessageBus`` (JSON payloads
keyed by symbol), so it is fully testable without a broker: tests inject
``FakeBus``. ``KafkaBus`` wraps confluent-kafka's Producer/Consumer; importing
it never opens a connection, so tests stay hermetic.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import Protocol

from confluent_kafka import Consumer, KafkaError, KafkaException, Producer


def _serialize(value: Mapping) -> bytes:
    return json.dumps(dict(value), separators=(",", ":")).encode("utf-8")


def _deserialize(payload: bytes | None) -> dict:
    if not payload:
        return {}
    return json.loads(payload.decode("utf-8"))


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

    def iter_consume(
        self,
        topics: str | Sequence[str],
        group_id: str,
        stop: threading.Event | None = None,
    ) -> Iterator[tuple[str, dict]]:
        consumer = Consumer(
            {
                "bootstrap.servers": self._bootstrap_servers,
                "group.id": group_id,
                "client.id": f"{self._client_id}-{group_id}",
                "auto.offset.reset": "latest",
                "enable.auto.commit": True,
            }
        )
        consumer.subscribe([topics] if isinstance(topics, str) else list(topics))
        try:
            while True:
                if stop is not None and stop.is_set():
                    return
                msg = consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise KafkaException(msg.error())
                yield msg.topic(), _deserialize(msg.value())
        finally:
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
