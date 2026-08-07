"""Online store (KV) abstraction over Redis, with an in-memory fake.

Redis is the online feature store: 1m ``live`` bars and 5m window features are
materialized here for sub-500ms serving. Consumers depend on ``KVStore`` so
tests inject ``FakeKV``; ``RedisKV`` is a thin wrapper over the redis client
(``decode_responses=True`` keeps values as JSON-safe strings).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import redis


class KVStore(Protocol):
    """Key → JSON value store with bounded lists for window history."""

    def get_json(self, key: str) -> Mapping | None: ...

    def set_json(self, key: str, value: Mapping) -> None: ...

    def push_json(self, key: str, value: Mapping, *, maxlen: int) -> None: ...

    def list_json(self, key: str, *, reverse: bool = False, maxlen: int = 50) -> list[Mapping]: ...


class RedisKV:
    def __init__(self, url: str) -> None:
        self._client = redis.Redis.from_url(url, decode_responses=True)

    def get_json(self, key: str) -> Mapping | None:
        raw = self._client.get(key)
        if not raw:
            return None
        import json

        return json.loads(raw)

    def set_json(self, key: str, value: Mapping) -> None:
        import json

        self._client.set(key, json.dumps(value))

    def push_json(self, key: str, value: Mapping, *, maxlen: int) -> None:
        import json

        pipe = self._client.pipeline()
        pipe.rpush(key, json.dumps(value))
        pipe.ltrim(key, -maxlen, -1)
        pipe.execute()

    def list_json(self, key: str, *, reverse: bool = False, maxlen: int = 50) -> list[Mapping]:
        import json

        raw = self._client.lrange(key, -maxlen, -1)
        items = [json.loads(entry) for entry in raw]
        return list(reversed(items)) if reverse else items


class FakeKV:
    """Dict-backed KV store for hermetic tests."""

    def __init__(self) -> None:
        self._values: dict[str, Mapping] = {}
        self._lists: dict[str, list[Mapping]] = {}

    def get_json(self, key: str) -> Mapping | None:
        return self._values.get(key)

    def set_json(self, key: str, value: Mapping) -> None:
        self._values[key] = dict(value)

    def push_json(self, key: str, value: Mapping, *, maxlen: int) -> None:
        self._lists.setdefault(key, []).append(dict(value))
        self._lists[key] = self._lists[key][-maxlen:]

    def list_json(self, key: str, *, reverse: bool = False, maxlen: int = 50) -> list[Mapping]:
        items = list(self._lists.get(key, []))
        items = items[-maxlen:]
        return list(reversed(items)) if reverse else items
