"""Production service clients — ClickHouse, Redis, Qdrant."""

from __future__ import annotations

import json
from typing import Any

import clickhouse_connect
import redis
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from shared.config import CLICKHOUSE, QDRANT, REDIS


class ClickHouseClient:
    """OLAP warehouse client (Snowflake/BigQuery class)."""

    def __init__(self):
        self._client = clickhouse_connect.get_client(
            host=CLICKHOUSE.host,
            port=CLICKHOUSE.port,
            username=CLICKHOUSE.user,
            password=CLICKHOUSE.password or None,
        )

    def ping(self) -> bool:
        return self._client.ping()

    def insert_df(self, table: str, df, database: str) -> None:
        self._client.insert_df(f"{database}.{table}", df)

    def query_df(self, sql: str):
        return self._client.query_df(sql)

    def command(self, sql: str) -> None:
        self._client.command(sql)

    def count(self, database: str, table: str) -> int:
        result = self._client.query(
            f"SELECT count() FROM {database}.{table}"
        )
        return int(result.result_rows[0][0])


class RedisFeatureStore:
    """Online feature store — Redis (Feast online store / low-latency serving)."""

    KEY_PREFIX = "feat:card:"

    def __init__(self):
        self._client = redis.Redis(
            host=REDIS.host,
            port=REDIS.port,
            db=REDIS.db,
            password=REDIS.password,
            decode_responses=True,
        )

    def ping(self) -> bool:
        return bool(self._client.ping())

    def set_features(self, card_id: str, features: dict[str, Any], ttl_sec: int = 86400) -> None:
        key = f"{self.KEY_PREFIX}{card_id}"
        self._client.setex(key, ttl_sec, json.dumps(features))

    def get_features(self, card_id: str) -> dict[str, Any] | None:
        raw = self._client.get(f"{self.KEY_PREFIX}{card_id}")
        return json.loads(raw) if raw else None


class QdrantPatternStore:
    """
    Vector DB for fraud pattern similarity.

    Why Qdrant here (and NOT for demand forecasting):
    - Fraud labs index known fraud transaction embeddings for nearest-neighbor search
    - When a live transaction vector is close to a known fraud cluster → boost score
    - Demand forecasting uses tabular lag features (XGBoost) — vectors add no value there
    """

    def __init__(self):
        self._client = QdrantClient(host=QDRANT.host, port=QDRANT.port)
        self._collection = QDRANT.collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=QDRANT.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_fraud_patterns(self, points: list[PointStruct]) -> None:
        self._client.upsert(collection_name=self._collection, points=points)

    def search_similar(self, vector: list[float], limit: int = 5) -> list[dict]:
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=limit,
        )
        return [
            {"id": h.id, "score": h.score, "payload": h.payload or {}}
            for h in hits
        ]

    def vector_fraud_score(self, vector: list[float], threshold: float = 0.85) -> float:
        """Return 0-1 score based on nearest known fraud pattern similarity."""
        hits = self.search_similar(vector, limit=1)
        if not hits:
            return 0.0
        return float(hits[0]["score"]) if hits[0]["score"] >= threshold else 0.0
