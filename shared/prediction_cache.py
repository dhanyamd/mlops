"""Redis prediction cache — avoids re-scoring identical transactions.

Why prediction caching:
  The same card may trigger multiple fraud score requests within seconds
  (e.g., retries, batch processing). Re-running XGBoost + Qdrant for an
  identical request wastes compute. A Redis cache returns the stored result
  in <1ms instead of running the full pipeline.

Cache key strategy:
  Hash of (card_id + amount + merchant_category) → deterministic, same inputs
  always produce the same cache key.

  We do NOT include timestamp in the key — a transaction at 14:32 and 14:33
  with identical amount/card/merchant should reuse the cached result since
  none of the features change within a short window.

TTL (Time to Live):
  60 seconds. After 60s, re-score. Velocity features change over time (a card
  that looked normal at 14:32 might have 15 more transactions by 14:33).
  60s is short enough to catch velocity changes, long enough to absorb retries.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Optional

import redis

from shared.config import REDIS
from shared.observability.logging import get_logger
from shared.observability.metrics import REDIS_FEATURE_FETCH_DURATION

log = get_logger(__name__)

_CACHE_PREFIX = "pred:fraud:"
_DEFAULT_TTL_SECONDS = 60


class PredictionCache:
    """Redis-backed prediction cache for fraud scoring results.

    Usage:
        cache = PredictionCache()

        cached = cache.get(card_id, amount, merchant_category)
        if cached:
            return {**cached, "cache_hit": True}

        result = run_full_inference(txn)
        cache.set(card_id, amount, merchant_category, result)
        return result
    """

    def __init__(self, ttl_seconds: int = _DEFAULT_TTL_SECONDS):
        self._client = redis.Redis(
            host=REDIS.host,
            port=REDIS.port,
            db=REDIS.db,
            password=REDIS.password,
            decode_responses=True,
            socket_connect_timeout=1.0,  # fail fast if Redis is down
            socket_timeout=0.5,
        )
        self.ttl_seconds = ttl_seconds

    def _make_key(self, card_id: str, amount: float, merchant_category: str) -> str:
        """Deterministic cache key from the inputs that define the fraud decision."""
        raw = f"{card_id}:{round(amount, 2)}:{merchant_category}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"{_CACHE_PREFIX}{digest}"

    def get(self, card_id: str, amount: float, merchant_category: str) -> Optional[dict[str, Any]]:
        """Return cached prediction or None on miss/error."""
        key = self._make_key(card_id, amount, merchant_category)
        start = time.monotonic()
        try:
            raw = self._client.get(key)
            elapsed = time.monotonic() - start
            REDIS_FEATURE_FETCH_DURATION.observe(elapsed)

            if raw:
                log.debug("prediction_cache_hit", card_id=card_id, key=key)
                return json.loads(raw)
            log.debug("prediction_cache_miss", card_id=card_id)
            return None
        except redis.RedisError as exc:
            # Cache miss is always safer than propagating errors
            log.warning("prediction_cache_get_error", error=str(exc))
            return None

    def set(
        self,
        card_id: str,
        amount: float,
        merchant_category: str,
        prediction: dict[str, Any],
    ) -> None:
        """Store prediction in cache. Silently swallows errors — cache is best-effort."""
        key = self._make_key(card_id, amount, merchant_category)
        try:
            self._client.setex(key, self.ttl_seconds, json.dumps(prediction, default=str))
            log.debug("prediction_cached", card_id=card_id, ttl=self.ttl_seconds)
        except redis.RedisError as exc:
            log.warning("prediction_cache_set_error", error=str(exc))

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except redis.RedisError:
            return False
