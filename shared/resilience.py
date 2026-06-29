"""Resilience patterns — circuit breaker, retry, fallback model.

Why resilience matters:
  In production, dependencies fail. Redis goes down. Qdrant times out.
  Without a circuit breaker, every request hangs for 30 seconds waiting for
  the dead service, backpressure builds, and your entire API goes down.

  With a circuit breaker:
    - After N failures, the breaker "opens" and immediately returns a fallback.
    - After a cooldown, it "half-opens" and tries one real request.
    - If that succeeds, it closes and goes back to normal.
    - Failure in one service is isolated — the rest of the system stays alive.

Circuit breaker states:
  CLOSED  → normal operation (requests go through)
  OPEN    → failing fast (requests immediately get fallback, no real call)
  HALF-OPEN → testing recovery (one request goes through, then re-evaluates)

This is the same pattern used at Netflix (Hystrix), Stripe, and Uber.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable

from shared.observability.logging import get_logger
from shared.observability.metrics import CIRCUIT_BREAKER_STATE

log = get_logger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"       # Healthy — requests flow through
    OPEN = "open"           # Failing — requests short-circuit to fallback
    HALF_OPEN = "half_open" # Recovering — one test request allowed


@dataclass
class CircuitBreaker:
    """Production circuit breaker with Prometheus metrics integration.

    Usage:
        breaker = CircuitBreaker(name="redis", failure_threshold=5, timeout=30)

        try:
            result = breaker.call(redis.get_features, card_id)
        except CircuitOpenError:
            result = None  # use fallback
    """

    name: str
    failure_threshold: int = 5       # Open circuit after this many consecutive failures
    success_threshold: int = 2       # Close circuit after this many consecutive successes in half-open
    timeout: float = 30.0            # Seconds before attempting recovery (half-open)

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.timeout:
                log.info("circuit_half_open", service=self.name)
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
        return self._state

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute func through the circuit breaker.

        Raises CircuitOpenError immediately if circuit is OPEN (don't call func).
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            CIRCUIT_BREAKER_STATE.labels(service=self.name).set(1)
            raise CircuitOpenError(f"Circuit breaker OPEN for service: {self.name}")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure(exc)
            raise

    def _on_success(self) -> None:
        self._failure_count = 0
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                log.info("circuit_closed", service=self.name)
                self._state = CircuitState.CLOSED
        CIRCUIT_BREAKER_STATE.labels(service=self.name).set(0)

    def _on_failure(self, exc: Exception) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        log.warning(
            "circuit_failure",
            service=self.name,
            failure_count=self._failure_count,
            error=str(exc),
        )
        if self._failure_count >= self.failure_threshold:
            if self._state != CircuitState.OPEN:
                log.error("circuit_opened", service=self.name, failures=self._failure_count)
                self._state = CircuitState.OPEN
            CIRCUIT_BREAKER_STATE.labels(service=self.name).set(1)


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is OPEN and a call is attempted."""


@dataclass
class FallbackModel:
    """Simple rule-based fallback when XGBoost model is unavailable.

    Why a fallback model exists:
      If MLflow is down or the champion model fails to load, we still need to
      make a decision. A rule-based heuristic is worse than XGBoost but better
      than crashing the service.

    In production: fallback to the previous model version (stored locally),
    not a heuristic. Here we use a heuristic for simplicity.
    """

    # Thresholds derived from domain knowledge
    HIGH_AMOUNT_THRESHOLD: float = 5000.0   # Unusual transaction amount
    VELOCITY_THRESHOLD: int = 10             # Too many transactions in 1h

    def predict_proba(self, features: dict) -> float:
        """Return a fraud probability using simple rules.

        Returns a score between 0 and 1.
        """
        score = 0.0

        # High amount signal
        amount = features.get("amount", 0)
        if amount > self.HIGH_AMOUNT_THRESHOLD:
            score += 0.4
        elif amount > 1000:
            score += 0.1

        # High velocity signal
        txn_count = features.get("txn_count_1h", 0)
        if txn_count > self.VELOCITY_THRESHOLD:
            score += 0.4
        elif txn_count > 5:
            score += 0.2

        # Night-time transaction
        if features.get("is_night", 0):
            score += 0.1

        # Deviation from normal spending
        deviation = features.get("amount_deviation", 1.0)
        if deviation > 5.0:
            score += 0.2

        return min(score, 1.0)
