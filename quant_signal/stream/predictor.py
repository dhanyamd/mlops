"""Online return prediction: River streaming regressor + conformal intervals.

Consumes the Flink 5m feature windows from Kafka and, per symbol, trains an
online regression model on the *realized* next-window return (learned only
once the next window closes — no lookahead). Predictions carry a conformal
interval whose nominal level is adapted online so that long-run coverage
tracks the target even as the market drifts:

  Adaptive Conformal Inference (Gibbs & Candès, 2021):
    C_t = [ y_hat_t - q_t,  y_hat_t + q_t ]
    q_t = (1 - alpha_t)-quantile of the trailing window of residuals |y - y_hat|
    alpha_{t+1} = alpha_t + gamma * (alpha - err_t),  err_t = 1{ y_t not in C_t }

Outputs land in the online store as ``prediction:crypto:5m:<SYMBOL>`` (a SET),
so the API serves them sub-500ms without touching Kafka or Snowflake.

Run with ``make stream-predictor``. Pure logic lives in ``handle`` so tests
drive it directly with ``FakeBus``/``FakeKV``; the River model is streamed
in-process, no external state.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from river import compose, linear_model, optim, preprocessing

from config.logging import configure_logging, get_logger
from config.settings import get_settings
from stream.bus import KafkaBus, MessageBus
from stream.kv import KVStore, RedisKV

logger = get_logger(__name__)


def prediction_key(prefix: str, symbol: str) -> str:
    return f"{prefix}:{symbol.upper()}"


def strategy_key(prefix: str, symbol: str) -> str:
    return f"{prefix}:{symbol.upper()}"


@dataclass
class ConformalInterval:
    """Adaptive Conformal Inference wrapper (Gibbs & Candès, 2021).

    ``alpha`` is the target miscoverage (e.g. 0.1 → 90% coverage), ``gamma``
    the adaptation step. ``alpha_t`` tracks the nominal level; ``err_t`` is
    coverage on the trailing window for observability.
    """

    alpha: float = 0.1
    gamma: float = 0.005
    residual_window: int = 200
    alpha_t: float = 0.1
    _residuals: deque[float] = field(default_factory=deque)
    _hits: deque[bool] = field(default_factory=deque)

    def predict(self, y_hat: float) -> tuple[float, float]:
        """Interval ``(low, high)`` around a point prediction (split conformal)."""
        if not self._residuals:
            return y_hat, y_hat
        n = len(self._residuals)
        q_index = min(n - 1, math.ceil((1 - self.alpha_t) * (n + 1)) - 1)
        q = sorted(self._residuals)[q_index]
        return y_hat - q, y_hat + q

    def update(self, y: float, y_hat: float, interval: tuple[float, float]) -> None:
        """Observe the realized target; adapt alpha toward nominal coverage."""
        low, high = interval
        hit = low <= y <= high
        self._residuals.append(abs(y - y_hat))
        self._hits.append(hit)
        while len(self._residuals) > self.residual_window:
            self._residuals.popleft()
            self._hits.popleft()
        self.alpha_t = min(
            0.99, max(0.01, self.alpha_t + self.gamma * (self.alpha - (0.0 if hit else 1.0)))
        )

    def coverage(self) -> float | None:
        """Empirical coverage on the trailing window (None while warming up)."""
        if len(self._hits) < 2:
            return None
        return sum(self._hits) / len(self._hits)


def _features(msg: dict) -> dict[str, float] | None:
    """Numeric feature dict from a Flink 5m window, or None if malformed.

    Uses only current-window values (no lookahead). Missing/NaN fields are
    dropped so River never sees a non-numeric feature.
    """
    features: dict[str, float] = {}
    close = msg.get("close")
    if not isinstance(close, (int, float)) or close != close or close == 0:
        return None
    if isinstance(msg.get("open"), (int, float)) and msg["open"]:
        features["ret_in_window"] = close / msg["open"] - 1.0
    if isinstance(msg.get("high"), (int, float)) and isinstance(msg.get("low"), (int, float)):
        features["range_pct"] = (msg["high"] - msg["low"]) / close
    if isinstance(msg.get("vwap"), (int, float)) and msg["vwap"]:
        features["vwap_spread"] = (msg["vwap"] - close) / close
    if isinstance(msg.get("volume"), (int, float)):
        features["log_volume"] = math.log1p(float(msg["volume"]))
    if isinstance(msg.get("bar_count"), int):
        features["bar_count"] = float(msg["bar_count"])
    return features or None


def _model() -> preprocessing.TargetStandardScaler:
    """Streaming regressor: standardized features → SGD linear regression.

    ``TargetStandardScaler`` normalizes the target (returns are small), so the
    learned weights are stable; predictions come back in original units.
    """
    pipeline = compose.Pipeline(
        ("scale", preprocessing.StandardScaler()),
        (
            "lin",
            linear_model.LinearRegression(optimizer=optim.SGD(0.01), l2=0.001),
        ),
    )
    return preprocessing.TargetStandardScaler(regressor=pipeline)


@dataclass
class _SymbolState:
    model: preprocessing.TargetStandardScaler = field(default_factory=_model)
    conformal: ConformalInterval = field(default_factory=ConformalInterval)
    last_features: dict[str, float] | None = None
    last_close: float | None = None
    last_y_hat: float | None = None
    last_interval: tuple[float, float] | None = None
    last_direction: str | None = None
    last_window_end_ms: int | None = None
    # Compounded strategy equity from realized directions, vs buy-and-hold.
    equity_strategy: list[float] = field(default_factory=lambda: [1.0])
    equity_buyhold: list[float] = field(default_factory=lambda: [1.0])
    n_trades: int = 0
    n_wins: int = 0
    n_windows: int = 0


def _direction(y_hat: float) -> str:
    """Trading rule: long/short when the predicted return is meaningful."""
    if y_hat > 1e-4:
        return "LONG"
    if y_hat < -1e-4:
        return "SHORT"
    return "FLAT"


class OnlinePredictor:
    """Per-symbol online return model fed by the 5m feature stream."""

    def __init__(
        self,
        kv: KVStore,
        *,
        prediction_prefix: str,
        strategy_prefix: str | None = None,
        strategy_maxlen: int = 500,
        alpha: float = 0.1,
        gamma: float = 0.005,
        residual_window: int = 200,
    ) -> None:
        self._kv = kv
        self._prediction_prefix = prediction_prefix
        self._strategy_prefix = strategy_prefix
        self._strategy_maxlen = strategy_maxlen
        self._states: dict[str, _SymbolState] = {}
        self._default = (alpha, gamma, residual_window)

    def _state(self, symbol: str) -> _SymbolState:
        state = self._states.get(symbol)
        if state is None:
            alpha, gamma, residual_window = self._default
            state = _SymbolState(
                conformal=ConformalInterval(
                    alpha=alpha, gamma=gamma, residual_window=residual_window
                )
            )
            self._states[symbol] = state
        return state

    def handle(self, msg: dict) -> None:
        """One feature window: learn from the *previous* window's target, then
        predict the next return and write the conformal interval to the store.
        """
        symbol = str(msg.get("symbol") or "").upper()
        if not symbol:
            return
        features = _features(msg)
        if features is None:
            return
        close = float(msg["close"])
        window_end = msg.get("window_end_ms")
        state = self._state(symbol)

        if state.last_features is not None and state.last_close is not None and close != 0:
            realized = close / state.last_close - 1.0
            state.model.learn_one(state.last_features, realized)
            if state.last_y_hat is not None and state.last_interval is not None:
                state.conformal.update(realized, state.last_y_hat, state.last_interval)
            self._record_period(state, state.last_direction, realized)

        y_hat = float(state.model.predict_one(features))
        interval = state.conformal.predict(y_hat)
        direction = _direction(y_hat)

        state.last_features = features
        state.last_close = close
        state.last_y_hat = y_hat
        state.last_interval = interval
        state.last_direction = direction
        state.last_window_end_ms = window_end

        self._kv.set_json(
            prediction_key(self._prediction_prefix, symbol),
            {
                "symbol": symbol,
                "window_end_ms": window_end,
                "predicted_return": round(y_hat, 6),
                "interval_low": round(interval[0], 6),
                "interval_high": round(interval[1], 6),
                "direction": direction,
                "alpha": round(state.conformal.alpha_t, 4),
                "coverage": state.conformal.coverage(),
                "updated_at": self._kv_now(),
            },
        )
        if self._strategy_prefix:
            self._write_strategy(state, symbol)

    def _record_period(self, state: _SymbolState, direction: str | None, realized: float) -> None:
        """Compound equity after the previous window's prediction matures.

        The previous direction earns ``+realized`` (LONG), ``-realized``
        (SHORT), or nothing (FLAT); the market earns ``+realized`` either way.
        """
        if direction == "LONG":
            strat = realized
        elif direction == "SHORT":
            strat = -realized
        else:
            strat = 0.0
        if direction in ("LONG", "SHORT"):
            state.n_trades += 1
            if strat > 0.0:
                state.n_wins += 1
        state.n_windows += 1
        state.equity_strategy.append(state.equity_strategy[-1] * (1.0 + strat))
        state.equity_buyhold.append(state.equity_buyhold[-1] * (1.0 + realized))
        if len(state.equity_strategy) > self._strategy_maxlen:
            del state.equity_strategy[: -self._strategy_maxlen]
        if len(state.equity_buyhold) > self._strategy_maxlen:
            del state.equity_buyhold[: -self._strategy_maxlen]

    def _write_strategy(self, state: _SymbolState, symbol: str) -> None:
        eq_strat = state.equity_strategy
        eq_buy = state.equity_buyhold
        self._kv.set_json(
            strategy_key(self._strategy_prefix or "", symbol),
            {
                "symbol": symbol,
                "n_windows": state.n_windows,
                "n_trades": state.n_trades,
                "n_wins": state.n_wins,
                "win_rate": (round(state.n_wins / state.n_trades, 4) if state.n_trades else None),
                "strategy_equity": [round(e, 6) for e in eq_strat],
                "buyhold_equity": [round(e, 6) for e in eq_buy],
                "total_return_strategy": round(eq_strat[-1] - 1.0, 6),
                "total_return_buyhold": round(eq_buy[-1] - 1.0, 6),
                "updated_at": self._kv_now(),
            },
        )

    @staticmethod
    def _kv_now() -> str:
        from datetime import UTC, datetime

        return datetime.now(UTC).isoformat()

    def run_forever(
        self,
        bus: MessageBus,
        features_topic: str,
        group_id: str,
        stop: threading.Event | None = None,
    ) -> None:
        for _topic, msg in bus.iter_consume(features_topic, group_id, stop=stop):
            self.handle(msg)

    def warm_start(self, features: list[Mapping]) -> None:
        """Rebuild per-symbol model state from the online store's history.

        A stateful online model must not start empty: replaying the already
        materialized feature windows (oldest first) calibrates residuals and
        intervals immediately instead of waiting for Kafka ``latest`` offsets.
        """
        for msg in features:
            self.handle(dict(msg))


def main() -> None:
    configure_logging()
    settings = get_settings()
    bus = KafkaBus(settings.stream_kafka_bootstrap_servers)
    kv = RedisKV(settings.stream_redis_url)
    predictor = OnlinePredictor(
        kv,
        prediction_prefix=settings.stream_redis_prediction_prefix,
        strategy_prefix=settings.stream_redis_strategy_prefix,
        strategy_maxlen=settings.stream_strategy_maxlen,
        alpha=settings.stream_prediction_alpha,
        gamma=settings.stream_prediction_gamma,
        residual_window=settings.stream_prediction_residual_window,
    )
    logger.info(
        "predictor consuming %s → %s (alpha=%s, gamma=%s)",
        settings.stream_kafka_topic_features,
        settings.stream_redis_prediction_prefix,
        settings.stream_prediction_alpha,
        settings.stream_prediction_gamma,
    )
    from config.settings import csv_list
    from stream.materializer import feature_key

    for symbol in csv_list(settings.ingest_default_crypto_symbols):
        history = kv.list_json(feature_key(settings.stream_redis_feature_prefix, symbol))
        if history:
            predictor.warm_start(history)
            logger.info("predictor warm-started %s from %d windows", symbol, len(history))
    try:
        predictor.run_forever(
            bus,
            settings.stream_kafka_topic_features,
            group_id="online-predictor",
        )
    except KeyboardInterrupt:
        logger.info("predictor stopped")


if __name__ == "__main__":
    main()
