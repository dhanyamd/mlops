"""Online return prediction: River streaming regressor + conformal intervals.

Consumes the Flink 1h feature windows from Kafka and, per symbol, trains an
online regression model on the *realized* next-window return (learned only
once the next window closes — no lookahead). Predictions carry a conformal
interval whose nominal level is adapted online so that long-run coverage
tracks the target even as the market drifts:

  Adaptive Conformal Inference (Gibbs & Candès, 2021):
    C_t = [ y_hat_t - q_t,  y_hat_t + q_t ]
    q_t = (1 - alpha_t)-quantile of the trailing window of residuals |y - y_hat|
    alpha_{t+1} = alpha_t + gamma * (alpha - err_t),  err_t = 1{ y_t not in C_t }

Outputs land in the online store as ``prediction:crypto:1h:<SYMBOL>`` (a SET),
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
from river.base.typing import FeatureName

from config.logging import configure_logging, get_logger
from config.settings import get_settings
from stream.bus import KafkaBus, MessageBus
from stream.kv import KVStore, RedisKV

logger = get_logger(__name__)

# Sanity bounds. A 1-hour crypto window with a ratio feature (open→close,
# high→low, vwap→close) beyond 50% is a corrupt bar, not a real move. Hourly
# BTC is far less wild: ~0.73% per-hour 1σ (Bysik & Ślepaczuk, "Machine
# Learning-Based Bitcoin Trading Under Transaction Costs", arXiv:2606.00060,
# 2026 — descriptive stats over 70,872 hourly observations: min −20.1%,
# max +16.0%), i.e. a 20% move is ~27σ and a 10% move ~14σ of a normal hour.
# The 5m-era bound (10%) was calibrated to 5-minute bars where a 10% move is
# ~77σ; at 1h it would reject real, tradeable shock bars. Rejecting only
# genuinely implausible windows stops the online scalers from being poisoned:
# River's maintainers document that its online StandardScaler is unstable when
# a feature changes abruptly (online-ml/river#335) — a few corrupt-window
# learns run the model away to absurd predictions that online learning can
# never forget.
_MAX_FEATURE_RATIO = 0.5
# A 1-hour close-to-close return beyond 50% is implausible (real hourly moves
# are <~5%, and 50% ≈ 68σ of hourly volatility); skip learning on it so a
# corrupt close can't explode the target scaler or the conformal residuals.
_MAX_REALIZED = 0.5


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


def _features(
    msg: Mapping, cross_returns: Mapping[str, float] | None = None
) -> dict[FeatureName, float] | None:
    """Numeric feature dict from a Flink 1h window, or None if malformed.

    Uses only current-window values (no lookahead). ``cross_returns`` carries
    each cross symbol's realized return over the window that closed BEFORE this
    one (computed by the caller from the cross-close registry, so only past
    closes are used). Missing/NaN fields are dropped so River never sees a
    non-numeric feature, and windows whose ratio features are implausible (a
    corrupt bar, e.g. an ``open`` or ``close`` off by orders of magnitude) are
    rejected outright so the online model can never learn on garbage.
    """
    features: dict[FeatureName, float] = {}
    close = msg.get("close")
    if not isinstance(close, (int, float)) or close != close or close == 0:
        return None
    if isinstance(msg.get("open"), (int, float)) and msg["open"]:
        ret_in_window = close / msg["open"] - 1.0
        if abs(ret_in_window) > _MAX_FEATURE_RATIO:
            return None
        features["ret_in_window"] = ret_in_window
    if isinstance(msg.get("high"), (int, float)) and isinstance(msg.get("low"), (int, float)):
        range_pct = (msg["high"] - msg["low"]) / close
        if abs(range_pct) > _MAX_FEATURE_RATIO:
            return None
        features["range_pct"] = range_pct
    if isinstance(msg.get("vwap"), (int, float)) and msg["vwap"]:
        vwap_spread = (msg["vwap"] - close) / close
        if abs(vwap_spread) > _MAX_FEATURE_RATIO:
            return None
        features["vwap_spread"] = vwap_spread
    if isinstance(msg.get("volume"), (int, float)):
        features["log_volume"] = math.log1p(float(msg["volume"]))
    if isinstance(msg.get("bar_count"), int):
        features["bar_count"] = float(msg["bar_count"])
    if cross_returns:
        for cross, ret in cross_returns.items():
            if isinstance(ret, (int, float)) and ret == ret and abs(ret) <= _MAX_FEATURE_RATIO:
                features[f"lag_{cross.lower()}_ret"] = float(ret)
    return features or None


def _model() -> compose.Pipeline:
    """Streaming regressor: standardized features → SGD linear regression.

    ``TargetStandardScaler`` normalizes the target (returns are small), so the
    learned weights are stable; predictions come back in original units. The
    feature scaler sits *outside* the target wrapper — River's documented shape
    (see its TargetStandardScaler examples) — because the wrapper expects a
    plain ``Regressor``, and the scaler must run before the linear model.
    """
    return preprocessing.StandardScaler() | preprocessing.TargetStandardScaler(
        regressor=linear_model.LinearRegression(optimizer=optim.SGD(0.01), l2=0.001)
    )


@dataclass
class _SymbolState:
    model: compose.Pipeline = field(default_factory=_model)
    conformal: ConformalInterval = field(default_factory=ConformalInterval)
    last_features: dict[FeatureName, float] | None = None
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


def _direction(y_hat: float, threshold: float) -> str:
    """Trading rule: long/short when the predicted return clears ±threshold.

    Below threshold the predicted move is within hourly noise (~0.73% vol), so
    the honest call is FLAT — trading every coin-flip lean just bleeds fees.
    """
    if y_hat > threshold:
        return "LONG"
    if y_hat < -threshold:
        return "SHORT"
    return "FLAT"


class OnlinePredictor:
    """Per-symbol online return model fed by the 1h feature stream.

    Cross-coin lead-lag: each symbol's model also sees the lagged returns of
    the ``cross_symbols`` (the majors, e.g. BTC/ETH). ``_closes`` is a rolling
    per-symbol registry of ``(window_end_ms, close)``; when a window arrives
    for a symbol, the return of each cross symbol over its most recent window
    that ended *before* this one is used as a feature — never the current
    window's close, so there is no lookahead and the online model learns the
    same spillover the lead-lag papers document. Three facts drive the feature
    choice, and the sign is deliberately NOT hardcoded:

      - Positive spillover, minute data, Binance: lagged returns of other
        coins predict a focal coin up to 10 minutes ahead; BTC is the reliable
        positive leader (~75 bps per hour per 1σ of its lagged return — above
        our 20 bps entry band), with BNB/TRX also leading (Guo, Sang, Tu &
        Wang, "Cross-Cryptocurrency Return Predictability", JEDC 163, 2024).
      - Negative ("seesaw") lead-lag intraday: the five largest coins
        negatively predict smaller coins — small coins do not predict large
        ones (Jia, Wu, Yan & Yin, "A Seesaw Effect in the Cryptocurrency
        Market", CUFE / JEF 74, 2023).
      - The direction is regime- and tail-dependent: systemic centrality does
        NOT track market cap (XRP/SOL-style coins transmit in calm markets,
        BTC flips from net receiver to top transmitter in crises), and
        spillovers amplify at both tails (Fu, Zhu & Liu, Jinan Univ. —
        SA-Log-HAR arXiv:2507.22409; TVTP-MS-HAR, Mathematics 13(15), 2025).

    Because the literature itself finds sign instability across horizons,
    coins and regimes, we feed the realized lagged cross return as a raw
    feature and let each symbol's online regressor learn its own sign online —
    no hardcoded direction. Trading the spillover through the cost-aware
    execution filter (|r̂| > λ·c) instead of raw high-frequency scalping is our
    novel extension: whether lead-lag survives hourly (and nets of taker
    fees) is untested in the literature.
    """

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
        direction_threshold: float = 0.002,
        cross_symbols: list[str] | None = None,
    ) -> None:
        self._kv = kv
        self._prediction_prefix = prediction_prefix
        self._strategy_prefix = strategy_prefix
        self._strategy_maxlen = strategy_maxlen
        self._direction_threshold = direction_threshold
        self._cross_symbols = [s.upper() for s in (cross_symbols or [])]
        self._states: dict[str, _SymbolState] = {}
        self._default = (alpha, gamma, residual_window)
        # Rolling cross-symbol close registry: symbol -> (window_end_ms, close).
        self._closes: dict[str, list[tuple[int, float]]] = {}

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

    def _record_close(self, symbol: str, window_end: int, close: float) -> None:
        """Append this symbol's close to the registry, trimmed to recent windows."""
        hist = self._closes.setdefault(symbol, [])
        if hist and hist[-1][0] >= window_end:
            return
        hist.append((window_end, close))
        # Keep ~24h of windows (1h cadence) for lag computation — a full day
        # of the leader's hourly closes.
        while len(hist) > 24:
            del hist[0]

    def _cross_returns(self, symbol: str, window_end: int) -> dict[str, float]:
        """Lagged returns of the cross symbols, from windows ending BEFORE this one.

        For each cross symbol, takes its last completed close strictly before
        ``window_end`` and the one before that → a realized return the model
        would actually know when this window opens. Missing/warm-up pairs are
        simply omitted (the feature set is stable because River treats an
        absent feature as no signal for that window).
        """
        returns: dict[str, float] = {}
        for cross in self._cross_symbols:
            if cross == symbol:
                continue
            hist = [h for h in self._closes.get(cross, []) if h[0] < window_end]
            if len(hist) < 2:
                continue
            older, newer = hist[-2], hist[-1]
            if older[0] >= newer[0]:
                continue
            if newer[1] != 0:
                returns[cross] = newer[1] / older[1] - 1.0
        return returns

    def handle(self, msg: dict) -> None:
        """One feature window: learn from the *previous* window's target, then
        predict the next return and write the conformal interval to the store.
        """
        symbol = str(msg.get("symbol") or "").upper()
        if not symbol:
            return
        close = float(msg["close"]) if isinstance(msg.get("close"), (int, float)) else None
        window_end = msg.get("window_end_ms")
        if isinstance(window_end, (int, float)):
            window_end = int(window_end)
        else:
            window_end = None
        if close is None or window_end is None:
            return
        # Register this close BEFORE computing cross returns so the leaders'
        # most-recent completed window is available to followers this bar.
        self._record_close(symbol, window_end, close)
        cross = self._cross_returns(symbol, window_end)
        features = _features(msg, cross)
        if features is None:
            return
        state = self._state(symbol)

        if state.last_features is not None and state.last_close is not None and close != 0:
            realized = close / state.last_close - 1.0
            if abs(realized) > _MAX_REALIZED:
                logger.warning(
                    "skipping learning: implausible realized=%.4f at %s (close=%s, prev_close=%s)",
                    realized,
                    window_end,
                    close,
                    state.last_close,
                )
            else:
                state.model.learn_one(state.last_features, realized)
                if state.last_y_hat is not None and state.last_interval is not None:
                    state.conformal.update(realized, state.last_y_hat, state.last_interval)
                self._record_period(state, state.last_direction, realized)

        y_hat = float(state.model.predict_one(features))
        interval = state.conformal.predict(y_hat)
        direction = _direction(y_hat, self._direction_threshold)

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
            self._write_strategy(state, symbol, window_end)

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

    def _write_strategy(self, state: _SymbolState, symbol: str, window_end: int | None) -> None:
        eq_strat = state.equity_strategy
        eq_buy = state.equity_buyhold
        self._kv.set_json(
            strategy_key(self._strategy_prefix or "", symbol),
            {
                "symbol": symbol,
                "window_end_ms": window_end,
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
    from config.settings import csv_list

    predictor = OnlinePredictor(
        kv,
        prediction_prefix=settings.stream_redis_prediction_prefix,
        strategy_prefix=settings.stream_redis_strategy_prefix,
        strategy_maxlen=settings.stream_strategy_maxlen,
        alpha=settings.stream_prediction_alpha,
        gamma=settings.stream_prediction_gamma,
        residual_window=settings.stream_prediction_residual_window,
        direction_threshold=settings.stream_prediction_direction_threshold,
        cross_symbols=csv_list(settings.stream_prediction_cross_symbols),
    )
    logger.info(
        "predictor consuming %s → %s (alpha=%s, gamma=%s, direction_threshold=%s, cross=%s)",
        settings.stream_kafka_topic_features,
        settings.stream_redis_prediction_prefix,
        settings.stream_prediction_alpha,
        settings.stream_prediction_gamma,
        settings.stream_prediction_direction_threshold,
        settings.stream_prediction_cross_symbols,
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
