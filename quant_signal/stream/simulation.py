"""Forward price simulation: geometric Brownian motion, Monte Carlo.

Runs ``N`` GBM paths forward from the latest feature close, calibrated from
the *realized* log returns of recent 5m windows (per-step mean/std — no
lookahead), and emits the fan-chart percentile bands plus tail risk metrics:

  S_{t+1} = S_t * exp((mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z),  Z ~ N(0,1)

``mu`` (per-step log-return drift, MLE) is dwarfed by ``sigma`` on 5m horizons,
so a ``drift`` toggle lets the demo run driftless (conservative) or with the
MLE drift estimated from the trailing windows.

Outputs land in the online store as ``simulation:crypto:5m:<SYMBOL>`` (a SET)
so the API serves the fan chart sub-500ms. Pure math lives in
``MonteCarloEngine`` for hermetic tests; ``run`` drives it from a feature
message.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Mapping
from statistics import mean, stdev

import numpy as np

from config.logging import configure_logging, get_logger
from config.settings import get_settings
from stream.bus import KafkaBus, MessageBus
from stream.kv import KVStore, RedisKV

logger = get_logger(__name__)

# Crypto trades 24/7: 365 days × 24 h × 12 five-minute windows per hour.
_PERIODS_PER_YEAR = 365 * 24 * 12

_PERCENTILES = [10, 25, 50, 75, 90]

# Bins for the terminal-return histogram (a display resolution, not a model knob).
_HIST_BINS = 24

# Number of raw paths shipped to the UI for the QuantPad-style "all paths"
# fan-chart spaghetti. The percentile statistics always use *all* paths; this
# is purely how many thin lines the browser renders (a display resolution).
_SAMPLE_PATHS = 200


def simulation_key(prefix: str, symbol: str) -> str:
    return f"{prefix}:{symbol.upper()}"


class MonteCarloEngine:
    """Deterministic-seed GBM path simulator with percentile + VaR/ES stats."""

    def __init__(
        self,
        *,
        n_paths: int = 10_000,
        horizon_steps: int = 12,
        vol_windows: int = 40,
        drift: bool = False,
        seed: int | None = None,
        sample_paths: int = _SAMPLE_PATHS,
    ) -> None:
        self._n_paths = n_paths
        self._horizon_steps = horizon_steps
        self._vol_windows = vol_windows
        self._drift = drift
        self._seed = seed
        self._sample_paths = sample_paths

    def calibrate(self, closes: list[float]) -> tuple[float, float] | None:
        """Per-step ``(mu, sigma)`` of log returns over the trailing windows.

        Returns None when there aren't enough closes to estimate volatility.
        """
        if len(closes) < 2:
            return None
        returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        tail = returns[-self._vol_windows :]
        if len(tail) < 2 or stdev(tail) == 0:
            return None
        sigma = stdev(tail)
        mu = (mean(tail) + sigma * sigma / 2) if self._drift else 0.0
        return mu, sigma

    def paths(self, s0: float, mu: float, sigma: float) -> np.ndarray:
        """``(horizon_steps + 1)`` × ``n_paths`` GBM price paths (S0 first).

        All paths are simulated in a single vectorized call (SIMD-parallel —
        the same trick QuantPad's "10k parallel simulations" relies on), so
        the whole fan chart costs microseconds rather than a Python loop.
        """
        rng = np.random.default_rng(self._seed)
        half_var = sigma * sigma / 2.0
        shocks = rng.normal(0.0, 1.0, size=(self._horizon_steps, self._n_paths))
        increments = np.exp((mu - half_var) + sigma * shocks)
        return s0 * np.vstack([np.ones(self._n_paths), np.cumprod(increments, axis=0)])

    def forecast(self, closes: list[float]) -> dict | None:
        """Full forecast payload from trailing closes, or None if uncalibrated."""
        calibrated = self.calibrate(closes)
        if calibrated is None:
            return None
        mu, sigma = calibrated
        s0 = closes[-1]
        paths = self.paths(s0, mu, sigma)

        band_rows = np.percentile(paths, _PERCENTILES, axis=1)  # (5, steps)
        bands = {
            str(pct): [round(float(v), 6) for v in row] for pct, row in zip(_PERCENTILES, band_rows)
        }
        finals = paths[-1]
        returns = finals / s0 - 1.0
        var95 = float(np.percentile(returns, 5))
        tail = returns[returns <= var95]
        es95 = float(np.mean(tail)) if tail.size else var95
        prob_up = float(np.mean(returns > 0.0))
        hist_counts, hist_edges = np.histogram(returns, bins=_HIST_BINS)

        # Evenly-spaced subsample of the raw paths for the QuantPad-style
        # "all paths (N)" fan-chart spaghetti — stats always use every path.
        sample_idx = np.linspace(0, self._n_paths - 1, self._sample_paths, dtype=int)
        sample_paths = paths[:, sample_idx].T  # (sample_paths, steps + 1)

        return {
            "symbol": None,  # filled by run()
            "base_price": round(s0, 6),
            "horizon_steps": self._horizon_steps,
            "n_paths": self._n_paths,
            "mu": round(mu, 8),
            "sigma": round(sigma, 6),
            "sigma_annualized": round(sigma * math.sqrt(_PERIODS_PER_YEAR), 6),
            "percentiles": bands,
            "median_path": bands["50"],
            "sample_paths": [p.tolist() for p in sample_paths],
            "var95": round(var95, 6),
            "es95": round(es95, 6),
            "prob_up": round(prob_up, 4),
            "returns_histogram": {
                "counts": [int(c) for c in hist_counts],
                "edges": [round(float(e), 6) for e in hist_edges],
            },
            "confidence_interval": {
                "p10": bands["10"][-1],
                "p90": bands["90"][-1],
            },
        }


def _closes_from_features(features: list[dict]) -> list[float]:
    """Chronological close prices from feature windows (oldest first)."""
    closes: deque[float] = deque()
    for window in features:
        close = window.get("close")
        if isinstance(close, (int, float)) and close == close:
            closes.append(float(close))
    return list(closes)


def run(
    kv: KVStore,
    engine: MonteCarloEngine,
    symbol: str,
    features: list[dict],
    *,
    simulation_prefix: str,
) -> dict | None:
    """Forecast from the feature history and land it in the online store."""
    payload = engine.forecast(_closes_from_features(features))
    if payload is None:
        return None
    payload["symbol"] = symbol
    kv.set_json(simulation_key(simulation_prefix, symbol), payload)
    return payload


class SimulationConsumer:
    """Consume the 5m feature stream and refresh the MC forecast per symbol."""

    def __init__(
        self,
        kv: KVStore,
        *,
        simulation_prefix: str,
        n_paths: int = 10_000,
        horizon_steps: int = 12,
        vol_windows: int = 40,
        drift: bool = False,
        seed: int | None = None,
    ) -> None:
        self._kv = kv
        self._simulation_prefix = simulation_prefix
        self._engine = MonteCarloEngine(
            n_paths=n_paths,
            horizon_steps=horizon_steps,
            vol_windows=vol_windows,
            drift=drift,
            seed=seed,
        )
        self._history: dict[str, deque[dict]] = {}
        self._maxlen = vol_windows * 2

    def handle(self, msg: dict) -> dict | None:
        symbol = str(msg.get("symbol") or "").upper()
        if not symbol:
            return None
        history = self._history.setdefault(symbol, deque())
        history.append(msg)
        while len(history) > self._maxlen:
            history.popleft()
        return run(
            self._kv,
            self._engine,
            symbol,
            list(history),
            simulation_prefix=self._simulation_prefix,
        )

    def warm_start(self, features: list[Mapping]) -> None:
        """Seed the per-symbol window history from the online store.

        The simulator estimates volatility from trailing windows, so it needs
        history before the first forecast; replaying the already-materialized
        feature windows (oldest first) avoids a cold ``latest``-offset start.
        """
        for msg in features:
            self.handle(dict(msg))

    def run_forever(
        self,
        bus: MessageBus,
        features_topic: str,
        group_id: str,
        stop: threading.Event | None = None,
    ) -> None:
        for _topic, msg in bus.iter_consume(features_topic, group_id, stop=stop):
            self.handle(msg)


def main() -> None:
    configure_logging()
    settings = get_settings()
    bus = KafkaBus(settings.stream_kafka_bootstrap_servers)
    kv = RedisKV(settings.stream_redis_url)
    consumer = SimulationConsumer(
        kv,
        simulation_prefix=settings.stream_redis_simulation_prefix,
        n_paths=settings.stream_simulation_paths,
        horizon_steps=settings.stream_simulation_horizon_steps,
        vol_windows=settings.stream_simulation_vol_windows,
    )
    from config.settings import csv_list
    from stream.materializer import feature_key

    for symbol in csv_list(settings.ingest_default_crypto_symbols):
        history = kv.list_json(feature_key(settings.stream_redis_feature_prefix, symbol))
        if history:
            consumer.warm_start(history)
            logger.info("simulator warm-started %s from %d windows", symbol, len(history))
    logger.info(
        "simulator consuming %s → %s (paths=%s, horizon=%s×5m)",
        settings.stream_kafka_topic_features,
        settings.stream_redis_simulation_prefix,
        settings.stream_simulation_paths,
        settings.stream_simulation_horizon_steps,
    )
    try:
        consumer.run_forever(
            bus,
            settings.stream_kafka_topic_features,
            group_id="monte-carlo-simulator",
        )
    except KeyboardInterrupt:
        logger.info("simulator stopped")


if __name__ == "__main__":
    main()
