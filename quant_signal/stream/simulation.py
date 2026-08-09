"""Forward price simulation: fat-tailed GBM (Student-t + EWMA), Monte Carlo.

Runs ``N`` paths forward from the latest feature close, calibrated from the
*realized* log returns of recent 5m windows (no lookahead), and emits the
fan-chart percentile bands plus tail risk metrics:

  S_{t+1} = S_t * exp((mu - sigma^2 / 2) + sigma * Z_t),  Z_t ~ std-t(nu)

Two model upgrades over a plain Normal-GBM (both research-backed):

  EWMA volatility   sigma_t^2 = lam * sigma_{t-1}^2 + (1-lam) * r_t^2, the
                    RiskMetrics recursion (λ = 0.94; J.P. Morgan 1994/2006,
                    Hendricks NY Fed 1996). Volatility clusters (Mandelbrot
                    1963; Fama 1965), so the equal-weight trailing stdev smears
                    a burst into the forecast; EWMA decays it geometrically.
  Student-t shocks  Z standardized to variance 1 with the df estimated from
                    the standardized residuals (Bollerslev 1987 GARCH-t). Fat
                    tails are the empirical norm and Normal-innovation models
                    under-cover tail risk while Student-t captures the tail
                    shape (Horváth & Šopov 2016); df→∞ recovers the Normal.

The exponential map with standardized-t shocks is the log-Student-t ("Gosset")
family used for fat-tailed MC pricing (Cassidy, Hamp & Ouyed 2010); with
per-window sigma of a few bps the drift-bias from the t-tail is negligible.

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
from scipy import stats

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

# Steps for the 3D density surface: the density is computed per forward step,
# so the UI can render the "probability mountain" forming in real time.
_SURFACE_STEPS = 12

# Number of raw paths shipped to the UI for the QuantPad-style "all paths"
# fan-chart spaghetti. The percentile statistics always use *all* paths; this
# is purely how many thin lines the browser renders (a display resolution).
_SAMPLE_PATHS = 200


def simulation_key(prefix: str, symbol: str) -> str:
    return f"{prefix}:{symbol.upper()}"


def _ewma_vol(returns, lam: float) -> float:
    """1-step-ahead RiskMetrics EWMA volatility of log returns.

    sigma_t^2 = lam * sigma_{t-1}^2 + (1-lam) * r_t^2 — the recursive EWMA of
    J.P. Morgan's RiskMetrics (1994/2006; Hendricks, NY Fed 1996). Seeded with
    the trailing sample mean-square, so with λ=0.94 a past shock decays
    geometrically (99.9% of its weight is gone after ~112 windows) instead of
    being smeared by an equal-weight window.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    var = float(np.mean(r * r))
    for x in r:
        var = lam * var + (1.0 - lam) * x * x
    return float(math.sqrt(var))


def _t_df(returns, lam: float, df_min: float, df_max: float) -> float:
    """Student-t innovation df from the standardized residuals (GARCH-t).

    Each log return is standardized by the EWMA conditional volatility known
    *before* it realized (GARCH-style), then the df is recovered from the
    excess kurtosis via the t moment identity ν = 4 + 6/kurtosis (Heikkinen &
    Kanto 2002; Aalto w369). df → df_max recovers the Normal-innovation model;
    note sample kurtosis is a biased-but-practical per-window estimator
    (Heracleous 2007) — fine for a self-checking monitor, not for exactness.
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 4:
        return df_max
    var = float(np.mean(r * r))
    stds = np.empty_like(r)
    for i, x in enumerate(r):
        stds[i] = math.sqrt(max(var, 1e-12))
        var = lam * var + (1.0 - lam) * x * x
    z = r / stds
    kurt = float(stats.kurtosis(z, fisher=True))
    if not math.isfinite(kurt) or kurt <= 0.0:
        return df_max
    return float(min(max(4.0 + 6.0 / kurt, df_min), df_max))


class MonteCarloEngine:
    """GBM Monte Carlo path simulator with percentile + VaR/ES stats.

    Uses a time-based rolling seed derived from the feature window_end_ms so
    each new 5m window produces reproducible paths unique to that window — no
    fixed global seed that would make every window look identical."""

    def __init__(
        self,
        *,
        n_paths: int = 10_000,
        horizon_steps: int = 12,
        vol_windows: int = 40,
        drift: bool = False,
        seed: int | None = None,
        sample_paths: int = _SAMPLE_PATHS,
        ewma_lambda: float = 0.94,
        t_df_min: float = 4.0,
        t_df_max: float = 30.0,
    ) -> None:
        self._n_paths = n_paths
        self._horizon_steps = horizon_steps
        self._vol_windows = vol_windows
        self._drift = drift
        self._seed = seed
        self._sample_paths = sample_paths
        self._ewma_lambda = ewma_lambda
        self._t_df_min = t_df_min
        self._t_df_max = t_df_max

    def calibrate_dist(self, closes):
        """Per-step (mu, sigma, nu) of log returns over the trailing windows.

        Volatility is the RiskMetrics EWMA 1-step forecast; ``nu`` is the
        Student-t df of the standardized residuals (GARCH-t). Returns None when
        there are not enough closes to estimate volatility.
        """
        if len(closes) < 2:
            return None
        returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        tail = returns[-self._vol_windows :]
        if len(tail) < 2 or stdev(tail) == 0:
            return None
        sigma = _ewma_vol(tail, self._ewma_lambda)
        if sigma == 0.0:
            return None
        nu = _t_df(tail, self._ewma_lambda, self._t_df_min, self._t_df_max)
        mu = (mean(tail) + sigma * sigma / 2) if self._drift else 0.0
        return mu, sigma, nu

    def calibrate(self, closes):
        """Per-step (mu, sigma) of log returns over the trailing windows.

        Returns None when there are not enough closes to estimate volatility.
        """
        dist = self.calibrate_dist(closes)
        if dist is None:
            return None
        mu, sigma, _nu = dist
        return mu, sigma

    def paths(
        self,
        s0: float,
        mu: float,
        sigma: float,
        seed: int | None = None,
        nu: float = float("inf"),
    ) -> np.ndarray:
        """(horizon_steps + 1) x n_paths fat-tailed GBM price paths (S0 first).

        Shocks are standardized Student-t (mean 0, variance 1) for finite df;
        ``nu=inf`` (the default) recovers Normal shocks. Vectorized simulation
        of all paths in one numpy call.
        """
        rng = np.random.default_rng(seed if seed is not None else self._seed)
        half_var = sigma * sigma / 2.0
        size = (self._horizon_steps, self._n_paths)
        if math.isinf(nu):
            shocks = rng.normal(0.0, 1.0, size=size)
        else:
            # Raw t has variance nu/(nu-2); scale by sqrt((nu-2)/nu) so the
            # shock has variance 1 for every df (fGarch 'std' / stochvol).
            shocks = rng.standard_t(nu, size=size) * math.sqrt((nu - 2.0) / nu)
        increments = np.exp((mu - half_var) + sigma * shocks)
        return s0 * np.vstack([np.ones(self._n_paths), np.cumprod(increments, axis=0)])

    def forecast(
        self,
        closes: list[float],
        window_end_ms: int | None = None,
        *,
        sigma_scale: float = 1.0,
        scenario: dict | None = None,
    ) -> dict | None:
        """Full forecast payload from trailing closes, or None if uncalibrated.

        ``sigma_scale`` applies a what-if stress to the *calibrated* volatility
        before generating paths (the engine recomputes everything downstream —
        surface, fan bands, VaR/ES — from the stressed ensemble, so the stress
        shows up in every visual). ``scenario`` is echoed into the payload so
        the UI can label what-if runs honestly.
        """
        calibrated = self.calibrate_dist(closes)
        if calibrated is None:
            return None
        mu, sigma, nu = calibrated
        sigma = sigma * sigma_scale
        s0 = closes[-1]
        seed = window_end_ms if window_end_ms is not None else self._seed
        paths = self.paths(s0, mu, sigma, seed=seed, nu=nu)

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

        # Per-step return density across ALL paths (not the UI sample): the 3D
        # "probability surface" bins every forward step over a *shared, fixed*
        # return grid that spans ±4σ·√steps. The span is derived from the
        # calibrated volatility (not the ensemble's min/max), so the mountain
        # visibly widens/narrows as volatility changes window to window and
        # under what-if stress — a display resolution, not a model knob. Paths
        # outside the grid are clipped into the extreme bins so every step row
        # still sums to n_paths.
        surface_steps = min(self._horizon_steps, _SURFACE_STEPS)
        step_returns = paths[1 : surface_steps + 1] / s0 - 1.0  # (steps, n_paths)
        span = 4.0 * sigma * math.sqrt(surface_steps)
        surface_edges = np.linspace(-span, span, _HIST_BINS + 1)
        surface = np.zeros((surface_steps, _HIST_BINS), dtype=int)
        for i in range(surface_steps):
            idx = np.clip(np.digitize(step_returns[i], surface_edges) - 1, 0, _HIST_BINS - 1)
            np.add.at(surface[i], idx, 1)

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
            "nu": round(nu, 2),
            "vol_model": "ewma",
            "ewma_lambda": self._ewma_lambda,
            "scenario": scenario,
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
            "surface_grid": {
                "steps": surface_steps,
                "edges": [round(float(e), 6) for e in surface_edges],
                "counts": [[int(c) for c in row] for row in surface],
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
    closes = _closes_from_features(features)
    window_end_ms = features[-1].get("window_end_ms") if features else None
    payload = engine.forecast(closes, window_end_ms=window_end_ms)
    if payload is None:
        return None
    payload["symbol"] = symbol
    payload["window_end_ms"] = window_end_ms
    payload["updated_at"] = _now_iso()
    kv.set_json(simulation_key(simulation_prefix, symbol), payload)
    return payload


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


class SimulationConsumer:
    """Consume the 5-minute feature stream and refresh the MC forecast per symbol."""

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
        sample_paths: int = _SAMPLE_PATHS,
        ewma_lambda: float = 0.94,
        t_df_min: float = 4.0,
        t_df_max: float = 30.0,
    ) -> None:
        self._kv = kv
        self._simulation_prefix = simulation_prefix
        self._engine = MonteCarloEngine(
            n_paths=n_paths,
            horizon_steps=horizon_steps,
            vol_windows=vol_windows,
            drift=drift,
            seed=seed,
            sample_paths=sample_paths,
            ewma_lambda=ewma_lambda,
            t_df_min=t_df_min,
            t_df_max=t_df_max,
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
        drift=settings.stream_simulation_drift,
        sample_paths=settings.stream_simulation_sample_paths,
        ewma_lambda=settings.stream_simulation_ewma_lambda,
        t_df_min=settings.stream_simulation_t_df_min,
        t_df_max=settings.stream_simulation_t_df_max,
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
