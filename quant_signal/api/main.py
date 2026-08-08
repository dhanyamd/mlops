"""Dashboard API — read-only FastAPI app over the live Silver/Gold layers.

Serves the same real numbers the CLI shows: market bars, PIT fundamentals,
the PEAD event study, pipeline latency, and macro series. No mocking, no
business values hardcoded — instruments come from ``INGEST_DEFAULT_TICKERS``.

Also runs the live market stream (Binance minute bars → WebSocket fan-out)
for the duration of the process, via the FastAPI lifespan.

Run:  uv run uvicorn api.main:app --port 8000
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api import db
from api.stream import MarketStream, start_stream, stop_stream
from config.logging import configure_logging
from config.settings import csv_list, get_settings
from scripts.pead_backtest import compute_pead
from stream.kv import KVStore, RedisKV
from stream.materializer import feature_key
from stream.mlflow_tracking import track_gate_report, track_validation
from stream.pipeline_health import pipeline_summary
from stream.predictive_eval import evaluate_predictor, passes_gate
from stream.predictor import prediction_key, strategy_key
from stream.simulation import simulation_key
from stream.strategy_mc import StrategyMonteCarlo

configure_logging()

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop the live market stream with the API process.

    ``app.state.kv`` is the Redis online store (lazy client, no I/O here), so
    the feature endpoint never touches Kafka or Snowflake.
    """
    stream = start_stream()
    if stream is not None:
        stream.start(asyncio.get_running_loop())
    app.state.stream = stream
    app.state.kv = RedisKV(settings.stream_redis_url) if settings.stream_enabled else None
    try:
        yield
    finally:
        stop_stream(app.state.stream)


app = FastAPI(title="Quant Signal Dashboard", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=csv_list(settings.api_cors_origins),
    allow_methods=["GET"],
    allow_headers=["*"],
)

# The PEAD study is a ~10s query; cache per parameter set for a short TTL.
_pead_cache: dict[str, tuple[float, dict]] = {}


def _stream() -> MarketStream | None:
    """The running MarketStream (if enabled) — None in tests that disable it."""
    return getattr(app.state, "stream", None)


def _kv() -> KVStore | None:
    """The Redis online store (if streaming is enabled) — None otherwise."""
    return getattr(app.state, "kv", None)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "quant-signal-api"}


@app.get("/api/tickers")
def tickers() -> dict:
    return {"tickers": db.default_tickers()}


@app.get("/api/metrics")
def metrics() -> dict:
    return {"metrics": db.default_metrics()}


@app.get("/api/macro/series")
def macro_series_list() -> dict:
    return {"series": db.default_macro_series()}


@app.get("/api/market/symbols")
def market_symbols() -> dict:
    """Tracked crypto symbols — never hardcoded, sourced from env."""
    return {"symbols": csv_list(settings.ingest_default_crypto_symbols)}


@app.get("/api/market/{symbol}")
def market(
    symbol: str,
    days: int = Query(default=750, ge=1, le=8000, description="trading days to return"),
) -> dict:
    bars = db.market_bars(symbol, days=days)
    return {"symbol": symbol.upper(), "count": len(bars), "bars": bars}


@app.get("/api/fundamentals/{ticker}")
def fundamentals(
    ticker: str,
    metric: str | None = Query(default=None, description="e.g. NetIncomeLoss"),
) -> dict:
    facts = db.fundamentals(ticker, metric=metric)
    return {"ticker": ticker.upper(), "metric": metric, "count": len(facts), "facts": facts}


@app.get("/api/pead")
def pead(
    metric: str = Query(default="NetIncomeLoss", description="US-GAAP metric"),
    windows: str = Query(default="1,5,20", description="post-filing trading-day windows (CSV)"),
    min_prior: int = Query(default=5, ge=1, description="min prior surprises before SUE"),
    quintiles: int = Query(default=5, ge=2, le=10, description="SUE groups for drift table"),
) -> dict:
    cache_key = f"{metric}|{windows}|{min_prior}|{quintiles}"
    hit = _pead_cache.get(cache_key)
    if hit and time.monotonic() - hit[0] < settings.api_pead_cache_ttl_seconds:
        return hit[1]
    result = compute_pead(
        metric,
        [int(w) for w in windows.split(",")],
        min_prior,
        quintiles,
    )
    _pead_cache[cache_key] = (time.monotonic(), result)
    return result


@app.get("/api/metrics/pipeline")
def pipeline_metrics(
    flow: str | None = Query(default=None, description="filter to one flow"),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict:
    return {"runs": db.pipeline_metrics(flow=flow, limit=limit)}


@app.get("/api/macro")
def macro(
    series: str | None = Query(default=None, description="FRED series ID, e.g. VIXCLS"),
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict:
    points = db.macro_series(series_id=series, limit=limit)
    return {"count": len(points), "points": points}


@app.get("/api/market/live/{symbol}")
def market_live(symbol: str) -> dict:
    """Ring-buffer snapshot from the live stream (no Kafka/Snowflake, no delay)."""
    stream = _stream()
    if stream is None:
        return {"symbol": symbol.upper(), "enabled": False, "count": 0, "bars": []}
    bars = stream.hub.snapshot(symbol)
    return {"symbol": symbol.upper(), "enabled": True, "count": len(bars), "bars": bars}


@app.get("/api/market/health/summary")
def market_health_summary() -> dict:
    """Per-stage pipeline freshness, derived from *event* timestamps only.

    Every artifact tags the feature window it came from (``window_end_ms``);
    ages compare those tags against the latest feature window, never the host
    clock (which drifts ~5h on this machine). The same event-time delta the
    watchdog alerts on, served to the UI.
    """
    kv = _kv()
    if kv is None:
        return {"enabled": False, "healthy": None, "stages": []}
    summary = pipeline_summary(
        kv,
        symbols=csv_list(settings.ingest_default_crypto_symbols),
        live_prefix=settings.stream_redis_live_prefix,
        feature_prefix=settings.stream_redis_feature_prefix,
        prediction_prefix=settings.stream_redis_prediction_prefix,
        simulation_prefix=settings.stream_redis_simulation_prefix,
        strategy_prefix=settings.stream_redis_strategy_prefix,
        staleness_threshold=settings.stream_watchdog_staleness_threshold_seconds,
    )
    return {"enabled": True, **summary}


@app.get("/api/market/features/{symbol}")
def market_features(
    symbol: str,
    limit: int = Query(default=12, ge=1, le=200, description="windows to return"),
) -> dict:
    """Window features from the Redis online store (Flink-computed, sub-500ms)."""
    kv = _kv()
    if kv is None:
        return {"symbol": symbol.upper(), "enabled": False, "count": 0, "features": []}
    rows = kv.list_json(
        feature_key(settings.stream_redis_feature_prefix, symbol.upper()),
        reverse=True,  # newest window first
        maxlen=limit,
    )
    return {"symbol": symbol.upper(), "enabled": True, "count": len(rows), "features": rows}


@app.get("/api/market/predict/{symbol}")
def market_predict(symbol: str) -> dict:
    """Next-return prediction + conformal interval from the online store."""
    kv = _kv()
    if kv is None:
        return {"symbol": symbol.upper(), "enabled": False, "prediction": None}
    prediction = kv.get_json(
        prediction_key(settings.stream_redis_prediction_prefix, symbol.upper())
    )
    return {"symbol": symbol.upper(), "enabled": True, "prediction": prediction}


@app.get("/api/market/simulation/{symbol}")
def market_simulation(symbol: str) -> dict:
    """Monte Carlo forward fan chart (percentiles, VaR/ES) from the online store."""
    kv = _kv()
    if kv is None:
        return {"symbol": symbol.upper(), "enabled": False, "simulation": None}
    simulation = kv.get_json(
        simulation_key(settings.stream_redis_simulation_prefix, symbol.upper())
    )
    return {"symbol": symbol.upper(), "enabled": True, "simulation": simulation}


@app.get("/api/market/strategy/{symbol}")
def market_strategy(symbol: str) -> dict:
    """Live strategy P&L curve (compounded equity vs buy-and-hold)."""
    kv = _kv()
    if kv is None:
        return {"symbol": symbol.upper(), "enabled": False, "strategy": None}
    strategy = kv.get_json(strategy_key(settings.stream_redis_strategy_prefix, symbol.upper()))
    return {"symbol": symbol.upper(), "enabled": True, "strategy": strategy}


@app.get("/api/market/validation/{symbol}")
def market_validation(symbol: str, track: bool = Query(default=False)) -> dict:
    """QuantPad-style pass probability: bootstrap the strategy's realized
    returns into simulated futures and score them against prop-firm rules.

    The Monte Carlo is seeded from the latest feature window's event time
    (``window_end_ms``), so every 5m window resamples the futures differently
    while staying reproducible per window — the seed is echoed in the payload
    for audit (prod-grade: reproducible, never OS-entropy, never frozen at a
    fixed seed).

    ``track=true`` logs this run to MLflow Tracking (params/metrics/artifacts).
    It defaults to off so the Signal Terminal's 15s polling never spams runs;
    pass it explicitly (one-off / automation) when you want a recorded run.
    """
    kv = _kv()
    if kv is None:
        return {"symbol": symbol.upper(), "enabled": False, "validation": None}
    strategy = kv.get_json(strategy_key(settings.stream_redis_strategy_prefix, symbol.upper()))
    if not strategy or not strategy.get("strategy_equity"):
        return {"symbol": symbol.upper(), "enabled": True, "validation": None}
    seed = strategy.get("window_end_ms") or settings.stream_validation_seed
    mc = StrategyMonteCarlo(
        n_sims=settings.stream_validation_sims,
        max_drawdown=settings.stream_validation_max_drawdown,
        target=settings.stream_validation_target,
        seed=seed,
    )
    validation = mc.validate(strategy["strategy_equity"])
    if track and validation is not None:
        track_validation(
            symbol.upper(),
            validation,
            target=settings.stream_validation_target,
            max_drawdown=settings.stream_validation_max_drawdown,
            seed=seed,
            n_sims=settings.stream_validation_sims,
        )
    return {
        "symbol": symbol.upper(),
        "enabled": True,
        "seed": seed,
        "window_end_ms": strategy.get("window_end_ms"),
        "validation": validation,
    }


@app.get("/api/market/validation/{symbol}/geometry")
def market_validation_geometry(symbol: str) -> dict:
    """Geometry Optimizer heat grid.

    Holds the strategy's realized daily EV constant while sweeping win-rate × R:R
    shape, showing pass-probability across 49 configurations. Research-backed
    insight (PropSim/QuantPad): trailing-DD rules are path-dependent, so two
    traders with identical edge can have 90% vs 10% pass rates.

    Seeded from the latest feature window's event time (``window_end_ms``) so
    each 5m window resamples the Monte Carlo differently — the curve and grid
    visibly move — while staying reproducible per window; the seed is echoed in
    the payload for audit.
    """
    kv = _kv()
    if kv is None:
        return {"symbol": symbol.upper(), "enabled": False, "grid": None}
    strategy = kv.get_json(strategy_key(settings.stream_redis_strategy_prefix, symbol.upper()))
    if not strategy or not strategy.get("strategy_equity"):
        return {"symbol": symbol.upper(), "enabled": True, "grid": None}
    from stream.strategy_mc import strategy_returns_from_equity

    returns = strategy_returns_from_equity(strategy["strategy_equity"])
    seed = strategy.get("window_end_ms") or settings.stream_validation_seed
    mc = StrategyMonteCarlo(
        n_sims=settings.stream_validation_sims,
        max_drawdown=settings.stream_validation_max_drawdown,
        target=settings.stream_validation_target,
        seed=seed,
    )
    grid = mc.geometry_sweep(
        returns=returns,
        n_periods=len(returns),
        target=settings.stream_validation_target,
        max_drawdown=settings.stream_validation_max_drawdown,
        sweep_rows=7,
        sweep_cols=7,
        n_sims=1000,
    )
    return {
        "symbol": symbol.upper(),
        "enabled": True,
        "seed": seed,
        "window_end_ms": strategy.get("window_end_ms"),
        "grid": grid,
    }


@app.get("/api/market/gate/{symbol}")
def market_gate(symbol: str, track: bool = Query(default=False)) -> dict:
    """Promotion-gate verdict for the symbol's online model.

    Replays the predictor's exact learn-then-predict loop over the stored
    feature-window history (oldest first) and scores it against the validation
    gate: enough scored windows, positive skill vs both naive baselines, IC
    and direction accuracy above floor, conformal coverage near nominal,
    strategy return clearing buy-and-hold after taker costs, and a Deflated
    Sharpe (multiple-testing-corrected) above the significance floor.

    Until ``gate.passes`` is true the live model may learn but must not emit
    tradeable directions — it is not a strategy, it is a guess wearing a
    number (see the research note in ``stream/predictive_eval.py``).

    ``track=true`` logs this run to MLflow Tracking (default off, so the
    Signal Terminal's 15s polling never spams runs; pass it explicitly for a
    recorded run).
    """
    kv = _kv()
    if kv is None:
        return {"symbol": symbol.upper(), "enabled": False, "gate": None}
    rows = kv.list_json(
        feature_key(settings.stream_redis_feature_prefix, symbol.upper()),
        reverse=False,  # oldest first: progressive validation is chronological
        maxlen=settings.stream_redis_feature_maxlen,
    )
    if not rows:
        return {"symbol": symbol.upper(), "enabled": True, "gate": None}
    report = evaluate_predictor(
        rows,
        alpha=settings.stream_prediction_alpha,
        gamma=settings.stream_prediction_gamma,
        residual_window=settings.stream_prediction_residual_window,
        taker_cost=settings.stream_gate_taker_cost,
    )
    passes, failures = passes_gate(report, settings, n_trials=settings.stream_gate_n_trials)
    if track:
        track_gate_report(
            symbol.upper(),
            report,
            failures,
            n_trials=settings.stream_gate_n_trials,
            taker_cost=settings.stream_gate_taker_cost,
            alpha=settings.stream_prediction_alpha,
            gamma=settings.stream_prediction_gamma,
            residual_window=settings.stream_prediction_residual_window,
        )
    # The raw per-window strategy returns are internal to the DSR; strip them
    # from the wire payload and attach the rejection reasons for the UI.
    gate = {k: v for k, v in report.items() if k != "_strat_rets"}
    gate["failures"] = failures
    return {"symbol": symbol.upper(), "enabled": True, "gate": gate}


def _default_live_symbol() -> str:
    """First tracked crypto symbol from env — never a hardcoded instrument."""
    symbols = csv_list(settings.ingest_default_crypto_symbols)
    return symbols[0] if symbols else ""


@app.websocket("/ws/market")
async def market_ws(websocket: WebSocket, symbol: str = Query(default="")) -> None:
    """Live minute bars for one symbol: snapshot, then deltas as they arrive."""
    if not symbol:
        symbol = _default_live_symbol()
    await websocket.accept()
    stream = _stream()
    if stream is None or not hasattr(stream, "hub"):
        await websocket.send_json({"type": "error", "message": "live stream disabled"})
        await websocket.close()
        return

    queue: asyncio.Queue[list[dict]] = asyncio.Queue(maxsize=1000)
    stream.hub.subscribe(queue)
    try:
        await websocket.send_json(
            {"type": "snapshot", "symbol": symbol.upper(), "bars": stream.hub.snapshot(symbol)}
        )
        while True:
            bars = await queue.get()
            for bar in bars:
                if bar["symbol"] == symbol.upper():
                    await websocket.send_json({"type": "bar", "symbol": symbol.upper(), "bar": bar})
    except WebSocketDisconnect:
        pass
    finally:
        stream.hub.unsubscribe(queue)
