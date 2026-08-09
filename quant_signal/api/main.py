"""Dashboard API — read-only FastAPI app over the live Silver/Gold layers.

Serves the same real numbers the CLI shows: market bars, PIT fundamentals,
the PEAD event study, pipeline latency, and macro series. No mocking, no
business values hardcoded — instruments come from ``INGEST_DEFAULT_TICKERS``.

Also runs the live market stream (venue minute bars → WebSocket fan-out)
for the duration of the process, via the FastAPI lifespan.

Run:  uv run uvicorn api.main:app --port 8000
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api import db
from api.stream import MarketStream, start_stream, stop_stream
from config.logging import configure_logging
from config.settings import csv_list, get_settings
from scripts.pead_backtest import compute_pead
from stream.execution import execution_key
from stream.kv import KVStore, RedisKV
from stream.materializer import feature_key
from stream.mlflow_tracking import track_gate_report, track_validation
from stream.pipeline_health import pipeline_summary
from stream.predictive_eval import evaluate_predictor, passes_gate
from stream.predictor import prediction_key, strategy_key
from stream.reality_check import reality_report
from stream.simulation import MonteCarloEngine, _closes_from_features, simulation_key
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


@app.get("/api/market/portfolio")
def market_portfolio() -> dict:
    """Aggregate the paper book across all tracked symbols.

    Total realized P&L (and fees), trade count, win rate, and the open
    positions marked to market — the portfolio-management half of the signal
    terminal. Defined above /api/market/{symbol} so it is never shadowed by
    the symbol catch-all.
    """
    kv = _kv()
    symbols = csv_list(settings.ingest_default_crypto_symbols)
    if kv is None:
        return {"enabled": False, "symbols": symbols, "total_pnl": None, "rows": []}

    rows: list[dict] = []
    total_pnl = 0.0
    total_fees = 0.0
    total_volume = 0.0
    n_trades = 0
    n_wins = 0
    for symbol in symbols:
        execution = kv.get_json(execution_key(settings.stream_redis_execution_prefix, symbol))
        if not execution:
            rows.append({"symbol": symbol, "present": False})
            continue
        position = execution.get("position")
        pnl = float(execution.get("realized_pnl", 0.0))
        unrealized = float(position.get("unrealized_pnl") or 0.0) if position else 0.0
        total_pnl += pnl
        total_fees += float(execution.get("total_fees", 0.0))
        total_volume += float(execution.get("gross_volume", 0.0))
        n_trades += int(execution.get("n_trades", 0))
        n_wins += int(execution.get("n_wins", 0))
        rows.append(
            {
                "symbol": symbol,
                "present": True,
                "n_trades": execution.get("n_trades"),
                "win_rate": execution.get("win_rate"),
                "realized_pnl": execution.get("realized_pnl"),
                "unrealized_pnl": round(unrealized, 2),
                "total_pnl": round(pnl + unrealized, 2),
                "total_fees": execution.get("total_fees"),
                "fees_pct_of_gross_pnl": execution.get("fees_pct_of_gross_pnl"),
                "gross_volume": execution.get("gross_volume"),
                "total_return": execution.get("total_return"),
                "signals_skipped": execution.get("signals_skipped"),
                "position": position,
            }
        )

    gross = sum(float(r.get("gross_volume") or 0.0) for r in rows if r.get("present"))
    return {
        "enabled": True,
        "symbols": symbols,
        "total_realized_pnl": round(total_pnl, 2),
        "total_unrealized_pnl": round(
            sum(float(r.get("unrealized_pnl") or 0.0) for r in rows if r.get("present")), 2
        ),
        "total_pnl": round(
            total_pnl
            + sum(float(r.get("unrealized_pnl") or 0.0) for r in rows if r.get("present")),
            2,
        ),
        "total_fees": round(total_fees, 2),
        "fees_pct_of_gross_pnl": (round(total_fees / gross * 100.0, 2) if gross else None),
        "gross_volume": round(total_volume, 2),
        "n_trades": n_trades,
        "n_wins": n_wins,
        "win_rate": (round(n_wins / n_trades, 4) if n_trades else None),
        "rows": rows,
    }


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
        execution_prefix=settings.stream_redis_execution_prefix,
        staleness_threshold=settings.stream_watchdog_staleness_threshold_seconds,
    )
    return {"enabled": True, **summary}


@app.get("/api/market/reality/{symbol}")
def market_reality(symbol: str) -> dict:
    """Forecast calibration monitor ("reality check").

    Replays the MC engine's *exact* 1-step-ahead predictive distribution over
    the stored feature history, point-in-time (same code path as the live
    consumer, seeded per window — no lookahead, offline/online parity), and
    scores it against the closes that actually materialized: empirical
    10–90-band coverage vs its nominal level, PIT histogram + Wasserstein/KS
    miscalibration, Kupiec + Christoffersen coverage backtests, and an
    anytime-valid e-process whose alarm is valid under continuous monitoring
    (Arnold–Henzi–Ziegel 2021 / Ville's inequality). 1-step-ahead only, so the
    formal tests are not contaminated by overlapping multi-step horizons; the
    multi-step fan + realized path is a labeled visual diagnostic.

    Research: Diebold, Gunther & Tay (1998); Gneiting, Balabdaoui & Raftery
    (2007); Arnold, Henzi & Ziegel (2021); Retzlaff (2025).
    """
    kv = _kv()
    if kv is None:
        return {"symbol": symbol.upper(), "enabled": False, "reality": None}
    symbol_u = symbol.upper()
    rows = kv.list_json(
        feature_key(settings.stream_redis_feature_prefix, symbol_u),
        reverse=False,  # chronological: replay is a forward walk
        maxlen=settings.stream_redis_feature_maxlen,
    )
    if not rows:
        return {"symbol": symbol_u, "enabled": True, "reality": None}
    engine = MonteCarloEngine(
        n_paths=settings.stream_simulation_paths,
        horizon_steps=settings.stream_simulation_horizon_steps,
        vol_windows=settings.stream_simulation_vol_windows,
        drift=settings.stream_simulation_drift,
        sample_paths=settings.stream_simulation_sample_paths,
        ewma_lambda=settings.stream_simulation_ewma_lambda,
        t_df_min=settings.stream_simulation_t_df_min,
        t_df_max=settings.stream_simulation_t_df_max,
        kelly_cap=settings.stream_simulation_kelly_cap,
        edge_min_sigma=settings.stream_simulation_edge_min_sigma,
    )
    reality = reality_report(
        [dict(w) for w in rows],
        engine=engine,
        nominal_coverage=settings.stream_reality_nominal_coverage,
        alpha=settings.stream_reality_evalue_alpha,
    )
    if reality is not None:
        reality["symbol"] = symbol_u
    return {"symbol": symbol_u, "enabled": True, "reality": reality}


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
def market_simulation(
    symbol: str,
    scenario: str | None = Query(
        default=None, description="what-if stress scenario (e.g. 'stress')"
    ),
) -> dict:
    """Monte Carlo forward fan chart (percentiles, VaR/ES) from the online store.

    ``scenario`` re-runs the *same* GBM engine on the real calibrated inputs
    with a what-if knob (e.g. volatility ×4) and labels the payload — a stress
    preview for the UI, never a fake data feed. Without it, serves the live
    artifact exactly as the consumer wrote it.
    """
    kv = _kv()
    if kv is None:
        return {"symbol": symbol.upper(), "enabled": False, "simulation": None}
    symbol_u = symbol.upper()
    if scenario:
        cfg = settings.stream_scenarios.get(scenario)
        if cfg is None:
            raise HTTPException(status_code=404, detail=f"unknown scenario '{scenario}'")
        history = kv.list_json(
            feature_key(settings.stream_redis_feature_prefix, symbol_u), reverse=False
        )
        closes = _closes_from_features([dict(w) for w in history])
        engine = MonteCarloEngine(
            n_paths=settings.stream_simulation_paths,
            horizon_steps=settings.stream_simulation_horizon_steps,
            vol_windows=settings.stream_simulation_vol_windows,
            drift=settings.stream_simulation_drift,
            sample_paths=settings.stream_simulation_sample_paths,
            ewma_lambda=settings.stream_simulation_ewma_lambda,
            t_df_min=settings.stream_simulation_t_df_min,
            t_df_max=settings.stream_simulation_t_df_max,
            kelly_cap=settings.stream_simulation_kelly_cap,
            edge_min_sigma=settings.stream_simulation_edge_min_sigma,
        )
        window_end = history[-1].get("window_end_ms") if history else None
        simulation = engine.forecast(
            closes,
            window_end_ms=window_end,
            sigma_scale=cfg["sigma_scale"],
            scenario={"name": scenario, **cfg},
        )
        return {
            "symbol": symbol_u,
            "enabled": True,
            "scenario": scenario,
            "simulation": simulation,
        }
    simulation = kv.get_json(simulation_key(settings.stream_redis_simulation_prefix, symbol_u))
    return {"symbol": symbol_u, "enabled": True, "simulation": simulation}


@app.get("/api/market/strategy/{symbol}")
def market_strategy(symbol: str) -> dict:
    """Live strategy P&L curve (compounded equity vs buy-and-hold)."""
    kv = _kv()
    if kv is None:
        return {"symbol": symbol.upper(), "enabled": False, "strategy": None}
    strategy = kv.get_json(strategy_key(settings.stream_redis_strategy_prefix, symbol.upper()))
    return {"symbol": symbol.upper(), "enabled": True, "strategy": strategy}


@app.get("/api/market/execution/{symbol}")
def market_execution(symbol: str) -> dict:
    """Paper-execution book: fills, realized/unrealized P&L, position, ledger.

    The execution engine consumes the predictor's *realized* directions and
    simulates fills at the next window's close with slippage + taker fees — no
    lookahead, deterministic by construction (see ``stream/execution.py``).
    """
    kv = _kv()
    if kv is None:
        return {"symbol": symbol.upper(), "enabled": False, "execution": None}
    execution = kv.get_json(execution_key(settings.stream_redis_execution_prefix, symbol.upper()))
    return {"symbol": symbol.upper(), "enabled": True, "execution": execution}


@app.get("/api/market/validation/{symbol}")
def market_validation(
    symbol: str,
    track: bool = Query(default=False),
    scenario: str | None = Query(
        default=None, description="what-if stress scenario (e.g. 'stress')"
    ),
) -> dict:
    """QuantPad-style pass probability: bootstrap the strategy's realized
    returns into simulated futures and score them against pass/fail rules.

    By default the rules are RISK-SCALED to the strategy's own realized
    terminal volatility (target/max-DD = target_sigma/max_drawdown_sigma ×
    σ_T) — a fixed 6%/8% contract is structurally unreachable for a 5m signal
    and pins the gauge at 0/100/0, so scaling restores a well-posed, moving
    question about realized edge relative to risk (research: the target-to-DD
    ratio, not absolute percentages, decides difficulty — OneTradeJournal/
    CrossTrade/PropFlux). The Monte Carlo is seeded from the latest feature
    window's event time (``window_end_ms``), so every 5m window resamples the
    futures differently while staying reproducible per window — the seed is
    echoed in the payload for audit.

    ``scenario`` re-runs the same bootstrap against an explicit what-if rule
    override (e.g. a 3% trailing stop / 6% target), so busted/red futures
    appear and the pass arc moves — a stress preview, clearly labeled, never a
    fake data feed.

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
    cfg = settings.stream_scenarios.get(scenario) if scenario else None
    if scenario and cfg is None:
        raise HTTPException(status_code=404, detail=f"unknown scenario '{scenario}'")
    mc = StrategyMonteCarlo(
        n_sims=settings.stream_validation_sims,
        target=cfg.get("target") if cfg else None,
        max_drawdown=cfg.get("max_drawdown") if cfg else None,
        target_sigma=settings.stream_validation_target_sigma,
        max_drawdown_sigma=settings.stream_validation_max_drawdown_sigma,
        seed=seed,
    )
    validation = mc.validate(strategy["strategy_equity"])
    if track and validation is not None:
        track_validation(
            symbol.upper(),
            validation,
            target=validation.get("target"),
            max_drawdown=validation.get("max_drawdown_rule") or 0.0,
            seed=seed,
            n_sims=settings.stream_validation_sims,
        )
    return {
        "symbol": symbol.upper(),
        "enabled": True,
        "seed": seed,
        "window_end_ms": strategy.get("window_end_ms"),
        "scenario": scenario,
        "validation": validation,
    }


@app.get("/api/market/validation/{symbol}/geometry")
def market_validation_geometry(symbol: str) -> dict:
    """Geometry Optimizer heat grid.

    Holds the strategy's realized EV constant while sweeping win-rate × R:R
    shape, showing pass-probability across 49 configurations. Research-backed
    insight (PropSim/QuantPad): trailing-DD rules are path-dependent, so two
    traders with identical edge can have 90% vs 10% pass rates.

    The rules use the same risk-scaling as the validation gauge (target/max-DD
    multiples of the strategy's realized σ_T), so the grid and the edge sweep
    answer the same well-posed question as the headline pass probability.

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
    from stream.strategy_mc import effective_rules, strategy_returns_from_equity

    returns = strategy_returns_from_equity(strategy["strategy_equity"])
    seed = strategy.get("window_end_ms") or settings.stream_validation_seed
    rules = effective_rules(
        returns,
        target=None,
        max_drawdown=None,
        target_sigma=settings.stream_validation_target_sigma,
        max_drawdown_sigma=settings.stream_validation_max_drawdown_sigma,
    )
    mc = StrategyMonteCarlo(
        n_sims=settings.stream_validation_sims,
        target_sigma=settings.stream_validation_target_sigma,
        max_drawdown_sigma=settings.stream_validation_max_drawdown_sigma,
        seed=seed,
    )
    grid = mc.geometry_sweep(
        returns=returns,
        n_periods=len(returns),
        target=rules["target"] or 0.0,
        max_drawdown=rules["max_drawdown"] or 0.0,
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
        "rules": rules,
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
