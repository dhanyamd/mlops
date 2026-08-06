"""Dashboard API — read-only FastAPI app over the live Silver/Gold layers.

Serves the same real numbers the CLI shows: market bars, PIT fundamentals,
the PEAD event study, pipeline latency, and macro series. No mocking, no
business values hardcoded — instruments come from ``INGEST_DEFAULT_TICKERS``.

Run:  uv run uvicorn api.main:app --port 8000
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from api import db
from config.logging import configure_logging
from config.settings import csv_list, get_settings
from scripts.pead_backtest import compute_pead

configure_logging()

settings = get_settings()

app = FastAPI(title="Quant Signal Dashboard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=csv_list(settings.api_cors_origins),
    allow_methods=["GET"],
    allow_headers=["*"],
)

# The PEAD study is a ~10s query; cache per parameter set for a short TTL.
_pead_cache: dict[str, tuple[float, dict]] = {}


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
