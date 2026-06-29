"""Fraud Detection FastAPI microservice.

Why FastAPI over Flask:
  - Async request handling: handles concurrent requests without blocking.
  - Pydantic integration: automatic request validation and response serialization.
  - Auto-generated OpenAPI docs at /docs — zero extra work.
  - Native support for dependency injection (database connections, model loading).
  - prometheus_fastapi_instrumentator: automatic p50/p95/p99 metrics with one line.

Architecture:
  POST /v1/score        → Score a single transaction
  POST /v1/score/batch  → Score up to 100 transactions in one request
  GET  /v1/explain/{id} → SHAP feature importance for a scored transaction
  GET  /health          → Liveness check (load balancer polls this)
  GET  /metrics         → Prometheus scrape endpoint

Request flow:
  Request → Pydantic validation → Prediction cache check (Redis)
         → Circuit breaker → Feature fetch (Redis online store)
         → XGBoost inference → Qdrant vector score → Combined score
         → Cache result → Prometheus metrics → Structured log → Response
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from fraud_detection.api.schemas import (
    BatchScoreResponse,
    BatchTransactionRequest,
    FraudScoreResponse,
    HealthResponse,
    TransactionRequest,
)
from shared.clients import QdrantPatternStore, RedisFeatureStore
from shared.mlflow_utils import setup_mlflow
from shared.observability.logging import get_logger, setup_logging
from shared.observability.metrics import (
    FRAUD_PREDICTIONS_TOTAL,
    FRAUD_INFERENCE_DURATION,
    FRAUD_RATE_GAUGE,
    FRAUD_SCORE_HISTOGRAM,
    REDIS_FEATURE_MISSES_TOTAL,
    REDIS_FEATURE_FETCH_DURATION,
)
from shared.observability.tracing import get_tracer, setup_tracing
from shared.prediction_cache import PredictionCache
from shared.resilience import CircuitBreaker, CircuitOpenError, FallbackModel

# ── Constants matching training.py FEATURE_COLS ───────────────────────────────
FEATURE_COLS: list[str] = [
    "amount",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sum_24h",
    "amount_mean_24h",
    "amount_std_24h",
    "velocity_ratio",
    "amount_deviation",
    "hour_of_day",
    "is_night",
]

MODEL_WEIGHT = 0.7
VECTOR_WEIGHT = 0.3
FRAUD_THRESHOLD = 0.5

# ── App-level state (initialised once at startup) ──────────────────────────────
log = get_logger(__name__)
tracer = get_tracer(__name__)
_start_time = time.monotonic()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all resources once at startup, release at shutdown.

    Why lifespan over @app.on_event("startup"):
      Lifespan is the modern FastAPI pattern (on_event is deprecated in 0.110+).
      Using asynccontextmanager ensures resources are released cleanly on shutdown.
    """
    setup_logging(json_logs=True)
    setup_tracing(service_name="fraud-api")

    log.info("api_startup", msg="Loading fraud detection model and connections")
    setup_mlflow()

    app.state.model = mlflow.xgboost.load_model("models:/fraud_detection_model@champion")
    app.state.model_version = "fraud_detection_model@champion"
    
    import feast
    from pathlib import Path
    repo_path = Path(__file__).parent.parent.resolve() / "feature_repo"
    log.info("initializing_feast_online_feature_store_api", repo_path=str(repo_path))
    app.state.feature_store = feast.FeatureStore(repo_path=str(repo_path))
    
    app.state.qdrant = QdrantPatternStore()
    app.state.cache = PredictionCache(ttl_seconds=60)
    app.state.fallback_model = FallbackModel()

    # Circuit breakers — one per external dependency
    app.state.redis_breaker = CircuitBreaker(name="redis", failure_threshold=5, timeout=30)
    app.state.qdrant_breaker = CircuitBreaker(name="qdrant", failure_threshold=3, timeout=60)
    app.state.model_breaker = CircuitBreaker(name="model", failure_threshold=3, timeout=120)

    log.info("api_ready", model_version=app.state.model_version)
    yield

    log.info("api_shutdown")


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring with XGBoost + Qdrant vector similarity.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Auto-instrument all endpoints: adds p50/p95/p99 latency histograms automatically
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health(request: Request):
    """Liveness probe. Load balancers and Kubernetes poll this endpoint."""
    return HealthResponse(
        status="healthy",
        model_version=request.app.state.model_version,
        redis_connected=request.app.state.cache.ping(),
        qdrant_connected=True,  # TODO: add Qdrant ping
        uptime_seconds=round(time.monotonic() - _start_time, 1),
    )


# ── Single transaction scoring ─────────────────────────────────────────────────

@app.post(
    "/v1/score",
    response_model=FraudScoreResponse,
    status_code=status.HTTP_200_OK,
    tags=["Scoring"],
    summary="Score a single transaction for fraud risk.",
)
async def score_transaction(
    txn: TransactionRequest,
    request: Request,
    explain: bool = Query(default=False, description="Include SHAP feature attributions in response."),
):
    """Score one transaction.

    Returns a fraud probability (0–1) combining:
    - XGBoost model score (trained on velocity + amount features)
    - Qdrant vector similarity score (proximity to known fraud patterns)
    """
    with tracer.start_as_current_span("score_transaction") as span:
        span.set_attribute("card_id", txn.card_id)
        span.set_attribute("amount", txn.amount)

        start = time.monotonic()

        # ── 1. Check prediction cache ────────────────────────────────────────
        cached = request.app.state.cache.get(txn.card_id, txn.amount, txn.merchant_category)
        if cached:
            FRAUD_PREDICTIONS_TOTAL.labels(
                model_version=request.app.state.model_version,
                result="fraud" if cached["fraud_label"] else "legit",
            ).inc()
            return FraudScoreResponse(**cached, cache_hit=True)

        # ── 2. Fetch online features from Feast ──────────────────────────────
        with tracer.start_as_current_span("feast_online_feature_fetch"):
            feat_start = time.monotonic()
            try:
                response = request.app.state.redis_breaker.call(
                    request.app.state.feature_store.get_online_features,
                    features=[
                        "transaction_features:txn_count_1h",
                        "transaction_features:txn_count_24h",
                        "transaction_features:amount_sum_24h",
                        "transaction_features:amount_mean_24h",
                        "transaction_features:amount_std_24h",
                        "transaction_features:velocity_ratio",
                        "transaction_features:amount_deviation",
                        "transaction_features:hour_of_day",
                        "transaction_features:is_night",
                    ],
                    entity_rows=[{"card_id": txn.card_id}]
                )
                feats_dict = response.to_dict()
                redis_feats = {k.split("__")[-1]: v[0] for k, v in feats_dict.items() if v}
            except CircuitOpenError:
                log.warning("redis_circuit_open", card_id=txn.card_id)
                redis_feats = {}
            except Exception as exc:
                log.warning("feast_fetch_failed", card_id=txn.card_id, error=str(exc))
                redis_feats = {}
            REDIS_FEATURE_FETCH_DURATION.observe(time.monotonic() - feat_start)

            if not redis_feats:
                REDIS_FEATURE_MISSES_TOTAL.inc()

        # ── 3. Build feature row (same order as FEATURE_COLS in training.py) ─
        row = {**txn.model_dump(), **redis_feats}
        row.setdefault("txn_count_1h", 1)
        row.setdefault("txn_count_24h", 1)
        row.setdefault("amount_sum_24h", txn.amount)
        row.setdefault("amount_mean_24h", txn.amount)
        row.setdefault("amount_std_24h", 0.0)
        row.setdefault("velocity_ratio", 1.0)
        row.setdefault("amount_deviation", 1.0)
        row.setdefault("hour_of_day", datetime.now(timezone.utc).hour)
        row.setdefault("is_night", int(datetime.now(timezone.utc).hour < 6 or datetime.now(timezone.utc).hour > 22))

        X = np.array([[float(row.get(c, 0.0)) for c in FEATURE_COLS]])

        # ── 4. XGBoost inference (with circuit breaker + fallback) ───────────
        with tracer.start_as_current_span("xgboost_inference"):
            try:
                model_score = float(
                    request.app.state.model_breaker.call(
                        request.app.state.model.predict_proba, X
                    )[0][1]
                )
            except CircuitOpenError:
                log.warning("model_circuit_open_using_fallback", card_id=txn.card_id)
                model_score = request.app.state.fallback_model.predict_proba(row)

        # ── 5. Qdrant vector similarity score ────────────────────────────────
        with tracer.start_as_current_span("qdrant_vector_score"):
            v_cols_values = [float(row.get(f"v{i}", 0)) for i in range(1, 29)]
            vector = v_cols_values + [txn.amount / 25000.0, 0.0]
            try:
                vector_score = request.app.state.qdrant_breaker.call(
                    request.app.state.qdrant.vector_fraud_score, vector
                )
            except CircuitOpenError:
                log.warning("qdrant_circuit_open", card_id=txn.card_id)
                vector_score = 0.0

        # ── 6. Combine scores ────────────────────────────────────────────────
        fraud_score = MODEL_WEIGHT * model_score + VECTOR_WEIGHT * vector_score
        is_fraud = fraud_score >= FRAUD_THRESHOLD

        result = {
            "transaction_id": txn.transaction_id,
            "card_id": txn.card_id,
            "amount": txn.amount,
            "fraud_score": round(fraud_score, 6),
            "model_score": round(model_score, 6),
            "vector_score": round(vector_score, 6),
            "fraud_label": is_fraud,
            "model_version": request.app.state.model_version,
            "scored_at": datetime.now(timezone.utc),
            "cache_hit": False,
        }

        # ── 7. Record Prometheus metrics ─────────────────────────────────────
        elapsed = time.monotonic() - start
        FRAUD_INFERENCE_DURATION.labels(model_version=request.app.state.model_version).observe(elapsed)
        FRAUD_PREDICTIONS_TOTAL.labels(
            model_version=request.app.state.model_version,
            result="fraud" if is_fraud else "legit",
        ).inc()
        FRAUD_SCORE_HISTOGRAM.labels(model_version=request.app.state.model_version).observe(fraud_score)

        # ── 8. Structured log ────────────────────────────────────────────────
        log.info(
            "transaction_scored",
            transaction_id=txn.transaction_id,
            card_id=txn.card_id,
            fraud_score=round(fraud_score, 4),
            fraud_label=is_fraud,
            latency_ms=round(elapsed * 1000, 2),
            model_version=request.app.state.model_version,
        )

        # ── 9. Cache result ──────────────────────────────────────────────────
        request.app.state.cache.set(txn.card_id, txn.amount, txn.merchant_category, {
            **result,
            "scored_at": result["scored_at"].isoformat(),
        })

        return FraudScoreResponse(**result)


# ── Batch scoring ──────────────────────────────────────────────────────────────

@app.post(
    "/v1/score/batch",
    response_model=BatchScoreResponse,
    tags=["Scoring"],
    summary="Score up to 100 transactions in one request.",
)
async def score_batch(batch: BatchTransactionRequest, request: Request):
    """Batch scoring for offline pipelines or bulk reprocessing."""
    start = time.monotonic()
    results = []
    for txn in batch.transactions:
        # Reuse single-score endpoint logic by constructing a minimal request
        resp = await score_transaction(txn, request, explain=False)
        results.append(resp)

    elapsed_ms = round((time.monotonic() - start) * 1000, 2)
    fraud_count = sum(1 for r in results if r.fraud_label)

    return BatchScoreResponse(
        results=results,
        total=len(results),
        fraud_count=fraud_count,
        processing_time_ms=elapsed_ms,
    )


# ── Error handlers ─────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled_exception", error=str(exc), path=str(request.url))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs for details."},
    )
