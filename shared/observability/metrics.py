"""Prometheus metrics for both ML systems.

Architecture:
  - metrics.py defines ALL Prometheus instruments in one place.
  - inference_service.py, FastAPI app, training pipeline all import from here.
  - Prometheus scrapes /metrics endpoint (exposed by FastAPI or a standalone server).

Why Prometheus:
  - Pull-based: Prometheus polls your /metrics endpoint on a schedule.
  - Counter, Histogram, Gauge cover every ML monitoring need.
  - Grafana reads from Prometheus to build dashboards.
  - Industry standard at Google, Cloudflare, Stripe, Uber.

Metric naming convention:
  <namespace>_<subsystem>_<name>_<unit>
  e.g. mlops_fraud_inference_duration_seconds
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# ── Fraud Detection Metrics ────────────────────────────────────────────────────

# How many predictions served (total throughput)
FRAUD_PREDICTIONS_TOTAL = Counter(
    name="mlops_fraud_predictions_total",
    documentation="Total fraud scoring requests processed.",
    labelnames=["model_version", "result"],  # result: fraud | legit
)

# End-to-end inference latency (p50/p95/p99 visible in Grafana)
FRAUD_INFERENCE_DURATION = Histogram(
    name="mlops_fraud_inference_duration_seconds",
    documentation="Time from Kafka message received to prediction produced.",
    labelnames=["model_version"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# Feature retrieval latency from Redis
REDIS_FEATURE_FETCH_DURATION = Histogram(
    name="mlops_redis_feature_fetch_seconds",
    documentation="Latency of Redis online feature store lookup.",
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
)

# Redis cache miss rate — high miss rate = stale feature store
REDIS_FEATURE_MISSES_TOTAL = Counter(
    name="mlops_redis_feature_misses_total",
    documentation="Redis feature store cache misses (card_id not found in online store).",
)

# Rolling fraud rate — business-level signal (not just technical metric)
FRAUD_RATE_GAUGE = Gauge(
    name="mlops_fraud_rate_rolling",
    documentation="Rolling fraud detection rate (last 1000 predictions).",
    labelnames=["model_version"],
)

# Model score distribution — tracks score drift over time
FRAUD_SCORE_HISTOGRAM = Histogram(
    name="mlops_fraud_score_distribution",
    documentation="Distribution of raw fraud model scores (0-1).",
    labelnames=["model_version"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# Circuit breaker state (0=closed/healthy, 1=open/failing)
CIRCUIT_BREAKER_STATE = Gauge(
    name="mlops_circuit_breaker_open",
    documentation="1 if circuit breaker is open (service degraded), 0 if healthy.",
    labelnames=["service"],  # service: model | redis | qdrant
)

# ── Demand Forecasting Metrics ─────────────────────────────────────────────────

# Training run outcome
TRAINING_RUNS_TOTAL = Counter(
    name="mlops_training_runs_total",
    documentation="Total model training runs.",
    labelnames=["project", "status"],  # status: success | failed
)

# Training duration — catch regressions (sudden 2x slowdown = data problem)
TRAINING_DURATION = Histogram(
    name="mlops_training_duration_seconds",
    documentation="Time to complete a full training pipeline run.",
    labelnames=["project"],
    buckets=(30, 60, 120, 300, 600, 1200, 3600),
)

# Model performance metrics logged per training run
MODEL_AUC_GAUGE = Gauge(
    name="mlops_model_auc",
    documentation="AUC-ROC of the latest champion model.",
    labelnames=["project", "model_version"],
)

MODEL_F1_GAUGE = Gauge(
    name="mlops_model_f1",
    documentation="F1 score of the latest champion model.",
    labelnames=["project", "model_version"],
)

# Forecast accuracy for demand forecasting
FORECAST_MAPE_GAUGE = Gauge(
    name="mlops_forecast_mape",
    documentation="Mean Absolute Percentage Error of the latest demand forecast.",
    labelnames=["model_name"],
)

# Pipeline health — catches silent failures
PIPELINE_RUNS_TOTAL = Counter(
    name="mlops_pipeline_runs_total",
    documentation="Total Prefect pipeline executions.",
    labelnames=["pipeline", "status"],
)

# ── Drift Metrics ──────────────────────────────────────────────────────────────

# KS statistic per feature — high values = distribution shift
FEATURE_DRIFT_SCORE = Gauge(
    name="mlops_feature_drift_ks_statistic",
    documentation="Kolmogorov-Smirnov statistic comparing feature distribution to training baseline.",
    labelnames=["feature_name", "project"],
)

# Data quality gate pass/fail
DATA_QUALITY_GATE = Gauge(
    name="mlops_data_quality_gate",
    documentation="1 if data quality gate passed, 0 if failed.",
    labelnames=["project", "checkpoint"],
)

# ── Grouped export (for easy import) ──────────────────────────────────────────

METRICS = {
    # Fraud
    "fraud_predictions_total": FRAUD_PREDICTIONS_TOTAL,
    "fraud_inference_duration": FRAUD_INFERENCE_DURATION,
    "redis_feature_fetch_duration": REDIS_FEATURE_FETCH_DURATION,
    "redis_feature_misses_total": REDIS_FEATURE_MISSES_TOTAL,
    "fraud_rate_gauge": FRAUD_RATE_GAUGE,
    "fraud_score_histogram": FRAUD_SCORE_HISTOGRAM,
    "circuit_breaker_state": CIRCUIT_BREAKER_STATE,
    # Training
    "training_runs_total": TRAINING_RUNS_TOTAL,
    "training_duration": TRAINING_DURATION,
    "model_auc_gauge": MODEL_AUC_GAUGE,
    "model_f1_gauge": MODEL_F1_GAUGE,
    "forecast_mape_gauge": FORECAST_MAPE_GAUGE,
    "pipeline_runs_total": PIPELINE_RUNS_TOTAL,
    # Drift
    "feature_drift_score": FEATURE_DRIFT_SCORE,
    "data_quality_gate": DATA_QUALITY_GATE,
}


def start_metrics_server(port: int = 8000) -> None:
    """Start a standalone Prometheus metrics HTTP server.

    Use this for non-FastAPI processes (Kafka consumer, training pipeline).
    FastAPI apps use prometheus_fastapi_instrumentator instead.

    After calling this, Prometheus can scrape http://host:8000/metrics.
    """
    start_http_server(port)
