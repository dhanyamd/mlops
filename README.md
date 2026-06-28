# MLOps Academy — End-to-End Learning Projects

Two production-grade ML systems built from the [ML Academy MLOps course](https://www.mlacademy.ai/articles/free-mlops-course-ml-production-system-breakdown), covering all 10 components of a production ML system.

| Project | Pattern | Storage | Streaming | Model |
|---------|---------|---------|-----------|-------|
| **Demand Forecasting** | Batch ETL + Prefect | ClickHouse + PostgreSQL | — | XGBoost + lag features |
| **Fraud Detection** | Kafka + Spark Streaming | ClickHouse + Redis + **Qdrant** | Kafka | XGBoost + vector ensemble |

## What You'll Learn

This repo is a **complete MLOps curriculum in code** — not toy notebooks. Every concept from the 5-day course is implemented:

### The 10 Production ML Components

| # | Component | Demand Forecasting | Fraud Detection |
|---|-----------|-------------------|-----------------|
| 1 | Data Storage | **ClickHouse** (sales) + **PostgreSQL** (catalog) | **ClickHouse** + **Qdrant** (vectors) |
| 2 | Data Processing | Prefect batch | **Kafka** + **Spark Structured Streaming** |
| 3 | Preprocessing & Feature Eng | Lag/rolling features | Velocity features (Spark) + PCA vectors |
| 4 | Training Pipeline | Time-based XGBoost | Imbalanced XGBoost from warehouse |
| 5 | Inference Pipeline | Batch → ClickHouse | Kafka service + Qdrant NN ensemble |
| 6 | Feature Store | ClickHouse (batch) | **Redis** online (Spark sink) |
| 7 | Model Registry | MLflow @champion | MLflow @champion |
| 8 | Experiment Tracking | MLflow + MinIO/S3 | MLflow + MinIO/S3 |
| 9 | Monitoring | Drift (KS/PSI) | Drift + Kafka alerts topic |
| 10 | CI/CD | GitHub Actions | GitHub Actions |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEMAND FORECASTING (Batch System)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Snowflake/CH ─┐                                                            │
│  PostgreSQL  ──┼──► ETL ──► Preprocess ──► Feature Eng ──► Training ──► MLflow
│  CSV (weather)─┘                              │                │            │
│                                               └──── Inference ◄─┘            │
│                                                      │                      │
│                                              PostgreSQL + Streamlit UI       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    FRAUD DETECTION (Real-Time System)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Kafka (live txns) ──► Spark Streaming* ──► Online Store (Redis)           │
│                              │                      │                       │
│  Historical DW ──► Batch ETL ──► Offline Store ─────┼──► Training ──► MLflow
│                                                     │                       │
│                              Inference ◄────────────┘                       │
│                                  │                                          │
│                          Kafka (predictions) ──► Alerting + DB Actioning    │
│  * Python consumer simulates Spark Streaming for local learning             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd mlops && make setup
make infra-up          # Kafka, ClickHouse, Qdrant, Redis, Spark, MLflow...
make bootstrap         # Download REAL datasets → ClickHouse + Postgres + Qdrant
make demand            # Demand forecasting pipeline
make fraud             # Fraud training pipeline
make stream            # Kafka → Spark → Redis → Inference → ClickHouse
```

**UIs:** Kafka UI `:8080` · ClickHouse `:8123` · Qdrant `:6333/dashboard` · MLflow `:5000` · Spark `:8081`

See [docs/DATABASE_GUIDE.md](docs/DATABASE_GUIDE.md) for why each database exists (including when Qdrant is the right call vs wrong).

## Project Structure

```
mlops/
├── demand_forecasting/          # Batch ML system
│   ├── data/                    # Synthetic data generators
│   ├── etl/                     # Multi-source ETL pipeline
│   ├── pipelines/               # Preprocess → Features → Train → Infer
│   ├── flows/                   # Prefect orchestration
│   └── app/                     # Streamlit dashboard
├── fraud_detection/             # Real-time ML system
│   ├── feature_repo/            # Feast feature definitions
│   ├── streaming/               # Kafka producer + consumer
│   ├── pipelines/               # Training pipeline
│   ├── etl/                     # Batch historical ETL
│   └── flows/                   # Prefect orchestration
├── shared/                      # Config, MLflow utils, drift monitoring
├── infra/                       # PostgreSQL init scripts
├── docker-compose.yml           # Local production stack
└── docs/ARCHITECTURE.md         # Deep-dive architecture guide
```

## Key Concepts Explained

### When to Use a Feature Store

From [Day 4 of the course](https://www.mlacademy.ai/articles/free-mlops-course-feature-store-model-registry-and-experiment-tracking):

- **Use it**: Real-time inference, shared features across models, strict training-serving consistency
- **Skip it**: Single-model batch inference (demand forecasting in this repo)

Demand forecasting uses versioned parquet datasets + orchestration instead — the simpler pattern most churn and forecasting systems use.

### Batch vs Real-Time Inference

| | Batch (Forecasting) | Real-Time (Fraud) |
|---|---|---|
| Trigger | Hourly/daily/weekly | Every transaction |
| Latency | Minutes to hours | Milliseconds |
| Features | Computed in pipeline | Pre-computed in online store |
| Output | Warehouse table | Kafka topic + API |

### Orchestration Triggers

Both systems support the three retraining triggers from the course:

1. **Scheduled** — Prefect flows run on cron
2. **Performance-based** — metric threshold in monitoring
3. **Drift-based** — `shared/monitoring/drift.py` KS test + PSI

### Model Registry Lifecycle

```
Experiment → Tracking Server → Staging → Production (@champion) → Archive
```

Implemented via MLflow with `@champion` alias for inference pipelines.

## Course Reference Map

| Course Day | Topic | Code Location |
|------------|-------|---------------|
| [Day 1](https://www.mlacademy.ai/articles/free-mlops-course-ml-production-system-breakdown) | 10 components overview | This README + `docs/ARCHITECTURE.md` |
| [Day 2](https://www.mlacademy.ai/articles/free-mlops-course-databases-and-processing) | Databases & processing | `docker-compose.yml`, ETL pipelines |
| [Day 3](https://www.mlacademy.ai/articles/free-mlops-course-machine-learning-pipelines) | ML pipelines | `*/pipelines/` |
| [Day 4](https://www.mlacademy.ai/articles/free-mlops-course-feature-store-model-registry-and-experiment-tracking) | Feature store & registry | `fraud_detection/feature_repo/`, `shared/mlflow_utils.py` |
| [Day 5](https://www.mlacademy.ai/articles/free-mlops-course-data-drift-and-model-monitoring-ci-cd-pipelines) | Monitoring & CI/CD | `shared/monitoring/`, `.github/workflows/` |

## Testing & CI

### Troubleshooting

**macOS + XGBoost:** If you see `libomp.dylib could not be loaded`, install OpenMP:

```bash
brew install libomp
```

**Sync dependencies after pulling changes:**

```bash
uv sync
```

```bash
make test    # uv run pytest
make lint    # uv run ruff
```

CI pipeline runs on every push: lint → unit tests → (Docker build ready).

## Learning Path

1. **Start with Demand Forecasting** — simpler batch pattern, understand ETL → pipelines → registry
2. **Read `docs/ARCHITECTURE.md`** — component-by-component deep dive
3. **Move to Fraud Detection** — adds Kafka, Feast, real-time inference
4. **Explore MLflow UI** — compare experiments, understand model promotion
5. **Modify drift thresholds** — trigger retraining in `flows/orchestration.py`
6. **Add a new feature** — practice the full FTI (Feature → Train → Infer) loop

## License

MIT — built for learning. Based on concepts from [ML Academy](https://www.mlacademy.ai).
