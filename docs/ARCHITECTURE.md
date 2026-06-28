# Architecture Deep Dive

This document maps every file and design decision to the ML Academy production ML system framework.

## Demand Forecasting — Batch System

Mirrors the **Demand Forecasting ML System Design** diagram from ML Academy.

### Data Sources (Simulated)


| Production Source                         | Local Stand-in                             | File                                            |
| ----------------------------------------- | ------------------------------------------ | ----------------------------------------------- |
| Snowflake / ClickHouse (historical sales) | Parquet files                              | `demand_forecasting/data/generate_data.py`      |
| PostgreSQL (product catalog)              | PostgreSQL `demand_forecasting.products`   | `infra/init-db.sql`                             |
| Salesforce (promotions)                   | PostgreSQL `demand_forecasting.promotions` | `infra/init-db.sql`                             |
| External (holidays, weather)              | CSV                                        | `data/demand_forecasting/raw/external_data.csv` |


### Pipeline DAG

```
generate_data → ETL → preprocessing → feature_engineering → training → inference
                     ↑                                              ↓
              drift_check ────────────────────────────── (retrain trigger) 
```

**Why no Feature Store?** Batch inference on a schedule with a single model does not need the complexity of Feast/Hopsworks. Versioned parquet + orchestration is the industry-standard simpler pattern for forecasting and churn systems.

### Feature Engineering Choices

Demand forecasting uses **lag features** and **rolling statistics** — the standard approach for time-series regression:

- `lag_1, lag_7, lag_14, lag_28` — autoregressive signal
- `roll_mean_7, roll_std_7` — smoothed trends
- Calendar features — day of week, month, weekend flag
- Price features — effective price after promotions

### Model Selection

**XGBoost** chosen over Prophet/DeepAR because:

- Handles mixed tabular features (promotions, weather, lags) natively
- Fast training for weekly retraining schedules
- Strong baseline for retail demand forecasting
- Aligns with course code examples

### Output Design

Forecasts written to:

1. `data/demand_forecasting/forecasts/latest_forecasts.parquet` — file output
2. `demand_forecasting.forecasts` PostgreSQL table — warehouse stand-in
3. Streamlit dashboard — business consumption

Schema: `product_id, forecast_date, predicted_demand, lower_bound, upper_bound, model_version, scored_at`

---

## Fraud Detection — Real-Time System

Mirrors the **Fraud Detection ML System Design** diagram.

### Data Sources


| Source                  | Technology                 | Purpose         |
| ----------------------- | -------------------------- | --------------- |
| Real-time transactions  | Kafka topic `transactions` | Live scoring    |
| Historical transactions | Parquet / PostgreSQL       | Training data   |
| Fraud labels            | PostgreSQL                 | Ground truth    |
| User profiles           | PostgreSQL                 | Static features |


### Feature Store Architecture (Feast)

```
                    ┌─────────────────┐
  Batch ETL ───────►│  Offline Store  │──────► Training Pipeline
                    │   (Parquet)     │
                    └────────┬────────┘
                             │ materialize
                    ┌────────▼────────┐
  Kafka Stream ────►│  Online Store   │──────► Inference Pipeline
  (feature compute) │    (Redis)      │
                    └─────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Feature Views  │  ← prevents training-serving skew
                    └─────────────────┘
```

**Feature Views defined in** `fraud_detection/feature_repo/features.py`:

- `transaction_features` — velocity, amount deviation, time-of-day
- `user_profile_features` — account age, avg spend, risk tier

### Feature Engineering Choices

Fraud detection uses **velocity** and **graph-like** features:

- `txn_count_1h, txn_count_24h` — transaction velocity
- `amount_deviation` — amount vs user historical mean
- `velocity_ratio` — burst detection (many txns in 1h vs 24h)
- `is_night` — time-based risk signal
- `risk_score` — from user profile tier

### Streaming Path

```
Producer → Kafka (transactions) → Consumer → Feature Computer → Model → Kafka (predictions)
                                                                      → PostgreSQL (actioning)
                                                                      → Alert (if fraud)
```

The Python consumer simulates **Spark Streaming** for local learning. In production, Spark Streaming/Flink would write features to the online store continuously.

### Model Selection

**XGBoost Classifier** with `scale_pos_weight=10` for imbalanced fraud data (~3% fraud rate). Alternatives considered:

- Isolation Forest — good for anomaly, less interpretable for labeled fraud
- Deep learning — overkill for tabular fraud with limited data

### Output Design & Actioning

Predictions schema:

```json
{
  "transaction_id": "TXN_RT_000001",
  "fraud_score": 0.87,
  "fraud_label": true,
  "model_version": "fraud_detection_model@champion",
  "scored_at": "2026-06-25T12:00:00"
}
```

Downstream:

- **Transaction Update Service** — writes to `fraud_detection.scored_transactions`
- **Alerting Service** — logs fraud alerts (extend to PagerDuty/Slack)

---

## Shared Infrastructure

### Docker Compose Services


| Service    | Port | Role                           |
| ---------- | ---- | ------------------------------ |
| PostgreSQL | 5432 | Relational storage             |
| Redis      | 6379 | Feast online store             |
| Kafka      | 9092 | Event streaming                |
| MLflow     | 5000 | Experiment tracking + registry |
| MinIO      | 9000 | S3-compatible artifact store   |
| Prefect    | 4200 | Orchestration UI               |


### MLflow Model Lifecycle

```python
# Training registers and promotes
promote_model("demand_forecast_model", version, stage="Production", alias="champion")

# Inference loads by alias
model = mlflow.pyfunc.load_model("models:/demand_forecast_model@champion")
```

### Monitoring & Drift

`shared/monitoring/drift.py` implements:

- **Kolmogorov-Smirnov test** — univariate distribution shift
- **Population Stability Index** — industry-standard drift metric
- **Retrain trigger** — fires when ≥2 features drift

Actions on drift (from Day 5):

1. Retrain on fresh data
2. Rollback to previous @champion
3. Turn off solution
4. Raise investigation

### CI/CD Pipeline

`.github/workflows/ci.yml`:

1. **Lint** — Ruff (code quality)
2. **Test** — Pytest (unit + integration)
3. **Build** — Docker images (extend for CD)

---

## Design Decisions & Trade-offs

### Local vs Cloud

This repo uses Docker Compose to simulate cloud services locally. Mapping to production:


| Local      | AWS                     | GCP                      |
| ---------- | ----------------------- | ------------------------ |
| PostgreSQL | RDS                     | Cloud SQL                |
| Kafka      | MSK                     | Pub/Sub                  |
| MinIO      | S3                      | GCS                      |
| MLflow     | SageMaker / self-hosted | Vertex AI                |
| Prefect    | Prefect Cloud           | Cloud Composer (Airflow) |
| Redis      | ElastiCache             | Memorystore              |


### When dbt + Airflow Is Enough

For batch systems without real-time requirements, many teams use:

- **dbt** — SQL transformations in the warehouse
- **Airflow** — scheduling

This repo uses **Prefect** + Python pipelines to match the course. The patterns are identical — swap the orchestrator, keep the pipeline structure.

### Extending This Repo

Suggested next steps for deeper learning:

1. Add Great Expectations for data quality checks in ETL
2. Wire Prefect deployments with cron schedules
3. Add Evidently dashboards for drift visualization
4. Deploy inference as a FastAPI microservice
5. Add GitHub Actions CD to push Docker images on merge

