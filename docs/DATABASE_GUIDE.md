# Database & Storage Guide — What Startups and Labs Actually Use

This project uses a **polyglot persistence** architecture: the right database for each job, not one DB for everything.

## Stack Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION ML PLATFORM                          │
├──────────────┬──────────────┬──────────────┬──────────────┬───────────┤
│  ClickHouse  │  PostgreSQL  │    Redis     │   Qdrant     │   Kafka   │
│  (OLAP)      │  (OLTP)      │  (Online)    │  (Vectors)   │  (Events) │
├──────────────┼──────────────┼──────────────┼──────────────┼───────────┤
│ Snowflake    │ Supabase/    │ Feast online │ Pinecone/    │ MSK /     │
│ BigQuery     │ Neon class   │ store, cache │ Weaviate     │ Confluent │
│ class        │              │              │ class        │ class     │
└──────────────┴──────────────┴──────────────┴──────────────┴───────────┘
         ▲              ▲              ▲              ▲
         │              │              │              │
   Batch analytics   Catalog/CRM   Real-time      Fraud pattern
   913k sales rows   products      features       similarity NN
```

## When to Use Each Database

| Database | Use Case in This Repo | Used By (Industry) | Why NOT everywhere? |
|----------|----------------------|--------------------|---------------------|
| **ClickHouse** | Historical sales, fraud txns, forecast output, audit logs | Uber, Cloudflare, eBay analytics | Bad at row updates, not for CRM |
| **PostgreSQL** | Product catalog, promotions, alert rules | Every startup (Supabase, Neon, RDS) | Too slow for billion-row scans |
| **Redis** | Online feature store (Spark → Redis) | Twitter, Stripe, Feast online store | Not durable analytics storage |
| **Qdrant** | Fraud pattern vector search | Fraud labs, semantic search startups | Useless for tabular forecasting |
| **Kafka** | Transaction stream, feature stream, alerts | LinkedIn, Netflix, every fintech | Overkill for nightly batch-only |
| **MinIO/S3** | Model artifacts, raw data lake | All cloud ML teams | Not queryable |
| **MLflow** | Experiment tracking + model registry | Databricks, most ML teams | Not a feature store |

## Why Vector DB (Qdrant) Here — and Why NOT for Forecasting

### Fraud Detection ✅ Qdrant makes sense

Modern fraud teams index **known fraud transaction embeddings** in a vector DB:

1. Historical fraud cases → 30-dim vectors (V1–V28 + Amount + Time)
2. Live transaction arrives → nearest-neighbor search in Qdrant
3. High similarity to known fraud cluster → boost ensemble score

This catches **fraud rings** and **pattern replay** that tabular XGBoost alone misses.

```python
# Ensemble scoring in inference_service.py
fraud_score = 0.7 * xgboost_score + 0.3 * qdrant_similarity_score
```

### Demand Forecasting ❌ Qdrant would be wrong

Demand forecasting uses **tabular lag features + XGBoost**. There are no embeddings to search. Adding Qdrant would be architecture theater.

Use ClickHouse for fast aggregations over 913k sales rows instead.

## Real Datasets (Not Synthetic)

| Project | Dataset | Source | Rows |
|---------|---------|--------|------|
| Demand Forecasting | Store Item Demand | [Kaggle / skforecast](https://github.com/skforecast/skforecast-datasets) | 913,000 |
| Fraud Detection | ULB Credit Card Fraud | [Kaggle ULB MLG](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 284,807 |

Download + ingest: `make bootstrap`

## Streaming Architecture (Fraud)

```
                    ┌─────────────────┐
  ClickHouse ──────►│  Batch Training │──────► MLflow Registry
  (historical)      └─────────────────┘
                              │
  Kafka(transactions)         │
         │                    ▼
         ▼            ┌─────────────────┐
  ┌──────────────┐  │ Spark Structured│──► Redis (online features)
  │   Producer   │──►│    Streaming    │──► Kafka(transaction_features)
  └──────────────┘  └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Inference Svc   │◄── Qdrant (vector NN)
                    │ XGBoost+Vectors │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         Kafka(alerts)  ClickHouse     Kafka(predictions)
                        (audit log)
```

Run: `make stream`

## Cloud Mapping

| Local (Docker) | AWS | GCP |
|----------------|-----|-----|
| ClickHouse | ClickHouse Cloud / Redshift | BigQuery |
| PostgreSQL | RDS / Aurora | Cloud SQL |
| Redis | ElastiCache | Memorystore |
| Qdrant | Qdrant Cloud | Qdrant Cloud |
| Kafka | MSK | Confluent / Pub/Sub |
| Spark | EMR / Databricks | Dataproc |
| MinIO | S3 | GCS |

## Alternatives Considered

| Instead of | You might use | When |
|------------|---------------|------|
| ClickHouse | DuckDB, Snowflake | DuckDB for solo dev; Snowflake at scale |
| Qdrant | Pinecone, Weaviate, Milvus | Managed Pinecone in prod |
| Redis | Dragonfly, KeyDB | Higher throughput cache |
| Kafka | Redpanda, Pulsar | Redpanda = Kafka API, simpler ops |
| Spark | Flink | Flink if sub-10ms latency required |
| PostgreSQL | CockroachDB | Global multi-region OLTP |
