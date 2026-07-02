"""
Spark Structured Streaming: Kafka → canonical velocity features → Feast online store + Delta Lake.

This job is the real-time feature-freshness path. It computes the SAME 10-feature
schema the model trains on (see fraud_detection/pipelines/training.py:FEATURE_COLS and
feature_repo/features.py:transaction_features) and pushes it into the Feast online
store (Redis) via `store.push()`. The inference service and the FastAPI app read those
features back with `feature_store.get_online_features(...)`, so producing them here keeps
online features fresh per-transaction instead of only as fresh as the last batch
`feast materialize`. A parallel Delta Lake sink archives the raw transactions.

Infra requirements (the Spark driver runs foreachBatch, so these apply to spark-master):
  - `feast` and `pandas` installed on the Spark image.
  - The Feast repo mounted at FEATURE_REPO_PATH (default /opt/feature_repo), including
    its registry.db, so the driver can resolve the pushed feature definitions.
  - Redis reachable at REDIS_HOST:6379 from inside the Spark network.

Submit (prod):
  docker compose exec spark-master spark-submit \
    --master spark://spark-master:7077 \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,io.delta:delta-spark_2.12:3.1.0,org.apache.hadoop:hadoop-aws:3.3.4 \
    /opt/spark/jobs/feature_streaming.py
"""

import os
import sys
from collections import defaultdict, deque
from datetime import timedelta

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from feast import FeatureStore
from feast.data_source import PushMode
from feast.repo_config import RepoConfig

KAFKA_BOOTSTRAP = sys.argv[1] if len(sys.argv) > 1 else "kafka:29092"
INPUT_TOPIC = "transactions"
REDIS_HOST = sys.argv[2] if len(sys.argv) > 2 else os.getenv("REDIS_HOST", "redis")

# Feast repo (mounted into the Spark container) — used to push features to the online store.
FEATURE_REPO = os.getenv("FEATURE_REPO_PATH", "/opt/feature_repo")
PUSH_SOURCE_NAME = "transaction_features_stream"

# S3 Data Lake connection configuration (MinIO in Docker network)
S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT_URL", "http://minio:9000")
if "localhost" in S3_ENDPOINT or "127.0.0.1" in S3_ENDPOINT:
    S3_ENDPOINT = S3_ENDPOINT.replace("localhost", "minio").replace("127.0.0.1", "minio")

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

spark = (
    SparkSession.builder.appName("fraud-feature-streaming")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.hadoop.fs.s3a.endpoint", S3_ENDPOINT)
    .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY)
    .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_KEY)
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .getOrCreate()
)

# Feast client — built in code (not from feature_store.yaml) so the online store points
# at the in-network Redis host regardless of the yaml's localhost connection string.
# entity_key_serialization_version is pinned to match feature_store.yaml so keys written
# here are readable by the API / inference service.
_repo_config = RepoConfig(
    project="fraud_detection",
    provider="local",
    registry=os.path.join(FEATURE_REPO, "data", "registry.db"),
    online_store={"type": "redis", "connection_string": f"{REDIS_HOST}:6379"},
    entity_key_serialization_version=2,
)
store = FeatureStore(config=_repo_config)

# Parse all 28 PCA components + time metadata + timestamp
schema = StructType(
    [
        StructField("transaction_id", StringType()),
        StructField("card_id", StringType()),
        StructField("amount", DoubleType()),
        StructField("time_seconds", DoubleType()),
        StructField("timestamp", StringType()),
    ]
    + [StructField(f"v{i}", DoubleType()) for i in range(1, 29)]
)

raw = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", INPUT_TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

parsed = (
    raw.select(F.from_json(F.col("value").cast("string"), schema).alias("data"))
    .select("data.*")
    .withColumn(
        "timestamp_parsed",
        F.coalesce(F.col("timestamp").cast("timestamp"), F.current_timestamp()),
    )
)

# ── Canonical velocity features (must mirror generate_data.compute_velocity_features) ──
# foreachBatch runs on the driver in a single process, so we maintain a bounded per-card
# rolling history here and compute the exact 10-feature schema the model expects.
WINDOW_1H = timedelta(hours=1)
WINDOW_24H = timedelta(hours=24)
_history: dict[str, deque] = defaultdict(deque)


def compute_features(card_id: str, amount: float, ts) -> dict:
    """Rolling 1h/24h velocity features for one transaction — matches the offline schema."""
    history = _history[card_id]
    history.append((ts, amount))

    # Prune anything older than the 24h window
    cutoff = ts - WINDOW_24H
    while history and history[0][0] < cutoff:
        history.popleft()

    amounts_24h = [a for _, a in history]
    amounts_1h = [a for t, a in history if t >= ts - WINDOW_1H]

    txn_1h = len(amounts_1h)
    txn_24h = len(amounts_24h)
    amt_sum = sum(amounts_24h)
    amt_mean = amt_sum / txn_24h if txn_24h else 0.0
    # Sample std (ddof=1) to match pandas .std() used in the offline pipeline
    amt_std = (
        (sum((a - amt_mean) ** 2 for a in amounts_24h) / (txn_24h - 1)) ** 0.5
        if txn_24h > 1
        else 0.0
    )

    return {
        "amount": float(amount),
        "txn_count_1h": int(txn_1h),
        "txn_count_24h": int(txn_24h),
        "amount_sum_24h": float(amt_sum),
        "amount_mean_24h": float(amt_mean),
        "amount_std_24h": float(amt_std),
        "velocity_ratio": float(txn_1h / max(txn_24h, 1)),
        "amount_deviation": float(amount / max(amt_mean, 1.0)),
        "hour_of_day": int(ts.hour),
        "is_night": int(ts.hour < 6 or ts.hour > 22),
    }


def push_to_feast(batch_df, batch_id):
    """foreachBatch sink: compute canonical features and push to the Feast online store."""
    rows = batch_df.select("card_id", "amount", "timestamp_parsed").collect()
    if not rows:
        return

    records = []
    for row in sorted(rows, key=lambda r: r["timestamp_parsed"]):
        feats = compute_features(row["card_id"], float(row["amount"]), row["timestamp_parsed"])
        records.append(
            {"card_id": row["card_id"], "event_timestamp": row["timestamp_parsed"], **feats}
        )

    pdf = pd.DataFrame(records)
    # Online store keeps only the latest row per entity — collapse to one row per card.
    pdf = pdf.sort_values("event_timestamp").groupby("card_id", as_index=False).last()
    store.push(PUSH_SOURCE_NAME, pdf, to=PushMode.ONLINE)


# 1. Feast online-store sink (Redis) — keeps served features fresh in real time.
feast_query = (
    parsed.writeStream.foreachBatch(push_to_feast)
    .option("checkpointLocation", "/tmp/spark-checkpoints/feast-features")
    .outputMode("append")
    .start()
)

# 2. Delta Lake sink to S3 data lake partitioned by date — raw transaction archive.
delta_df = parsed.withColumn("date", F.to_date(F.col("timestamp_parsed")))
delta_query = (
    delta_df.writeStream.format("delta")
    .partitionBy("date")
    .option("checkpointLocation", "s3a://ml-data-lake/checkpoints/raw_transactions_delta")
    .outputMode("append")
    .start("s3a://ml-data-lake/delta/raw_transactions")
)

spark.streams.awaitAnyTermination()
