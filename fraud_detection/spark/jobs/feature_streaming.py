"""
Spark Structured Streaming: Kafka → velocity features → Redis + Kafka + Delta Lake.

Submit (prod):
  docker compose exec spark-master spark-submit \
    --master spark://spark-master:7077 \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,io.delta:delta-spark_2.12:3.1.0,org.apache.hadoop:hadoop-aws:3.3.4 \
    /opt/spark/jobs/feature_streaming.py
"""

import json
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

KAFKA_BOOTSTRAP = sys.argv[1] if len(sys.argv) > 1 else "kafka:29092"
INPUT_TOPIC = "transactions"
OUTPUT_TOPIC = "transaction_features"
REDIS_HOST = sys.argv[2] if len(sys.argv) > 2 else "redis"

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
        F.coalesce(F.col("timestamp").cast("timestamp"), F.current_timestamp())
    )
)

# Windowed velocity features (Spark — not in-memory Python)
windowed = (
    parsed.withWatermark("timestamp_parsed", "1 hour")
    .groupBy(F.col("card_id"), F.window(F.col("timestamp_parsed"), "1 hour"))
    .agg(
        F.count("*").alias("txn_count_1h"),
        F.sum("amount").alias("amount_sum_1h"),
        F.avg("amount").alias("amount_mean_1h"),
        F.stddev("amount").alias("amount_std_1h"),
        F.max("amount").alias("amount_max_1h"),
    )
)

features = windowed.select(
    F.col("card_id"),
    F.col("txn_count_1h"),
    F.col("amount_sum_1h"),
    F.col("amount_mean_1h"),
    F.coalesce(F.col("amount_std_1h"), F.lit(0.0)).alias("amount_std_1h"),
    F.col("amount_max_1h"),
    F.to_json(
        F.struct(
            F.col("card_id"),
            F.col("txn_count_1h"),
            F.col("amount_sum_1h"),
            F.col("amount_mean_1h"),
            F.col("amount_std_1h"),
            F.col("amount_max_1h"),
        )
    ).alias("value"),
)


def write_to_redis(batch_df, batch_id):
    """foreachBatch sink: push features to Redis online store."""
    import redis

    r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
    for row in batch_df.collect():
        payload = {
            "txn_count_1h": row["txn_count_1h"],
            "amount_sum_1h": float(row["amount_sum_1h"] or 0),
            "amount_mean_1h": float(row["amount_mean_1h"] or 0),
            "amount_std_1h": float(row["amount_std_1h"] or 0),
            "amount_max_1h": float(row["amount_max_1h"] or 0),
        }
        r.setex(f"feat:card:{row['card_id']}", 86400, json.dumps(payload))


# 1. Kafka sink for downstream inference service
kafka_query = (
    features.select(F.col("card_id").alias("key"), F.col("value"))
    .writeStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("topic", OUTPUT_TOPIC)
    .option("checkpointLocation", "/tmp/spark-checkpoints/kafka-features")
    .outputMode("update")
    .start()
)

# 2. Redis sink for online serving
redis_query = (
    features.writeStream.foreachBatch(write_to_redis)
    .option("checkpointLocation", "/tmp/spark-checkpoints/redis-features")
    .outputMode("update")
    .start()
)

# 3. Delta Lake sink to S3 data lake partitioned by date
delta_df = parsed.withColumn("date", F.to_date(F.col("timestamp_parsed")))
delta_query = (
    delta_df.writeStream.format("delta")
    .partitionBy("date")
    .option("checkpointLocation", "s3a://ml-data-lake/checkpoints/raw_transactions_delta")
    .outputMode("append")
    .start("s3a://ml-data-lake/delta/raw_transactions")
)

spark.streams.awaitAnyTermination()

