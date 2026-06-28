"""
Spark Structured Streaming: Kafka → velocity features → Redis + Kafka.

Submit (prod):
  docker compose exec spark-master spark-submit \\
    --master spark://spark-master:7077 \\
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \\
    /opt/spark/jobs/feature_streaming.py
"""

import json
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

KAFKA_BOOTSTRAP = sys.argv[1] if len(sys.argv) > 1 else "kafka:29092"
INPUT_TOPIC = "transactions"
OUTPUT_TOPIC = "transaction_features"
REDIS_HOST = sys.argv[2] if len(sys.argv) > 2 else "redis"

spark = (
    SparkSession.builder.appName("fraud-feature-streaming")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

schema = StructType(
    [
        StructField("transaction_id", StringType()),
        StructField("card_id", StringType()),
        StructField("amount", DoubleType()),
        StructField("time_seconds", DoubleType()),
        StructField("v1", DoubleType()),
        StructField("v2", DoubleType()),
        StructField("v3", DoubleType()),
        StructField("v4", DoubleType()),
        StructField("v5", DoubleType()),
    ]
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
)

# Windowed velocity features (Spark — not in-memory Python)
windowed = (
    parsed.withWatermark("time_seconds", "1 hour")
    .groupBy(F.col("card_id"), F.window(F.col("time_seconds").cast("timestamp"), "1 hour"))
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


# Kafka sink for downstream inference service
kafka_query = (
    features.select(F.col("card_id").alias("key"), F.col("value"))
    .writeStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("topic", OUTPUT_TOPIC)
    .option("checkpointLocation", "/tmp/spark-checkpoints/kafka-features")
    .outputMode("update")
    .start()
)

redis_query = (
    features.writeStream.foreachBatch(write_to_redis)
    .option("checkpointLocation", "/tmp/spark-checkpoints/redis-features")
    .outputMode("update")
    .start()
)

spark.streams.awaitAnyTermination()
