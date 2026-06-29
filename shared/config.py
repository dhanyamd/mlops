"""Production service configuration — all connections via env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_LAKE = Path(os.getenv("DATA_LAKE", PROJECT_ROOT / "data_lake"))


@dataclass(frozen=True)
class PostgresConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    user: str = os.getenv("POSTGRES_USER", "mlops")
    password: str = os.getenv("POSTGRES_PASSWORD", "mlops")
    database: str = os.getenv("POSTGRES_DB", "mlops")

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class ClickHouseConfig:
    host: str = os.getenv("CLICKHOUSE_HOST", "localhost")
    port: int = int(os.getenv("CLICKHOUSE_PORT", "8123"))
    user: str = os.getenv("CLICKHOUSE_USER", "default")
    password: str = os.getenv("CLICKHOUSE_PASSWORD", "")
    database: str = os.getenv("CLICKHOUSE_DATABASE", "default")


@dataclass(frozen=True)
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    password: str | None = os.getenv("REDIS_PASSWORD")

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


@dataclass(frozen=True)
class QdrantConfig:
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    collection: str = os.getenv("QDRANT_FRAUD_COLLECTION", "fraud_patterns")
    vector_size: int = int(os.getenv("QDRANT_VECTOR_SIZE", "30"))  # V1-V28 + Amount + Time


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    transactions_topic: str = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
    features_topic: str = os.getenv("KAFKA_FEATURES_TOPIC", "transaction_features")
    predictions_topic: str = os.getenv("KAFKA_PREDICTIONS_TOPIC", "fraud_predictions")
    alerts_topic: str = os.getenv("KAFKA_ALERTS_TOPIC", "fraud_alerts")


@dataclass(frozen=True)
class SparkConfig:
    master: str = os.getenv("SPARK_MASTER", "spark://localhost:7077")
    app_name: str = os.getenv("SPARK_APP_NAME", "fraud-streaming")


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    s3_endpoint: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")


@dataclass(frozen=True)
class S3Config:
    endpoint_url: str = os.getenv("AWS_S3_ENDPOINT_URL", os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"))
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    region_name: str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    bucket_name: str = os.getenv("DATA_LAKE_BUCKET", "ml-data-lake")


POSTGRES = PostgresConfig()
CLICKHOUSE = ClickHouseConfig()
REDIS = RedisConfig()
QDRANT = QdrantConfig()
KAFKA = KafkaConfig()
SPARK = SparkConfig()
MLFLOW_CFG = MLflowConfig()
S3_CFG = S3Config()

# Public dataset URLs (real data — not synthetic)
DATASETS = {
    "store_sales": os.getenv(
        "STORE_SALES_URL",
        "https://raw.githubusercontent.com/skforecast/skforecast-datasets/main/data/store_sales.csv",
    ),
    "credit_card_fraud": os.getenv(
        "CREDIT_CARD_FRAUD_URL",
        "https://raw.githubusercontent.com/SimoCs/Fraudulent-Transactions-Java/main/creditcard.csv",
    ),
}

SAMPLE_ROWS = int(os.getenv("MLOPS_SAMPLE_ROWS", "0"))  # 0 = full dataset
