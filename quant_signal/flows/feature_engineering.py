"""Spark feature-engineering batch over BRONZE bars -> GOLD.FEATURES.

Computes rolling/as-of features over the raw BRONZE bars and lands them in the
GOLD analytics mart. Feature math lives in ``flows.features.compute_features``
(the single source of truth, hermetic-unit-tested); this job only orchestrates
it at scale.

Two run modes with the *identical* output schema:

  --source snowflake  Real PySpark batch: SparkSession reads BRONZE through the
                      spark-snowflake connector, applies ``compute_features``
                      per symbol with a grouped-map pandas UDF, and writes
                      GOLD.FEATURES. Needs Java + the connector JAR:

      uv run python flows/feature_engineering.py --source snowflake

  --source pandas     No-Java fallback: reads BRONZE with the Snowflake
                      connector, computes the same features in pandas, and
                      MERGE-upserts GOLD.FEATURES. This is what ``make
                      features`` runs locally:

      uv run python flows/feature_engineering.py --source pandas

Output contract: the exact ``OUTPUT_COLUMNS`` tuple below, so both paths and
every consumer see one stable schema regardless of how the batch ran.
"""

from __future__ import annotations

import argparse
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd

from config.logging import configure_logging, get_logger
from config.settings import get_settings
from ingest.store import BAR_TABLE, CRYPTO_TABLE, write_features

if TYPE_CHECKING:
    from db.snowflake import SnowflakeClient

log = get_logger("flows.feature_engineering")

# Canonical GOLD.FEATURES schema — both run paths emit exactly these columns.
OUTPUT_COLUMNS = [
    "symbol",
    "ts",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "log_return",
    "ret_1",
    "vol_20",
    "mom_20",
    "zscore_20",
    "volume_zscore_20",
    "loaded_at",
]

# spark-snowflake connector JAR (Maven artifact pulled via spark.jars.packages).
# 3.2.1-spark_4.1 is the newest release with Spark 4.x support (Scala 2.13,
# JDK 17+); verified to run on the Spark 4.2 wheel this project pins.
SPARK_SNOWFLAKE_CONNECTOR = "net.snowflake:spark-snowflake_2.13:3.2.1-spark_4.1"


def _read_bronze(client: "SnowflakeClient", schema: str) -> pd.DataFrame:
    """Union the BRONZE bar tables (equity + crypto) into one feature frame.

    Both tables share the bar contract (symbol, ts, timeframe, OHLCV, ...), so
    the union feeds per-symbol rolling features over every instrument.
    """
    return client.query_df(
        f'SELECT * FROM "{schema}".{BAR_TABLE} UNION ALL SELECT * FROM "{schema}".{CRYPTO_TABLE}'
    ).rename(columns=str.lower)


def _finalize(features: pd.DataFrame) -> pd.DataFrame:
    """Stamp the load timestamp and narrow to the canonical output contract."""
    features["loaded_at"] = datetime.now(UTC).replace(tzinfo=None)
    return features[OUTPUT_COLUMNS]


def _pandas_run(schema: str, gold_schema: str, window: int) -> int:
    """No-Java path: BRONZE via Snowflake connector, features in pandas."""
    from db.snowflake import SnowflakeClient
    from flows.features import compute_features

    client = SnowflakeClient()
    bars = _read_bronze(client, schema)
    if bars.empty:
        log.info("feature_engineering_no_rows", table=f"{gold_schema}.FEATURES")
        return 0
    features = _finalize(compute_features(bars, window=window))
    written = write_features(features)
    log.info(
        "feature_engineering_pandas_done",
        rows=len(features),
        table=f"{gold_schema}.FEATURES",
    )
    return written


def _spark_run(schema: str, gold_schema: str, window: int) -> int:
    """Spark path: BRONZE via spark-snowflake, features via a grouped pandas UDF."""
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.functions import current_timestamp
    from pyspark.sql.types import (
        DoubleType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    from flows.features import compute_features

    settings = get_settings()
    key_file = settings.snowflake_private_key_file
    if key_file is not None:
        with open(key_file) as f:
            pem = f.read()
        # The connector expects the unencrypted PKCS8 DER key base64-encoded
        # without the PEM armor — same key the Python connector loads from file.
        pem_private_key = re.sub(r"-----BEGIN [^-]+-----|-----END [^-]+-----|\s", "", pem)
        snowflake_options = {
            "sfUrl": f"https://{settings.snowflake_account}.snowflakecomputing.com",
            "sfUser": settings.snowflake_user,
            "pem_private_key": pem_private_key,
            "sfDatabase": settings.snowflake_database,
            "sfWarehouse": settings.snowflake_warehouse,
            "sfRole": settings.snowflake_role,
        }
    else:
        snowflake_options = {
            "sfUrl": f"https://{settings.snowflake_account}.snowflakecomputing.com",
            "sfUser": settings.snowflake_user,
            "sfPassword": settings.snowflake_password or "",
            "sfDatabase": settings.snowflake_database,
            "sfWarehouse": settings.snowflake_warehouse,
            "sfRole": settings.snowflake_role,
        }

    spark = (
        SparkSession.builder.appName("quant_signal_feature_engineering")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.jars.packages", SPARK_SNOWFLAKE_CONNECTOR)
        .getOrCreate()
    )

    try:
        bars = (
            spark.read.format("snowflake")
            .options(**snowflake_options)
            .option("sfSchema", schema)
            .option("dbtable", BAR_TABLE)
            .load()
        )
        crypto = (
            spark.read.format("snowflake")
            .options(**snowflake_options)
            .option("sfSchema", schema)
            .option("dbtable", CRYPTO_TABLE)
            .load()
        )
        bars = bars.unionByName(crypto)
        bars = bars.select([F.col(c).alias(c.lower()) for c in bars.columns]).drop(
            "provider", "loaded_at"
        )

        # Grouped-map pandas UDF output: OUTPUT_COLUMNS minus loaded_at (stamped
        # in Spark below, not in the UDF, so both paths produce identical rows).
        field_types = {
            "symbol": StringType(),
            "ts": TimestampType(),
            "timeframe": StringType(),
        }
        out_schema = StructType(
            [
                StructField(c, field_types.get(c, DoubleType()), True)
                for c in OUTPUT_COLUMNS
                if c != "loaded_at"
            ]
        )

        def _features_udf(group: pd.DataFrame) -> pd.DataFrame:
            return compute_features(group, window=window)[[c.name for c in out_schema.fields]]

        featured = (
            bars.groupBy("symbol")
            .applyInPandas(_features_udf, schema=out_schema)
            .withColumn("loaded_at", current_timestamp())
        )

        featured.write.format("snowflake").options(**snowflake_options).option(
            "sfSchema", gold_schema
        ).option("dbtable", "FEATURES").mode("overwrite").save()
        log.info("feature_engineering_spark_done", table=f"{gold_schema}.FEATURES")
        return 1
    finally:
        spark.stop()


def run(source: str, window: int) -> int:
    configure_logging()
    settings = get_settings()
    schema = settings.snowflake_schema
    gold = settings.snowflake_gold_schema
    if source == "pandas":
        return _pandas_run(schema, gold, window)
    if source == "snowflake":
        return _spark_run(schema, gold, window)
    raise ValueError(f"unknown source: {source!r} (choose from pandas, snowflake)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute rolling/as-of features over BRONZE bars into GOLD.FEATURES."
    )
    parser.add_argument(
        "--source",
        choices=["pandas", "snowflake"],
        default="pandas",
        help="pandas = no-Java fallback; snowflake = PySpark batch (needs Java + connector)",
    )
    parser.add_argument("--window", type=int, default=20, help="rolling window length")
    args = parser.parse_args()
    run(source=args.source, window=args.window)
