"""Iceberg lake tier: version the Snowflake mart to an open-table-format lake.

Apache Iceberg is storage-agnostic: the catalog (SQLite locally, JDBC/Postgres
in a shared deployment) tracks schema + snapshots while the Parquet data files
live on any S3-compatible object store (MinIO in dev, real S3 on the AWS
deploy). The same tables therefore move to S3 unchanged by swapping
``LAKE_ENDPOINT`` — the AWS IaC storage module already provisions the bucket.

Each ``--source`` export runs an idempotent *overwrite*: one new snapshot per
run, and old snapshots stay readable (time travel). A mart rebuild is a safe
re-run. ``--verify`` round-trips the table through the catalog to prove the
files, schema and snapshot history are intact.

Run with ``make lake-export`` / ``make lake-query`` (needs the ``lake`` extra:
``uv sync --extra lake``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.logging import configure_logging, get_logger
from config.settings import Settings, get_settings

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
    from pyiceberg.catalog.sql import SqlCatalog

log = get_logger("flows.lake_export")


def catalog_properties(settings: Settings) -> dict[str, str]:
    """SqlCatalog properties: SQLAlchemy backend + S3-compatible FileIO config.

    ``py-io-impl`` is PyArrowFileIO — pyarrow's S3FileSystem honours the
    ``s3.endpoint`` override, which is how Iceberg talks to any S3-compatible
    store (MinIO locally, AWS S3 later). No hardcoded values: everything comes
    from Settings.
    """
    return {
        "type": "sql",
        "uri": settings.lake_catalog_uri,
        "warehouse": f"s3://{settings.lake_bucket}",
        "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO",
        "s3.endpoint": settings.lake_endpoint,
        "s3.access-key-id": settings.lake_access_key or "",
        "s3.secret-access-key": settings.lake_secret_key or "",
        "s3.region": settings.lake_region,
    }


def ensure_bucket(settings: Settings) -> str:
    """Idempotently create the lake bucket on the S3-compatible store.

    The classic MinIO gotcha is a missing bucket: Iceberg fails its first
    HeadObject with an opaque 400. Creating it up front keeps export
    declarative — a ``terraform apply`` on the real S3 deploy makes this a
    no-op.
    """
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - guarded by the lake extra
        raise RuntimeError("boto3 required for lake bucket bootstrap (lake extra)") from exc
    client = boto3.client(
        "s3",
        endpoint_url=settings.lake_endpoint,
        aws_access_key_id=settings.lake_access_key,
        aws_secret_access_key=settings.lake_secret_key,
        region_name=settings.lake_region,
    )
    buckets = {b["Name"] for b in client.list_buckets()["Buckets"]}
    if settings.lake_bucket not in buckets:
        client.create_bucket(Bucket=settings.lake_bucket)
        log.info("lake_bucket_created", bucket=settings.lake_bucket)
    return settings.lake_bucket


def _catalog(settings: Settings) -> "SqlCatalog":
    """Lazy pyiceberg SqlCatalog (the ``lake`` extra is optional)."""
    try:
        from pyiceberg.catalog.sql import SqlCatalog
    except ImportError as exc:  # pragma: no cover - guarded by the lake extra
        raise RuntimeError("pyiceberg not installed; run `uv sync --extra lake`") from exc
    uri = settings.lake_catalog_uri
    if uri.startswith("sqlite:///"):
        path = uri.removeprefix("sqlite:///")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqlCatalog("quant_lake", **catalog_properties(settings))


def _to_arrow(df: "pd.DataFrame") -> "pa.Table":
    """pandas → Arrow with Iceberg-safe timestamp precision.

    pandas materializes datetimes as ``timestamp[ns]``, which pyiceberg does
    not support; Iceberg stores timestamps at microsecond precision, so we
    cast any ``timestamp[ns]`` column down to ``timestamp[us]`` (bar/minutely
    data loses nothing at this precision).
    """
    import pyarrow as pa

    table = pa.Table.from_pandas(df, preserve_index=False)
    target = pa.schema(
        [
            pa.field(
                name,
                pa.timestamp("us") if pa.types.is_timestamp(t) and t.unit == "ns" else t,
            )
            for name, t in zip(table.schema.names, table.schema.types)
        ]
    )
    return table.cast(target) if target != table.schema else table


def export_features(
    settings: Settings,
    *,
    source_table: str = "features",
    source_schema: str | None = None,
) -> dict[str, Any]:
    """Version a Snowflake mart table into the Iceberg lake.

    Reads ``{GOLD_SCHEMA}.{source_table}`` via the Snowflake client, then
    overwrites the Iceberg table ``{namespace}.{table}`` with one new snapshot
    (old snapshots retained = time travel). Returns row count + snapshot id.
    """
    from db.snowflake import SnowflakeClient

    ensure_bucket(settings)
    client = SnowflakeClient(settings)
    gold = source_schema or settings.snowflake_gold_schema
    df = client.query_df(f'SELECT * FROM "{settings.snowflake_database}"."{gold}"."{source_table}"')
    if df.empty:
        log.info("lake_export_no_rows", table=f"{gold}.{source_table}")
        return {"rows": 0, "snapshot_id": None}

    arrow = _to_arrow(df)
    catalog = _catalog(settings)
    catalog.create_namespace_if_not_exists(settings.lake_namespace)
    identifier = (settings.lake_namespace, settings.lake_table_features)
    table = catalog.create_table_if_not_exists(identifier, schema=arrow.schema)
    table.overwrite(arrow)
    snapshot = table.current_snapshot()

    log.info(
        "lake_export_done",
        table=f"{settings.lake_namespace}.{settings.lake_table_features}",
        rows=len(df),
        snapshot_id=snapshot.snapshot_id if snapshot else None,
    )
    return {
        "table": f"{settings.lake_namespace}.{settings.lake_table_features}",
        "rows": len(df),
        "snapshot_id": snapshot.snapshot_id if snapshot else None,
    }


def verify(settings: Settings) -> dict[str, Any]:
    """Round-trip proof: read the Iceberg table back through the catalog.

    Returns row count from a fresh scan (from object storage, not Snowflake),
    the current snapshot, and the retained snapshot history — the time-travel
    evidence that overwrite-per-run is versioning, not replacement.
    """
    catalog = _catalog(settings)
    identifier = (settings.lake_namespace, settings.lake_table_features)
    table = catalog.load_table(identifier)
    scanned = table.scan().to_pandas()
    current = table.current_snapshot()
    history = [
        {"snapshot_id": snap.snapshot_id, "timestamp_ms": snap.timestamp_ms}
        for snap in (table.snapshots() or [])
    ]

    log.info(
        "lake_verify_done",
        table=identifier,
        rows=len(scanned),
        snapshots=len(history),
        current_snapshot_id=current.snapshot_id if current else None,
    )
    return {
        "table": identifier,
        "rows": len(scanned),
        "current_snapshot_id": current.snapshot_id if current else None,
        "snapshots": history,
    }


def run(mode: str, source_table: str) -> None:
    configure_logging()
    settings = get_settings()
    if not settings.lake_enabled:
        raise RuntimeError(
            "LAKE_ENABLED must be true (with LAKE_ACCESS_KEY/LAKE_SECRET_KEY) "
            "to use the Iceberg lake tier"
        )
    start = time.perf_counter()
    if mode == "export":
        result = export_features(settings, source_table=source_table)
    else:
        result = verify(settings)
    log.info("lake_run", mode=mode, elapsed_ms=round((time.perf_counter() - start) * 1000))
    print(f"lake_{mode}={result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Version the Snowflake mart to an Apache Iceberg lake (MinIO/S3)."
    )
    parser.add_argument(
        "--mode",
        choices=["export", "verify"],
        default="export",
        help="export = overwrite mart -> new Iceberg snapshot; verify = round-trip read",
    )
    parser.add_argument(
        "--source-table",
        default="FEATURES",
        help="Snowflake GOLD table to export (default FEATURES)",
    )
    args = parser.parse_args()
    run(mode=args.mode, source_table=args.source_table)
