"""Iceberg lake tier: version the Snowflake mart to an open-table-format lake.

Apache Iceberg is storage-agnostic: the catalog (SQLite locally, JDBC/Postgres
in a shared deployment) tracks schema + snapshots while the Parquet data files
live on any S3-compatible object store (MinIO in dev, real S3 on the AWS
deploy). The same tables therefore move to S3 unchanged by swapping
``LAKE_ENDPOINT`` — the AWS IaC storage module already provisions the bucket.

Each ``--source`` export runs an idempotent *overwrite*: one new snapshot per
run, and old snapshots stay readable (time travel) — up to the configured
retention, after which they are expired so metadata can't grow without bound
(the Iceberg scaling trap). New tables are partitioned on a coarse low-
cardinality column (``LAKE_PARTITION_BY``, default SYMBOL = 32 values) per
the "partition coarsely, stay well under a few thousand partitions" rule.
``--verify`` round-trips the table through the catalog to prove the files,
schema and snapshot history are intact.

Run with ``make lake-export`` / ``make lake-query`` (needs the ``lake`` extra:
``uv sync --extra lake``).
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.logging import configure_logging, get_logger
from config.settings import Settings, get_settings

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.table import StaticTable

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


def _partition_column(settings: Settings, schema_names: list[str]) -> str | None:
    """The configured partition column if it exists in the exported schema.

    Returns None when partitioning is not configured or the column is absent
    (an export would otherwise create an unpartitioned table despite a bogus
    ``LAKE_PARTITION_BY``).
    """
    column = settings.lake_partition_by
    if not column:
        return None
    if column not in schema_names:
        log.warning("lake_partition_column_missing", column=column, columns=schema_names)
        return None
    return column


def _create_or_load(
    catalog: "SqlCatalog",
    identifier: tuple[str, str],
    arrow_schema: "pa.Schema",
    settings: Settings,
) -> "StaticTable":
    """Create a (partitioned) table if absent, otherwise load it.

    Partitioning is applied by column *name* via ``create_table_transaction``
    + ``update_spec().add_identity`` — the pyiceberg-recommended path. An
    id-based ``PartitionSpec`` cannot be paired with a pyarrow schema: the
    catalog converts it to an Iceberg schema whose field ids are -1
    placeholders, so a spec's ``source_id`` never resolves (``ValueError:
    Could not find in old schema``, apache/iceberg-python#1100). Matching by
    name sidesteps field-id assignment entirely, and Iceberg only applies a
    spec at create time.
    """
    if not catalog.table_exists(identifier):
        column = _partition_column(settings, arrow_schema.names)
        with catalog.create_table_transaction(identifier, schema=arrow_schema) as txn:
            if column:
                with txn.update_spec() as update_spec:
                    update_spec.add_identity(column)
    return catalog.load_table(identifier)


def _partition_status(settings: Settings, table: "StaticTable") -> str:
    """Human summary of the table's actual partition spec.

    A table created before partitioning was enabled stays unpartitioned (Iceberg
    applies a spec at create time); report the truth rather than claiming the
    configured column is in effect.
    """
    fields = table.spec().fields
    if not fields:
        return "unpartitioned"
    return ", ".join(f"{f.name}: {f.transform}" for f in fields)


def _expire_snapshots(
    catalog: "SqlCatalog",
    identifier: tuple[str, str],
    settings: Settings,
) -> int:
    """Expire snapshots older than the retention window (metadata hygiene).

    Iceberg keeps every snapshot by default; without expiry the metadata log
    grows without bound and scan planning slows — the classic lake-scale trap.
    The cutoff is wall-clock based, so time travel is bounded to the retention
    window (default 7 days) regardless of run frequency. Returns the number of
    snapshots removed.
    """
    retention_hours = settings.lake_snapshot_retention_hours
    if retention_hours <= 0:
        log.info("lake_retention_disabled", retention_hours=retention_hours)
        return 0
    from pyiceberg.table import Transaction
    from pyiceberg.table.update.snapshot import ExpireSnapshots

    table = catalog.load_table(identifier)
    before = len(table.snapshots() or [])
    if before == 0:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
    ExpireSnapshots(transaction=Transaction(table, autocommit=True)).older_than(cutoff).commit()
    after = len(catalog.load_table(identifier).snapshots() or [])
    expired = before - after
    if expired:
        log.info(
            "lake_snapshots_expired",
            expired=expired,
            retained=after,
            retention_hours=retention_hours,
            cutoff=cutoff.isoformat(),
        )
    return expired


def export_features(
    settings: Settings,
    *,
    source_table: str = "features",
    source_schema: str | None = None,
) -> dict[str, Any]:
    """Version a Snowflake mart table into the Iceberg lake.

    Reads ``{GOLD_SCHEMA}.{source_table}`` via the Snowflake client, then
    overwrites the Iceberg table ``{namespace}.{table}`` with one new snapshot
    (old snapshots retained = time travel, bounded by the configured retention,
    after which ``_expire_snapshots`` removes them). Returns row count,
    snapshot id and expiry stats.
    """
    from db.snowflake import SnowflakeClient

    ensure_bucket(settings)
    client = SnowflakeClient(settings)
    gold = source_schema or settings.snowflake_gold_schema
    df = client.query_df(f'SELECT * FROM "{settings.snowflake_database}"."{gold}"."{source_table}"')
    if df.empty:
        log.info("lake_export_no_rows", table=f"{gold}.{source_table}")
        return {"rows": 0, "snapshot_id": None, "partitioned_by": None, "expired_snapshots": 0}

    arrow = _to_arrow(df)
    catalog = _catalog(settings)
    catalog.create_namespace_if_not_exists(settings.lake_namespace)
    identifier = (settings.lake_namespace, settings.lake_table_features)
    table = _create_or_load(catalog, identifier, arrow.schema, settings)
    partition_status = _partition_status(settings, table)
    if partition_status == "unpartitioned" and settings.lake_partition_by:
        log.warning(
            "lake_table_not_partitioned",
            table=identifier,
            column=settings.lake_partition_by,
            hint="created before partitioning was enabled; drop or migrate the "
            "table to apply LAKE_PARTITION_BY (Iceberg specs are fixed at create time)",
        )
    table.overwrite(arrow)
    snapshot = table.current_snapshot()
    expired = _expire_snapshots(catalog, identifier, settings)

    log.info(
        "lake_export_done",
        table=f"{settings.lake_namespace}.{settings.lake_table_features}",
        rows=len(df),
        snapshot_id=snapshot.snapshot_id if snapshot else None,
        partition=partition_status,
        expired_snapshots=expired,
    )
    return {
        "table": f"{settings.lake_namespace}.{settings.lake_table_features}",
        "rows": len(df),
        "snapshot_id": snapshot.snapshot_id if snapshot else None,
        "partitioned_by": settings.lake_partition_by or None,
        "partition_status": partition_status,
        "expired_snapshots": expired,
    }


def verify(settings: Settings) -> dict[str, Any]:
    """Round-trip proof: read the Iceberg table back through the catalog.

    Returns row count from a fresh scan (from object storage, not Snowflake),
    the current snapshot, the retained snapshot history — the time-travel
    evidence that overwrite-per-run is versioning, not replacement — and the
    actual partition spec + retention window in effect.
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
    partition_status = _partition_status(settings, table)

    log.info(
        "lake_verify_done",
        table=identifier,
        rows=len(scanned),
        snapshots=len(history),
        partition=partition_status,
        retention_hours=settings.lake_snapshot_retention_hours,
        current_snapshot_id=current.snapshot_id if current else None,
    )
    return {
        "table": identifier,
        "rows": len(scanned),
        "current_snapshot_id": current.snapshot_id if current else None,
        "partition_status": partition_status,
        "retention_hours": settings.lake_snapshot_retention_hours,
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
