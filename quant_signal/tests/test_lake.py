"""Lake tier tests: catalog config shape + settings validation (hermetic).

The Iceberg writes themselves go against object storage, so the unit suite
covers the pure config and settings guards. An opt-in integration test
(QUANTSIGNAL_TEST_LAKE=1) round-trips a tiny table against the running MinIO
to prove the catalog + S3-compatible FileIO path end-to-end.
"""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from config.settings import PROJECT_ROOT, Settings
from flows.lake_export import catalog_properties


def _settings(**overrides) -> Settings:
    base = {
        "snowflake_account": "GULXCKK-PI01025",
        "snowflake_user": "devuser",
        "snowflake_password": "pw",
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)


def test_lake_disabled_by_default_without_keys() -> None:
    # A plain config (lake off) must not demand object-store credentials.
    s = _settings()
    assert s.lake_enabled is False


def test_lake_enabled_requires_credentials() -> None:
    with pytest.raises(ValidationError):
        _settings(lake_enabled=True)


def test_catalog_properties_shape() -> None:
    s = _settings(
        lake_enabled=True,
        lake_access_key="minioadmin",
        lake_secret_key="minioadmin",
    )
    props = catalog_properties(s)
    assert props["type"] == "sql"
    assert props["uri"] == s.lake_catalog_uri
    assert props["warehouse"] == "s3://quant-lake"
    assert props["py-io-impl"] == "pyiceberg.io.pyarrow.PyArrowFileIO"
    assert props["s3.endpoint"] == "http://localhost:9000"
    assert props["s3.access-key-id"] == "minioadmin"
    assert props["s3.secret-access-key"] == "minioadmin"
    assert props["s3.region"] == "us-east-1"


def test_lake_catalog_uri_default_is_absolute_under_project() -> None:
    s = _settings()
    assert s.lake_catalog_uri == f"sqlite:///{PROJECT_ROOT}/lake/catalog.db"


def test_to_arrow_casts_ns_timestamps_to_us() -> None:
    # Regression: pandas datetime64[ns] is unsupported by pyiceberg; the
    # export must downcast to timestamp[us] (Iceberg microsecond precision).
    import pandas as pd

    from flows import lake_export

    df = pd.DataFrame({"symbol": ["BTCUSDT"], "ts": [pd.Timestamp("2026-01-01 12:00:00")]})
    arrow = lake_export._to_arrow(df)
    assert arrow.schema.field("ts").type.unit == "us"
    assert arrow.schema.field("symbol").type == "string"


def test_partition_column_resolves_from_schema() -> None:
    # Partitioning is applied by column name (the pyiceberg-recommended path);
    # the configured column is used only when it exists in the exported schema.
    from flows import lake_export

    assert (
        lake_export._partition_column(_settings(lake_partition_by="SYMBOL"), ["SYMBOL", "CLOSE"])
        == "SYMBOL"
    )


def test_partition_column_absent_or_unconfigured_is_none() -> None:
    from flows import lake_export

    assert (
        lake_export._partition_column(_settings(lake_partition_by="TIMESTAMP"), ["SYMBOL"]) is None
    )
    assert lake_export._partition_column(_settings(lake_partition_by=""), ["SYMBOL"]) is None


@pytest.mark.skipif(
    os.environ.get("QUANTSIGNAL_TEST_LAKE") != "1",
    reason="set QUANTSIGNAL_TEST_LAKE=1 to round-trip against the local MinIO",
)
def test_lake_round_trip_against_minio() -> None:
    import time
    import uuid

    import pandas as pd
    import pyarrow as pa

    from flows import lake_export

    settings = _settings(
        lake_enabled=True,
        lake_access_key="minioadmin",
        lake_secret_key="minioadmin",
        lake_bucket="quant-lake-test",
        lake_namespace=f"test_ns_{uuid.uuid4().hex[:8]}",
        lake_catalog_uri="sqlite:////tmp/quant_lake_test_catalog.db",
        lake_partition_by="symbol",
        lake_snapshot_retention_hours=1e-6,
    )
    lake_export.ensure_bucket(settings)
    catalog = lake_export._catalog(settings)
    catalog.create_namespace_if_not_exists(settings.lake_namespace)
    identifier = (settings.lake_namespace, "probe")

    arrow = pa.Table.from_pandas(
        pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT"], "value": [1.0, 2.0]}),
        preserve_index=False,
    )
    table = lake_export._create_or_load(catalog, identifier, arrow.schema, settings)
    # coarse identity partition applied at create time (spec is immutable)
    assert table.spec().fields and table.spec().fields[0].name == "symbol"
    table.overwrite(arrow)

    back = catalog.load_table(identifier).scan().to_pandas()
    assert len(back) == 2
    assert sorted(back["symbol"]) == ["BTCUSDT", "ETHUSDT"]
    assert catalog.load_table(identifier).current_snapshot() is not None

    # a second overwrite then expires snapshots older than the retention window
    # (sub-millisecond), proving overwrite-per-run stays bounded time travel:
    # old snapshots gone, exactly the current one retained
    time.sleep(0.1)
    table.overwrite(arrow)
    assert lake_export._expire_snapshots(catalog, identifier, settings) >= 1
    snapshots = catalog.load_table(identifier).snapshots() or []
    assert len(snapshots) == 1
    assert snapshots[0].snapshot_id == catalog.load_table(identifier).current_snapshot().snapshot_id
