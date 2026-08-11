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


@pytest.mark.skipif(
    os.environ.get("QUANTSIGNAL_TEST_LAKE") != "1",
    reason="set QUANTSIGNAL_TEST_LAKE=1 to round-trip against the local MinIO",
)
def test_lake_round_trip_against_minio() -> None:
    import pandas as pd
    import pyarrow as pa

    from flows import lake_export

    settings = _settings(
        lake_enabled=True,
        lake_access_key="minioadmin",
        lake_secret_key="minioadmin",
        lake_bucket="quant-lake-test",
        lake_namespace="test_ns",
        lake_catalog_uri="sqlite:////tmp/quant_lake_test_catalog.db",
    )
    lake_export.ensure_bucket(settings)
    catalog = lake_export._catalog(settings)
    catalog.create_namespace_if_not_exists(settings.lake_namespace)
    identifier = (settings.lake_namespace, "probe")

    arrow = pa.Table.from_pandas(
        pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT"], "value": [1.0, 2.0]}),
        preserve_index=False,
    )
    table = catalog.create_table_if_not_exists(identifier, schema=arrow.schema)
    table.overwrite(arrow)

    back = catalog.load_table(identifier).scan().to_pandas()
    assert len(back) == 2
    assert sorted(back["symbol"]) == ["BTCUSDT", "ETHUSDT"]
    assert catalog.load_table(identifier).current_snapshot() is not None
