"""Pipeline latency telemetry — the timer records stage timing; flush is best-effort."""

from __future__ import annotations

import time

import pandas as pd

from ingest.metrics import PipelineMetrics


def test_stage_records_timing_and_rows() -> None:
    metrics = PipelineMetrics(flow="test-flow")
    with metrics.stage("fetch"):
        time.sleep(0.05)
    with metrics.stage("validate", rows=10):
        pass
    assert len(metrics._stages) == 2
    fetch, validate = metrics._stages
    assert fetch["flow"] == "test-flow"
    assert fetch["stage"] == "fetch"
    assert fetch["n_rows"] is None
    assert fetch["elapsed_ms"] >= 40
    assert validate["n_rows"] == 10
    assert validate["run_id"] == fetch["run_id"]  # same run


def test_flush_is_noop_without_stages() -> None:
    metrics = PipelineMetrics(flow="test-flow")
    assert metrics.flush() == 0  # no stages -> no DB write


def test_flush_never_raises_on_failure(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import ingest.metrics as metrics_mod

    def boom(self, df: pd.DataFrame, table_name: str, **kwargs) -> int:  # type: ignore[no-untyped-def]
        raise RuntimeError("snowflake is down")

    monkeypatch.setattr(metrics_mod.SnowflakeClient, "insert_df", boom)
    metrics = PipelineMetrics(flow="test-flow")
    with metrics.stage("fetch"):
        pass
    assert metrics.flush() == 0  # telemetry failure swallowed, not raised


def test_flush_builds_schema_with_n_rows_and_loaded_at(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import ingest.metrics as metrics_mod

    captured: dict = {}

    def capture(self, df: pd.DataFrame, table_name: str, **kwargs) -> int:  # type: ignore[no-untyped-def]
        captured["df"] = df
        return len(df)

    monkeypatch.setattr(metrics_mod.SnowflakeClient, "insert_df", capture)
    metrics = PipelineMetrics(flow="test-flow")
    with metrics.stage("fetch", rows=5):
        pass
    assert metrics.flush() == 1
    df = captured["df"]
    # Contract for BRONZE.PIPELINE_METRICS: n_rows (ROWS is reserved), + loaded_at.
    assert set(df.columns) == {
        "run_id",
        "flow",
        "stage",
        "started_at",
        "elapsed_ms",
        "n_rows",
        "loaded_at",
    }
    assert df.iloc[0]["n_rows"] == 5
    assert df.iloc[0]["loaded_at"] is not None
