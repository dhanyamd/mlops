"""Pipeline latency telemetry — persist per-stage timing to Bronze.

The flows already emit ``elapsed_ms`` in structured logs; this module turns
that into queryable infrastructure data in ``BRONZE.PIPELINE_METRICS`` so we
can measure E2E latency, detect SLO breaches, and attribute spend per stage.
Telemetry writes are best-effort: a metrics failure never fails the run.
"""

from __future__ import annotations

import datetime as dt
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import pandas as pd

from config.logging import get_logger
from config.settings import Settings, get_settings
from db.snowflake import SnowflakeClient

log = get_logger("ingest.metrics")

METRICS_TABLE = "PIPELINE_METRICS"


@dataclass
class PipelineMetrics:
    """Collect per-stage timing for one flow run, flush to Bronze at the end.

    Usage::

        metrics = PipelineMetrics(flow="ingest-market-data")
        with metrics.stage("fetch"):
            raw = fetch_bars(...)
        with metrics.stage("validate", rows=len(raw)):
            good, bad = validate(raw)
        metrics.flush()
    """

    flow: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _stages: list[dict] = field(default_factory=list, init=False)

    @contextmanager
    def stage(self, name: str, rows: int | None = None) -> Iterator[None]:
        started_at = dt.datetime.now(dt.timezone.utc)
        start = time.perf_counter()
        try:
            yield
        finally:
            self._stages.append(
                {
                    "run_id": self.run_id,
                    "flow": self.flow,
                    "stage": name,
                    "started_at": started_at,
                    "elapsed_ms": round((time.perf_counter() - start) * 1000),
                    # n_rows (not "rows"): ROWS is an ANSI-reserved word in
                    # Snowflake and would need quoting on every reference.
                    "n_rows": rows,
                }
            )

    def flush(self, settings: Settings | None = None) -> int:
        """Write collected stages to Bronze. Never raises — telemetry is optional."""
        if not self._stages:
            return 0
        try:
            client = SnowflakeClient(settings or get_settings())
            rows = [
                {**stage, "loaded_at": dt.datetime.now(dt.timezone.utc)} for stage in self._stages
            ]
            return client.insert_df(
                pd.DataFrame(rows),
                METRICS_TABLE,
                schema=client._settings.snowflake_schema,
            )
        except Exception as exc:  # noqa: BLE001 - telemetry must never break a run
            log.warning("metrics_flush_failed", error=str(exc))
            return 0
