"""Data-quality CLI: score every symbol's window history and persist the report.

Computes the five-pillar quality dimensions (freshness, volume, uniqueness,
ordering, gaps, validity, accuracy, consistency — Soda/Elementary model) over
the materialized window history in the online store, plus the stage-by-stage
lineage manifest, then persists the cross-symbol summary to
``data:quality:summary`` and each symbol's report to
``data:quality:<SYMBOL>``. The batch warehouse sink (``make ch-materialize``)
copies these keys into ClickHouse for Grafana.

Run with ``make stream-quality`` (or ``python -m scripts.run_quality``).
"""

from __future__ import annotations

from config.logging import configure_logging, get_logger
from config.settings import get_settings
from stream.data_quality import quality_summary
from stream.kv import RedisKV

log = get_logger(__name__)


def main() -> None:
    configure_logging()
    settings = get_settings()
    kv = RedisKV(settings.stream_redis_url)

    summary = quality_summary(kv, settings)
    kv.set_json(f"{settings.quality_prefix}:summary", summary)
    for report in summary.get("symbols", []):
        kv.set_json(f"{settings.quality_prefix}:{report['symbol']}", report)

    log.info(
        "quality_summary_persisted",
        key=f"{settings.quality_prefix}:summary",
        overall=summary.get("overall_score"),
        healthy=summary.get("healthy"),
        symbols=len(summary.get("symbols", [])),
    )


if __name__ == "__main__":
    main()
