"""Run dbt build and record its wall-clock as a pipeline stage.

Usage (replaces the raw ``dbt build`` in ``make dbt-run``)::

    uv run python scripts/run_dbt.py build

Exit code mirrors dbt's, so CI/Makefile behavior is unchanged; the timing is
persisted to BRONZE.PIPELINE_METRICS so E2E latency is measurable end to end.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from config.settings import PROJECT_ROOT
from ingest.metrics import PipelineMetrics

DBT_DIR = PROJECT_ROOT / "dbt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", nargs="?", default="build", help="dbt command (build|run|test)")
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--profiles-dir", default=".")
    args = parser.parse_args()

    metrics = PipelineMetrics(flow="dbt")
    cmd = [
        "uv",
        "run",
        "dbt",
        args.command,
        "--project-dir",
        args.project_dir,
        "--profiles-dir",
        args.profiles_dir,
    ]
    with metrics.stage(f"dbt-{args.command}"):
        proc = subprocess.run(cmd, cwd=DBT_DIR)
    metrics.flush()
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
