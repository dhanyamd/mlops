"""Feast feature store registration and materialization orchestrator.

Why materialization:
  The offline feature store contains historical logs. The online store (Redis)
  serves them at low latency (<5ms). Materialization is the sync process that
  copies the latest features from Parquet/ClickHouse into Redis.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from shared.observability.logging import get_logger

log = get_logger(__name__)


def run_cmd(cmd: list[str], cwd: Path) -> None:
    """Run a shell command and handle failure gracefully."""
    log.info("running_feast_command", cmd=" ".join(cmd), cwd=str(cwd))
    import os
    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = "minioadmin"
    env["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    env["AWS_ENDPOINT_URL"] = "http://localhost:9000"
    env["AWS_ENDPOINT_URL_S3"] = "http://localhost:9000"
    env["AWS_S3_ENDPOINT_URL"] = "http://localhost:9000"
    env["FEAST_S3_ENDPOINT_URL"] = "http://localhost:9000"
    env["AWS_DEFAULT_REGION"] = "us-east-1"
    
    res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        log.error("feast_command_failed", error=res.stderr)
        raise RuntimeError(f"Feast command failed: {res.stderr}")
    log.info("feast_command_success", output=res.stdout.strip())


def materialize_online_store() -> None:
    """Register definitions (feast apply) and sync offline data to Redis."""
    repo_dir = Path(__file__).parent.resolve()
    
    # 1. Feast apply: registers/updates features.py to the registry database
    run_cmd(["feast", "apply"], cwd=repo_dir)
    
    # 2. Feast materialize-incremental: syncs newest data to Redis
    now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    run_cmd(["feast", "materialize-incremental", now_str], cwd=repo_dir)


if __name__ == "__main__":
    try:
        materialize_online_store()
    except Exception as exc:
        log.error("materialization_failed", error=str(exc))
        sys.exit(1)
