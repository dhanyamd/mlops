"""MLflow helpers: experiment setup, model promotion, artifact loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import mlflow
from mlflow.tracking import MlflowClient

from shared.config import MLFLOW_CFG, PROJECT_ROOT

LOCAL_MLFLOW_URI = (PROJECT_ROOT / "mlruns").as_uri()


def setup_mlflow() -> str:
    """Configure MLflow tracking; fall back to local file store if server is down."""
    os.environ.setdefault("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"))
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"))
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", MLFLOW_CFG.s3_endpoint)
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    try:
        response = httpx.get(f"{MLFLOW_CFG.tracking_uri}/health", timeout=2.0)
        if response.status_code == 200:
            mlflow.set_tracking_uri(MLFLOW_CFG.tracking_uri)
            return MLFLOW_CFG.tracking_uri
    except Exception:
        pass

    Path(PROJECT_ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(LOCAL_MLFLOW_URI)
    return LOCAL_MLFLOW_URI


def promote_model(
    model_name: str,
    version: str,
    stage: str = "Production",
    alias: str | None = "champion",
) -> None:
    """Promote a registered model version to Production (with optional alias)."""
    try:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True,
        )
        if alias:
            client.set_registered_model_alias(model_name, alias, version)
    except Exception as exc:
        from shared.observability.logging import get_logger
        log = get_logger(__name__)
        log.warning("model_promotion_skipped", error=str(exc))


def load_production_model(model_name: str, alias: str = "champion"):
    """Load the current production model by alias."""
    setup_mlflow()
    model_uri = f"models:/{model_name}@{alias}"
    return mlflow.pyfunc.load_model(model_uri)


def log_pipeline_params(params: dict[str, Any]) -> None:
    """Log a flat dict of pipeline configuration parameters."""
    mlflow.log_params({k: str(v) for k, v in params.items()})
