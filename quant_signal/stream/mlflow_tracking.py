"""MLflow experiment tracking for offline validation runs.

The served models are online learners (River) + Monte Carlo, which have no
classic train → register → stage → serve cycle, so the *model registry* is not
used. What IS useful is MLflow *tracking*: recording each validation run's
params, metrics, and artifacts so the QuantPad-style pass probability is
reproducible and comparable across symbols and rule sets.

mlflow is an optional extra, imported lazily: without it (or with tracking
disabled) every call here is a safe no-op, keeping the default venv light and
the test suite hermetic.
"""

from __future__ import annotations

from config.logging import get_logger
from config.settings import get_settings

log = get_logger("stream.mlflow_tracking")

# Validation metrics worth tracking verbatim from the StrategyMonteCarlo
# payload (all floats already — log_metrics rejects nested/str values).
_METRICS = (
    "pass_probability",
    "bust_rate",
    "neutral_rate",
    "expected_return",
    "median_terminal",
    "best10_terminal",
    "worst10_terminal",
    "avg_max_drawdown",
    "median_max_drawdown",
    "p95_max_drawdown",
)
# Float-valued metrics repeated as ints where that's the natural unit.
_INT_METRICS = ("n_sims", "n_periods")


def _try_import_mlflow():
    """Return the mlflow module or None (optional dependency)."""
    try:
        import mlflow  # type: ignore[no-redef] - optional extra
    except ImportError:  # pragma: no cover - depends on the venv
        return None
    return mlflow


def track_validation(
    symbol: str,
    validation: dict,
    *,
    target: float | None,
    max_drawdown: float,
    seed: int | None,
    n_sims: int,
) -> str | None:
    """Log one strategy-validation run to MLflow Tracking.

    Returns the run id, or None when tracking is disabled or mlflow is not
    installed. Never raises — tracking must not fail the calling request.
    """
    settings = get_settings()
    if not settings.mlflow_tracking_enabled:
        return None
    mlflow = _try_import_mlflow()
    if mlflow is None:
        log.info("mlflow_tracking_unavailable", symbol=symbol)
        return None
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        with mlflow.start_run(run_name=f"strategy-validation-{symbol.lower()}"):
            mlflow.log_params(
                {
                    "symbol": symbol,
                    "target": target if target is not None else "",
                    "max_drawdown": max_drawdown,
                    "seed": seed if seed is not None else "",
                    "n_sims": n_sims,
                }
            )
            mlflow.log_metrics(
                {
                    key: float(validation[key])
                    for key in _METRICS + _INT_METRICS
                    if key in validation
                }
            )
            mlflow.log_dict(validation.get("terminal_histogram", {}), "terminal_histogram.json")
            mlflow.log_dict(validation.get("drawdown_histogram", {}), "drawdown_histogram.json")
            log.info(
                "mlflow_tracking_logged",
                symbol=symbol,
                pass_probability=validation.get("pass_probability"),
            )
            return mlflow.active_run().info.run_id
    except Exception:  # noqa: BLE001 - tracking must never break the API
        log.exception("mlflow_tracking_failed", symbol=symbol)
        return None
