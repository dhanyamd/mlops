"""Structured logging with structlog.

Why structlog over print() / logging.basicConfig():
  - Every log is JSON → parseable by Loki, Datadog, CloudWatch, Splunk.
  - Context variables (request_id, model_version, card_id) are bound ONCE
    and appear on every subsequent log line in that scope automatically.
  - No string formatting — you log key-value pairs, the renderer decides format.
  - Correlation: trace_id from OpenTelemetry is injected into every log line,
    letting you jump from a Grafana alert → exact log line → exact trace.

Usage:
    from shared.observability.logging import get_logger
    log = get_logger(__name__)

    # Bind context once per request/transaction
    log = log.bind(card_id="abc123", model_version="v42")
    log.info("score_computed", fraud_score=0.91, latency_ms=12)

    # Output (JSON):
    # {"event": "score_computed", "card_id": "abc123", "model_version": "v42",
    #  "fraud_score": 0.91, "latency_ms": 12, "timestamp": "...", "level": "info"}
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(
    level: str = "INFO",
    json_logs: bool = True,
) -> None:
    """Configure structlog for the entire application.

    Call once at process startup (before any logging occurs).
    In development: pretty console output.
    In production (json_logs=True): machine-readable JSON for log aggregators.
    """
    shared_processors: list = [
        # Add log level (info, warning, error) to every event
        structlog.stdlib.add_log_level,
        # Add ISO timestamp to every event
        structlog.processors.TimeStamper(fmt="iso"),
        # If an exception is present, render it cleanly
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        # Thread/coroutine context
        structlog.contextvars.merge_contextvars,
    ]

    if json_logs:
        # Production: JSON output for Loki / CloudWatch
        renderer = structlog.processors.JSONRenderer()
    else:
        # Development: coloured pretty output
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging so third-party libs (kafka, mlflow) emit JSON
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(level.upper()),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a bound structlog logger for the given module.

    Usage:
        log = get_logger(__name__)
        log.info("model_loaded", model_version="v42")
    """
    return structlog.get_logger(name)
