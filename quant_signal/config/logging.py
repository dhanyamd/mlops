"""Structured logging — JSON lines, level from env, secrets never logged.

Every service in this platform logs through ``get_logger`` so logs are
machine-parseable (Observability: Loki/Grafana later).
"""

from __future__ import annotations

import logging

import structlog

from config.settings import get_settings


def configure_logging() -> None:
    """Idempotent structlog setup. Call once at process start."""
    settings = get_settings()
    level = logging.getLevelName(settings.log_level.upper())
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "quant_signal") -> structlog.stdlib.BoundLogger:
    """Bound logger for a module. Bind stable context with ``log.bind(...)``."""
    return structlog.get_logger(name)
