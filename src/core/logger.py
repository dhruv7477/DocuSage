"""
src/core/logger.py
──────────────────
Structured logging using structlog.
Outputs JSON in production, colored console in development.
Import: `from src.core.logger import get_logger; log = get_logger(__name__)`
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor


def _add_service_name(logger: Any, method: str, event_dict: EventDict) -> EventDict:
    """Inject app name into every log entry for log aggregation filtering."""
    event_dict["service"] = "privateai"
    return event_dict


def configure_logging(log_level: str = "INFO", is_production: bool = False) -> None:
    """
    Call once at app startup (in main.py lifespan).
    - Development: pretty colored console output
    - Production: JSON lines (compatible with Datadog, CloudWatch, Loki)
    """
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        _add_service_name,
    ]

    if is_production:
        # JSON output for log aggregators
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        # Human-readable colored output for dev
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Factory function — use this instead of logging.getLogger()."""
    return structlog.get_logger(name)
