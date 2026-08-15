from __future__ import annotations

import logging
import sys
from typing import Any

from app.core.config import get_settings


class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage() and "/ready" not in record.getMessage()


def setup_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)

    logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

    for logger_name in ("uvicorn", "uvicorn.error", "fastapi"):
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
