"""Structured logging setup for OSCAR.

Uses Loguru for ergonomic structured logging. Falls back gracefully if
loguru is not installed (uses stdlib `logging`).

The logger is namespaced under `oscar.<module>`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from loguru import logger as _loguru_logger


class _InterceptHandler(logging.Handler):
    """Route stdlib `logging` records into Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_back and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(
    level: str = "INFO",
    log_to_file: bool = False,
    log_dir: Path | None = None,
    json_format: bool = False,
) -> None:
    """Configure OSCAR logging.

    Args:
        level: Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL).
        log_to_file: Whether to write logs to a rotating file.
        log_dir: Directory for log file (default: ./logs).
        json_format: Use JSON formatter instead of human-readable.
    """
    _loguru_logger.remove()

    if json_format:
        fmt = (
            '{{"ts":"{time:YYYY-MM-DD HH:mm:ss.SSS}",'
            '"level":"{level}",'
            '"module":"{name}",'
            '"message":"{message}"}}'
        )
    else:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    _loguru_logger.add(
        sys.stderr,
        format=fmt,
        level=level.upper(),
        colorize=not json_format,
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )

    if log_to_file:
        log_dir = log_dir or Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        _loguru_logger.add(
            log_dir / "oscar.log",
            format=fmt,
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=False,
            enqueue=True,
        )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for noisy in ("urllib3", "transformers", "sentence_transformers", "streamlit"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> Any:
    """Return a bound logger.

    Args:
        name: Optional module/sub-namespace (e.g. "ingestion.gdelt").

    Returns:
        Loguru logger instance, optionally bound with `name`.
    """
    if name:
        return _loguru_logger.bind(module=name)
    return _loguru_logger


__all__ = ["configure_logging", "get_logger"]
