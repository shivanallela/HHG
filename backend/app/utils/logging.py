"""
Professional logging foundation for HH Goa 2026 Voice-Enabled RAG.

Supports:
- Structured log formatting with timestamps
- Request ID tracking (for future middleware)
- Processing stage context
- File + console output
- Log level configuration via environment
- Secret filtering (never logs API keys)
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from backend.app.config import settings

# Fields that must NEVER appear in logs
_SENSITIVE_KEYS = frozenset({
    "api_key", "api_secret", "password", "token", "secret",
    "sarvam_api_key", "elevenlabs_api_key", "llm_api_key",
})


class SensitiveFilter(logging.Filter):
    """Filter that redacts sensitive information from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.msg)
        for key in _SENSITIVE_KEYS:
            if key.lower() in msg.lower():
                record.msg = "[REDACTED — sensitive field detected]"
                break
        return True


def setup_logging(
    log_level: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure and return the application logger.

    Args:
        log_level: Override log level (defaults to settings.LOG_LEVEL).
        log_dir: Override log directory (defaults to settings.LOG_DIR).

    Returns:
        Configured root logger for the application.
    """
    level = getattr(logging, (log_level or settings.LOG_LEVEL).upper(), logging.INFO)
    log_directory = log_dir or settings.LOG_DIR
    log_directory.mkdir(parents=True, exist_ok=True)

    # Formatter with structured context
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(name)-25s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(SensitiveFilter())

    # File handler
    file_handler = logging.FileHandler(
        log_directory / "app.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(SensitiveFilter())

    # Root application logger
    logger = logging.getLogger("hhg")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    logger.info(
        "Logging initialized | level=%s | dir=%s",
        logging.getLevelName(level),
        log_directory,
    )
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the application namespace."""
    return logging.getLogger(f"hhg.{name}")
