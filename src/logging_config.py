"""Structured logging configuration for experiments and optimization pipelines."""

import logging
import os
from pathlib import Path


def setup_logger(
    name: str = "anderson",
    log_dir: str = "results",
    log_file: str | None = "run.log",
    level: str | None = None,
) -> logging.Logger:
    """
    Configure and return a structured logger with console and file output.

    Args:
        name: Logger name.
        log_dir: Directory where log files are stored.
        log_file: Name of the log file, or None to disable file logging.
        level: Logging level (e.g. 'DEBUG', 'INFO'). Overrides LOG_LEVEL env var.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    log_level_str = level or os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-7s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        out_dir = Path(log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(out_dir / log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
