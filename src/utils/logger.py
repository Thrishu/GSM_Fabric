"""
Logging utilities for the fabric GSM pipeline.

Provides centralized logging configuration with file and console output.
Designed for production use with structured logging practices.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_dir: Path,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return a logger with file and console handlers.

    Args:
        name: Logger name (typically module name via __name__)
        log_dir: Directory path where log files will be written
        level: Logging level (default INFO)

    Returns:
        Configured Logger instance

    Raises:
        OSError: If log directory cannot be created or accessed
    """
    # Ensure log directory exists
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if logger is already configured
    if logger.hasHandlers():
        return logger

    # Formatter for consistent log output
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler: writes all logs to file
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / f"{name}.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler: writes to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
