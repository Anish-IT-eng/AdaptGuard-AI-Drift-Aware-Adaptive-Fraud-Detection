"""
AdaptGuard AI — Centralized Logger
Provides structured logging with file + console output.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger that writes to both console and a rotating file.

    Args:
        name:    Module/component name (e.g., 'drift.monitor', 'adaptation.controller').
        log_dir: Directory where log files are stored.
        level:   Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:          # Avoid duplicate handlers on re-import
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handler (10 MB per file, keep last 5)
    log_file = os.path.join(log_dir, f"adaptguard_{datetime.now().strftime('%Y%m%d')}.log")
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    return logger
