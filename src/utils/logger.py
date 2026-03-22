"""
Centralized logging setup.
All modules should call `get_logger(__name__)` to get a child logger.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_INITIALIZED = False


def setup_logging(level: str = "INFO", log_dir: str | Path | None = None) -> None:
    """Configure root logger with console + optional file handler."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler (optional)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path / "pipeline.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call setup_logging() first in main."""
    return logging.getLogger(name)
