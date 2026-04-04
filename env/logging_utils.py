"""Logging helpers for SmartOps AI."""

from __future__ import annotations

import logging


def get_logger(name: str = "smartops_ai_env", level: str = "INFO") -> logging.Logger:
    """Return a package logger configured once."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(level.upper())
    return logger
