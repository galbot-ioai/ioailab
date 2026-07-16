"""ioailab logging configuration."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``ioailab``."""

    if name.startswith("ioailab"):
        return logging.getLogger(name)
    return logging.getLogger(f"ioailab.{name}")


def configure(level: str = "INFO") -> None:
    """Configure root ioailab logger with a human-readable format."""

    logger = logging.getLogger("ioailab")
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
