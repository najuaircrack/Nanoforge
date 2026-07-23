"""Small stdlib-logging helper used across Nanoforge.

Keeps a single configured root-style logger for the ``nanoforge`` namespace. Level
is taken from the ``NANOFORGE_LOG_LEVEL`` environment variable (default ``INFO``),
so scripts and CI can raise/lower verbosity without code changes.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str | int | None = None) -> None:
    """Attach a stream handler to the ``nanoforge`` logger once.

    ``level`` overrides ``NANOFORGE_LOG_LEVEL``; both default to ``INFO``.
    """
    global _CONFIGURED
    resolved = level if level is not None else os.environ.get("NANOFORGE_LOG_LEVEL", "INFO")
    if isinstance(resolved, str):
        resolved = getattr(logging, resolved.upper(), logging.INFO)
    logger = logging.getLogger("nanoforge")
    logger.setLevel(resolved)
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    if not name.startswith("nanoforge"):
        name = f"nanoforge.{name}"
    return logging.getLogger(name)
