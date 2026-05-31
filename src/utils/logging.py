"""Logger configuration shared across the codebase.

Every module calls get_logger(__name__) rather than wiring up its own
handlers, so log format and level stay consistent everywhere.
"""

from __future__ import annotations

import logging
import sys


_CONFIGURED = False


def configure_root_logger(level: str = "INFO") -> None:
    """Install a single stream handler on the root logger.

    Safe to call multiple times; the second and later calls do nothing.

    Args:
        level: Standard logging level name (DEBUG, INFO, WARNING, ERROR).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger; ensures the root handler is configured first."""
    if not _CONFIGURED:
        configure_root_logger()
    return logging.getLogger(name)
