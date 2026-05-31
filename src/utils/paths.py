"""Filesystem path helpers.

Centralizes resolution of artifact directories so no other module
hardcodes a path. Directories are created lazily on first access.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the project root, two levels above this file."""
    return Path(__file__).resolve().parent.parent.parent


def resolve(path: str | Path) -> Path:
    """Resolve a path against the project root if it is not absolute.

    Args:
        path: Either an absolute path or one relative to the project root.

    Returns:
        An absolute Path object.
    """
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    return p


def ensure_dir(path: str | Path) -> Path:
    """Resolve `path` and create the directory tree if needed.

    Always returns the absolute Path. Safe to call repeatedly.
    """
    p = resolve(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
