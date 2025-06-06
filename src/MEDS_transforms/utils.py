"""Utility helpers for path resolution and miscellaneous tasks."""

import os
from importlib.resources import files
from pathlib import Path

PKG_PFX = "pkg://"


def resolve_pkg_path(pkg_path: str) -> Path:
    """Resolve a ``pkg://`` path into an on-disk :class:`~pathlib.Path`.

    Args:
        pkg_path: Path in ``pkg://`` notation.

    Returns:
        The resolved path to the package resource on disk.

    Raises:
        ValueError: If the package specified in ``pkg_path`` does not exist.

    Examples:
        >>> resolve_pkg_path("pkg://MEDS_transforms.configs.pipeline.py").suffix
        '.py'
    """
    parts = pkg_path[len(PKG_PFX) :].split(".")
    pkg_name = parts[0]
    suffix = parts[-1]
    relative_path = Path(os.path.join(*parts[1:-1])).with_suffix(f".{suffix}")
    try:
        return files(pkg_name) / relative_path
    except ModuleNotFoundError as e:
        raise ValueError(f"Package '{pkg_name}' not found. Please check the package name.") from e
