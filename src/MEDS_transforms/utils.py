"""Utility helpers for path resolution and miscellaneous tasks."""

import os
import sys
from importlib.resources import files
from pathlib import Path

from . import __package_name__

PKG_PFX = "pkg://"


def invocation_name(module: str = __package_name__, argv_0: str | None = None) -> str:
    """Return a user-facing program name for how this process was invoked.

    Console scripts (``MEDS_transform-stage``, ``MEDS_transform-pipeline``) put their own name in
    ``sys.argv[0]``, which reads well in a usage line. ``python -m <module>`` instead puts the absolute
    path of the executed ``.py`` file there, which does not, so we render the module form in that case.

    Args:
        module: The dotted module name to suggest for ``python -m`` invocations. Defaults to the package
            itself, which is the ``-m`` target for the package's ``__main__.py``; modules that are their
            own ``-m`` target (``MEDS_transforms.runner``) pass their own name.
        argv_0: The value to interpret. Defaults to ``sys.argv[0]``.

    Returns:
        The program name to show the user.

    Examples:
        >>> invocation_name(argv_0="MEDS_transform-stage")
        'MEDS_transform-stage'
        >>> invocation_name(argv_0="/usr/local/bin/MEDS_transform-stage")
        'MEDS_transform-stage'
        >>> invocation_name(argv_0="/src/MEDS_transforms/__main__.py")
        'python -m MEDS_transforms'
        >>> invocation_name("MEDS_transforms.runner", "/src/MEDS_transforms/runner.py")
        'python -m MEDS_transforms.runner'
        >>> invocation_name(argv_0="")
        'python -m MEDS_transforms'

        With no `argv_0`, the live `sys.argv[0]` is read:

        >>> with patch.object(sys, "argv", ["/src/MEDS_transforms/__main__.py"]):
        ...     invocation_name()
        'python -m MEDS_transforms'
    """

    if argv_0 is None:
        argv_0 = sys.argv[0]

    name = Path(argv_0).name
    if not name or name.endswith(".py"):
        return f"python -m {module}"
    return name


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
