"""This file contains utilities for pretty printing stage related things."""

from __future__ import annotations

from pathlib import Path

_SPACE = "    "
_BRANCH = "│   "
_TEE = "├── "
_LAST = "└── "


def pretty_list_directory(path: Path, prefix: str | None = None) -> list[str]:
    """Pretty prints the contents of a directory.

    Args:
        path: The path to the directory to list.
        prefix: Used for the recursive prefixing of subdirectories. Defaults to None.

    Returns:
        A list of strings representing the contents of the directory. To be printed with newlines separating
        them.

    Raises:
        ValueError: If the path is not a directory.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir)
        ...     (path / "file1.txt").touch()
        ...     (path / "foo").mkdir()
        ...     (path / "bar").mkdir()
        ...     (path / "bar" / "baz.csv").touch()
        ...     for l in pretty_list_directory(path): print(l) # This is just used as newlines break doctests
        ├── bar
        │   └── baz.csv
        ├── file1.txt
        └── foo
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir)
        ...     pretty_list_directory(path / "foo")
        Traceback (most recent call last):
            ...
        ValueError: Path /tmp/tmp.../foo does not exist.
        >>> with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        ...     path = Path(tmp.name)
        ...     pretty_list_directory(path)
        Traceback (most recent call last):
            ...
        ValueError: Path /tmp/tmp....txt is not a directory.
        >>> pretty_list_directory("foo")
        Traceback (most recent call last):
            ...
        ValueError: Expected a Path object, got <class 'str'>: foo
    """

    if not isinstance(path, Path):
        raise ValueError(f"Expected a Path object, got {type(path)}: {path}")

    if not path.exists():
        raise ValueError(f"Path {path} does not exist.")

    if not path.is_dir():
        raise ValueError(f"Path {path} is not a directory.")

    if prefix is None:
        prefix = ""

    lines = []

    children = sorted(list(path.iterdir()))

    for i, child in enumerate(children):
        is_last = i == len(children) - 1

        node_prefix = _LAST if is_last else _TEE
        subdir_prefix = _SPACE if is_last else _BRANCH

        if child.is_file():
            lines.append(f"{prefix}{node_prefix}{child.name}")
        elif child.is_dir():
            lines.append(f"{prefix}{node_prefix}{child.name}")
            lines.extend(pretty_list_directory(child, prefix=prefix + subdir_prefix))
    return lines
