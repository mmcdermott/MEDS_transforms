"""In-memory execution mode: skip parquet round-trips between MAP stages.

This module provides a process-local ``FrameRegistry`` keyed by the logical filesystem path a
stage would otherwise read/write, plus a context manager that activates it for the duration of an
in-process pipeline run. While a registry is active:

- The default ``read_df`` / ``write_df`` (in :mod:`MEDS_transforms.dataframe`) transparently
  round-trip through the registry instead of touching parquet.
- ``rwlock_wrap`` (in :mod:`MEDS_transforms.mapreduce.rwlock`) skips its filesystem ``FileLock``
  and ``out_fp_checker``, because in a single-process in-memory run there is no cross-worker
  contention to arbitrate.

Stages that register their own ``read_fn`` / ``write_fn`` at ``Stage.register`` time bypass the
default and therefore also bypass this mechanism — they keep doing whatever they did before.

Scope (Phase 0): MAP stages only. MAPREDUCE stages still perform their file-based reduce
handshake; wiring the reducer through the registry is a follow-up. A ``MAIN`` stage that reads
and writes parquet directly also stays on disk.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..dataframe.types import DF_T


class FrameRegistry:
    """Thread-safe map from (logical) output path to in-memory ``DataFrame`` / ``LazyFrame``.

    Paths are used only as opaque keys — they are never opened or written to. The registry
    normalizes incoming paths with ``Path()`` so callers may pass ``str`` or ``Path``
    interchangeably.

    Examples:
        >>> reg = FrameRegistry()
        >>> reg.has(Path("/tmp/nope.parquet"))
        False
        >>> df = pl.DataFrame({"a": [1, 2, 3]})
        >>> reg.put(Path("/tmp/shard.parquet"), df)
        >>> reg.has("/tmp/shard.parquet")
        True
        >>> reg.get("/tmp/shard.parquet")
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        │ 2   │
        │ 3   │
        └─────┘
        >>> sorted(str(p) for p in reg.keys())
        ['/tmp/shard.parquet']
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[Path, DF_T] = {}

    def put(self, fp: str | Path, df: DF_T) -> None:
        with self._lock:
            self._frames[Path(fp)] = df

    def get(self, fp: str | Path) -> DF_T:
        with self._lock:
            return self._frames[Path(fp)]

    def has(self, fp: str | Path) -> bool:
        with self._lock:
            return Path(fp) in self._frames

    def keys(self) -> list[Path]:
        with self._lock:
            return list(self._frames.keys())


_registry: FrameRegistry | None = None


def active_registry() -> FrameRegistry | None:
    """Return the currently installed ``FrameRegistry``, or ``None`` if disk mode is active.

    Examples:
        >>> active_registry() is None
        True
        >>> with in_memory_mode() as reg:
        ...     active_registry() is reg
        True
        >>> active_registry() is None
        True
    """
    return _registry


def in_memory_read(fp: str | Path) -> DF_T:
    """Fetch a frame from the active registry.

    Raises if no registry is active.
    """
    reg = active_registry()
    if reg is None:  # pragma: no cover - defensive
        raise RuntimeError("in_memory_read called outside of an in_memory_mode context")
    return reg.get(fp)


def in_memory_write(df: DF_T, fp: str | Path) -> None:
    """Store a frame into the active registry. Raises if no registry is active.

    Collects lazy frames eagerly so each stage's output is materialized at the point of write, preventing
    later stages from accidentally re-triggering an earlier stage's compute graph.
    """
    reg = active_registry()
    if reg is None:  # pragma: no cover - defensive
        raise RuntimeError("in_memory_write called outside of an in_memory_mode context")
    if isinstance(df, pl.LazyFrame):
        df = df.collect().lazy()
    reg.put(fp, df)


@contextmanager
def in_memory_mode(registry: FrameRegistry | None = None) -> Iterator[FrameRegistry]:
    """Activate in-memory execution for the duration of the ``with`` block.

    Args:
        registry: An existing registry to install. If ``None``, a fresh one is created. Passing
            an existing registry lets a caller seed input shards before entering the context or
            inspect outputs after exiting.

    Yields:
        The active ``FrameRegistry``.

    Raises:
        RuntimeError: If a registry is already active in the current process.

    Examples:
        >>> with in_memory_mode() as reg:
        ...     reg.put("in/shard_0.parquet", pl.DataFrame({"x": [1, 2]}))
        ...     print(active_registry() is reg)
        True

        A registry can be constructed outside the block and handed in:

        >>> seeded = FrameRegistry()
        >>> seeded.put("in/shard_0.parquet", pl.DataFrame({"x": [1, 2]}))
        >>> with in_memory_mode(seeded) as reg:
        ...     print(reg is seeded)
        True

        Nested activation is rejected — the mechanism is process-global state, and two layers of
        in-memory mode would be ambiguous:

        >>> with in_memory_mode():
        ...     with in_memory_mode():
        ...         pass
        Traceback (most recent call last):
            ...
        RuntimeError: in_memory_mode is not re-entrant
    """
    global _registry
    if _registry is not None:
        raise RuntimeError("in_memory_mode is not re-entrant")
    _registry = registry if registry is not None else FrameRegistry()
    try:
        yield _registry
    finally:
        _registry = None
