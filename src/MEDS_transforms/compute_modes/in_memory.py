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
        # Paths currently being written by a worker; used by ``try_reserve_write`` to preserve
        # the mutual-exclusion semantics that ``FileLock`` provides in disk mode.
        self._in_progress: set[Path] = set()

    def put(self, fp: str | Path, df: DF_T) -> None:
        with self._lock:
            self._frames[Path(fp)] = df

    def get(self, fp: str | Path) -> DF_T:
        """Fetch the frame keyed by ``fp``. Raises ``KeyError`` with a registry-scoped message.

        The bare-dict ``KeyError`` would just show the Path repr; this version makes it obvious
        that the failure is "registry wasn't seeded" rather than "file missing on disk".
        """
        key = Path(fp)
        with self._lock:
            if key not in self._frames:
                raise KeyError(
                    f"In-memory FrameRegistry has no frame registered for path: {key!s}. "
                    f"Known keys: {sorted(str(k) for k in self._frames)}"
                )
            return self._frames[key]

    def has(self, fp: str | Path) -> bool:
        with self._lock:
            return Path(fp) in self._frames

    def keys(self) -> list[Path]:
        with self._lock:
            return list(self._frames.keys())

    def try_reserve_write(self, fp: str | Path) -> bool:
        """Reserve ``fp`` for writing if no other worker is currently holding it.

        Returns ``True`` if the caller obtained the reservation (and must pair it with
        ``release_write``). Returns ``False`` if the path is already in progress — mirroring
        disk mode where ``FileLock.acquire(timeout=0)`` raises ``Timeout`` and
        ``rwlock_wrap`` returns ``False`` without running ``compute_fn``.

        Examples:
            >>> reg = FrameRegistry()
            >>> reg.try_reserve_write("/virtual/out.parquet")
            True
            >>> reg.try_reserve_write("/virtual/out.parquet")
            False
            >>> reg.release_write("/virtual/out.parquet")
            >>> reg.try_reserve_write("/virtual/out.parquet")
            True
        """
        key = Path(fp)
        with self._lock:
            if key in self._in_progress:
                return False
            self._in_progress.add(key)
            return True

    def release_write(self, fp: str | Path) -> None:
        """Release a reservation previously obtained via ``try_reserve_write``."""
        with self._lock:
            self._in_progress.discard(Path(fp))


_registry: FrameRegistry | None = None
_registry_lock = threading.Lock()


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
    """Fetch a frame from the active registry. Raises if no registry is active.

    Examples:
        >>> with in_memory_mode() as reg:
        ...     reg.put("/virtual/shard.parquet", pl.LazyFrame({"a": [1, 2]}))
        ...     print(in_memory_read("/virtual/shard.parquet").collect())
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        │ 2   │
        └─────┘
    """
    reg = active_registry()
    if reg is None:  # pragma: no cover - defensive
        raise RuntimeError("in_memory_read called outside of an in_memory_mode context")
    return reg.get(fp)


def in_memory_write(df: DF_T, fp: str | Path) -> None:
    """Store a frame into the active registry. Raises if no registry is active.

    Materializes the frame and stores it as a fresh ``LazyFrame`` so reads via ``read_df`` return
    the same type they would in disk mode (where ``pl.scan_parquet`` yields ``LazyFrame``). This
    prevents type drift across chained MAP stages regardless of whether the stage's compute
    function returns an eager ``pl.DataFrame`` or a ``pl.LazyFrame``. Collecting at write time
    also severs any lingering compute-graph references to the previous stage.

    Examples:
        >>> with in_memory_mode() as reg:
        ...     in_memory_write(pl.DataFrame({"a": [1, 2]}), "/eager/shard.parquet")
        ...     in_memory_write(pl.LazyFrame({"a": [3, 4]}), "/lazy/shard.parquet")
        ...     print(type(reg.get("/eager/shard.parquet")).__name__)
        ...     print(type(reg.get("/lazy/shard.parquet")).__name__)
        LazyFrame
        LazyFrame
    """
    reg = active_registry()
    if reg is None:  # pragma: no cover - defensive
        raise RuntimeError("in_memory_write called outside of an in_memory_mode context")
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    reg.put(fp, df.lazy())


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
    with _registry_lock:
        if _registry is not None:
            raise RuntimeError("in_memory_mode is not re-entrant")
        _registry = registry if registry is not None else FrameRegistry()
        installed = _registry
    try:
        yield installed
    finally:
        with _registry_lock:
            _registry = None
