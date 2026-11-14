"""Hold stages' data shards in memory instead of on disk.

In-memory mode (``MEDS_transform-pipeline --in_memory``) keeps each stage's data shards in a `FrameStore`
rather than writing them as parquet files, so a pipeline's intermediate data never touches disk. Only the
directories registered with the store as *roots* (each in-memory stage's ``data/`` output directory) live
in memory: every read, write, existence check and directory listing of a path under a root goes to the
store, and everything else (metadata, scratch files, logs, and the pipeline's input) stays on disk. Keeping
the store to data shards means stages that read metadata files directly, rather than through the
package's read and write functions, work unchanged.

A store entry is either a `polars.DataFrame` or the `Path` of a parquet file on disk holding the shard's
contents; the latter lets a stage's output carry over input shards that are still on disk without reading
them.

The store is process-local, so in-memory mode always runs stages in process, one worker at a time. It does
not coordinate with other launches of the same pipeline, which could not see its shards anyway.

Examples:
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     root = Path(tmpdir) / "stage" / "data"
    ...     with in_memory_mode() as store:
    ...         store.add_root(root)
    ...         store.put(root / "train" / "0.parquet", pl.DataFrame({"a": [1]}))
    ...         print(owning_store(root / "train" / "0.parquet") is store, owning_store(tmpdir))
    ...         print(sorted(p.as_posix() for p in store.files(root)))
    ...     print(active_store(), root.exists())
    True None
    ['train/0.parquet']
    None False
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

ENTRY_T = pl.DataFrame | Path

_ACTIVE_STORE: FrameStore | None = None
_ACTIVE_LOCK = threading.Lock()


def store_key(fp: str | Path) -> Path:
    """The store key for ``fp``: its absolute, normalized path (symlinks are not resolved)."""
    return Path(os.path.abspath(fp))


class FrameStore:
    """An in-memory stand-in for the parquet files under a set of root directories.

    Examples:
        >>> store = FrameStore()
        >>> store.add_root("out/stage/data")
        >>> store.owns("out/stage/data/train/0.parquet"), store.owns("out/stage/metadata/codes.parquet")
        (True, False)
        >>> store.put("out/stage/data/train/0.parquet", pl.DataFrame({"a": [1]}))
        >>> store.has("out/stage/data/train/0.parquet"), store.has("out/stage/data/train/1.parquet")
        (True, False)
        >>> store.read("out/stage/data/train/0.parquet").collect()["a"].to_list()
        [1]
        >>> store.put("out/elsewhere.parquet", pl.DataFrame())
        Traceback (most recent call last):
            ...
        ValueError: ... is not under any of this store's roots.

        An entry can stand for a parquet file on disk, which is read lazily:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     disk_fp = Path(tmpdir) / "0.parquet"
        ...     pl.DataFrame({"a": [2]}).write_parquet(disk_fp)
        ...     store.put("out/stage/data/train/1.parquet", disk_fp)
        ...     print(store.read("out/stage/data/train/1.parquet").collect()["a"].to_list())
        [2]

        Removing a root drops its entries:

        >>> store.remove_root("out/stage/data")
        >>> store.owns("out/stage/data/train/0.parquet"), len(store)
        (False, 0)
    """

    def __init__(self) -> None:
        self._roots: set[Path] = set()
        self._entries: dict[Path, ENTRY_T] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def add_root(self, root: str | Path) -> None:
        """Hold the files under ``root`` in memory from now on."""
        self._roots.add(store_key(root))

    def remove_root(self, root: str | Path) -> None:
        """Stop holding ``root`` in memory, dropping its entries."""
        root = store_key(root)
        self._roots.discard(root)
        for fp in [fp for fp in self._entries if fp.is_relative_to(root)]:
            del self._entries[fp]

    def owns(self, fp: str | Path) -> bool:
        """Whether ``fp`` is (or would be) held in this store rather than on disk."""
        fp = store_key(fp)
        return any(fp.is_relative_to(root) for root in self._roots)

    def _ownedstore_key(self, fp: str | Path) -> Path:
        key = store_key(fp)
        if not self.owns(key):
            raise ValueError(f"{key} is not under any of this store's roots.")
        return key

    def has(self, fp: str | Path) -> bool:
        return store_key(fp) in self._entries

    def get(self, fp: str | Path) -> ENTRY_T:
        """The entry for ``fp``.

        Raises:
            FileNotFoundError: If there is none.
        """
        try:
            return self._entries[store_key(fp)]
        except KeyError:
            raise FileNotFoundError(f"No in-memory frame for {fp}") from None

    def read(self, fp: str | Path) -> pl.LazyFrame:
        entry = self.get(fp)
        return pl.scan_parquet(entry, glob=False) if isinstance(entry, Path) else entry.lazy()

    def put(self, fp: str | Path, entry: ENTRY_T | pl.LazyFrame) -> None:
        """Store ``entry`` for ``fp``, collecting it first if it is lazy."""
        if isinstance(entry, pl.LazyFrame):
            entry = entry.collect()
        self._entries[self._ownedstore_key(fp)] = entry

    def delete(self, fp: str | Path) -> None:
        self._entries.pop(store_key(fp), None)

    def files(self, directory: str | Path, suffix: str = "") -> dict[Path, Path]:
        """The entries under ``directory`` ending in ``suffix``, as ``{relative path: key}``.

        Hidden paths (any part starting with ``.``) are skipped, as on disk.

        Examples:
            >>> store = FrameStore()
            >>> store.add_root("out/data")
            >>> for fp in ("train/0.parquet", "held_out/0.parquet", ".tmp/x.parquet"):
            ...     store.put(f"out/data/{fp}", pl.DataFrame())
            >>> sorted(p.as_posix() for p in store.files("out/data", ".parquet"))
            ['held_out/0.parquet', 'train/0.parquet']
            >>> sorted(p.as_posix() for p in store.files("out/data/train"))
            ['0.parquet']
        """
        directory = store_key(directory)
        out = {}
        for fp in sorted(self._entries):
            if not fp.is_relative_to(directory) or not fp.name.endswith(suffix):
                continue
            rel = fp.relative_to(directory)
            if not any(part.startswith(".") for part in rel.parts):
                out[rel] = fp
        return out

    @staticmethod
    def disk_snapshot(root: str | Path, suffix: str = ".parquet") -> dict[Path, tuple[int, int]]:
        """The non-hidden ``suffix`` files on disk under ``root``, with their sizes and modification times."""
        root = Path(root)
        snapshot = {}
        for fp in sorted(root.rglob(f"*{suffix}")) if root.is_dir() else []:
            if not any(part.startswith(".") for part in fp.relative_to(root).parts):
                st = fp.stat()
                snapshot[fp] = (st.st_size, st.st_mtime_ns)
        return snapshot

    def adopt_disk_files(
        self, root: str | Path, before: dict[Path, tuple[int, int]], suffix: str = ".parquet"
    ) -> list[Path]:
        """Take over files written directly to disk under ``root`` (an in-memory root) since ``before``.

        A stage with its own write function writes its shards to disk even under in-memory mode. This makes
        the store refer to those files, so later stages (which read ``root`` from the store) see them. Files
        unchanged since the `disk_snapshot` ``before``, such as a previous on-disk run's, are left out.

        Args:
            root: The in-memory root to look under on disk.
            before: A `disk_snapshot` of ``root`` taken before the stage ran.
            suffix: The suffix of the files to take over.

        Returns:
            The keys of the files taken over.

        Examples:
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     root = Path(tmpdir) / "data"
            ...     (root / "train").mkdir(parents=True)
            ...     pl.DataFrame({"a": [0]}).write_parquet(root / "train" / "old.parquet")
            ...     store = FrameStore()
            ...     store.add_root(root)
            ...     before = store.disk_snapshot(root)
            ...     pl.DataFrame({"a": [1]}).write_parquet(root / "train" / "0.parquet")
            ...     adopted = store.adopt_disk_files(root, before)
            ...     print([p.relative_to(root).as_posix() for p in adopted])
            ...     print(store.read(root / "train" / "0.parquet").collect()["a"].to_list())
            ['train/0.parquet']
            [1]
        """
        adopted = []
        for fp, stat in self.disk_snapshot(root, suffix).items():
            if before.get(fp) != stat:
                self.put(fp, fp)
                adopted.append(store_key(fp))
        return adopted

    def write_to_disk(self, fp: str | Path, dst: Path) -> None:
        """Write the entry for ``fp`` to the parquet file ``dst`` on disk."""
        from .dataframe import write_df
        from .stage_outputs import link_or_copy

        entry = self.get(fp)
        if isinstance(entry, Path):
            link_or_copy(entry, dst)
        else:
            write_df(entry, dst)


def active_store() -> FrameStore | None:
    """The store of the current `in_memory_mode` block, if any."""
    return _ACTIVE_STORE


def owning_store(fp: str | Path) -> FrameStore | None:
    """The active store if it holds ``fp``, else ``None`` (so ``fp`` lives on disk)."""
    store = _ACTIVE_STORE
    return store if store is not None and store.owns(fp) else None


@contextmanager
def in_memory_mode() -> Iterator[FrameStore]:
    """Activate a fresh `FrameStore` for the duration of the block.

    Raises:
        RuntimeError: If a store is already active; blocks do not nest.

    Examples:
        >>> with in_memory_mode() as store:
        ...     with in_memory_mode():
        ...         pass
        Traceback (most recent call last):
            ...
        RuntimeError: An in-memory store is already active.
        >>> print(active_store())
        None
    """
    global _ACTIVE_STORE
    with _ACTIVE_LOCK:
        if _ACTIVE_STORE is not None:
            raise RuntimeError("An in-memory store is already active.")
        _ACTIVE_STORE = store = FrameStore()
    try:
        yield store
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_STORE = None
