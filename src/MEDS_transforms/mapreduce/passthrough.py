"""Carry a stage's untouched half of the dataset (data or metadata) forward into its output directory.

Every stage in a pipeline writes a complete MEDS dataset: a ``data/`` directory and a ``metadata/``
directory. A stage only computes one of those halves (data stages write data shards; metadata stages write
``codes.parquet``). The other half is carried over from the stage's inputs by the helpers here, so the next
stage can always read ``<prior stage>/data`` and ``<prior stage>/metadata`` as a pair.

Files are hard-linked where the filesystem allows it, so carrying data forward costs no disk space or copy
time, and are copied otherwise. Hard links (rather than symlinks) keep each stage directory valid on its
own: deleting an earlier stage's directory does not break a later one. Every output in MEDS-Transforms is
written to a temporary file that is then renamed into place, so a re-run that writes over a hard-linked file
replaces the link rather than altering the earlier stage's copy.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def link_or_copy(src: Path, dst: Path) -> None:
    """Make ``dst`` a hard link to ``src``, or a copy of it if hard links are not possible.

    The link (or copy) is staged at a temporary path next to ``dst`` and renamed into place, so an existing
    ``dst`` is replaced atomically and its old contents (which may be another stage's file, via a link) are
    never written through.

    Args:
        src: The file to carry forward.
        dst: Where it should appear. Parent directories are created as needed.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     src = Path(tmpdir) / "codes.parquet"
        ...     _ = src.write_text("original")
        ...     dst = Path(tmpdir) / "out" / "nested" / "codes.parquet"
        ...     link_or_copy(src, dst)
        ...     print(dst.read_text(), dst.samefile(src))
        original True

        If hard links fail (for instance because ``dst`` is on another filesystem), the file is copied:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     src = Path(tmpdir) / "codes.parquet"
        ...     _ = src.write_text("original")
        ...     dst = Path(tmpdir) / "copy.parquet"
        ...     with patch("os.link", side_effect=OSError("cross-device link")):
        ...         link_or_copy(src, dst)
        ...     print(dst.read_text(), dst.samefile(src))
        original False

        Replacing an existing destination does not change the file it used to point to:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     old_src = Path(tmpdir) / "old.parquet"
        ...     _ = old_src.write_text("old")
        ...     new_src = Path(tmpdir) / "new.parquet"
        ...     _ = new_src.write_text("new")
        ...     dst = Path(tmpdir) / "dst.parquet"
        ...     link_or_copy(old_src, dst)
        ...     link_or_copy(new_src, dst)
        ...     print(dst.read_text(), old_src.read_text())
        new old
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_fp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            os.link(src, tmp_fp)
        except OSError:
            shutil.copy2(src, tmp_fp)
        os.replace(tmp_fp, dst)
    except BaseException:
        tmp_fp.unlink(missing_ok=True)
        raise


def carry_forward(src_dir: Path, dst_dir: Path, replace_existing: bool = False) -> list[Path]:
    """Carry every file in ``src_dir`` into ``dst_dir``, keeping relative paths.

    Hidden files and directories (names starting with ``.``) are skipped, as they hold bookkeeping (e.g.
    logs or in-progress temporary files), not dataset contents.

    Args:
        src_dir: The directory to carry forward. If it does not exist, nothing is done.
        dst_dir: The directory to carry it into.
        replace_existing: If ``False``, files already in ``dst_dir`` are left alone; use this when the stage
            may have written some of these files itself. If ``True``, existing files that are not already
            the same file as their source are replaced; use this when the stage never writes into
            ``dst_dir``, so anything there is a carry-over from an earlier run.

    Returns:
        The destination paths that were (re-)created.

    Examples:
        Text files stand in for parquet files in these examples, as their contents are never parsed.

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     src = Path(tmpdir) / "in"
        ...     dst = Path(tmpdir) / "out"
        ...     src.mkdir()
        ...     dst.mkdir()
        ...     _ = yaml_disk({"codes.txt": "codes", "notes.txt": "notes", ".logs/": {"run.txt": "log"}}, src)
        ...     _ = yaml_disk({"codes.txt": "written by the stage"}, dst)
        ...     created = carry_forward(src, dst)
        ...     print([p.relative_to(dst).as_posix() for p in created])
        ...     print_directory(dst)
        ...     print((dst / "codes.txt").read_text())
        ['notes.txt']
        ├── codes.txt
        └── notes.txt
        written by the stage

        With ``replace_existing=True``, files that differ from their source are replaced and files that are
        already linked to their source are left in place:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     src = Path(tmpdir) / "in"
        ...     dst = Path(tmpdir) / "out"
        ...     src.mkdir()
        ...     _ = yaml_disk({"train/": {"0.txt": "a", "1.txt": "b"}}, src)
        ...     _ = carry_forward(src, dst)
        ...     (src / "train" / "1.txt").unlink()
        ...     _ = (src / "train" / "1.txt").write_text("new b")
        ...     created = carry_forward(src, dst, replace_existing=True)
        ...     print([p.relative_to(dst).as_posix() for p in created])
        ...     print((dst / "train" / "1.txt").read_text())
        ['train/1.txt']
        new b

        A missing source directory is a no-op:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     carry_forward(Path(tmpdir) / "missing", Path(tmpdir) / "out")
        []
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.is_dir():
        return []

    created = []
    for src_fp in sorted(src_dir.rglob("*")):
        rel = src_fp.relative_to(src_dir)
        if not src_fp.is_file() or any(part.startswith(".") for part in rel.parts):
            continue

        dst_fp = dst_dir / rel
        if dst_fp.exists() and (not replace_existing or dst_fp.samefile(src_fp)):
            continue

        link_or_copy(src_fp, dst_fp)
        created.append(dst_fp)

    return created


def carry_forward_unchanged_outputs(cfg: DictConfig, is_metadata: bool, wrote_metadata: bool = True) -> None:
    """Fill in a stage's ``data/`` and ``metadata/`` output directories with whatever it did not produce.

    A data stage never writes metadata, so its whole ``metadata/`` input directory is carried into its
    ``metadata/`` output directory, replacing stale copies. A metadata stage never writes data, so its whole
    ``data/`` input is carried into its ``data/`` output. A metadata stage also writes only some of the
    metadata files (normally just ``codes.parquet``), so the remaining input metadata files (e.g.
    ``dataset.json`` and ``subject_splits.parquet``) are carried over without touching the stage's own
    outputs.

    This is a no-op unless the stage config defines ``data_output_dir`` and ``metadata_output_dir``, which
    `PipelineConfig` always does but hand-built configs (as in many doctests) may not.

    Args:
        cfg: The full stage configuration, with a ``stage_cfg`` node.
        is_metadata: Whether the stage is a metadata stage.
        wrote_metadata: For metadata stages, whether this worker has finished writing the stage's metadata
            outputs. If ``False`` (e.g. for map-only workers of a map-reduce stage, which exit before the
            reduction finishes), the metadata directory is left alone so that the carried-over input
            ``codes.parquet`` cannot be mistaken for the stage's own output.

    Examples:
        We'll use a helper that lays out input and output directories and builds the matching config. Text
        files stand in for parquet files, as their contents are never parsed.

        >>> def make_cfg(root: Path, outputs: dict) -> DictConfig:
        ...     (root / "in").mkdir()
        ...     (root / "out").mkdir()
        ...     inputs = {
        ...         "data/train/": {"0.txt": "shard"},
        ...         "metadata/": {"codes.txt": "input codes", "dataset.txt": "dataset"},
        ...     }
        ...     _ = yaml_disk(inputs, root / "in")
        ...     if outputs:
        ...         _ = yaml_disk(outputs, root / "out")
        ...     return DictConfig({
        ...         "stage_cfg": {
        ...             "data_input_dir": str(root / "in" / "data"),
        ...             "metadata_input_dir": str(root / "in" / "metadata"),
        ...             "data_output_dir": str(root / "out" / "data"),
        ...             "metadata_output_dir": str(root / "out" / "metadata"),
        ...         }
        ...     })

        A metadata stage keeps the metadata it wrote and gains the data and the rest of the metadata:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     root = Path(tmpdir)
        ...     cfg = make_cfg(root, {"metadata/": {"codes.txt": "new codes"}})
        ...     carry_forward_unchanged_outputs(cfg, is_metadata=True)
        ...     print_directory(root / "out")
        ...     print((root / "out" / "metadata" / "codes.txt").read_text())
        ├── data
        │   └── train
        │       └── 0.txt
        └── metadata
            ├── codes.txt
            └── dataset.txt
        new codes

        A map-only worker of a metadata stage only carries the data forward:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     root = Path(tmpdir)
        ...     cfg = make_cfg(root, {})
        ...     carry_forward_unchanged_outputs(cfg, is_metadata=True, wrote_metadata=False)
        ...     print_directory(root / "out")
        └── data
            └── train
                └── 0.txt

        A data stage keeps the data it wrote and gains all of the metadata:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     root = Path(tmpdir)
        ...     cfg = make_cfg(root, {"data/train/": {"0.txt": "new shard"}})
        ...     carry_forward_unchanged_outputs(cfg, is_metadata=False)
        ...     print_directory(root / "out")
        ...     print((root / "out" / "data" / "train" / "0.txt").read_text())
        ├── data
        │   └── train
        │       └── 0.txt
        └── metadata
            ├── codes.txt
            └── dataset.txt
        new shard

        Without the output directory keys, nothing happens:

        >>> carry_forward_unchanged_outputs(DictConfig({"stage_cfg": {}}), is_metadata=False)
    """
    stage_cfg = cfg.stage_cfg
    data_output_dir = stage_cfg.get("data_output_dir", None)
    metadata_output_dir = stage_cfg.get("metadata_output_dir", None)
    if data_output_dir is None or metadata_output_dir is None:
        return

    data_input_dir = Path(stage_cfg.data_input_dir)
    metadata_input_dir = Path(stage_cfg.metadata_input_dir)
    data_output_dir = Path(data_output_dir)
    metadata_output_dir = Path(metadata_output_dir)

    if is_metadata:
        if data_input_dir.resolve() != data_output_dir.resolve():
            n = len(carry_forward(data_input_dir, data_output_dir, replace_existing=True))
            logger.info(f"Carried {n} data file(s) forward from {data_input_dir} to {data_output_dir}")
        if wrote_metadata and metadata_input_dir.resolve() != metadata_output_dir.resolve():
            n = len(carry_forward(metadata_input_dir, metadata_output_dir, replace_existing=False))
            logger.info(f"Carried {n} metadata file(s) forward to {metadata_output_dir}")
    elif metadata_input_dir.resolve() != metadata_output_dir.resolve():
        n = len(carry_forward(metadata_input_dir, metadata_output_dir, replace_existing=True))
        logger.info(
            f"Carried {n} metadata file(s) forward from {metadata_input_dir} to {metadata_output_dir}"
        )
