"""Keep each stage's output directory a complete, current MEDS dataset.

Every stage reads a MEDS dataset (``stage_cfg.data_input_dir`` and ``stage_cfg.metadata_input_dir``) and
writes one (``stage_cfg.data_output_dir`` and ``stage_cfg.metadata_output_dir``). A stage usually computes
only one half of its output: data stages write data shards, and metadata stages write ``codes.parquet``. The
helpers here finish the job:

- `prune_stale_outputs` deletes data shards that an earlier run of the stage wrote but the current run does
  not produce (for instance because the input's sharding changed).
- `carry_forward_unchanged_outputs` fills in whatever the stage did not compute from its input, so the next
  stage (or the user) can always read the output as a complete dataset. The stage launcher
  (``MEDS_transform-stage``) runs it once, after all of the stage's workers have finished.

Carried files are hard links where the filesystem allows it, so they cost no disk space or copy time, and
copies otherwise. Hard links (rather than symlinks) keep every stage directory valid on its own: deleting an
earlier stage's directory does not break a later one. Every output in MEDS-Transforms is written to a
temporary file that is then renamed into place, so a re-run that writes over a linked file replaces the link
rather than altering the earlier stage's copy.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from hydra.experimental.callback import Callback

if TYPE_CHECKING:
    from collections.abc import Iterable

    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _is_hidden(rel: Path) -> bool:
    return any(part.startswith(".") for part in rel.parts)


def _dataset_files(root: Path) -> dict[Path, Path]:
    """The non-hidden files under ``root``, keyed by their path relative to it."""
    if not root.is_dir():
        return {}
    files = {}
    for fp in sorted(root.rglob("*")):
        rel = fp.relative_to(root)
        if fp.is_file() and not _is_hidden(rel):
            files[rel] = fp
    return files


def _is_same_content(src: Path, dst: Path) -> bool:
    """Whether ``dst`` is already a link to, or an unmodified copy of, ``src``.

    Copies are made with `shutil.copy2`, which preserves modification times, so a copy whose size and
    modification time both match its source is taken to be current. This keeps repeated carry-overs cheap
    on filesystems without hard links.
    """
    if dst.samefile(src):
        return True
    src_stat, dst_stat = src.stat(), dst.stat()
    return src_stat.st_size == dst_stat.st_size and src_stat.st_mtime_ns == dst_stat.st_mtime_ns


def link_or_copy(src: Path, dst: Path) -> None:
    """Make ``dst`` a hard link to ``src``, or a copy of it if hard links are not possible.

    The link (or copy) is staged at a hidden temporary path next to ``dst`` and renamed into place, so an
    existing ``dst`` is replaced atomically and its old contents (which may be another stage's file, via a
    link) are never written through.

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

        If the final rename fails, the staged link is removed and the error is raised:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     src = Path(tmpdir) / "codes.parquet"
        ...     _ = src.write_text("original")
        ...     dst = Path(tmpdir) / "out" / "codes.parquet"
        ...     with patch("os.replace", side_effect=OSError("rename failed")):
        ...         try:
        ...             link_or_copy(src, dst)
        ...         except OSError as e:
        ...             print(e)
        ...     print(list((Path(tmpdir) / "out").iterdir()))
        rename failed
        []
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


def carry_forward(src_dir: Path, dst_dir: Path, mirror: bool = False) -> list[Path]:
    """Carry every file in ``src_dir`` into ``dst_dir``, keeping relative paths.

    Hidden files and directories (names starting with ``.``) are ignored on both sides, as they hold
    bookkeeping (logs, locks, in-progress temporary files), not dataset contents.

    Args:
        src_dir: The directory to carry forward. If it does not exist, it is treated as empty.
        dst_dir: The directory to carry it into.
        mirror: If ``False``, files already in ``dst_dir`` are left alone; use this when the stage may have
            written some of these files itself. If ``True``, ``dst_dir`` is made to match ``src_dir``
            exactly: files that differ from their source are replaced, and files with no source are
            deleted. Use this only when the stage never writes into ``dst_dir``, so anything there is a
            carry-over from an earlier run.

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

        With ``mirror=True``, files that differ from their source are replaced, files already linked to
        their source are left in place, and files whose source is gone are deleted:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     src = Path(tmpdir) / "in"
        ...     dst = Path(tmpdir) / "out"
        ...     src.mkdir()
        ...     _ = yaml_disk({"train/": {"0.txt": "a", "1.txt": "b", "2.txt": "c"}}, src)
        ...     _ = carry_forward(src, dst)
        ...     (src / "train" / "1.txt").unlink()
        ...     _ = (src / "train" / "1.txt").write_text("new b")
        ...     (src / "train" / "2.txt").unlink()
        ...     created = carry_forward(src, dst, mirror=True)
        ...     print([p.relative_to(dst).as_posix() for p in created])
        ...     print_directory(dst)
        ...     print((dst / "train" / "1.txt").read_text())
        ['train/1.txt']
        └── train
            ├── 0.txt
            └── 1.txt
        new b

        A missing source directory carries nothing:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     carry_forward(Path(tmpdir) / "missing", Path(tmpdir) / "out")
        []
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    src_files = _dataset_files(src_dir)

    if mirror:
        for rel, dst_fp in _dataset_files(dst_dir).items():
            if rel not in src_files:
                logger.info(f"Removing {dst_fp}, which is no longer in {src_dir}")
                dst_fp.unlink()

    created = []
    for rel, src_fp in src_files.items():
        dst_fp = dst_dir / rel
        if dst_fp.exists() and (not mirror or _is_same_content(src_fp, dst_fp)):
            continue

        link_or_copy(src_fp, dst_fp)
        created.append(dst_fp)

    return created


def prune_stale_outputs(output_dir: Path, keep: Iterable[Path], suffix: str = ".parquet") -> list[Path]:
    """Delete files in ``output_dir`` ending in ``suffix`` that are not in ``keep``.

    A stage that is re-run after its input changed (for instance, re-sharded) would otherwise leave behind
    the shards it no longer produces, and the next stage would read them as part of the dataset. Hidden
    files and directories, and files with other suffixes (locks, temporary files), are never touched.

    Args:
        output_dir: The stage's data output directory. It must belong to the stage alone.
        keep: The output files the current run produced.
        suffix: The suffix of the stage's output files.

    Returns:
        The files that were deleted.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     out = Path(tmpdir)
        ...     _ = yaml_disk(
        ...         {
        ...             "train/": {"0.txt": "new", "1.txt": "stale"},
        ...             "held_out/": {"0.txt": "new", "0.txt.lock": ""},
        ...             ".scratch/": {"partial.txt": "kept"},
        ...         },
        ...         out,
        ...     )
        ...     keep = [out / "train" / "0.txt", out / "held_out" / "0.txt"]
        ...     removed = prune_stale_outputs(out, keep, suffix=".txt")
        ...     print([p.relative_to(out).as_posix() for p in removed])
        ...     print_directory(out)
        ['train/1.txt']
        ├── .scratch
        │   └── partial.txt
        ├── held_out
        │   ├── 0.txt
        │   └── 0.txt.lock
        └── train
            └── 0.txt
    """
    output_dir = Path(output_dir)
    keep = {Path(fp).resolve() for fp in keep}
    removed = []
    for fp in _dataset_files(output_dir).values():
        if fp.name.endswith(suffix) and fp.resolve() not in keep:
            logger.info(f"Removing stale output {fp}")
            fp.unlink()
            removed.append(fp)
    return removed


def carry_forward_unchanged_outputs(cfg: DictConfig, stage_type: str, is_metadata: bool) -> None:
    """Fill in a stage's ``data/`` and ``metadata/`` output directories with whatever it did not produce.

    The stage launcher calls this once all of a stage's workers have finished. What gets carried depends on
    what the stage can write:

    - A map (data) stage writes every data shard and no metadata, so its ``metadata/`` output is made to
      mirror its ``metadata/`` input exactly.
    - A map-reduce (metadata) stage writes no data, so its ``data/`` output mirrors its ``data/`` input. It
      writes some metadata files (normally just ``codes.parquet``), so the remaining input metadata files
      (e.g. ``dataset.json`` and ``subject_splits.parquet``) are added without touching the stage's own.
    - A stage with a custom main function may write anything. If it wrote no data shards, its input data is
      carried over in full; if it wrote any, it is taken to have written all of them, and no input shard is
      added (an input shard that happens to share a name with none of the stage's would otherwise sneak in,
      duplicating subjects). Input metadata files are added where the stage wrote nothing at the same path.

    This is a no-op unless the stage config defines all four of ``data_input_dir``, ``metadata_input_dir``,
    ``data_output_dir`` and ``metadata_output_dir``. `PipelineConfig` always defines them; hand-built configs
    may not.

    Args:
        cfg: The full stage configuration, with a ``stage_cfg`` node.
        stage_type: The stage's `StageType` (``"main"``, ``"map"`` or ``"mapreduce"``).
        is_metadata: Whether the stage is a metadata stage.

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
        ...     carry_forward_unchanged_outputs(cfg, "mapreduce", is_metadata=True)
        ...     print_directory(root / "out")
        ...     print((root / "out" / "metadata" / "codes.txt").read_text())
        ├── data
        │   └── train
        │       └── 0.txt
        └── metadata
            ├── codes.txt
            └── dataset.txt
        new codes

        A data stage keeps the data it wrote, and its metadata is made to match its input, dropping metadata
        left over from an earlier run:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     root = Path(tmpdir)
        ...     cfg = make_cfg(
        ...         root, {"data/train/": {"0.txt": "new shard"}, "metadata/": {"old.txt": "stale"}}
        ...     )
        ...     carry_forward_unchanged_outputs(cfg, "map", is_metadata=False)
        ...     print_directory(root / "out")
        ...     print((root / "out" / "data" / "train" / "0.txt").read_text())
        ├── data
        │   └── train
        │       └── 0.txt
        └── metadata
            ├── codes.txt
            └── dataset.txt
        new shard

        A stage with a custom main function keeps everything it wrote, in either half. If it wrote no data,
        it gains all of its input data:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     root = Path(tmpdir)
        ...     cfg = make_cfg(root, {"metadata/": {"codes.txt": "custom codes"}})
        ...     carry_forward_unchanged_outputs(cfg, "main", is_metadata=False)
        ...     print_directory(root / "out")
        ...     print((root / "out" / "metadata" / "codes.txt").read_text())
        ├── data
        │   └── train
        │       └── 0.txt
        └── metadata
            ├── codes.txt
            └── dataset.txt
        custom codes

        If it wrote any data (here, a re-sharding into differently named shards), no input data is added:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     root = Path(tmpdir)
        ...     cfg = make_cfg(root, {"data/train/": {"a.txt": "resharded"}})
        ...     carry_forward_unchanged_outputs(cfg, "main", is_metadata=False)
        ...     print_directory(root / "out")
        ├── data
        │   └── train
        │       └── a.txt
        └── metadata
            ├── codes.txt
            └── dataset.txt

        Without the directory keys, nothing happens:

        >>> carry_forward_unchanged_outputs(DictConfig({"stage_cfg": {}}), "map", is_metadata=False)
    """
    stage_cfg = cfg.stage_cfg
    keys = ("data_input_dir", "metadata_input_dir", "data_output_dir", "metadata_output_dir")
    dirs = {k: stage_cfg.get(k, None) for k in keys}
    if any(v is None for v in dirs.values()):
        return
    dirs = {k: Path(v) for k, v in dirs.items()}

    if stage_type == "main":
        data_mode = "mirror" if not _dataset_files(dirs["data_output_dir"]) else None
        metadata_mode = "fill"
    elif is_metadata:
        data_mode, metadata_mode = "mirror", "fill"
    else:
        data_mode, metadata_mode = None, "mirror"

    for half, mode in (("data", data_mode), ("metadata", metadata_mode)):
        src, dst = dirs[f"{half}_input_dir"], dirs[f"{half}_output_dir"]
        if mode is None or (src.exists() and dst.exists() and src.samefile(dst)):
            continue
        n = len(carry_forward(src, dst, mirror=(mode == "mirror")))
        logger.info(f"Carried {n} {half} file(s) forward from {src} to {dst}")


class CaptureStageConfig(Callback):
    """A Hydra callback that keeps the stage launcher's composed config for use after Hydra returns.

    The stage launcher (``MEDS_transform-stage``) runs `carry_forward_unchanged_outputs` once, after Hydra has
    run every worker. With ``--multirun``, the workers' configs live in the worker processes (or Slurm jobs),
    so the launcher records the config Hydra composes in its own process when the run starts. Sweep
    parameters (such as ``worker``) are left at their defaults in that config, which doesn't matter here:
    the directories it resolves are the same for every worker.

    Examples:
        >>> cfg = DictConfig({"stage_cfg": {"output_dir": "out"}})
        >>> CaptureStageConfig().on_multirun_start(cfg)
        >>> CaptureStageConfig.config is cfg
        True
        >>> CaptureStageConfig().on_run_start(DictConfig({}))
        >>> CaptureStageConfig.config
        {}
        >>> CaptureStageConfig.config = None
    """

    config: ClassVar[DictConfig | None] = None

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        CaptureStageConfig.config = config

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        CaptureStageConfig.config = config
