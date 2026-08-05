"""Regression tests for run-scoped `do_overwrite` (issue #399).

Parallel map stages divide work by having every worker walk the same `(input -> output)` list and letting
`rwlock_wrap`'s "output already exists, skip it" branch arbitrate. `do_overwrite=True` used to disable that
branch outright, with two consequences:

  1. Workers stopped sharing work — each redid the whole list rather than 1/N of it.
  2. Each redo `unlink()`ed an output a sibling had already finished, so anything reading those outputs
     concurrently (a reducer joining its own stage's map outputs, say) could hit a path that had just
     vanished — the `FileNotFoundError` reported in the issue.

These tests drive `rwlock_wrap`/`map_over` directly rather than through a live multirun, so the
interleaving that matters is exercised deterministically instead of being raced for.
"""

import subprocess
import threading
from pathlib import Path

import polars as pl
import pytest
from omegaconf import DictConfig

from MEDS_transforms.dataframe import read_df, write_df
from MEDS_transforms.mapreduce.mapper import map_over
from MEDS_transforms.mapreduce.rwlock import (
    RUN_MARKER_DIRNAME,
    _run_marker_fp,
    run_marker_dir,
    rwlock_wrap,
)
from MEDS_transforms.mapreduce.shard_iteration import InOutFilePair


@pytest.fixture
def shards(tmp_path: Path) -> list[InOutFilePair]:
    """Three input shards on disk, paired with the output paths a map stage would write."""

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    pairs = []
    for i in range(3):
        in_fp = in_dir / f"{i}.parquet"
        pl.DataFrame({"subject_id": [i], "value": [i]}).write_parquet(in_fp)
        pairs.append(InOutFilePair(in_fp, out_dir / f"{i}.parquet"))
    return pairs


def counting_map_fn(counts: list[Path]):
    """A map function that records which output each invocation was for."""

    def map_fn(df: pl.DataFrame) -> pl.DataFrame:
        counts.append(df)
        return df

    return map_fn


def test_workers_do_not_redo_each_others_work_under_do_overwrite(shards, tmp_path: Path):
    """Two workers walking the same shard list compute each output once between them, not once each.

    Worker shard orders are shuffled per worker, so a worker routinely reaches an entry a sibling has already
    finished. That is the case this test pins.
    """

    marker_dir = tmp_path / "markers" / "run-1"

    computed = []
    map_fn = counting_map_fn(computed)

    worker_a = list(shards)
    worker_b = list(reversed(shards))

    map_over(worker_a, map_fn, do_overwrite=True, marker_dir=marker_dir)
    n_after_a = len(computed)

    map_over(worker_b, map_fn, do_overwrite=True, marker_dir=marker_dir)

    assert n_after_a == 3, "Worker A should have computed all three shards."
    assert len(computed) == 3, (
        "Worker B recomputed outputs worker A had already finished; work is not being shared."
    )


def test_do_overwrite_false_still_skips_existing_outputs(shards, tmp_path: Path):
    """The default path is untouched: an existing output is a cache hit, run markers or not.

    This is how workers have always divided work, and it is the behavior `do_overwrite=True` should now
    match rather than defeat.
    """

    computed = []
    map_fn = counting_map_fn(computed)

    map_over(list(shards), map_fn)
    map_over(list(reversed(shards)), map_fn)

    assert len(computed) == 3
    assert not (tmp_path / "out" / ".logs").exists(), "No run bookkeeping without do_overwrite."


def test_without_run_scoping_workers_duplicate_work(shards, tmp_path: Path):
    """The pre-fix behavior, kept as a guard on what `marker_dir=None` still means.

    `rwlock_wrap` is public API with callers outside this repo, so omitting `marker_dir` must keep
    overwriting unconditionally rather than silently acquiring the new semantics.
    """

    computed = []
    map_fn = counting_map_fn(computed)

    map_over(list(shards), map_fn, do_overwrite=True)
    map_over(list(reversed(shards)), map_fn, do_overwrite=True)

    assert len(computed) == 6


def test_a_sibling_worker_never_unlinks_a_fresh_output(tmp_path: Path):
    """A revisit leaves the output in place, so a concurrent reader can never miss it.

    Without run scoping the revisit `unlink()`s first and only then recomputes, leaving a window in
    which the output does not exist. `observed_during_compute` samples exactly that window.
    """

    in_fp = tmp_path / "in.parquet"
    out_fp = tmp_path / "out.parquet"
    pl.DataFrame({"subject_id": [1], "value": [1]}).write_parquet(in_fp)

    observed_during_compute = []

    def compute_fn(df: pl.DataFrame) -> pl.DataFrame:
        observed_during_compute.append(out_fp.is_file())
        return df

    marker_dir = tmp_path / "markers" / "run-1"
    kwargs = {"do_overwrite": True, "marker_dir": marker_dir}

    assert rwlock_wrap(in_fp, out_fp, read_df, write_df, compute_fn, **kwargs) is True
    assert observed_during_compute == [False]  # nothing there yet on the first pass

    # A sibling worker reaching the same entry: no recompute, and crucially no unlink.
    assert rwlock_wrap(in_fp, out_fp, read_df, write_df, compute_fn, **kwargs) is False
    assert observed_during_compute == [False]
    assert out_fp.is_file()

    # Contrast: the unscoped path deletes the output and is mid-recompute while it is absent.
    assert rwlock_wrap(in_fp, out_fp, read_df, write_df, compute_fn, do_overwrite=True) is True
    assert observed_during_compute == [False, False], (
        "Unscoped do_overwrite should still have unlinked the output before recomputing."
    )


def test_concurrent_reader_never_sees_a_missing_output(tmp_path: Path):
    """The reported crash: a reducer reading its own stage's map outputs while a worker revisits them.

    The reducer polls until every map output is readable and then reads them. With run scoping the
    revisiting worker neither deletes nor rewrites those outputs, so the read is safe.
    """

    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    pairs = []
    for i in range(6):
        in_fp = in_dir / f"{i}.parquet"
        pl.DataFrame({"subject_id": [i], "value": [i]}).write_parquet(in_fp)
        pairs.append(InOutFilePair(in_fp, out_dir / f"{i}.parquet"))

    marker_dir = tmp_path / "markers" / "run-1"
    out_fps = [p.out_fp for p in pairs]

    # Worker 0 maps everything, then reduces; worker 1 keeps sweeping the list behind it.
    map_over(pairs, lambda df: df, do_overwrite=True, marker_dir=marker_dir)

    errors: list[Exception] = []
    stop = threading.Event()

    def resweeping_worker():
        try:
            while not stop.is_set():
                map_over(list(reversed(pairs)), lambda df: df, do_overwrite=True, marker_dir=marker_dir)
        except Exception as e:  # pragma: no cover - only on regression
            errors.append(e)

    def reducer():
        try:
            for _ in range(50):
                pl.concat([read_df(fp).lazy() for fp in out_fps]).collect()
        except Exception as e:  # pragma: no cover - only on regression
            errors.append(e)

    sweeper = threading.Thread(target=resweeping_worker, daemon=True)
    sweeper.start()
    try:
        reducer_thread = threading.Thread(target=reducer)
        reducer_thread.start()
        reducer_thread.join(timeout=60)
        assert not reducer_thread.is_alive(), "Reducer did not finish."
    finally:
        stop.set()
        sweeper.join(timeout=60)

    assert errors == [], f"Concurrent map/reduce raised: {errors}"


def test_run_marker_dir_from_stage_config(tmp_path: Path):
    """`run_marker_dir` reads the two keys a stage config actually carries."""

    cfg = DictConfig({"run_id": "deadbeef", "log_dir": str(tmp_path / "out" / ".logs")})
    assert run_marker_dir(cfg) == tmp_path / "out" / ".logs" / ".run_markers" / "deadbeef"

    # A stage invoked without the dispatcher (programmatic use, in-process tests) has no run_id, and
    # must keep the historical behavior rather than silently pointing markers somewhere arbitrary.
    assert run_marker_dir(DictConfig({"log_dir": str(tmp_path)})) is None


def test_output_finished_while_waiting_for_the_lock_is_not_recomputed(tmp_path: Path):
    """The existence check is repeated under the lock, closing the gap before acquiring it.

    A worker can pass the cheap pre-check, block on the lock while a sibling computes the same output,
    and then acquire it moments after that sibling released. Recomputing at that point would be pure
    waste, so the check runs again with the lock held. `out_fp_checker` stands in for the sibling here,
    since the interleaving is otherwise not reproducible on demand.
    """

    in_fp = tmp_path / "in.parquet"
    out_fp = tmp_path / "out.parquet"
    pl.DataFrame({"subject_id": [1], "value": [1]}).write_parquet(in_fp)

    marker_dir = tmp_path / "markers" / "run-1"
    marker_fp = _run_marker_fp(marker_dir, out_fp)

    checks = []

    def sibling_finishes_after_our_first_check(fp: Path) -> bool:
        checks.append(fp)
        if len(checks) == 1:
            return False  # our pre-check: nothing there yet
        marker_fp.parent.mkdir(parents=True, exist_ok=True)
        marker_fp.write_text(str(out_fp))
        return True

    def compute_fn(df: pl.DataFrame) -> pl.DataFrame:  # pragma: no cover - must not run
        raise AssertionError("Recomputed an output a sibling worker had already finished.")

    ran = rwlock_wrap(
        in_fp,
        out_fp,
        read_df,
        write_df,
        compute_fn,
        do_overwrite=True,
        out_fp_checker=sibling_finishes_after_our_first_check,
        marker_dir=marker_dir,
    )

    assert ran is False
    assert len(checks) == 2, "Expected a pre-check and a re-check under the lock."
    assert not (tmp_path / "out.parquet.lock").exists(), "Lock file should still be cleaned up."


@pytest.mark.parallelized
def test_multirun_workers_all_inherit_one_run_id(simple_static_MEDS, tmp_path: Path):
    """End-to-end: a real `--multirun` sweep stamps one `run_id` and shares it across every worker.

    The run scoping above is only worth anything if the workers of one invocation agree on the marker
    directory. They do because `MEDS_transform-stage` stamps `run_id` once, in the dispatcher process,
    before Hydra fans out — so this asserts on the artifact of that: exactly one marker directory, and
    one marker in it per output written.
    """

    output_dir = tmp_path / "output"
    pipeline_fp = tmp_path / "pipeline.yaml"
    pipeline_fp.write_text(
        f"input_dir: {simple_static_MEDS}\n"
        f"output_dir: {output_dir}\n"
        "stages:\n"
        "  - filter_subjects:\n"
        "      min_events_per_subject: 1\n"
    )

    result = subprocess.run(
        [
            "MEDS_transform-stage",
            str(pipeline_fp),
            "filter_subjects",
            "--multirun",
            "stage=filter_subjects",
            "do_overwrite=True",
            "worker=range(0,2)",
            "hydra/launcher=joblib",
        ],
        check=False,
        capture_output=True,
    )
    assert result.returncode == 0, (
        f"Stage failed.\nStdout:\n{result.stdout.decode()}\nStderr:\n{result.stderr.decode()}"
    )

    marker_roots = list(output_dir.glob(f"**/{RUN_MARKER_DIRNAME}"))
    assert len(marker_roots) == 1, f"Expected one marker root, got {marker_roots}"

    run_dirs = list(marker_roots[0].iterdir())
    assert len(run_dirs) == 1, f"Workers disagreed on the run id: {run_dirs}"

    n_outputs = len(list((output_dir / "data").rglob("*.parquet")))
    assert n_outputs > 0
    assert len(list(run_dirs[0].iterdir())) == n_outputs


def test_markers_stay_out_of_the_data_directory(shards, tmp_path: Path):
    """Bookkeeping lands under `log_dir`, never beside the MEDS outputs it describes."""

    marker_dir = tmp_path / "out" / ".logs" / ".run_markers" / "run-1"
    map_over(shards, lambda df: df, do_overwrite=True, marker_dir=marker_dir)

    out_dir = tmp_path / "out"
    data_files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    assert data_files == ["0.parquet", "1.parquet", "2.parquet"]
    assert len(list(marker_dir.iterdir())) == 3
