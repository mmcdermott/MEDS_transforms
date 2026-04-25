"""Regression test for https://github.com/mmcdermott/MEDS_transforms/issues/373.

``reduce_over`` used to poll for input readiness with ``fp.is_file()``, which returns True as soon
as the mapper creates the output file — well before ``df.write_parquet(...)`` has flushed a valid
parquet footer. The reducer then read a partial file and raised ``polars.exceptions.ComputeError:
parquet: File out of specification``. These tests lock in the fix (atomic ``write_df`` publish plus
a completeness-aware readiness poll).
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from MEDS_transforms.dataframe import read_df, write_df
from MEDS_transforms.mapreduce.reducer import reduce_over

if TYPE_CHECKING:
    from pathlib import Path

# Window during which the reducer must observe the empty-but-existing parquet file. On a machine
# where this isn't enough, the whole test suite is already in trouble.
_RACE_WINDOW_SECONDS = 0.3


def _publish_before_complete_write(
    df: pl.DataFrame,
    fp: Path,
    touched: threading.Event,
    race_window: float,
) -> None:
    """Simulate a publish-before-complete-write interleaving.

    Creates ``fp`` (so ``is_file()`` returns True), signals ``touched``, then holds
    ``race_window`` seconds before writing real parquet content. This is a synthetic interleaving,
    not a literal model of ``df.write_parquet``, but it exposes the same contract violation: the
    reducer treats file existence as publication.
    """
    fp.touch()
    touched.set()
    time.sleep(race_window)
    write_df(df, fp)


def _reduce_fn(*dfs: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    return pl.concat(dfs, how="vertical")


def test_reduce_over_waits_for_complete_parquet(tmp_path: Path) -> None:
    """Reducer should wait for valid parquet, not just file existence."""
    in_fps = [tmp_path / f"in_{i}.parquet" for i in range(2)]
    out_fp = tmp_path / "out.parquet"

    df0 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df1 = pl.DataFrame({"a": [5, 6], "b": [7, 8]})

    write_df(df0, in_fps[0])

    touched = threading.Event()
    slow_writer = threading.Thread(
        target=_publish_before_complete_write,
        args=(df1, in_fps[1], touched, _RACE_WINDOW_SECONDS),
    )
    slow_writer.start()
    try:
        # Explicit handshake: proceed only once the partial file exists.
        assert touched.wait(timeout=5.0), "writer thread never created the partial file"
        reduce_over(
            in_fps=in_fps,
            out_fp=out_fp,
            read_fn=read_df,
            write_fn=write_df,
            reduce_fn=_reduce_fn,
            polling_time=0.005,
        )
    finally:
        slow_writer.join(timeout=5.0)

    # ``read_df`` returns a LazyFrame; collect before comparing.
    result = read_df(out_fp).collect().sort("a")
    expected = pl.concat([df0, df1], how="vertical").sort("a")
    # check_dtypes=False because ``reduce_over`` calls ``shrink_dtype`` on numeric columns.
    assert_frame_equal(result, expected, check_dtypes=False)


def test_reduce_over_times_out_on_permanently_invalid_input(tmp_path: Path) -> None:
    """A permanently invalid (empty) parquet should time out, not hang forever."""
    in_fps = [tmp_path / "ok.parquet", tmp_path / "broken.parquet"]
    out_fp = tmp_path / "out.parquet"

    write_df(pl.DataFrame({"a": [1]}), in_fps[0])
    in_fps[1].touch()  # exists but invalid parquet, never fixed

    with pytest.raises(TimeoutError, match=r"present but unreadable.*broken\.parquet") as excinfo:
        reduce_over(
            in_fps=in_fps,
            out_fp=out_fp,
            read_fn=read_df,
            write_fn=write_df,
            reduce_fn=_reduce_fn,
            polling_time=0.01,
            max_poll_time=0.3,
        )
    # The "missing" branch should NOT fire here — the file exists (just isn't valid parquet).
    assert "missing:" not in str(excinfo.value)


def test_reduce_over_times_out_on_missing_input(tmp_path: Path) -> None:
    """An input path that never appears should time out with a ``missing:`` message."""
    in_fps = [tmp_path / "ok.parquet", tmp_path / "never_created.parquet"]
    out_fp = tmp_path / "out.parquet"

    write_df(pl.DataFrame({"a": [1]}), in_fps[0])
    # Deliberately do not create in_fps[1].

    with pytest.raises(TimeoutError, match=r"missing:.*never_created\.parquet") as excinfo:
        reduce_over(
            in_fps=in_fps,
            out_fp=out_fp,
            read_fn=read_df,
            write_fn=write_df,
            reduce_fn=_reduce_fn,
            polling_time=0.01,
            max_poll_time=0.3,
        )
    # The "present but unreadable" branch should NOT fire here — the file never existed.
    assert "present but unreadable" not in str(excinfo.value)


def test_reduce_over_rejects_max_poll_time_not_larger_than_polling_time(tmp_path: Path) -> None:
    """``max_poll_time`` must exceed ``polling_time`` to avoid spurious timeouts on the first poll."""
    in_fps = [tmp_path / "a.parquet"]
    out_fp = tmp_path / "out.parquet"

    with pytest.raises(ValueError, match="must be greater than"):
        reduce_over(
            in_fps=in_fps,
            out_fp=out_fp,
            read_fn=read_df,
            write_fn=write_df,
            reduce_fn=_reduce_fn,
            polling_time=1.0,
            max_poll_time=1.0,
        )
