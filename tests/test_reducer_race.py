"""Regression test for https://github.com/mmcdermott/MEDS_transforms/issues/373.

`reduce_over` polls for input readiness with `fp.is_file()`, which returns True
as soon as the mapper creates the output file — well before `df.write_parquet(...)`
has flushed a valid parquet footer. The reducer then reads a partial file and
raises `polars.exceptions.ComputeError: parquet: File out of specification`.
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from MEDS_transforms.dataframe import read_df, write_df
from MEDS_transforms.mapreduce.reducer import reduce_over

# Window during which the reducer must observe the empty-but-existing parquet
# file, after which the writer finishes and the race would no longer reproduce.
_RACE_WINDOW_SECONDS = 0.3


def _publish_before_complete_write(
    df: pl.DataFrame,
    fp: Path,
    touched: threading.Event,
    race_window: float,
) -> None:
    """Simulate a publish-before-complete-write interleaving.

    Creates ``fp`` (so ``is_file()`` returns True), signals ``touched``, then
    holds ``race_window`` seconds before writing real parquet content. This is
    a synthetic interleaving, not a literal model of ``df.write_parquet``, but
    it exposes the same contract violation: the reducer treats file existence
    as publication.
    """
    fp.touch()
    touched.set()
    time.sleep(race_window)
    write_df(df, fp)


def _reduce_fn(*dfs: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    return pl.concat(dfs, how="vertical")


def test_reduce_over_waits_for_complete_parquet() -> None:
    """Reducer should wait for valid parquet, not just file existence.

    Currently fails with ``polars.exceptions.ComputeError`` because ``reduce_over``
    polls ``fp.is_file()`` and reads the partial parquet file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_fps = [tmp / f"in_{i}.parquet" for i in range(2)]
        out_fp = tmp / "out.parquet"

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
            slow_writer.join()

        # ``read_df`` returns a LazyFrame; collect before comparing.
        result = read_df(out_fp).collect().sort("a")
        expected = pl.concat([df0, df1], how="vertical").sort("a")
        assert_frame_equal(result, expected)
