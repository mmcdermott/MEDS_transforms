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

from MEDS_transforms.mapreduce.reducer import reduce_over


def _slow_mapper_write(df: pl.DataFrame, fp: Path, pre_write_delay: float) -> None:
    """Simulate a mapper whose parquet write is not atomic: create the file,
    hold for `pre_write_delay` seconds, then actually write the content.

    This models the real behavior of `df.write_parquet(fp)` where the file exists
    on disk before the parquet footer is flushed.
    """
    fp.touch()
    time.sleep(pre_write_delay)
    df.write_parquet(fp)


def _reduce_fn(*dfs: pl.DataFrame) -> pl.DataFrame:
    return pl.concat(dfs, how="vertical")


def test_reduce_over_waits_for_complete_parquet() -> None:
    """Reducer should wait for valid parquet, not just file existence.

    Currently fails with ``polars.exceptions.ComputeError`` because ``reduce_over``
    polls ``fp.is_file()`` and reads the mapper's partial parquet file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_fps = [tmp / f"in_{i}.parquet" for i in range(2)]
        out_fp = tmp / "out.parquet"

        df0 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        df1 = pl.DataFrame({"a": [5, 6], "b": [7, 8]})

        df0.write_parquet(in_fps[0])
        slow_writer = threading.Thread(target=_slow_mapper_write, args=(df1, in_fps[1], 0.3))
        slow_writer.start()
        try:
            time.sleep(0.02)  # let the touch() land
            reduce_over(
                in_fps=in_fps,
                out_fp=out_fp,
                read_fn=pl.read_parquet,
                write_fn=pl.DataFrame.write_parquet,
                reduce_fn=_reduce_fn,
                polling_time=0.005,
            )
        finally:
            slow_writer.join()

        result = pl.read_parquet(out_fp).sort("a")
        expected = pl.concat([df0, df1], how="vertical").sort("a")
        assert result.equals(expected), f"Reducer output differs:\n{result}\nvs expected:\n{expected}"
