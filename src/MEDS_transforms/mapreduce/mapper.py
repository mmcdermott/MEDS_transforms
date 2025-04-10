"""Basic utilities for parallelizable map operations on sharded MEDS datasets with caching and locking."""

import logging
from pathlib import Path

from ..compute_modes import COMPUTE_FN_T
from ..dataframe import READ_FN_T, WRITE_FN_T, read_df, write_df
from .rwlock import rwlock_wrap
from .shard_iteration import InOutFilePair

logger = logging.getLogger(__name__)


def map_over(
    shards: list[InOutFilePair],
    map_fn: COMPUTE_FN_T,
    read_fn: READ_FN_T = read_df,
    write_fn: WRITE_FN_T = write_df,
    do_overwrite: bool = False,
) -> list[Path]:
    """Performs a map operation over a list of input file paths, writing the outputs to output files.

    Internally, for each operation, this uses `rwlock_wrap` to ensure that multiple competing workers do not
    attempt to compute over the same input/output pair at once. This means that this function can be run
    safely in parallel without subdividing the set of shards manually in advance.

    Args:
        shards: List of input-output file pairs to process.
        map_fn: Function to perform the mapping operation on the data. It should take one dataframe
            argument and return one dataframe.
        read_fn: Function to read data from the input file paths. If `None`, the default read function
            reads parquet files.
        write_fn: Function to write the mapped data to the output file path.
        do_overwrite: Should this overwrite an existing out file?

    Returns:
        The list of output files written by this operation.

    Examples:
        >>> def map_fn(df: pl.DataFrame) -> pl.DataFrame:
        ...     return df.with_columns(pl.col("A") * 2)
        >>> dfs = [pl.DataFrame({"A": [i, i + 1, i + 2], "B": [1, 2, 3]}) for i in range(3)]
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     in_fps = [Path(tmp) / f"input_{i}.parquet" for i in range(3)]
        ...     out_fps = [Path(tmp) / f"output_{i}.parquet" for i in range(3)]
        ...     for fp, df in zip(in_fps, dfs):
        ...         df.write_parquet(fp)
        ...     shards = list(zip(in_fps, out_fps))
        ...     out_fps = map_over(shards, map_fn)
        ...     for fp in out_fps:
        ...         print(fp.relative_to(tmp))
        ...         print(pl.read_parquet(fp))
        output_0.parquet
        shape: (3, 2)
        ┌─────┬─────┐
        │ A   ┆ B   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 0   ┆ 1   │
        │ 2   ┆ 2   │
        │ 4   ┆ 3   │
        └─────┴─────┘
        output_1.parquet
        shape: (3, 2)
        ┌─────┬─────┐
        │ A   ┆ B   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 2   ┆ 1   │
        │ 4   ┆ 2   │
        │ 6   ┆ 3   │
        └─────┴─────┘
        output_2.parquet
        shape: (3, 2)
        ┌─────┬─────┐
        │ A   ┆ B   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 4   ┆ 1   │
        │ 6   ┆ 2   │
        │ 8   ┆ 3   │
        └─────┴─────┘
    """

    all_out_fps = []
    for in_fp, out_fp in shards:
        logger.info(f"Processing {str(in_fp.resolve())} into {str(out_fp.resolve())}")
        rwlock_wrap(
            in_fp,
            out_fp,
            read_fn,
            write_fn,
            compute_fn=map_fn,
            do_overwrite=do_overwrite,
        )
        all_out_fps.append(out_fp)
    return all_out_fps
