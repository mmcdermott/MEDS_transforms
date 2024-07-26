#!/usr/bin/env python
import gzip
import warnings
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

import polars as pl
from loguru import logger


class SupportedFileFormats(StrEnum):
    """The supported file formats for dataframes we can read in, in priority order.

    The values of the enum are the allowed file suffix for the format.
    """

    PARQUET = ".parquet"
    CSV_GZ = ".csv.gz"
    CSV = ".csv"


def scan_csv_gz(fp: Path, **kwargs) -> pl.LazyFrame:
    with gzip.open(fp, mode="rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return pl.read_csv(f, **kwargs).lazy()


READERS = {
    SupportedFileFormats.PARQUET: pl.scan_parquet,
    SupportedFileFormats.CSV_GZ: scan_csv_gz,
    SupportedFileFormats.CSV: pl.scan_csv,
}


DF_T = TypeVar("DF_T")


def get_supported_fp(root_dir: Path, file_prefix: str | Path) -> tuple[Path, Callable[[Path], DF_T]]:
    """This function finds the best file path to read for a given root_dir and prefix.

    Args:
        root_dir: The root directory to search for files.
        file_prefix: The file prefix to search for.

    Raises:
        FileNotFoundError: If no files are found with the given prefix and an allowed suffix.

    Returns:
        The filepath with the matching prefix and the most preferred allowed suffix and an appropriate reader
        function for that file type.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, schema={"a": pl.UInt8, "b": pl.Int64})
        >>> with TemporaryDirectory() as tmpdir:
        ...     tmpdir = Path(tmpdir)
        ...     fp = tmpdir / "test.csv"
        ...     df.write_csv(fp)
        ...     fp, reader = get_supported_fp(tmpdir, "test")
        ...     print(str(fp.relative_to(tmpdir)), reader(fp).collect())
        test.csv shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        └─────┴─────┘
        >>> with TemporaryDirectory() as tmpdir:
        ...     tmpdir = Path(tmpdir)
        ...     fp = tmpdir / "test.parquet"
        ...     csv_fp = tmpdir / "test.csv"
        ...     df.write_parquet(fp)
        ...     df.write_csv(csv_fp)
        ...     fp, reader = get_supported_fp(tmpdir, "test")
        ...     print(str(fp.relative_to(tmpdir)), reader(fp).collect())
        test.parquet shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ u8  ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        └─────┴─────┘
        >>> import gzip
        >>> with TemporaryDirectory() as tmpdir:
        ...     tmpdir = Path(tmpdir)
        ...     fp = tmpdir / "test.csv.gz"
        ...     with gzip.open(fp, mode="wb") as f:
        ...         with warnings.catch_warnings():
        ...             warnings.simplefilter("ignore", category=UserWarning)
        ...             df.write_csv(f)
        ...     fp, reader = get_supported_fp(tmpdir, "test")
        ...     print(str(fp.relative_to(tmpdir)), reader(fp).collect())
        test.csv.gz shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        └─────┴─────┘
        >>> with TemporaryDirectory() as tmpdir:
        ...     tmpdir = Path(tmpdir)
        ...     fp = tmpdir / "test.json"
        ...     df.write_json(fp)
        ...     get_supported_fp(tmpdir, "test") # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        FileNotFoundError: No files found with prefix: test and allowed suffixes
            ['.parquet', '.csv.gz', '.csv']...
    """

    for suffix in list(SupportedFileFormats):
        fp = root_dir / f"{file_prefix}{suffix.value}"
        if fp.exists():
            logger.debug(f"Found file: {str(fp.resolve())}")
            return fp, READERS[suffix]
    raise FileNotFoundError(
        f"No files found with prefix: {file_prefix} and allowed suffixes "
        f"{[x.value for x in SupportedFileFormats]} in root dir {str(root_dir.resolve())}"
    )
