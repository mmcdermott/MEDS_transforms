"""Tests for the atomic-publish behavior of :func:`MEDS_transforms.dataframe.write_df`."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from MEDS_transforms.dataframe import write_df


def test_write_df_publishes_final_file(tmp_path: Path) -> None:
    """Happy path: the final file exists, the staging .tmp file does not."""
    out_fp = tmp_path / "out.parquet"
    write_df(pl.DataFrame({"a": [1, 2, 3]}), out_fp)

    assert out_fp.is_file()
    assert not list(tmp_path.glob("*.tmp"))


def test_write_df_creates_parent_directory(tmp_path: Path) -> None:
    """Missing parent dirs should be created by write_df (covers the mkdir branch)."""
    out_fp = tmp_path / "nested" / "deeper" / "out.parquet"
    assert not out_fp.parent.exists()
    write_df(pl.DataFrame({"a": [1]}), out_fp)
    assert out_fp.is_file()


def test_write_df_collects_lazyframe(tmp_path: Path) -> None:
    """A LazyFrame should be collected and written as parquet."""
    out_fp = tmp_path / "out.parquet"
    write_df(pl.LazyFrame({"a": [1, 2, 3]}), out_fp)
    assert out_fp.is_file()
    assert pl.read_parquet(out_fp)["a"].to_list() == [1, 2, 3]


def test_write_df_cleans_up_tmp_file_on_rename_failure(tmp_path: Path) -> None:
    """If ``os.replace`` raises after the tmp parquet is written, the tmp file is unlinked and the original
    exception propagates.

    The final ``out_fp`` should remain untouched.
    """
    out_fp = tmp_path / "out.parquet"

    with (
        patch("MEDS_transforms.dataframe.write_fn.os.replace", side_effect=OSError("boom")),
        pytest.raises(OSError, match="boom"),
    ):
        write_df(pl.DataFrame({"a": [1]}), out_fp)

    assert not list(tmp_path.glob("*.tmp")), "staging .tmp file should be cleaned up on failure"
    assert not out_fp.exists(), "final output must not appear when rename failed"


def test_write_df_cleans_up_tmp_file_on_write_failure(tmp_path: Path) -> None:
    """If ``df.write_parquet`` raises, any partial tmp file should be cleaned up too."""
    out_fp = tmp_path / "out.parquet"

    def _explode(self, fp, *args, **kwargs):
        # Simulate a mid-write failure: create the tmp file, then raise.
        Path(fp).write_bytes(b"partial")
        raise RuntimeError("write_parquet exploded")

    with (
        patch("polars.DataFrame.write_parquet", new=_explode),
        pytest.raises(RuntimeError, match="exploded"),
    ):
        write_df(pl.DataFrame({"a": [1]}), out_fp)

    assert not list(tmp_path.glob("*.tmp")), "staging .tmp file should be cleaned up on failure"
    assert not out_fp.exists()


def test_write_df_concurrent_writers_do_not_collide(tmp_path: Path) -> None:
    """Two writers of the same output stage to different files, so neither renames the other's away."""
    out_fp = tmp_path / "out.parquet"
    staged = []
    real_replace = os.replace

    def _record(src, dst):
        staged.append(Path(src).name)
        real_replace(src, dst)

    with patch("MEDS_transforms.dataframe.write_fn.os.replace", side_effect=_record):
        write_df(pl.DataFrame({"a": [1]}), out_fp)
        write_df(pl.DataFrame({"a": [2]}), out_fp)

    assert len(set(staged)) == 2
    assert pl.read_parquet(out_fp)["a"].to_list() == [2]
