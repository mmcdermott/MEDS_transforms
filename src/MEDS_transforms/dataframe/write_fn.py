import os
from collections.abc import Callable
from pathlib import Path

import polars as pl

from .types import DF_T

WRITE_FN_T = Callable[[DF_T, Path], None]


def write_df(df: DF_T, out_fp: Path) -> None:
    """Atomically write a dataframe, either lazy or eager, to a parquet file.

    Content is staged at ``<out_fp>.tmp`` and then moved into place with
    ``os.replace`` so concurrent readers never observe a partial parquet file at
    the final path. The rename is atomic on POSIX and Windows as long as
    ``out_fp`` and its staging path share a filesystem.

    When an in-memory ``FrameRegistry`` is active (see
    :func:`MEDS_transforms.compute_modes.in_memory_mode`), the frame is stored in the registry
    keyed by ``out_fp`` instead — no parquet is written and no directory is created.
    """
    from ..compute_modes.in_memory import active_registry, in_memory_write

    if active_registry() is not None:
        in_memory_write(df, out_fp)
        return

    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    tmp_fp = out_fp.with_suffix(out_fp.suffix + ".tmp")
    try:
        df.write_parquet(tmp_fp, use_pyarrow=True)
        os.replace(tmp_fp, out_fp)
    except BaseException:
        tmp_fp.unlink(missing_ok=True)
        raise
