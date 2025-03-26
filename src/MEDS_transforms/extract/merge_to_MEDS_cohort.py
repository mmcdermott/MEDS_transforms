#!/usr/bin/env python
import logging
from functools import partial
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig, OmegaConf

from MEDS_transforms.extract import CONFIG_YAML
from MEDS_transforms.mapreduce import map_over, shard_iterator_by_shard_map

logger = logging.getLogger(__name__)


def merge_subdirs_and_sort(
    sp_dir: Path,
    event_subsets: list[str],
    unique_by: list[str] | str | None,
    additional_sort_by: list[str] | None = None,
) -> pl.LazyFrame:
    """This function reads all parquet files in subdirs of `sp_dir` and merges them into a single dataframe.

    Args:
        sp_dir: The directory containing the subdirs with parquet files to be merged.
        event_subsets: The list of event table paths passed to maintain the order in event_configs.yaml
            while merging the events.
        unique_by: The list of columns that should be ensured to be unique after the dataframes are merged. If
            `None`, this is ignored. If `*`, all columns are used. If a list of strings, only the columns in
            the list are used. If a column is not found in the dataframe, it is omitted from the unique-by, a
            warning is logged, but an error is *not* raised. Which rows are retained if the uniqeu-by columns
            are not all columns is not guaranteed, but is also *not* random, so this may have statistical
            implications.
        additional_sort_by: Additional columns to sort by, in addition to the default sorting by subject ID
            and time. If `None`, only subject ID and time are used. If a list of strings, these
            columns are used in addition to the default sorting. If a column is not found in the dataframe, it
            is omitted from the sort-by, a warning is logged, but an error is *not* raised. This functionality
            is useful both for deterministic testing and in cases where a data owner wants to impose
            intra-event measurement ordering in the data, though this is not recommended in general.

    Returns:
        A single dataframe containing all the data from the parquet files in the subdirs of `sp_dir`. These
        files will be concatenated diagonally, taking the union of all rows in all dataframes and all unique
        columns in all dataframes to form the merged output. The returned dataframe will be made unique by the
        columns specified in `unique_by` and sorted by first subject ID, then time, then all columns in
        `additional_sort_by`, if any.

    Raises:
        FileNotFoundError: If no parquet files are found in the subdirs of `sp_dir`.
        ValueError: If `unique_by` is not `None`, `*`, or a list of strings

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> df1 = pl.DataFrame({"subject_id": [1, 2], "time": [10, 20], "code": ["A", "B"]})
        >>> df2 = pl.DataFrame({
        ...     "subject_id":      [1,   1,    3],
        ...     "time":       [2,   1,    8],
        ...     "code":            ["C", "D",  "E"],
        ...     "numeric_value": [None, 2.0, None],
        ... })
        >>> df3 = pl.DataFrame({
        ...     "subject_id":      [1,   1,    3],
        ...     "time":       [2,   2,    8],
        ...     "code":            ["C", "D",  "E"],
        ...     "numeric_value": [6.2, 2.0, None],
        ... })
        >>> with TemporaryDirectory() as tmpdir:
        ...     sp_dir = Path(tmpdir)
        ...     merge_subdirs_and_sort(sp_dir, event_subsets=[], unique_by=None)
        Traceback (most recent call last):
            ...
        FileNotFoundError: No parquet files found in ...
        >>> with TemporaryDirectory() as tmpdir:
        ...     sp_dir = Path(tmpdir)
        ...     (sp_dir / "subdir1").mkdir()
        ...     df1.write_parquet(sp_dir / "subdir1" / "file1.parquet")
        ...     df2.write_parquet(sp_dir / "subdir1" / "file2.parquet")
        ...     (sp_dir / "subdir2").mkdir()
        ...     df3.write_parquet(sp_dir / "subdir2" / "df.parquet")
        ...     merge_subdirs_and_sort(
        ...         sp_dir,
        ...         event_subsets=["subdir1", "subdir2"],
        ...         unique_by=None,
        ...         additional_sort_by=["code", "numeric_value", "missing_col_will_not_error"]
        ...     ).collect()
        shape: (8, 4)
        ┌────────────┬──────┬──────┬───────────────┐
        │ subject_id ┆ time ┆ code ┆ numeric_value │
        │ ---        ┆ ---  ┆ ---  ┆ ---           │
        │ i64        ┆ i64  ┆ str  ┆ f64           │
        ╞════════════╪══════╪══════╪═══════════════╡
        │ 1          ┆ 1    ┆ D    ┆ 2.0           │
        │ 1          ┆ 2    ┆ C    ┆ null          │
        │ 1          ┆ 2    ┆ C    ┆ 6.2           │
        │ 1          ┆ 2    ┆ D    ┆ 2.0           │
        │ 1          ┆ 10   ┆ A    ┆ null          │
        │ 2          ┆ 20   ┆ B    ┆ null          │
        │ 3          ┆ 8    ┆ E    ┆ null          │
        │ 3          ┆ 8    ┆ E    ┆ null          │
        └────────────┴──────┴──────┴───────────────┘
        >>> with TemporaryDirectory() as tmpdir:
        ...     sp_dir = Path(tmpdir)
        ...     (sp_dir / "subdir1").mkdir()
        ...     df1.write_parquet(sp_dir / "subdir1" / "file1.parquet")
        ...     df2.write_parquet(sp_dir / "subdir1" / "file2.parquet")
        ...     (sp_dir / "subdir2").mkdir()
        ...     df3.write_parquet(sp_dir / "subdir2" / "df.parquet")
        ...     merge_subdirs_and_sort(
        ...         sp_dir,
        ...         event_subsets=["subdir1", "subdir2"],
        ...         unique_by="*",
        ...         additional_sort_by=["code", "numeric_value"]
        ...     ).collect()
        shape: (7, 4)
        ┌────────────┬──────┬──────┬───────────────┐
        │ subject_id ┆ time ┆ code ┆ numeric_value │
        │ ---        ┆ ---  ┆ ---  ┆ ---           │
        │ i64        ┆ i64  ┆ str  ┆ f64           │
        ╞════════════╪══════╪══════╪═══════════════╡
        │ 1          ┆ 1    ┆ D    ┆ 2.0           │
        │ 1          ┆ 2    ┆ C    ┆ null          │
        │ 1          ┆ 2    ┆ C    ┆ 6.2           │
        │ 1          ┆ 2    ┆ D    ┆ 2.0           │
        │ 1          ┆ 10   ┆ A    ┆ null          │
        │ 2          ┆ 20   ┆ B    ┆ null          │
        │ 3          ┆ 8    ┆ E    ┆ null          │
        └────────────┴──────┴──────┴───────────────┘
        >>> with TemporaryDirectory() as tmpdir:
        ...     sp_dir = Path(tmpdir)
        ...     (sp_dir / "subdir1").mkdir()
        ...     df1.write_parquet(sp_dir / "subdir1" / "file1.parquet")
        ...     df2.write_parquet(sp_dir / "subdir1" / "file2.parquet")
        ...     (sp_dir / "subdir2").mkdir()
        ...     df3.write_parquet(sp_dir / "subdir2" / "df.parquet")
        ...     # We just display the subject ID, time, and code columns as the numeric value column
        ...     # is not guaranteed to be deterministic in the output given some rows will be dropped due to
        ...     # the unique-by constraint.
        ...     merge_subdirs_and_sort(
        ...         sp_dir,
        ...         event_subsets=["subdir1", "subdir2"],
        ...         unique_by=["subject_id", "time", "code", "missing_col_will_not_error"],
        ...         additional_sort_by=["code", "numeric_value"]
        ...     ).select("subject_id", "time", "code").collect()
        shape: (6, 3)
        ┌────────────┬──────┬──────┐
        │ subject_id ┆ time ┆ code │
        │ ---        ┆ ---  ┆ ---  │
        │ i64        ┆ i64  ┆ str  │
        ╞════════════╪══════╪══════╡
        │ 1          ┆ 1    ┆ D    │
        │ 1          ┆ 2    ┆ C    │
        │ 1          ┆ 2    ┆ D    │
        │ 1          ┆ 10   ┆ A    │
        │ 2          ┆ 20   ┆ B    │
        │ 3          ┆ 8    ┆ E    │
        └────────────┴──────┴──────┘
        >>> with TemporaryDirectory() as tmpdir:
        ...     sp_dir = Path(tmpdir)
        ...     (sp_dir / "subdir1").mkdir()
        ...     df1.write_parquet(sp_dir / "subdir1" / "file1.parquet")
        ...     df2.write_parquet(sp_dir / "subdir1" / "file2.parquet")
        ...     (sp_dir / "subdir2").mkdir()
        ...     df3.write_parquet(sp_dir / "subdir2" / "df.parquet")
        ...     # We just display the subject ID, time, and code columns as the numeric value column
        ...     # is not guaranteed to be deterministic in the output given some rows will be dropped due to
        ...     # the unique-by constraint.
        ...     merge_subdirs_and_sort(
        ...         sp_dir,
        ...         event_subsets=["subdir1", "subdir2"],
        ...         unique_by=352.2, # This will error
        ...     )
        Traceback (most recent call last):
            ...
        ValueError: Invalid unique_by value: 352.2
        >>> with TemporaryDirectory() as tmpdir:
        ...     sp_dir = Path(tmpdir)
        ...     (sp_dir / "subdir1").mkdir()
        ...     df1.write_parquet(sp_dir / "subdir1" / "file1.parquet")
        ...     df2.write_parquet(sp_dir / "subdir1" / "file2.parquet")
        ...     (sp_dir / "subdir2").mkdir()
        ...     df3.write_parquet(sp_dir / "subdir2" / "df.parquet")
        ...     # We just display the subject ID, time, and code columns as the numeric value column
        ...     # is not guaranteed to be deterministic in the output given some rows will be dropped due to
        ...     # the unique-by constraint.
        ...     merge_subdirs_and_sort(
        ...         sp_dir,
        ...         event_subsets=["subdir1", "subdir2", "subdir3", "this is missing so will error"],
        ...         unique_by=None,
        ...     )
        Traceback (most recent call last):
            ...
        RuntimeError: Number of found subsets (2) does not match number of subsets in event_config (4): ...
    """
    files_to_read = [fp for es in event_subsets for fp in (sp_dir / es).glob("*.parquet")]
    if not files_to_read:
        raise FileNotFoundError(f"No parquet files found in {sp_dir}/**/*.parquet.")

    if len(dirs_to_read := {fp.parent for fp in files_to_read}) != len(event_subsets):
        raise RuntimeError(
            "Number of found subsets ({}) does not match "
            "number of subsets in event_config ({}): {}".format(len(dirs_to_read), len(event_subsets), sp_dir)
        )

    file_strs = "\n".join(f"  - {str(fp.resolve())}" for fp in files_to_read)
    logger.info(f"Reading {len(files_to_read)} files:\n{file_strs}")

    dfs = [pl.scan_parquet(fp, glob=False) for fp in files_to_read]
    df = pl.concat(dfs, how="diagonal_relaxed")

    df_columns = set(df.collect_schema().names())

    match unique_by:
        case None:
            pass
        case "*":
            df = df.unique(maintain_order=True)
        case list() if len(unique_by) > 0 and all(isinstance(u, str) for u in unique_by):
            subset = []
            for u in unique_by:
                if u in df_columns:
                    subset.append(u)
                else:
                    logger.warning(f"Column {u} not found in dataframe. Omitting from unique-by subset.")
            df = df.unique(maintain_order=True, subset=subset)
        case _:
            raise ValueError(f"Invalid unique_by value: {unique_by}")

    sort_by = ["subject_id", "time"]
    if additional_sort_by is not None:
        for s in additional_sort_by:
            if s in df_columns:
                sort_by.append(s)
            else:
                logger.warning(f"Column {s} not found in dataframe. Omitting from sort-by list.")

    return df.sort(by=sort_by, maintain_order=True, multithreaded=False)


@hydra.main(version_base=None, config_path=str(CONFIG_YAML.parent), config_name=CONFIG_YAML.stem)
def main(cfg: DictConfig):
    """Merges the subject sub-sharded events into a single parquet file per subject shard.

    This function takes all dataframes (in parquet files) in any subdirs of the `cfg.stage_cfg.input_dir` and
    merges them into a single dataframe. All dataframes in the subdirs are assumed to be in the unnested, MEDS
    format, and cover the same group of subjects (specific to the shard being processed). The merged dataframe
    will also be sorted by subject ID and time.

    All arguments are specified through the command line into the `cfg` object through Hydra.

    The `cfg.stage_cfg` object is a special key that is imputed by OmegaConf to contain the stage-specific
    configuration arguments based on the global, pipeline-level configuration file. It cannot be overwritten
    directly on the command line, but can be overwritten implicitly by overwriting components of the
    `stage_configs.merge_to_MEDS_cohort` key.

    Args:
        stage_configs.merge_to_MEDS_cohort.unique_by: The list of columns that should be ensured to be unique
            after the dataframes are merged. Defaults to `"*"`, which means all columns are used.
        stage_configs.merge_to_MEDS_cohort.additional_sort_by: Additional columns to sort by, in addition to
            the default sorting by subject ID and time. Defaults to `None`, which means only subject ID
            and time are used.

    Returns:
        Writes the merged dataframes to the shard-specific output filepath in the `cfg.stage_cfg.output_dir`.
    """
    event_conversion_cfg = OmegaConf.load(cfg.event_conversion_config_fp)
    event_conversion_cfg.pop("subject_id_col", None)

    read_fn = partial(
        merge_subdirs_and_sort,
        event_subsets=list(event_conversion_cfg.keys()),
        unique_by=cfg.stage_cfg.get("unique_by", None),
        additional_sort_by=cfg.stage_cfg.get("additional_sort_by", None),
    )

    map_over(
        cfg,
        read_fn=read_fn,
        shard_iterator_fntr=shard_iterator_by_shard_map,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
