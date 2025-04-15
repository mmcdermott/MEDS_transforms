"""Basic code for a mapreduce stage."""

import logging
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import polars as pl
from meds import code_field, subject_id_field, subject_splits_filepath
from omegaconf import DictConfig

from ..compute_modes import (
    ANY_COMPUTE_FN_T,
    COMPUTE_FN_T,
    bind_compute_fn,
    is_match_revise,
    match_revise_fntr,
)
from ..dataframe import DF_T, READ_FN_T, WRITE_FN_T, read_and_filter_fntr, read_df, write_df
from .mapper import map_over
from .reducer import REDUCE_FN_T, reduce_over
from .shard_iteration import SHARD_ITR_FNTR_T, shard_iterator

logger = logging.getLogger(__name__)


def map_stage(
    cfg: DictConfig,
    map_fn: COMPUTE_FN_T,
    read_fn: READ_FN_T = read_df,
    write_fn: WRITE_FN_T = write_df,
    shard_iterator_fntr: SHARD_ITR_FNTR_T = shard_iterator,
) -> list[Path]:
    """Performs a mapping stage operation on shards produced by the shard iterator.

    Args:
        cfg: Configuration dictionary containing stage_cfg, input_dir, and other necessary parameters.
        map_fn: Function to apply to each shard.
        read_fn: Function to read data from the input file paths. Defaults to reading parquet files with
            polars.
        write_fn: Function to write the transformed data to the output file paths. Defaults to writing
            parquet files with polars.
        shard_iterator_fntr: Function to create the shard iterator. Defaults to the default shard iterator,
            which iterates over the parquet files in the data input directory within the `cfg.stage_cfg` in a
            pseudo-random, worker-dependent manner.

    Returns:
        List of output file paths that were written.

    Raises:
        FileNotFoundError: If train_only is True but subject split file doesn't exist.
        ValueError: If includes_only_train is True but train_only is False.

    Examples:
        >>> from meds_testing_helpers.dataset import MEDSDataset
        >>> D = MEDSDataset(root_dir=simple_static_MEDS)

    We'll show an example of this function using the `simple_static_MEDS` dataset provided as a pytest
    fixture and loaded in this doctest via our `conftest.py` file in the
    [`meds_testing_helpers`](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers) package.

    To see this data, let's inspect it, first on disk (which is merely the typical MEDS fashion):

        >>> print_directory_contents(simple_static_MEDS)
        ├── data
        │   ├── held_out
        │   │   └── 0.parquet
        │   ├── train
        │   │   ├── 0.parquet
        │   │   └── 1.parquet
        │   └── tuning
        │       └── 0.parquet
        └── metadata
            ├── codes.parquet
            ├── dataset.json
            └── subject_splits.parquet

    As well as its actual contents:

        >>> for k, df in D._pl_shards.items():
        ...     print(f"{k}:")
        ...     print(df)
        held_out/0:
        shape: (11, 4)
        ┌────────────┬─────────────────────┬───────────────────────┬───────────────┐
        │ subject_id ┆ time                ┆ code                  ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---                   ┆ ---           │
        │ i64        ┆ datetime[μs]        ┆ str                   ┆ f32           │
        ╞════════════╪═════════════════════╪═══════════════════════╪═══════════════╡
        │ 1500733    ┆ null                ┆ EYE_COLOR//BROWN      ┆ null          │
        │ 1500733    ┆ null                ┆ HEIGHT                ┆ 158.601318    │
        │ 1500733    ┆ 1986-07-20 00:00:00 ┆ DOB                   ┆ null          │
        │ 1500733    ┆ 2010-06-03 14:54:38 ┆ ADMISSION//ORTHOPEDIC ┆ null          │
        │ 1500733    ┆ 2010-06-03 14:54:38 ┆ HR                    ┆ 91.400002     │
        │ …          ┆ …                   ┆ …                     ┆ …             │
        │ 1500733    ┆ 2010-06-03 15:39:49 ┆ HR                    ┆ 84.400002     │
        │ 1500733    ┆ 2010-06-03 15:39:49 ┆ TEMP                  ┆ 100.300003    │
        │ 1500733    ┆ 2010-06-03 16:20:49 ┆ HR                    ┆ 90.099998     │
        │ 1500733    ┆ 2010-06-03 16:20:49 ┆ TEMP                  ┆ 100.099998    │
        │ 1500733    ┆ 2010-06-03 16:44:26 ┆ DISCHARGE             ┆ null          │
        └────────────┴─────────────────────┴───────────────────────┴───────────────┘
        train/0:
        shape: (30, 4)
        ┌────────────┬─────────────────────┬────────────────────┬───────────────┐
        │ subject_id ┆ time                ┆ code               ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---                ┆ ---           │
        │ i64        ┆ datetime[μs]        ┆ str                ┆ f32           │
        ╞════════════╪═════════════════════╪════════════════════╪═══════════════╡
        │ 239684     ┆ null                ┆ EYE_COLOR//BROWN   ┆ null          │
        │ 239684     ┆ null                ┆ HEIGHT             ┆ 175.271118    │
        │ 239684     ┆ 1980-12-28 00:00:00 ┆ DOB                ┆ null          │
        │ 239684     ┆ 2010-05-11 17:41:51 ┆ ADMISSION//CARDIAC ┆ null          │
        │ 239684     ┆ 2010-05-11 17:41:51 ┆ HR                 ┆ 102.599998    │
        │ …          ┆ …                   ┆ …                  ┆ …             │
        │ 1195293    ┆ 2010-06-20 20:24:44 ┆ HR                 ┆ 107.699997    │
        │ 1195293    ┆ 2010-06-20 20:24:44 ┆ TEMP               ┆ 100.0         │
        │ 1195293    ┆ 2010-06-20 20:41:33 ┆ HR                 ┆ 107.5         │
        │ 1195293    ┆ 2010-06-20 20:41:33 ┆ TEMP               ┆ 100.400002    │
        │ 1195293    ┆ 2010-06-20 20:50:04 ┆ DISCHARGE          ┆ null          │
        └────────────┴─────────────────────┴────────────────────┴───────────────┘
        train/1:
        shape: (14, 4)
        ┌────────────┬─────────────────────┬───────────────────────┬───────────────┐
        │ subject_id ┆ time                ┆ code                  ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---                   ┆ ---           │
        │ i64        ┆ datetime[μs]        ┆ str                   ┆ f32           │
        ╞════════════╪═════════════════════╪═══════════════════════╪═══════════════╡
        │ 68729      ┆ null                ┆ EYE_COLOR//HAZEL      ┆ null          │
        │ 68729      ┆ null                ┆ HEIGHT                ┆ 160.395309    │
        │ 68729      ┆ 1978-03-09 00:00:00 ┆ DOB                   ┆ null          │
        │ 68729      ┆ 2010-05-26 02:30:56 ┆ ADMISSION//PULMONARY  ┆ null          │
        │ 68729      ┆ 2010-05-26 02:30:56 ┆ HR                    ┆ 86.0          │
        │ …          ┆ …                   ┆ …                     ┆ …             │
        │ 814703     ┆ 1976-03-28 00:00:00 ┆ DOB                   ┆ null          │
        │ 814703     ┆ 2010-02-05 05:55:39 ┆ ADMISSION//ORTHOPEDIC ┆ null          │
        │ 814703     ┆ 2010-02-05 05:55:39 ┆ HR                    ┆ 170.199997    │
        │ 814703     ┆ 2010-02-05 05:55:39 ┆ TEMP                  ┆ 100.099998    │
        │ 814703     ┆ 2010-02-05 07:02:30 ┆ DISCHARGE             ┆ null          │
        └────────────┴─────────────────────┴───────────────────────┴───────────────┘
        tuning/0:
        shape: (7, 4)
        ┌────────────┬─────────────────────┬──────────────────────┬───────────────┐
        │ subject_id ┆ time                ┆ code                 ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---                  ┆ ---           │
        │ i64        ┆ datetime[μs]        ┆ str                  ┆ f32           │
        ╞════════════╪═════════════════════╪══════════════════════╪═══════════════╡
        │ 754281     ┆ null                ┆ EYE_COLOR//BROWN     ┆ null          │
        │ 754281     ┆ null                ┆ HEIGHT               ┆ 166.22261     │
        │ 754281     ┆ 1988-12-19 00:00:00 ┆ DOB                  ┆ null          │
        │ 754281     ┆ 2010-01-03 06:27:59 ┆ ADMISSION//PULMONARY ┆ null          │
        │ 754281     ┆ 2010-01-03 06:27:59 ┆ HR                   ┆ 142.0         │
        │ 754281     ┆ 2010-01-03 06:27:59 ┆ TEMP                 ┆ 99.800003     │
        │ 754281     ┆ 2010-01-03 08:22:13 ┆ DISCHARGE            ┆ null          │
        └────────────┴─────────────────────┴──────────────────────┴───────────────┘
        >>> D._pl_subject_splits
        shape: (6, 2)
        ┌────────────┬──────────┐
        │ subject_id ┆ split    │
        │ ---        ┆ ---      │
        │ i64        ┆ str      │
        ╞════════════╪══════════╡
        │ 239684     ┆ train    │
        │ 1195293    ┆ train    │
        │ 68729      ┆ train    │
        │ 814703     ┆ train    │
        │ 754281     ┆ tuning   │
        │ 1500733    ┆ held_out │
        └────────────┴──────────┘

    We'll also make a simple helper too to print the output for us

        >>> def profile_map_stage(test_dir: str, in_MEDS_dir: Path | None = None, **kwargs):
        ...     '''Makes a test config, adds the output directory, runs the mapping stage, & shows outputs'''
        ...     test_cfg = DictConfig(cfg)
        ...     test_cfg.stage_cfg.output_dir = Path(test_dir)
        ...     if in_MEDS_dir is not None:
        ...         test_cfg.input_dir = str(in_MEDS_dir)
        ...         test_cfg.stage_cfg.data_input_dir = str(in_MEDS_dir / "data")
        ...         test_cfg.stage_cfg.metadata_input_dir = str(in_MEDS_dir / "metadata")
        ...     out_fps = map_stage(cfg=test_cfg, **kwargs)
        ...     print("Output directory:")
        ...     print_directory_contents(test_cfg.stage_cfg.output_dir)
        ...     print("------------------")
        ...     print("Output files:")
        ...     print("------------------")
        ...     for fp in out_fps:
        ...         print(f"  - {fp.relative_to(test_cfg.stage_cfg.output_dir)}:")
        ...         print(pl.read_parquet(fp))

    To map over this data, we need a configuration file that will point our data input dir to this dataset.
    Normally, the stage configuration is handled automatically by the Stage objects and the `PipelineConfig`
    class, but we'll just fudge it here for the sake of the example. Note we haven't added an output dir yet
    -- that will be a temporary directory we'll create just before we run the stage.

        >>> cfg = DictConfig(
        ...     {
        ...         "worker": 0,
        ...         "do_overwrite": False,
        ...         "input_dir": str(simple_static_MEDS),
        ...         "stage_cfg": {
        ...             "data_input_dir": str(simple_static_MEDS / "data"),
        ...             "metadata_input_dir": str(simple_static_MEDS / "metadata"),
        ...             "output_dir": "???",
        ...         },
        ...     }
        ... )

    We'll also need a mapping function that will be applied to each shard. For this example, we'll just count
    the number of occurrences of each code:

        >>> def count_codes(df: pl.LazyFrame) -> pl.LazyFrame:
        ...     return df.group_by("code", maintain_order=True).agg(pl.len().alias("count"))

    Now we can run the mapping stage:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     profile_map_stage(tmpdir, map_fn=count_codes)
        Output directory:
        ├── held_out
        │   └── 0.parquet
        ├── train
        │   ├── 0.parquet
        │   └── 1.parquet
        └── tuning
            └── 0.parquet
        ------------------
        Output files:
        ------------------
          - held_out/0.parquet:
        shape: (7, 2)
        ┌───────────────────────┬───────┐
        │ code                  ┆ count │
        │ ---                   ┆ ---   │
        │ str                   ┆ u32   │
        ╞═══════════════════════╪═══════╡
        │ EYE_COLOR//BROWN      ┆ 1     │
        │ HEIGHT                ┆ 1     │
        │ DOB                   ┆ 1     │
        │ ADMISSION//ORTHOPEDIC ┆ 1     │
        │ HR                    ┆ 3     │
        │ TEMP                  ┆ 3     │
        │ DISCHARGE             ┆ 1     │
        └───────────────────────┴───────┘
          - tuning/0.parquet:
        shape: (7, 2)
        ┌──────────────────────┬───────┐
        │ code                 ┆ count │
        │ ---                  ┆ ---   │
        │ str                  ┆ u32   │
        ╞══════════════════════╪═══════╡
        │ EYE_COLOR//BROWN     ┆ 1     │
        │ HEIGHT               ┆ 1     │
        │ DOB                  ┆ 1     │
        │ ADMISSION//PULMONARY ┆ 1     │
        │ HR                   ┆ 1     │
        │ TEMP                 ┆ 1     │
        │ DISCHARGE            ┆ 1     │
        └──────────────────────┴───────┘
          - train/1.parquet:
        shape: (8, 2)
        ┌───────────────────────┬───────┐
        │ code                  ┆ count │
        │ ---                   ┆ ---   │
        │ str                   ┆ u32   │
        ╞═══════════════════════╪═══════╡
        │ EYE_COLOR//HAZEL      ┆ 2     │
        │ HEIGHT                ┆ 2     │
        │ DOB                   ┆ 2     │
        │ ADMISSION//PULMONARY  ┆ 1     │
        │ HR                    ┆ 2     │
        │ TEMP                  ┆ 2     │
        │ DISCHARGE             ┆ 2     │
        │ ADMISSION//ORTHOPEDIC ┆ 1     │
        └───────────────────────┴───────┘
          - train/0.parquet:
        shape: (8, 2)
        ┌────────────────────┬───────┐
        │ code               ┆ count │
        │ ---                ┆ ---   │
        │ str                ┆ u32   │
        ╞════════════════════╪═══════╡
        │ EYE_COLOR//BROWN   ┆ 1     │
        │ HEIGHT             ┆ 2     │
        │ DOB                ┆ 2     │
        │ ADMISSION//CARDIAC ┆ 2     │
        │ HR                 ┆ 10    │
        │ TEMP               ┆ 10    │
        │ DISCHARGE          ┆ 2     │
        │ EYE_COLOR//BLUE    ┆ 1     │
        └────────────────────┴───────┘

    If we set `train_only` to `True`, we can see that the held-out and tuning splits are not included in the
    output:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cfg.stage_cfg.train_only = True
        ...     profile_map_stage(tmpdir, map_fn=count_codes)
        ...     cfg.stage_cfg.train_only = False
        Output directory:
        └── train
            ├── 0.parquet
            └── 1.parquet
        ------------------
        Output files:
        ------------------
          - train/1.parquet:
        shape: (8, 2)
        ┌───────────────────────┬───────┐
        │ code                  ┆ count │
        │ ---                   ┆ ---   │
        │ str                   ┆ u32   │
        ╞═══════════════════════╪═══════╡
        │ EYE_COLOR//HAZEL      ┆ 2     │
        │ HEIGHT                ┆ 2     │
        │ DOB                   ┆ 2     │
        │ ADMISSION//PULMONARY  ┆ 1     │
        │ HR                    ┆ 2     │
        │ TEMP                  ┆ 2     │
        │ DISCHARGE             ┆ 2     │
        │ ADMISSION//ORTHOPEDIC ┆ 1     │
        └───────────────────────┴───────┘
          - train/0.parquet:
        shape: (8, 2)
        ┌────────────────────┬───────┐
        │ code               ┆ count │
        │ ---                ┆ ---   │
        │ str                ┆ u32   │
        ╞════════════════════╪═══════╡
        │ EYE_COLOR//BROWN   ┆ 1     │
        │ HEIGHT             ┆ 2     │
        │ DOB                ┆ 2     │
        │ ADMISSION//CARDIAC ┆ 2     │
        │ HR                 ┆ 10    │
        │ TEMP               ┆ 10    │
        │ DISCHARGE          ┆ 2     │
        │ EYE_COLOR//BLUE    ┆ 1     │
        └────────────────────┴───────┘

    Note that, here, the held-out and tuning shards outright are not included in the output, as files or
    otherwise -- this is because the `shard_iterator` function can tell by the prefix that this dataset is
    sharded by split. What if that weren't the case? To show that, we need to copy our MEDS dataset to a form
    not sharded by split. We'll use a simple helper to do that.

        >>> import shutil
        >>> def copy_MEDS_without_split_sharding(output_dir: Path):
        ...     '''Copy the MEDS dataset to a new location without split sharding.'''
        ...     for fp in Path(simple_static_MEDS).rglob("*.*"):
        ...         relative_path = fp.relative_to(simple_static_MEDS)
        ...         if relative_path.parts[0] == "data":
        ...             # Remove the {split}/ prefix
        ...             relative_path = Path("data/") / "_".join(relative_path.parts[1:])
        ...         out_fp = output_dir / relative_path
        ...         out_fp.parent.mkdir(parents=True, exist_ok=True)
        ...         shutil.copy(fp, out_fp)
        >>> with tempfile.TemporaryDirectory() as MEDS_dir, tempfile.TemporaryDirectory() as out_dir:
        ...     copy_MEDS_without_split_sharding(Path(MEDS_dir))
        ...     cfg.stage_cfg.train_only = True
        ...     profile_map_stage(out_dir, in_MEDS_dir=Path(MEDS_dir), map_fn=count_codes)
        ...     cfg.stage_cfg.train_only = False
        Output directory:
        ├── held_out_0.parquet
        ├── train_0.parquet
        ├── train_1.parquet
        └── tuning_0.parquet
        ------------------
        Output files:
        ------------------
          - train_1.parquet:
        shape: (8, 2)
        ┌───────────────────────┬───────┐
        │ code                  ┆ count │
        │ ---                   ┆ ---   │
        │ str                   ┆ u32   │
        ╞═══════════════════════╪═══════╡
        │ EYE_COLOR//HAZEL      ┆ 2     │
        │ HEIGHT                ┆ 2     │
        │ DOB                   ┆ 2     │
        │ ADMISSION//PULMONARY  ┆ 1     │
        │ HR                    ┆ 2     │
        │ TEMP                  ┆ 2     │
        │ DISCHARGE             ┆ 2     │
        │ ADMISSION//ORTHOPEDIC ┆ 1     │
        └───────────────────────┴───────┘
          - tuning_0.parquet:
        shape: (0, 2)
        ┌──────┬───────┐
        │ code ┆ count │
        │ ---  ┆ ---   │
        │ str  ┆ u32   │
        ╞══════╪═══════╡
        └──────┴───────┘
          - held_out_0.parquet:
        shape: (0, 2)
        ┌──────┬───────┐
        │ code ┆ count │
        │ ---  ┆ ---   │
        │ str  ┆ u32   │
        ╞══════╪═══════╡
        └──────┴───────┘
          - train_0.parquet:
        shape: (8, 2)
        ┌────────────────────┬───────┐
        │ code               ┆ count │
        │ ---                ┆ ---   │
        │ str                ┆ u32   │
        ╞════════════════════╪═══════╡
        │ EYE_COLOR//BROWN   ┆ 1     │
        │ HEIGHT             ┆ 2     │
        │ DOB                ┆ 2     │
        │ ADMISSION//CARDIAC ┆ 2     │
        │ HR                 ┆ 10    │
        │ TEMP               ┆ 10    │
        │ DISCHARGE          ┆ 2     │
        │ EYE_COLOR//BLUE    ┆ 1     │
        └────────────────────┴───────┘

    We can see that it keeps the files in this case (as it doesn't know they are only non-train files) but
    they are empty, as the non-train subjects have been removed. This is because the subject-split data is
    stored in the `metadata/subject_splits.parquet` file. If we remove this file as well, we'd get an error:

        >>> with tempfile.TemporaryDirectory() as MEDS_dir, tempfile.TemporaryDirectory() as out_dir:
        ...     copy_MEDS_without_split_sharding(Path(MEDS_dir))
        ...     (Path(MEDS_dir) / subject_splits_filepath).unlink()
        ...     cfg.stage_cfg.train_only = True
        ...     try:
        ...         profile_map_stage(out_dir, in_MEDS_dir=Path(MEDS_dir), map_fn=count_codes)
        ...     finally:
        ...         cfg.stage_cfg.train_only = False
        Traceback (most recent call last):
            ...
        FileNotFoundError: Train split requested, but shard prefixes can't be used and subject split file not
        found at /.../metadata/subject_splits.parquet.

    The examples so far have used the default shard iterator. You can read more about this in the
    `shard_iteration.py` file and its documentation, but we can also pass in our own shard iterators here, to
    show some more edge cases. For example, if we don't request train only, but the shard iterator asserts it
    is only returning train samples, that will cause an error:

        >>> def bad_shard_iterator(cfg: DictConfig) -> tuple[list[tuple[Path, Path]], bool]:
        ...     out_fps = []
        ...     for fp in Path(cfg.stage_cfg.data_input_dir).rglob("*.parquet"):
        ...         shard_name = fp.relative_to(cfg.stage_cfg.data_input_dir).as_posix()
        ...         if not shard_name.startswith("train/"):
        ...             # Skip the held-out and tuning splits as this is broken!
        ...             continue
        ...         out_fps.append((fp, cfg.stage_cfg.output_dir / shard_name))
        ...     return out_fps, True
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cfg.stage_cfg.train_only = False
        ...     profile_map_stage(tmpdir, map_fn=count_codes, shard_iterator_fntr=bad_shard_iterator)
        Traceback (most recent call last):
            ...
        ValueError: All splits should be used, but shard iterator is returning only train splits?!?
    """

    start = datetime.now(tz=UTC)

    train_only = cfg.stage_cfg.get("train_only", False)

    shards, includes_only_train = shard_iterator_fntr(cfg)

    if train_only:
        split_fp = Path(cfg.input_dir) / subject_splits_filepath
        if includes_only_train:
            logger.info(
                f"Processing train split only via shard prefix. Not filtering with {split_fp.resolve()!s}."
            )
        elif split_fp.exists():
            logger.info(f"Processing train split only by filtering read dfs via {split_fp.resolve()!s}")
            train_subjects = (
                pl.scan_parquet(split_fp)
                .filter(pl.col("split") == "train")
                .select(subject_id_field)
                .collect()[subject_id_field]
                .to_list()
            )
            read_fn = read_and_filter_fntr(pl.col("subject_id").is_in(train_subjects), read_fn)
        else:
            raise FileNotFoundError(
                f"Train split requested, but shard prefixes can't be used and "
                f"subject split file not found at {split_fp.resolve()!s}."
            )
    elif includes_only_train:
        raise ValueError("All splits should be used, but shard iterator is returning only train splits?!?")

    if is_match_revise(cfg.stage_cfg):
        map_fn = match_revise_fntr(cfg, cfg.stage_cfg, map_fn)
    else:
        map_fn = bind_compute_fn(cfg, cfg.stage_cfg, map_fn)

    all_out_fps = map_over(
        shards=shards,
        map_fn=map_fn,
        read_fn=read_fn,
        write_fn=write_fn,
        do_overwrite=cfg.do_overwrite,
    )
    logger.info(f"Finished mapping in {datetime.now(tz=UTC) - start}")
    return all_out_fps


def join_and_replace(new: DF_T, old: DF_T, join_cols: list[str]) -> DF_T:
    """Join two dataframes and replace the old columns with the new columns.

    Args:
        new: The new dataframe to join.
        old: The old dataframe to join.
        join_cols: The columns to join on.

    Returns:
        Adds the columns from `old` that are not in `new` into new, without replacing any values in `new` or
        changing the order of `new`, but while matching the join columns.

    Examples:
        >>> old = pl.DataFrame({"code": ["a", "b", "c"], "A": [1, 2, 3], "B": [4, 5, 6]})
        >>> new = pl.DataFrame({"code": ["c", "b", "d"], "A": [7, None, 9], "C": [10, 11, None]})
        >>> join_and_replace(new, old, ["code"])
        shape: (3, 4)
        ┌──────┬──────┬──────┬──────┐
        │ code ┆ A    ┆ C    ┆ B    │
        │ ---  ┆ ---  ┆ ---  ┆ ---  │
        │ str  ┆ i64  ┆ i64  ┆ i64  │
        ╞══════╪══════╪══════╪══════╡
        │ c    ┆ 7    ┆ 10   ┆ 6    │
        │ b    ┆ null ┆ 11   ┆ 5    │
        │ d    ┆ 9    ┆ null ┆ null │
        └──────┴──────┴──────┴──────┘
    """

    new_cols = new.collect_schema().names()
    old_cols = old.collect_schema().names()

    return new.join(
        old.drop(*[c for c in old_cols if c in set(new_cols) - set(join_cols)]),
        on=join_cols,
        how="left",
        coalesce=True,
    )


def mapreduce_stage(
    cfg: DictConfig,
    map_fn: ANY_COMPUTE_FN_T,
    reduce_fn: REDUCE_FN_T,
    merge_fn: REDUCE_FN_T | None = None,
    read_fn: READ_FN_T = read_df,
    write_fn: WRITE_FN_T = write_df,
    shard_iterator_fntr: SHARD_ITR_FNTR_T = shard_iterator,
):
    """Performs a map-stage over shards produced by the shard iterator, then reduces over those outputs.

    Args:
        cfg: Configuration dictionary containing stage_cfg, input_dir, and other necessary parameters.
        map_fn: Function to apply to each shard.
        reduce_fn: Function to reduce the mapped data down to a single output.
        merge_fn: Function to merge the reduced data with the original data. Defaults to None, which resolves
            to `join_and_replace`, joining by the code column and any code modifiers.
        read_fn: Function to read data from the input file paths. Defaults to reading parquet files with
            polars.
        write_fn: Function to write the transformed data to the output file paths. Defaults to writing
            parquet files with polars.
        shard_iterator_fntr: Function to create the shard iterator. Defaults to the default shard iterator,
            which iterates over the parquet files in the data input directory within the `cfg.stage_cfg` in a
            pseudo-random, worker-dependent manner.

    Examples:
        >>> from meds_testing_helpers.dataset import MEDSDataset
        >>> D = MEDSDataset(root_dir=simple_static_MEDS)

    We'll show an example of this function using the `simple_static_MEDS` dataset provided as a pytest fixture
    and loaded in this doctest via our `conftest.py` file in the
    [`meds_testing_helpers`](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers) package.

    To see this data, let's inspect it, first on disk (which is merely the typical MEDS fashion):

        >>> print_directory_contents(simple_static_MEDS)
        ├── data
        │   ├── held_out
        │   │   └── 0.parquet
        │   ├── train
        │   │   ├── 0.parquet
        │   │   └── 1.parquet
        │   └── tuning
        │       └── 0.parquet
        └── metadata
            ├── codes.parquet
            ├── dataset.json
            └── subject_splits.parquet

    As well as its actual contents:

        >>> for k, df in D._pl_shards.items():
        ...     print(f"{k}:")
        ...     print(df)
        held_out/0:
        shape: (11, 4)
        ┌────────────┬─────────────────────┬───────────────────────┬───────────────┐
        │ subject_id ┆ time                ┆ code                  ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---                   ┆ ---           │
        │ i64        ┆ datetime[μs]        ┆ str                   ┆ f32           │
        ╞════════════╪═════════════════════╪═══════════════════════╪═══════════════╡
        │ 1500733    ┆ null                ┆ EYE_COLOR//BROWN      ┆ null          │
        │ 1500733    ┆ null                ┆ HEIGHT                ┆ 158.601318    │
        │ 1500733    ┆ 1986-07-20 00:00:00 ┆ DOB                   ┆ null          │
        │ 1500733    ┆ 2010-06-03 14:54:38 ┆ ADMISSION//ORTHOPEDIC ┆ null          │
        │ 1500733    ┆ 2010-06-03 14:54:38 ┆ HR                    ┆ 91.400002     │
        │ …          ┆ …                   ┆ …                     ┆ …             │
        │ 1500733    ┆ 2010-06-03 15:39:49 ┆ HR                    ┆ 84.400002     │
        │ 1500733    ┆ 2010-06-03 15:39:49 ┆ TEMP                  ┆ 100.300003    │
        │ 1500733    ┆ 2010-06-03 16:20:49 ┆ HR                    ┆ 90.099998     │
        │ 1500733    ┆ 2010-06-03 16:20:49 ┆ TEMP                  ┆ 100.099998    │
        │ 1500733    ┆ 2010-06-03 16:44:26 ┆ DISCHARGE             ┆ null          │
        └────────────┴─────────────────────┴───────────────────────┴───────────────┘
        train/0:
        shape: (30, 4)
        ┌────────────┬─────────────────────┬────────────────────┬───────────────┐
        │ subject_id ┆ time                ┆ code               ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---                ┆ ---           │
        │ i64        ┆ datetime[μs]        ┆ str                ┆ f32           │
        ╞════════════╪═════════════════════╪════════════════════╪═══════════════╡
        │ 239684     ┆ null                ┆ EYE_COLOR//BROWN   ┆ null          │
        │ 239684     ┆ null                ┆ HEIGHT             ┆ 175.271118    │
        │ 239684     ┆ 1980-12-28 00:00:00 ┆ DOB                ┆ null          │
        │ 239684     ┆ 2010-05-11 17:41:51 ┆ ADMISSION//CARDIAC ┆ null          │
        │ 239684     ┆ 2010-05-11 17:41:51 ┆ HR                 ┆ 102.599998    │
        │ …          ┆ …                   ┆ …                  ┆ …             │
        │ 1195293    ┆ 2010-06-20 20:24:44 ┆ HR                 ┆ 107.699997    │
        │ 1195293    ┆ 2010-06-20 20:24:44 ┆ TEMP               ┆ 100.0         │
        │ 1195293    ┆ 2010-06-20 20:41:33 ┆ HR                 ┆ 107.5         │
        │ 1195293    ┆ 2010-06-20 20:41:33 ┆ TEMP               ┆ 100.400002    │
        │ 1195293    ┆ 2010-06-20 20:50:04 ┆ DISCHARGE          ┆ null          │
        └────────────┴─────────────────────┴────────────────────┴───────────────┘
        train/1:
        shape: (14, 4)
        ┌────────────┬─────────────────────┬───────────────────────┬───────────────┐
        │ subject_id ┆ time                ┆ code                  ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---                   ┆ ---           │
        │ i64        ┆ datetime[μs]        ┆ str                   ┆ f32           │
        ╞════════════╪═════════════════════╪═══════════════════════╪═══════════════╡
        │ 68729      ┆ null                ┆ EYE_COLOR//HAZEL      ┆ null          │
        │ 68729      ┆ null                ┆ HEIGHT                ┆ 160.395309    │
        │ 68729      ┆ 1978-03-09 00:00:00 ┆ DOB                   ┆ null          │
        │ 68729      ┆ 2010-05-26 02:30:56 ┆ ADMISSION//PULMONARY  ┆ null          │
        │ 68729      ┆ 2010-05-26 02:30:56 ┆ HR                    ┆ 86.0          │
        │ …          ┆ …                   ┆ …                     ┆ …             │
        │ 814703     ┆ 1976-03-28 00:00:00 ┆ DOB                   ┆ null          │
        │ 814703     ┆ 2010-02-05 05:55:39 ┆ ADMISSION//ORTHOPEDIC ┆ null          │
        │ 814703     ┆ 2010-02-05 05:55:39 ┆ HR                    ┆ 170.199997    │
        │ 814703     ┆ 2010-02-05 05:55:39 ┆ TEMP                  ┆ 100.099998    │
        │ 814703     ┆ 2010-02-05 07:02:30 ┆ DISCHARGE             ┆ null          │
        └────────────┴─────────────────────┴───────────────────────┴───────────────┘
        tuning/0:
        shape: (7, 4)
        ┌────────────┬─────────────────────┬──────────────────────┬───────────────┐
        │ subject_id ┆ time                ┆ code                 ┆ numeric_value │
        │ ---        ┆ ---                 ┆ ---                  ┆ ---           │
        │ i64        ┆ datetime[μs]        ┆ str                  ┆ f32           │
        ╞════════════╪═════════════════════╪══════════════════════╪═══════════════╡
        │ 754281     ┆ null                ┆ EYE_COLOR//BROWN     ┆ null          │
        │ 754281     ┆ null                ┆ HEIGHT               ┆ 166.22261     │
        │ 754281     ┆ 1988-12-19 00:00:00 ┆ DOB                  ┆ null          │
        │ 754281     ┆ 2010-01-03 06:27:59 ┆ ADMISSION//PULMONARY ┆ null          │
        │ 754281     ┆ 2010-01-03 06:27:59 ┆ HR                   ┆ 142.0         │
        │ 754281     ┆ 2010-01-03 06:27:59 ┆ TEMP                 ┆ 99.800003     │
        │ 754281     ┆ 2010-01-03 08:22:13 ┆ DISCHARGE            ┆ null          │
        └────────────┴─────────────────────┴──────────────────────┴───────────────┘
        >>> D._pl_subject_splits
        shape: (6, 2)
        ┌────────────┬──────────┐
        │ subject_id ┆ split    │
        │ ---        ┆ ---      │
        │ i64        ┆ str      │
        ╞════════════╪══════════╡
        │ 239684     ┆ train    │
        │ 1195293    ┆ train    │
        │ 68729      ┆ train    │
        │ 814703     ┆ train    │
        │ 754281     ┆ tuning   │
        │ 1500733    ┆ held_out │
        └────────────┴──────────┘

    For mapreduce stages, the input code metadata is also relevant:

        >>> D._pl_code_metadata
        shape: (5, 3)
        ┌──────────────────┬─────────────────────────────────┬──────────────────┐
        │ code             ┆ description                     ┆ parent_codes     │
        │ ---              ┆ ---                             ┆ ---              │
        │ str              ┆ str                             ┆ list[str]        │
        ╞══════════════════╪═════════════════════════════════╪══════════════════╡
        │ EYE_COLOR//BLUE  ┆ Blue Eyes. Less common than br… ┆ null             │
        │ EYE_COLOR//BROWN ┆ Brown Eyes. The most common ey… ┆ null             │
        │ EYE_COLOR//HAZEL ┆ Hazel eyes. These are uncommon  ┆ null             │
        │ HR               ┆ Heart Rate                      ┆ ["LOINC/8867-4"] │
        │ TEMP             ┆ Body Temperature                ┆ ["LOINC/8310-5"] │
        └──────────────────┴─────────────────────────────────┴──────────────────┘

    We'll also make a simple helper too to print the output for us

        >>> def profile_mapreduce_stage(test_dir: str, worker: int | None = None, **kwargs):
        ...     '''Makes a test config, adds the output directory, runs the mapping stage, & shows outputs'''
        ...     test_cfg = DictConfig(cfg)
        ...     test_cfg.stage_cfg.output_dir = Path(test_dir) / "data_output"
        ...     test_cfg.stage_cfg.reducer_output_dir = Path(test_dir) / "reducer_output"
        ...     if worker is not None:
        ...         test_cfg.worker = worker
        ...     mapreduce_stage(cfg=test_cfg, **kwargs)
        ...     print("Data output directory:")
        ...     print_directory_contents(test_cfg.stage_cfg.output_dir)
        ...     if test_cfg.stage_cfg.reducer_output_dir.exists():
        ...         print("Reducer output directory:")
        ...         print_directory_contents(test_cfg.stage_cfg.reducer_output_dir)
        ...     out_fp = test_cfg.stage_cfg.reducer_output_dir / "codes.parquet"
        ...     if out_fp.exists():
        ...         print("------------------")
        ...         print("Reduced Output:")
        ...         print("------------------")
        ...         print(pl.read_parquet(out_fp))
        ...     else:
        ...         print("No reduced output found.")

    To mapreduce over this data, we need a configuration file that will point our data input dir to this
    dataset. Normally, the stage configuration is handled automatically by the Stage objects and the
    `PipelineConfig` class, but we'll just fudge it here for the sake of the example. Note we haven't added an
    output dir yet -- that will be a temporary directory we'll create just before we run the stage.

        >>> cfg = DictConfig(
        ...     {
        ...         "worker": 0,
        ...         "polling_time": 0.01,
        ...         "do_overwrite": False,
        ...         "input_dir": str(simple_static_MEDS),
        ...         "stage_cfg": {
        ...             "data_input_dir": str(simple_static_MEDS / "data"),
        ...             "metadata_input_dir": str(simple_static_MEDS / "metadata"),
        ...             "output_dir": "???",
        ...             "reducer_output_dir": "???",
        ...         },
        ...     }
        ... )

    We'll also need a mapping function that will be applied to each shard. For this example, we'll just count
    the number of occurrences of each code:

        >>> def count_codes(df: pl.LazyFrame) -> pl.LazyFrame:
        ...     return df.group_by("code", maintain_order=True).agg(pl.len().alias("count"))

    For our reducer function, we'll just sum the counts of each code across all shards:

        >>> def sum_counts(*dfs: pl.LazyFrame) -> pl.LazyFrame:
        ...     return pl.concat(dfs).group_by("code", maintain_order=True).agg(pl.sum("count"))

    Now we can run the mapreduce stage:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     profile_mapreduce_stage(tmpdir, map_fn=count_codes, reduce_fn=sum_counts)
        Data output directory:
        ├── held_out
        │   └── 0.parquet
        ├── train
        │   ├── 0.parquet
        │   └── 1.parquet
        └── tuning
            └── 0.parquet
        Reducer output directory:
        └── codes.parquet
        ------------------
        Reduced Output:
        ------------------
        shape: (11, 4)
        ┌───────────────────────┬───────┬─────────────────────────────────┬──────────────────┐
        │ code                  ┆ count ┆ description                     ┆ parent_codes     │
        │ ---                   ┆ ---   ┆ ---                             ┆ ---              │
        │ str                   ┆ u8    ┆ str                             ┆ list[str]        │
        ╞═══════════════════════╪═══════╪═════════════════════════════════╪══════════════════╡
        │ EYE_COLOR//BROWN      ┆ 3     ┆ Brown Eyes. The most common ey… ┆ null             │
        │ HEIGHT                ┆ 6     ┆ null                            ┆ null             │
        │ DOB                   ┆ 6     ┆ null                            ┆ null             │
        │ ADMISSION//ORTHOPEDIC ┆ 2     ┆ null                            ┆ null             │
        │ HR                    ┆ 16    ┆ Heart Rate                      ┆ ["LOINC/8867-4"] │
        │ …                     ┆ …     ┆ …                               ┆ …                │
        │ DISCHARGE             ┆ 6     ┆ null                            ┆ null             │
        │ ADMISSION//PULMONARY  ┆ 2     ┆ null                            ┆ null             │
        │ EYE_COLOR//HAZEL      ┆ 2     ┆ Hazel eyes. These are uncommon  ┆ null             │
        │ ADMISSION//CARDIAC    ┆ 2     ┆ null                            ┆ null             │
        │ EYE_COLOR//BLUE       ┆ 1     ┆ Blue Eyes. Less common than br… ┆ null             │
        └───────────────────────┴───────┴─────────────────────────────────┴──────────────────┘

    If we set `train_only` to `True`, we can see that the held-out and tuning splits are not included in the
    output. This is generally desired for stages aggregating over codes, as you don't want to include held out
    data in your normalization computation

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cfg.stage_cfg.train_only = True
        ...     profile_mapreduce_stage(tmpdir, map_fn=count_codes, reduce_fn=sum_counts)
        ...     cfg.stage_cfg.train_only = False
        Data output directory:
        └── train
            ├── 0.parquet
            └── 1.parquet
        Reducer output directory:
        └── codes.parquet
        ------------------
        Reduced Output:
        ------------------
        shape: (11, 4)
        ┌───────────────────────┬───────┬─────────────────────────────────┬──────────────────┐
        │ code                  ┆ count ┆ description                     ┆ parent_codes     │
        │ ---                   ┆ ---   ┆ ---                             ┆ ---              │
        │ str                   ┆ u8    ┆ str                             ┆ list[str]        │
        ╞═══════════════════════╪═══════╪═════════════════════════════════╪══════════════════╡
        │ EYE_COLOR//HAZEL      ┆ 2     ┆ Hazel eyes. These are uncommon  ┆ null             │
        │ HEIGHT                ┆ 4     ┆ null                            ┆ null             │
        │ DOB                   ┆ 4     ┆ null                            ┆ null             │
        │ ADMISSION//PULMONARY  ┆ 1     ┆ null                            ┆ null             │
        │ HR                    ┆ 12    ┆ Heart Rate                      ┆ ["LOINC/8867-4"] │
        │ …                     ┆ …     ┆ …                               ┆ …                │
        │ DISCHARGE             ┆ 4     ┆ null                            ┆ null             │
        │ ADMISSION//ORTHOPEDIC ┆ 1     ┆ null                            ┆ null             │
        │ EYE_COLOR//BROWN      ┆ 1     ┆ Brown Eyes. The most common ey… ┆ null             │
        │ ADMISSION//CARDIAC    ┆ 2     ┆ null                            ┆ null             │
        │ EYE_COLOR//BLUE       ┆ 1     ┆ Blue Eyes. Less common than br… ┆ null             │
        └───────────────────────┴───────┴─────────────────────────────────┴──────────────────┘

    Lastly, note that mapreduce is special in that the reduction happens only in one worker -- so if the
    config has `worker` set to anything other than 0, it will exit after the mapping stage, and the reduction
    won't be completed:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     profile_mapreduce_stage(tmpdir, worker=1, map_fn=count_codes, reduce_fn=sum_counts)
        Data output directory:
        ├── held_out
        │   └── 0.parquet
        ├── train
        │   ├── 0.parquet
        │   └── 1.parquet
        └── tuning
            └── 0.parquet
        No reduced output found.
    """

    _out_fps = map_stage(
        cfg=cfg, map_fn=map_fn, read_fn=read_fn, write_fn=write_fn, shard_iterator_fntr=shard_iterator
    )

    if cfg.worker != 0:
        logger.info(f"Mapping completed. Exiting as am worker {cfg.worker}, not reducer (0).")
        return

    logger.info("Starting reduction process")
    start = datetime.now(tz=UTC)

    merge_fp = Path(cfg.stage_cfg.metadata_input_dir) / "codes.parquet"
    reduce_stage_out_fp = Path(cfg.stage_cfg.reducer_output_dir) / "codes.parquet"

    if merge_fn is None:
        join_cols = [code_field, *cfg.get("code_modifier_cols", [])]
        merge_fn = partial(join_and_replace, join_cols=join_cols)

    reduce_fn = bind_compute_fn(cfg, cfg.stage_cfg, reduce_fn)

    reduce_over(
        in_fps=_out_fps,
        out_fp=reduce_stage_out_fp,
        read_fn=read_fn,
        write_fn=write_fn,
        reduce_fn=reduce_fn,
        merge_fp=merge_fp,
        merge_fn=merge_fn,
        do_overwrite=cfg.do_overwrite,
        polling_time=cfg.polling_time,
    )
    logger.info(f"Finished reduction in {datetime.now(tz=UTC) - start}")
