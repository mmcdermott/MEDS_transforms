"""Basic code for a mapreduce stage."""

import logging
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path

import polars as pl
from meds import code_field, subject_id_field, subject_splits_filepath
from omegaconf import DictConfig

from ..utils import write_lazyframe
from .compute_fn import ANY_COMPUTE_FN_T, COMPUTE_FN_T, bind_compute_fn
from .mapper import map_over
from .match_revise import is_match_revise, match_revise_fntr
from .read_fn import read_and_filter_fntr
from .reducer import REDUCE_FN_T, reduce_over
from .shard_iteration import SHARD_ITR_FNTR_T, shard_iterator
from .types import DF_T

logger = logging.getLogger(__name__)


def map_stage(
    cfg: DictConfig,
    map_fn: COMPUTE_FN_T,
    read_fn: Callable[[Path], DF_T] = partial(pl.scan_parquet, glob=False),
    write_fn: Callable[[DF_T, Path], None] = write_lazyframe,
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

        We'll show an example of this function using the `simple_static_MEDS` dataset provided as a pytest
        fixture and loaded in this doctest via our `conftest.py` file in the
        [`meds_testing_helpers`](https://github.com/Medical-Event-Data-Standard/meds_testing_helpers) package.

        To see this data, let's inspect it:

        >>> from meds_testing_helpers.dataset import MEDSDataset
        >>> D = MEDSDataset(root_dir=simple_static_MEDS)

        We can see how it is arranged on disk (which is merely the typical MEDS fashion):

        >>> from MEDS_transforms.stages.examples import pretty_list_directory
        >>> for line in pretty_list_directory(simple_static_MEDS): print(line)
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

        And inspect its contents directly:

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

        >>> def profile_mapping_stage(test_dir: str, in_MEDS_dir: Path | None = None, **kwargs):
        ...     '''Makes a test config, adds the output directory, runs the mapping stage, & shows outputs'''
        ...     test_cfg = DictConfig(cfg)
        ...     test_cfg.stage_cfg.output_dir = Path(test_dir)
        ...     if in_MEDS_dir is not None:
        ...         test_cfg.input_dir = str(in_MEDS_dir)
        ...         test_cfg.stage_cfg.data_input_dir = str(in_MEDS_dir / "data")
        ...         test_cfg.stage_cfg.metadata_input_dir = str(in_MEDS_dir / "metadata")
        ...     out_fps = map_stage(cfg=test_cfg, **kwargs)
        ...     print("Output directory:")
        ...     for line in pretty_list_directory(test_cfg.stage_cfg.output_dir):
        ...         print(line)
        ...     print("------------------")
        ...     print("Output files:")
        ...     print("------------------")
        ...     for fp in out_fps:
        ...         print(f"  - {fp.relative_to(test_cfg.stage_cfg.output_dir)}:")
        ...         print(pl.read_parquet(fp))

        To map over this data, we need a configuration file that will point our data input dir to this
        dataset. Normally, the stage configuration is handled automatically by the Stage objects and the
        `PipelineConfig` class, but we'll just fudge it here for the sake of the example. Note we haven't
        added an output dir yet -- that will be a temporary directory we'll create just before we run the
        stage.

        >>> cfg = DictConfig({
        ...     "worker": 0,
        ...     "do_overwrite": False,
        ...     "input_dir": str(simple_static_MEDS),
        ...     "stage_cfg": {
        ...         "data_input_dir": str(simple_static_MEDS / "data"),
        ...         "metadata_input_dir": str(simple_static_MEDS / "metadata"),
        ...         "output_dir": "???",
        ...     }
        ... })

        We'll also need a mapping function that will be applied to each shard. For this example, we'll just
        count the number of occurrences of each code:

        >>> def count_codes(df: pl.DataFrame) -> pl.DataFrame:
        ...     return df.group_by("code", maintain_order=True).agg(pl.len().alias("count"))

        Now we can run the mapping stage:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     profile_mapping_stage(tmpdir, map_fn=count_codes)
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

        If we set `train_only` to `True`, we can see that the held-out and tuning splits are not included in
        the output:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cfg.stage_cfg.train_only = True
        ...     profile_mapping_stage(tmpdir, map_fn=count_codes)
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
        sharded by split. What if that weren't the case? To show that, we need to copy our MEDS dataset to a
        form not sharded by split. We'll use a simple helper to do that.

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
        ...     profile_mapping_stage(out_dir, in_MEDS_dir=Path(MEDS_dir), map_fn=count_codes)
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
        stored in the `metadata/subject_splits.parquet` file. If we remove this file as well, we'd get an
        error:

        >>> with tempfile.TemporaryDirectory() as MEDS_dir, tempfile.TemporaryDirectory() as out_dir:
        ...     copy_MEDS_without_split_sharding(Path(MEDS_dir))
        ...     (Path(MEDS_dir) / subject_splits_filepath).unlink()
        ...     cfg.stage_cfg.train_only = True
        ...     try:
        ...         profile_mapping_stage(out_dir, in_MEDS_dir=Path(MEDS_dir), map_fn=count_codes)
        ...     finally:
        ...         cfg.stage_cfg.train_only = False
        Traceback (most recent call last):
            ...
        FileNotFoundError: Train split requested, but shard prefixes can't be used and subject split file not
        found at /.../metadata/subject_splits.parquet.


        The examples so far have used the default shard iterator. You can read more about this in the
        `shard_iteration.py` file and its documentation, but we can also pass in our own shard iterators here,
        to show some more edge cases. For example, if we don't request train only, but the shard iterator
        asserts it is only returning train samples, that will cause an error:

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
        ...     profile_mapping_stage(tmpdir, map_fn=count_codes, shard_iterator_fntr=bad_shard_iterator)
        Traceback (most recent call last):
            ...
        ValueError: All splits should be used, but shard iterator is returning only train splits?!?
    """

    start = datetime.now()

    train_only = cfg.stage_cfg.get("train_only", False)

    shards, includes_only_train = shard_iterator_fntr(cfg)

    if train_only:
        split_fp = Path(cfg.input_dir) / subject_splits_filepath
        if includes_only_train:
            logger.info(
                f"Processing train split only via shard prefix. Not filtering with {str(split_fp.resolve())}."
            )
        elif split_fp.exists():
            logger.info(f"Processing train split only by filtering read dfs via {str(split_fp.resolve())}")
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
                f"subject split file not found at {str(split_fp.resolve())}."
            )
    elif includes_only_train:
        raise ValueError("All splits should be used, but shard iterator is returning only train splits?!?")

    if is_match_revise(cfg.stage_cfg):
        map_fn = match_revise_fntr(cfg, cfg.stage_cfg, map_fn)
    else:
        map_fn = bind_compute_fn(cfg, cfg.stage_cfg, map_fn)

    all_out_fps = map_over(
        shards=shards,
        read_fn=read_fn,
        write_fn=write_fn,
        map_fn=map_fn,
        do_overwrite=cfg.do_overwrite,
    )
    logger.info(f"Finished mapping in {datetime.now() - start}")
    return all_out_fps


def join_and_replace(new: pl.DataFrame, old: pl.DataFrame, join_cols: list[str]) -> pl.DataFrame:
    """Join two dataframes and replace the old columns with the new columns."""
    return new.join(
        old.drop(*[c for c in old.columns if c in set(new.columns) - set(join_cols)]),
        on=join_cols,
        how="left",
        coalesce=True,
    )


def mapreduce_stage(
    cfg: DictConfig,
    map_fn: ANY_COMPUTE_FN_T,
    reduce_fn: REDUCE_FN_T,
    merge_fn: REDUCE_FN_T | None = None,
    read_fn: Callable[[Path], DF_T] = partial(pl.scan_parquet, glob=False),
    write_fn: Callable[[DF_T, Path], None] = write_lazyframe,
    shard_iterator_fntr: SHARD_ITR_FNTR_T = shard_iterator,
):

    map_stage_out_fps = map_stage(
        cfg=cfg, map_fn=map_fn, read_fn=read_fn, write_fn=write_fn, shard_iterator_fntr=shard_iterator
    )

    if cfg.worker != 0:
        logger.info(f"Mapping completed. Exiting as am worker {cfg.worker}, not reducer (0).")
        return

    logger.info("Starting reduction process")
    start = datetime.now()

    merge_fp = Path(cfg.stage_cfg.metadata_input_dir) / "codes.parquet"
    reduce_stage_out_fp = Path(cfg.stage_cfg.reducer_output_dir) / "codes.parquet"

    if merge_fn is None:
        join_cols = [code_field, *cfg.get("code_modifier_cols", [])]
        merge_fn = partial(join_and_replace, join_cols=join_cols)

    reduce_fn = bind_compute_fn(cfg, cfg.stage_cfg, reduce_fn)

    reduce_over(
        in_fps=map_stage_out_fps,
        out_fp=reduce_stage_out_fp,
        read_fn=read_fn,
        write_fn=write_fn,
        reduce_fn=reduce_fn,
        merge_fp=merge_fp,
        merge_fn=merge_fn,
        do_overwrite=cfg.do_overwrite,
        polling_time=cfg.polling_time,
    )
    logger.info(f"Finished reduction in {datetime.now() - start}")
