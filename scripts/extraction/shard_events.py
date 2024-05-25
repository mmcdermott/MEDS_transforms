#!/usr/bin/env python

import gzip
import inspect
import random
from collections.abc import Sequence
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.utils import hydra_loguru_init, is_col_field

ROW_IDX_NAME = "__row_idx"

from collections.abc import Callable


def check_kwargs(func: Callable, kwargs: dict) -> dict:
    """Checks if the kwargs are valid for a function and logs then removes invalid kwargs.

    Args:
        func: The function to check the kwargs against.
        kwargs: The kwargs to check.

    Returns:
        A dictionary containing only the kwargs that are valid for the function.
    """
    valid_keywords = inspect.signature(func).parameters.keys()

    valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_keywords}
    invalid_kwargs = {k: v for k, v in kwargs.items() if k not in valid_keywords}

    if invalid_kwargs:
        kwarg_strs = "\n".join(f"  * {k}: {v}" for k, v in invalid_kwargs.items())
        logger.warning(
            f"Removing unused kwargs for {func.__name__}:\n{kwarg_strs}\n"
            f"Valid kwargs are: {', '.join(valid_keywords)}\n"
            "This behavior may be expected depending on the use case."
        )
    return valid_kwargs


def scan_with_row_idx(fp: Path, columns: Sequence[str], **scan_kwargs) -> pl.LazyFrame:
    kwargs = {"row_index_name": ROW_IDX_NAME, **scan_kwargs}
    match fp.suffix.lower():
        case ".csv.gz":
            logger.debug(f"Reading {str(fp.resolve())} as compressed CSV.")
            logger.warning("Reading compressed CSV files may be slow and limit parallelizability.")

            if columns:
                kwargs["columns"] = columns

            kwargs = check_kwargs(pl.read_csv, kwargs)
            with gzip.open(fp, mode="rb") as f:
                return pl.read_csv(f, **kwargs).lazy()
        case ".csv":
            logger.debug(f"Reading {str(fp.resolve())} as CSV.")
            kwargs = check_kwargs(pl.scan_csv, kwargs)
            df = pl.scan_csv(fp, **kwargs)
        case ".parquet":
            logger.debug(f"Reading {str(fp.resolve())} as Parquet.")
            kwargs = check_kwargs(pl.scan_parquet, kwargs)
            df = pl.scan_parquet(fp, **kwargs)
        case _:
            raise ValueError(f"Unsupported file type: {fp.suffix}")

    return df.select(columns) if columns else df


def parse_col_field(field: str) -> str:
    # Extracts the actual column name from a string formatted as "col(column_name)".
    return field[4:-1]


def retrieve_columns(
    files: Sequence[Path], cfg: DictConfig, event_conversion_cfg: DictConfig
) -> dict[Path, list[str]]:
    """Extracts and organizes column names from configuration for a list of files.

    This function processes each file specified in the 'files' list, reading the
    event conversion configurations that are specific to each file based on its
    stem (filename without the extension). It compiles a list of column names
    needed for each file from the configuration, which includes both general
    columns like row index and patient ID, as well as specific columns defined
    for medical events and timestamps formatted in a special 'col(column_name)' syntax.

    Args:
        files (Sequence[Path]): A sequence of Path objects representing the
            file paths to process.
        cfg (DictConfig): A dictionary configuration that might be used for
            further expansion (not used in the current implementation).
        event_conversion_cfg (DictConfig): A dictionary configuration where
            each key is a filename stem and each value is a dictionary containing
            configuration details for different codes or events, specifying
            which columns to retrieve for each file.

    Returns:
        dict[Path, list[str]]: A dictionary mapping each file path to a list
        of unique column names necessary for processing the file.
        The list of columns includes generic columns and those specified in the 'event_conversion_cfg'.
    """

    # Initialize a dictionary to store file paths as keys and lists of column names as values.
    file_to_columns = {}

    for f in files:
        # Access the event conversion config specific to the stem (filename without extension) of the file.
        file_meds_cfg = event_conversion_cfg[f.stem]

        # Start with a list containing default columns such as row index and patient ID column.
        file_columns = [ROW_IDX_NAME, event_conversion_cfg.patient_id_col]

        # Loop through each configuration item for the current file.
        for event_cfg in file_meds_cfg.values():
            # If the config has a 'code' key and it contains column fields, parse and add them.
            for key in ["code", "timestamp", "numerical_value"]:
                if key in event_cfg:
                    fields = event_cfg[key]
                    # make sure fields is a list
                    if isinstance(fields, str) or fields is None:
                        fields = [fields]
                    # append fields that are in the `col(<COLUMN_NAME>)`` format
                    file_columns += [parse_col_field(field) for field in fields if is_col_field(field)]

        # Store unique column names for each file to prevent duplicates.
        file_to_columns[f] = list(set(file_columns))

    return file_to_columns


def filter_to_row_chunk(df: pl.LazyFrame, start: int, end: int) -> pl.LazyFrame:
    return df.filter(pl.col(ROW_IDX_NAME).is_between(start, end, closed="left")).drop(ROW_IDX_NAME)


def write_fn(df: pl.LazyFrame, out_fp: Path) -> None:
    df.collect().write_parquet(out_fp, use_pyarrow=True)


def get_shard_prefix(base_path: Path, fp: Path) -> str:
    """Extracts the shard prefix from a file path by removing the raw_cohort_dir.

    Args:
        base_path: The base path to remove.
        fp: The file path to extract the shard prefix from.

    Returns:
        The shard prefix (the file path relative to the base path with the suffix removed).

    Examples:
        >>> get_shard_prefix(Path("/a/b/c"), Path("/a/b/c/d.parquet"))
        'd'
        >>> get_shard_prefix(Path("/a/b/c"), Path("/a/b/c/d/e.csv.gz"))
        'd/e'
    """

    relative_path = fp.relative_to(base_path)
    relative_parent = relative_path.parent
    file_name = relative_path.name.split(".")[0]

    return str(relative_parent / file_name)


@hydra.main(version_base=None, config_path="../../configs", config_name="extraction")
def main(cfg: DictConfig):
    """Runs the input data re-sharding process. Can be parallelized across output shards.

    Output shards are simply row-chunks of the input data. There is no randomization or re-ordering of the
    input data. Read contention on the input files may render additional parallelism beyond one worker per
    input file ineffective.
    """
    hydra_loguru_init()

    raw_cohort_dir = Path(cfg.raw_cohort_dir)
    MEDS_cohort_dir = Path(cfg.MEDS_cohort_dir)
    row_chunksize = cfg.row_chunksize

    seen_files = set()
    input_files_to_subshard = []
    for fmt in ["parquet", "csv", "csv.gz"]:
        files_in_fmt = list(raw_cohort_dir.glob(f"**/*.{fmt}"))
        for f in files_in_fmt:
            if get_shard_prefix(raw_cohort_dir, f) in seen_files:
                logger.warning(f"Skipping {f} as it has already been added in a preferred format.")
            else:
                input_files_to_subshard.append(f)
                seen_files.add(get_shard_prefix(raw_cohort_dir, f))

    # Select subset of files that we wish to pull events from
    event_conversion_cfg_fp = Path(cfg.event_conversion_config_fp)
    if not event_conversion_cfg_fp.exists():
        raise FileNotFoundError(f"Event conversion config file not found: {event_conversion_cfg_fp}")
    logger.info(f"Reading event conversion config from {event_conversion_cfg_fp}")
    event_conversion_cfg = OmegaConf.load(event_conversion_cfg_fp)
    input_files_to_subshard = [f for f in input_files_to_subshard if f.stem in event_conversion_cfg.keys()]
    table_to_columns = retrieve_columns(input_files_to_subshard, cfg, event_conversion_cfg)

    logger.info(f"Starting event sub-sharding. Sub-sharding {len(input_files_to_subshard)} files.")
    logger.info(
        f"Will read raw data from {str(raw_cohort_dir.resolve())}/$IN_FILE.parquet and write sub-sharded "
        f"data to {str(MEDS_cohort_dir.resolve())}/sub_sharded/$IN_FILE/$ROW_START-$ROW_END.parquet"
    )

    random.shuffle(input_files_to_subshard)

    start = datetime.now()
    for input_file in input_files_to_subshard:
        columns = table_to_columns[input_file]
        out_dir = MEDS_cohort_dir / "sub_sharded" / get_shard_prefix(raw_cohort_dir, input_file)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing {input_file} to {out_dir}.")

        df = scan_with_row_idx(input_file, columns, infer_schema_length=cfg["infer_schema_length"])
        row_count = df.select(pl.len()).collect().item()

        row_shards = list(range(0, row_count, row_chunksize))
        random.shuffle(row_shards)
        logger.info(f"Splitting {input_file} into {len(row_shards)} row-chunks of size {row_chunksize}.")

        for i, st in enumerate(row_shards):
            end = min(st + row_chunksize, row_count)
            out_fp = out_dir / f"[{st}-{end}).parquet"

            compute_fn = partial(filter_to_row_chunk, start=st, end=end)
            logger.info(
                f"Writing file {i+1}/{len(row_shards)}: {input_file} row-chunk [{st}-{end}) to {out_fp}."
            )
            rwlock_wrap(
                input_file,
                out_fp,
                partial(scan_with_row_idx, columns, cfg),
                write_fn,
                compute_fn,
                do_overwrite=cfg.do_overwrite,
            )
    end = datetime.now()
    logger.info(f"Sub-sharding completed in {datetime.now() - start}")


if __name__ == "__main__":
    main()
