#!/usr/bin/env python

"""Performs pre-MEDS data wrangling for MIMIC-IV."""

from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_transforms.extract.utils import get_supported_fp
from MEDS_transforms.utils import get_shard_prefix, hydra_loguru_init, write_lazyframe


def add_dot(code: pl.Expr, position: int) -> pl.Expr:
    """Adds a dot to the code expression at the specified position.

    Args:
        code: The code expression.
        position: The position to add the dot.

    Returns:
        The expression which would yield the code string with a dot added at the specified position

    Example:
        >>> pl.select(add_dot(pl.lit("12345"), 3))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ 123.45  │
        └─────────┘
        >>> pl.select(add_dot(pl.lit("12345"), 1))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ 1.2345  │
        └─────────┘
        >>> pl.select(add_dot(pl.lit("12345"), 6))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ 12345   │
        └─────────┘
    """
    return (
        pl.when(code.str.len_chars() > position)
        .then(code.str.slice(0, position) + "." + code.str.slice(position))
        .otherwise(code)
    )


def add_icd_diagnosis_dot(icd_version: pl.Expr, icd_code: pl.Expr) -> pl.Expr:
    """Adds the appropriate dot to the ICD diagnosis codebased on the version.

    Args:
        icd_version: The ICD version.
        icd_code: The ICD code.

    Returns:
        The ICD code with appropriate dot syntax based on the version.

    Examples:
        >>> pl.select(add_icd_diagnosis_dot(pl.lit("9"), pl.lit("12345")))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ 123.45  │
        └─────────┘
        >>> pl.select(add_icd_diagnosis_dot(pl.lit("9"), pl.lit("E1234")))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ E123.4  │
        └─────────┘
        >>> pl.select(add_icd_diagnosis_dot(pl.lit("9"), pl.lit("F1234")))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ F12.34  │
        └─────────┘
        >>> pl.select(add_icd_diagnosis_dot(pl.lit("10"), pl.lit("12345")))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ 123.45  │
        └─────────┘
        >>> pl.select(add_icd_diagnosis_dot(pl.lit("10"), pl.lit("E1234")))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ E12.34  │
        └─────────┘
    """

    icd9_code = (
        pl.when(icd_code.str.starts_with("E")).then(add_dot(icd_code, 4)).otherwise(add_dot(icd_code, 3))
    )

    icd10_code = add_dot(icd_code, 3)

    return pl.when(icd_version == "9").then(icd9_code).otherwise(icd10_code)


def add_icd_procedure_dot(icd_version: pl.Expr, icd_code: pl.Expr) -> pl.Expr:
    """Adds the appropriate dot to the ICD procedure code based on the version.

    Args:
        icd_version: The ICD version.
        icd_code: The ICD code.

    Returns:
        The ICD code with appropriate dot syntax based on the version.

    Examples:
        >>> pl.select(add_icd_procedure_dot(pl.lit("9"), pl.lit("12345")))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ 12.345  │
        └─────────┘
        >>> pl.select(add_icd_procedure_dot(pl.lit("10"), pl.lit("12345")))
        shape: (1, 1)
        ┌─────────┐
        │ literal │
        │ ---     │
        │ str     │
        ╞═════════╡
        │ 12345   │
        └─────────┘
    """

    icd9_code = add_dot(icd_code, 2)
    icd10_code = icd_code

    return pl.when(icd_version == "9").then(icd9_code).otherwise(icd10_code)


def add_discharge_time_by_hadm_id(
    df: pl.LazyFrame, discharge_time_df: pl.LazyFrame, out_column_name: str = "hadm_discharge_time"
) -> pl.LazyFrame:
    """Joins the two dataframes by ``"hadm_id"`` and adds the discharge time to the original dataframe."""

    discharge_time_df = discharge_time_df.select("hadm_id", pl.col("dischtime").alias(out_column_name))
    return df.join(discharge_time_df, on="hadm_id", how="left")


def fix_static_data(raw_static_df: pl.LazyFrame, death_times_df: pl.LazyFrame) -> pl.LazyFrame:
    """Fixes the static data by adding the death time to the static data and fixes the DOB nonsense.

    Args:
        raw_static_df: The raw static data.
        death_times_df: The death times data.

    Returns:
        The fixed static data.
    """

    death_times_df = death_times_df.group_by("subject_id").agg(pl.col("deathtime").min())

    return raw_static_df.join(death_times_df, on="subject_id", how="left").select(
        "subject_id",
        pl.coalesce(pl.col("deathtime"), pl.col("dod")).alias("dod"),
        (pl.col("anchor_year") - pl.col("anchor_age")).cast(str).alias("year_of_birth"),
        "gender",
    )


FUNCTIONS = {
    "hosp/diagnoses_icd": (add_discharge_time_by_hadm_id, ("hosp/admissions", ["hadm_id", "dischtime"])),
    "hosp/drgcodes": (add_discharge_time_by_hadm_id, ("hosp/admissions", ["hadm_id", "dischtime"])),
    "hosp/patients": (fix_static_data, ("hosp/admissions", ["subject_id", "deathtime"])),
}

ICD_DFS_TO_FIX = [
    ("hosp/d_icd_diagnoses", add_icd_diagnosis_dot),
    ("hosp/d_icd_procedures", add_icd_procedure_dot),
]


@hydra.main(version_base=None, config_path="configs", config_name="pre_MEDS")
def main(cfg: DictConfig):
    """Performs pre-MEDS data wrangling for MIMIC-IV.

    Inputs are the raw MIMIC files, read from the `input_dir` config parameter. Output files are either
    symlinked (if they are not modified) or written in processed form to the `MEDS_input_dir` config
    parameter. Hydra is used to manage configuration parameters and logging.
    """

    hydra_loguru_init()

    input_dir = Path(cfg.input_dir)
    MEDS_input_dir = Path(cfg.cohort_dir)

    done_fp = MEDS_input_dir / ".done"
    if done_fp.is_file() and not cfg.do_overwrite:
        logger.info(
            f"Pre-MEDS transformation already complete as {done_fp} exists and "
            f"do_overwrite={cfg.do_overwrite}. Returning."
        )
        exit(0)

    all_fps = list(input_dir.glob("**/*.*"))

    dfs_to_load = {}
    seen_fps = {}

    for in_fp in all_fps:
        pfx = get_shard_prefix(input_dir, in_fp)

        try:
            fp, read_fn = get_supported_fp(input_dir, pfx)
        except FileNotFoundError:
            logger.info(f"Skipping {pfx} @ {str(in_fp.resolve())} as no compatible dataframe file was found.")
            continue

        if fp.suffix in [".csv", ".csv.gz"]:
            read_fn = partial(read_fn, infer_schema_length=100000)

        if str(fp.resolve()) in seen_fps:
            continue
        else:
            seen_fps[str(fp.resolve())] = read_fn

        out_fp = MEDS_input_dir / fp.relative_to(input_dir)

        if out_fp.is_file():
            print(f"Done with {pfx}. Continuing")
            continue

        out_fp.parent.mkdir(parents=True, exist_ok=True)

        if pfx not in FUNCTIONS and pfx not in [p for p, _ in ICD_DFS_TO_FIX]:
            logger.info(
                f"No function needed for {pfx}: " f"Symlinking {str(fp.resolve())} to {str(out_fp.resolve())}"
            )
            relative_in_fp = fp.relative_to(out_fp.resolve().parent, walk_up=True)
            out_fp.symlink_to(relative_in_fp)
            continue
        else:
            out_fp = MEDS_input_dir / f"{pfx}.parquet"
            if out_fp.is_file():
                print(f"Done with {pfx}. Continuing")
                continue

            fn, need_df = FUNCTIONS[pfx]
            if not need_df:
                st = datetime.now()
                logger.info(f"Processing {pfx}...")
                df = read_fn(fp)
                logger.info(f"  Loaded raw {fp} in {datetime.now() - st}")
                processed_df = fn(df)
                write_lazyframe(processed_df, out_fp)
                logger.info(f"  Processed and wrote to {str(out_fp.resolve())} in {datetime.now() - st}")
            else:
                needed_pfx, needed_cols = need_df
                if needed_pfx not in dfs_to_load:
                    dfs_to_load[needed_pfx] = {"fps": set(), "cols": set()}

                dfs_to_load[needed_pfx]["fps"].add(fp)
                dfs_to_load[needed_pfx]["cols"].update(needed_cols)

    for df_to_load_pfx, fps_and_cols in dfs_to_load.items():
        fps = fps_and_cols["fps"]
        cols = list(fps_and_cols["cols"])

        df_to_load_fp, df_to_load_read_fn = get_supported_fp(input_dir, df_to_load_pfx)

        st = datetime.now()

        logger.info(f"Loading {str(df_to_load_fp.resolve())} for manipulating other dataframes...")
        if df_to_load_fp.suffix in [".csv.gz"]:
            df = read_fn(df_to_load_fp, columns=cols)
        else:
            df = read_fn(df_to_load_fp)
        logger.info(f"  Loaded in {datetime.now() - st}")

        for fp in fps:
            pfx = get_shard_prefix(input_dir, fp)
            out_fp = MEDS_input_dir / f"{pfx}.parquet"

            logger.info(f"  Processing dependent df @ {pfx}...")
            fn, _ = FUNCTIONS[pfx]

            fp_st = datetime.now()
            logger.info(f"    Loading {str(fp.resolve())}...")
            fp_df = seen_fps[str(fp.resolve())](fp)
            logger.info(f"    Loaded in {datetime.now() - fp_st}")
            processed_df = fn(fp_df, df)
            write_lazyframe(processed_df, out_fp)
            logger.info(f"    Processed and wrote to {str(out_fp.resolve())} in {datetime.now() - fp_st}")

    for pfx, fn in ICD_DFS_TO_FIX:
        fp, read_fn = get_supported_fp(input_dir, pfx)
        out_fp = MEDS_input_dir / f"{pfx}.parquet"

        if out_fp.is_file():
            print(f"Done with {pfx}. Continuing")
            continue

        st = datetime.now()
        logger.info(f"Processing {pfx}...")
        processed_df = read_fn(fp).with_columns(
            fn(pl.col("icd_version"), pl.col("icd_code")).alias("icd_code")
        )
        write_lazyframe(processed_df, out_fp)
        logger.info(f"  Processed and wrote to {str(out_fp.resolve())} in {datetime.now() - st}")

    logger.info(f"Done! All dataframes processed and written to {str(MEDS_input_dir.resolve())}")
    done_fp.write_text(f"Finished at {datetime.now()}")


if __name__ == "__main__":
    main()
