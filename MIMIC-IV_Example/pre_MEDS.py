#!/usr/bin/env python

"""Performs pre-MEDS data wrangling for MIMIC-IV."""
import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import gzip
from datetime import datetime
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_polars_functions.utils import (
    get_shard_prefix,
    hydra_loguru_init,
    write_lazyframe,
)


def load_raw_mimic_file(fp: Path, **kwargs) -> pl.LazyFrame:
    """Load a raw MIMIC file into a Polars DataFrame.

    Args:
        fp: The path to the MIMIC file.

    Returns:
        The Polars DataFrame containing the MIMIC data.
    """

    with gzip.open(fp, mode="rb") as f:
        return pl.read_csv(f, infer_schema_length=100000, **kwargs).lazy()


def add_discharge_time_by_hadm_id(
    df: pl.LazyFrame, discharge_time_df: pl.LazyFrame, out_column_name: str = "hadm_discharge_time"
) -> pl.LazyFrame:
    """Joins the two dataframes by ``"hadm_id"`` and adds the discharge time to the original dataframe."""

    discharge_time_df.select("hadm_id", pl.col("dischtime").alias(out_column_name))
    return df.join(discharge_time_df, on="hadm_id", how="left")


def fix_static_data(raw_static_df: pl.LazyFrame, death_times_df: pl.LazyFrame) -> pl.LazyFrame:
    """Fixes the static data by adding the death time to the static data and fixes the DOB nonsense.

    Args:
        raw_static_df: The raw static data.
        death_times_df: The death times data.

    Returns:
        The fixed static data.
    """

    raise NotImplementedError


FUNCTIONS = {
    "hosp/diagnoses_icd": (add_discharge_time_by_hadm_id, ("hosp/admissions", ["hadm_id", "dischtime"])),
    "hosp/drgcodes": (add_discharge_time_by_hadm_id, ("hosp/admissions", ["hadm_id", "dischtime"])),
    "hosp/patients": (fix_static_data, ("hosp/admissions", ["subject_id", "deathtime"])),
}


@hydra.main(version_base=None, config_path="configs", config_name="pre_MEDS")
def main(cfg: DictConfig):
    """Performs pre-MEDS data wrangling for MIMIC-IV.

    Inputs are the raw MIMIC files, read from the `raw_cohort_dir` config parameter. Output files are either
    symlinked (if they are not modified) or written in processed form to the `MEDS_input_dir` config
    parameter. Hydra is used to manage configuration parameters and logging.
    """

    hydra_loguru_init()

    raw_cohort_dir = Path(cfg.raw_cohort_dir)
    MEDS_input_dir = Path(cfg.MEDS_cohort_dir)

    all_fps = list(raw_cohort_dir.glob("**/*.csv.gz"))

    dfs_to_load = {}

    for in_fp in all_fps:
        pfx = get_shard_prefix(raw_cohort_dir, in_fp)

        out_fp = MEDS_input_dir / in_fp.relative_to(raw_cohort_dir)
        out_fp.parent.mkdir(parents=True, exist_ok=True)

        if pfx not in FUNCTIONS:
            logger.info(
                f"No function needed for {pfx}: "
                f"Symlinking {str(in_fp.resolve())} to {str(out_fp.resolve())}"
            )
            out_fp.symlink_to(in_fp)
            continue
        else:
            fn, need_df = FUNCTIONS[pfx]
            if not need_df:
                st = datetime.now()
                logger.info(f"Processing {pfx}...")
                df = load_raw_mimic_file(in_fp)
                logger.info(f"  Loaded raw {in_fp} in {datetime.now() - st}")
                processed_df = fn(df)
                write_lazyframe(processed_df, out_fp)
                logger.info(f"  Processed and wrote to {str(out_fp.resolve())} in {datetime.now() - st}")
            else:
                needed_pfx, needed_cols = need_df
                if needed_pfx not in dfs_to_load:
                    dfs_to_load[needed_pfx] = {"fps": set(), "cols": set()}

                dfs_to_load[needed_pfx]["fps"].add(in_fp)
                dfs_to_load[needed_pfx]["cols"].update(needed_cols)

    for df_to_load_pfx, fps_and_cols in dfs_to_load.items():
        fps = fps_and_cols["fps"]
        cols = list(fps_and_cols["cols"])

        df_to_load_fp = raw_cohort_dir / f"{df_to_load_pfx}.csv.gz"

        st = datetime.now()

        logger.info(f"Loading {str(df_to_load_fp.resolve())} for manipulating other dataframes...")
        df = load_raw_mimic_file(df_to_load_fp, columns=cols)
        logger.info(f"  Loaded in {datetime.now() - st}")

        for fp in fps:
            pfx = get_shard_prefix(raw_cohort_dir, fp)
            out_fp = MEDS_input_dir / fp.relative_to(raw_cohort_dir)

            logger.info(f"  Processing dependent df @ {pfx}...")
            fn, _ = FUNCTIONS[pfx]

            fp_st = datetime.now()
            logger.info(f"    Loading {str(fp.resolve())}...")
            fp_df = load_raw_mimic_file(fp)
            logger.info(f"    Loaded in {datetime.now() - fp_st}")
            processed_df = fn(fp_df, df)
            write_lazyframe(processed_df, out_fp)
            logger.info(f"    Processed and wrote to {str(out_fp.resolve())} in {datetime.now() - fp_st}")

    logger.info(f"Done! All dataframes processed and written to {str(MEDS_input_dir.resolve())}")


if __name__ == "__main__":
    main()
