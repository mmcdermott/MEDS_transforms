#!/usr/bin/env python

"""Performs pre-MEDS data wrangling for eICU.

See the docstring of `main` for more information.
"""
import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import gzip
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_polars_functions.utils import (
    get_shard_prefix,
    hydra_loguru_init,
    write_lazyframe,
)

HEALTH_SYSTEM_STAY_ID = "patientHealthSystemStayID"
UNIT_STAY_ID = "patientUnitStayID"
PATIENT_ID = "uniquepid"

# The end of year date, used for year-only timestamps in eICU. The time is set to midnight as we'll add a
# 24-hour time component from other columns in the data.
END_OF_YEAR = {"month": 12, "day": 31, "hour": 0, "minute": 0, "second": 0}


def load_raw_eicu_file(fp: Path, **kwargs) -> pl.LazyFrame:
    """Load a raw MIMIC file into a Polars DataFrame.

    Args:
        fp: The path to the MIMIC file.

    Returns:
        The Polars DataFrame containing the MIMIC data.
    """

    with gzip.open(fp, mode="rb") as f:
        return pl.read_csv(f, infer_schema_length=100000, **kwargs).lazy()


def check_timestamps_agree(df: pl.LazyFrame, pseudotime_col: pl.Expr, given_24htime_col: str):
    expected_time = pl.col(given_24htime_col).str.strptime(pl.Time, "%H:%M:%S")

    time_deltas_min = (pseudotime_col.dt.time() - expected_time).dt.total_minutes()

    # Check that the time deltas are all within 1 minute
    logger.info(
        "Checking that stated 24h times are consistent given offsets between {pseudotime_col.name} and "
        f"{given_24htime_col}..."
    )
    max_time_deltas_min = df.select(time_deltas_min.abs().max()).collect().item()
    if max_time_deltas_min > 1:
        raise ValueError(
            f"Max number of minutes between {pseudotime_col.name} and {given_24htime_col} is "
            f"{max_time_deltas_min}. Should be <= 1."
        )


def process_patient_table(df: pl.LazyFrame, hospital_df: pl.LazyFrame) -> pl.LazyFrame:
    """Takes the patient table and converts it to a form that includes timestamps.

    As eICU stores only offset times, note here that we add a CONSTANT TIME ACROSS ALL PATIENTS for the true
    timestamp of their health system admission. This is acceptable because in eICU ONLY RELATIVE TIME
    DIFFERENCES ARE MEANINGFUL, NOT ABSOLUTE TIMES.

    The output of this process is ultimately converted to events via the `patient` key in the
    `configs/event_configs.yaml` file.
    """

    hospital_discharge_pseudotime = pl.datetime(year=pl.col("hospitalDischargeYear"), **END_OF_YEAR) + pl.col(
        "hospitalDischargeTime24"
    ).str.strptime(pl.Time, "%H:%M:%S")

    unit_admit_pseudotime = hospital_discharge_pseudotime - pl.duration(
        minutes=pl.col("hospitalDischargeOffset")
    )

    unit_discharge_pseudotime = unit_admit_pseudotime + pl.duration(minutes=pl.col("unitDischargeOffset"))

    hospital_admit_pseudotime = unit_admit_pseudotime + pl.duration(minutes=pl.col("hospitalAdmitOffset"))

    age_in_years = pl.when(pl.col("age") == "> 89").then(90).otherwise(pl.col("age").cast(pl.UInt16))
    age_in_days = age_in_years * 365.25
    # We assume that the patient was born at the midpoint of the year as we don't know the actual birthdate
    pseudo_date_of_birth = unit_admit_pseudotime - pl.duration(days=(age_in_days - 365.25 / 2))

    # Check the times
    start = datetime.now()
    logger.info(
        "Checking that the 24h times are consistent. If this is extremely slow, consider refactoring to have "
        "only one `.collect()` call."
    )
    check_timestamps_agree(df, hospital_discharge_pseudotime, "hospitalDischargeTime24")
    check_timestamps_agree(df, hospital_admit_pseudotime, "hospitalAdmitTime24")
    check_timestamps_agree(df, unit_admit_pseudotime, "unitAdmitTime24")
    check_timestamps_agree(df, unit_discharge_pseudotime, "unitDischargeTime24")
    logger.info(f"Validated 24h times in {datetime.now() - start}")

    logger.warning("NOT validating the `unitVisitNumber` column as that isn't implemented yet.")

    logger.warning(
        "NOT SURE ABOUT THE FOLLOWING. Check with the eICU team:\n"
        "  - `apacheAdmissionDx` is not selected from the patients table as we grab it from `admissiondx`. "
        "Is this right?\n"
        "  - `admissionHeight` and `admissionWeight` are interpreted as **unit** admission height/weight, "
        "not hospital admission height/weight. Is this right?\n"
        "  - `age` is interpreted as the age at the time of the unit stay, not the hospital stay. "
        "Is this right?\n"
        "  - `What is the actual mean age for those > 89? Here we assume 90.\n"
    )

    return df.join(hospital_df, left_on="hospitalID", right_on="hospitalid", how="left").select(
        # 1. Static variables
        "uniquepid",
        "gender",
        pseudo_date_of_birth.alias("dateOfBirth"),
        "ethnicity",
        # 2. Health system stay parameters
        "patientHealthSystemStayID",
        "hospitalID",
        pl.col("numbedscategory").alias("hospitalNumBedsCategory"),
        pl.col("teachingstatus").alias("hospitalTeachingStatus"),
        pl.col("region").alias("hospitalRegion"),
        # 2.1 Admission parameters
        hospital_admit_pseudotime.alias("hospitalAdmitTimestamp"),
        "hospitalAdmitSource",
        # 2.2 Discharge parameters
        hospital_discharge_pseudotime.alias("hospitalDischargeTimestamp"),
        "hospitalDischargeLocation",
        "hospitalDischargeStatus",
        # 3. Unit stay parameters
        "patientUnitStayID",  # The unit stay ID
        "wardID",
        # 3.1 Admission parameters
        unit_admit_pseudotime.alias("unitAdmitTimestamp"),
        "unitAdmitSource",
        "unitStayType",
        pl.col("admissionHeight").alias("unitAdmissionHeight"),
        pl.col("admissionWeight").alias("unitAdmissionWeight"),
        # 3.2 Discharge parameters
        unit_discharge_pseudotime.alias("unitDischargeTimestamp"),
        "unitDischargeLocation",
        "unitDischargeStatus",
        pl.col("dischargeWeight").alias("unitDischargeWeight"),
    )


class PreProcessor(NamedTuple):
    """A preprocessor function and its dependencies.

    Args:
      function: TODO
      dependencies: A two-element tuple containing the prefix of the dependent dataframe and a list of
        columns needed from that dataframe.
    """

    function: Callable[[Sequence[pl.LazyFrame]], pl.LazyFrame]
    dependencies: tuple[str, list[str]]


FUNCTIONS: dict[str, PreProcessor] = {
    "patient": PreProcessor(
        process_patient_table, ("hospital", ["hospitalid", "numbedscategory", "teachingstatus", "region"])
    ),
}

# From MIMIC
# "hosp/diagnoses_icd": (add_discharge_time_by_hadm_id, ("hosp/admissions", ["hadm_id", "dischtime"])),
# "hosp/drgcodes": (add_discharge_time_by_hadm_id, ("hosp/admissions", ["hadm_id", "dischtime"])),
# "hosp/patients": (fix_static_data, ("hosp/admissions", ["subject_id", "deathtime"])),


@hydra.main(version_base=None, config_path="configs", config_name="pre_MEDS")
def main(cfg: DictConfig):
    """Performs pre-MEDS data wrangling for eICU.

    Inputs are the raw eICU files, read from the `raw_cohort_dir` config parameter. Output files are either
    symlinked (if they are not modified) or written in processed form to the `MEDS_input_dir` config
    parameter. Hydra is used to manage configuration parameters and logging.

    Note that eICU has only a tentative ability to identify true relative admission times for even the same
    patient, as health system stay IDs are only temporally ordered at the *year* level. As such, to properly
    parse this dataset in a longitudinal form, you must do one of the following:
      1. Not operate at the level of patients at all, but instead at the level of health system stays, as
         individual events within a health system stay can be well ordered.
      2. Restrict the analysis to only patients who do not have multiple health system stays within a single
         year (as health system stays across years can be well ordered, provided we assume to distinct stays
         within a single health system cannot overlap).

    In this pipeline, we choose to operate at the level of health system stays, as this is the most general
    approach. The only downside is that we lose the ability to track individual patients across health system
    stays, and thus can only explore questions of limited longitudinal scope.

    We ignore the following tables for the given reasons:
      1. `admissiondrug`: This table is noted in the
         [documentation](https://eicu-crd.mit.edu/eicutables/admissiondrug/) as being "Extremely infrequently
         used".

    Args (all as part of the config file):
        raw_cohort_dir: The directory containing the raw eICU files.
        output_dir: The directory to write the processed files to.
    """

    raise NotImplementedError("This script is not yet implemented for eICU.")

    hydra_loguru_init()

    raw_cohort_dir = Path(cfg.raw_cohort_dir)
    MEDS_input_dir = Path(cfg.output_dir)

    all_fps = list(raw_cohort_dir.glob("**/*.csv.gz"))

    dfs_to_load = {}

    for in_fp in all_fps:
        pfx = get_shard_prefix(raw_cohort_dir, in_fp)

        out_fp = MEDS_input_dir / in_fp.relative_to(raw_cohort_dir)

        if out_fp.is_file():
            print(f"Done with {pfx}. Continuing")
            continue

        out_fp.parent.mkdir(parents=True, exist_ok=True)

        if pfx not in FUNCTIONS:
            logger.info(
                f"No function needed for {pfx}: "
                f"Symlinking {str(in_fp.resolve())} to {str(out_fp.resolve())}"
            )
            relative_in_fp = in_fp.relative_to(out_fp.parent, walk_up=True)
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
                df = load_raw_eicu_file(in_fp)
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
        df = load_raw_eicu_file(df_to_load_fp, columns=cols)
        logger.info(f"  Loaded in {datetime.now() - st}")

        for fp in fps:
            pfx = get_shard_prefix(raw_cohort_dir, fp)
            out_fp = MEDS_input_dir / f"{pfx}.parquet"

            logger.info(f"  Processing dependent df @ {pfx}...")
            fn, _ = FUNCTIONS[pfx]

            fp_st = datetime.now()
            logger.info(f"    Loading {str(fp.resolve())}...")
            fp_df = load_raw_eicu_file(fp)
            logger.info(f"    Loaded in {datetime.now() - fp_st}")
            processed_df = fn(fp_df, df)
            write_lazyframe(processed_df, out_fp)
            logger.info(f"    Processed and wrote to {str(out_fp.resolve())} in {datetime.now() - fp_st}")

    logger.info(f"Done! All dataframes processed and written to {str(MEDS_input_dir.resolve())}")


if __name__ == "__main__":
    main()
