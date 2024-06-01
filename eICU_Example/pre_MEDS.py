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
from omegaconf import DictConfig, OmegaConf

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


def process_patient(df: pl.LazyFrame, hospital_df: pl.LazyFrame) -> pl.LazyFrame:
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
        PATIENT_ID,
        "gender",
        pseudo_date_of_birth.alias("dateOfBirth"),
        "ethnicity",
        # 2. Health system stay parameters
        HEALTH_SYSTEM_STAY_ID,
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
        UNIT_STAY_ID,
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


def join_and_get_pseudotime_fntr(
    table_name: str,
    offset_col: str,
    pseudotime_col: str,
    output_data_cols: list[str] | None = None,
    warning_items: list[str] | None = None,
) -> Callable[[pl.LazyFrame, pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that joins a dataframe to the `patient` table and adds pseudotimes.

    Also raises specified warning strings via the logger for uncertain columns.

    TODO
    """

    if output_data_cols is None:
        output_data_cols = []

    def fn(df: pl.LazyFrame, patient_df: pl.LazyFrame) -> pl.LazyFrame:
        f"""Takes the {table_name} table and converts it to a form that includes pseudo-timestamps.

        The output of this process is ultimately converted to events via the `{table_name}` key in the
        `configs/event_configs.yaml` file.
        """

        pseudotime = pl.col("unitAdmitTimestamp") + pl.duration(minutes=pl.col(offset_col))

        if warning_items:
            warning_lines = [
                f"NOT SURE ABOUT THE FOLLOWING for {table_name} table. Check with the eICU team:",
                *(f"  - {item}" for item in warning_items),
            ]
            logger.warning("\n".join(warning_lines))

        return df.join(patient_df, on=UNIT_STAY_ID, how="inner").select(
            HEALTH_SYSTEM_STAY_ID,
            UNIT_STAY_ID,
            pseudotime.alias(pseudotime_col),
            *output_data_cols,
        )

    return fn


class PreProcessor(NamedTuple):
    """A preprocessor function and its dependencies.

    Args:
      function: TODO
      dependencies: A two-element tuple containing the prefix of the dependent dataframe and a list of
        columns needed from that dataframe.
    """

    function: Callable[[Sequence[pl.LazyFrame]], pl.LazyFrame]
    dependencies: tuple[str, list[str]]


NEEDED_PATIENT_COLS = [UNIT_STAY_ID, HEALTH_SYSTEM_STAY_ID, "unitAdmitTimestamp"]
PATIENT_DEPENDENCY = ("patient", NEEDED_PATIENT_COLS)

# Generic "copy from patients" functions are stored in `configs/table_preprocessors.yaml` and loaded in
# `main`.
SPECIALTY_FUNCTIONS: dict[str, PreProcessor] = {
    "patient": PreProcessor(
        process_patient, ("hospital", ["hospitalid", "numbedscategory", "teachingstatus", "region"])
    ),
}


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
      2. `apacheApsVar`: This table is a sort of "meta-table" that contains variables used to compute the
         APACHE score; we won't use these raw variables from this table, but instead will use the raw data.
      3. `apachePatientResult`: This table has pre-computed APACHE score variables; we won't use these and
         will use the raw data directly.
      4. `apachePredVar`: This table contains variables used to compute the APACHE score; we won't use these
         in favor of the raw data directly.
      5. `carePlanCareProvider`: This table contains information about the provider for given care-plan
         entries; however, as we can't link this table to the particular care-plan entries, we don't use it
         here. It also is not clear (to the author of this script; the eICU team may know more) how reliable
         the time-offsets are for this table as they merely denote when a provider was entered into the care
         plan.
      6. `customLab`: The documentation for this table is very sparse, so we skip it.
      7. `intakeOutput`: There are a number of significant warnings about duplicates, cumulative values, and
         more in the documentation for this table, so for now we skip it.
      8. `microLab`: We don't use this because the culture taken time != culture result time, so seeing this
         data would give a model an advantage over any possible real-world implementation. Plus, the docs say
         it is not well populated.

    Args (all as part of the config file):
        raw_cohort_dir: The directory containing the raw eICU files.
        output_dir: The directory to write the processed files to.
    """

    raise NotImplementedError("This script is not yet implemented for eICU.")

    hydra_loguru_init()

    functions = {**SPECIALTY_FUNCTIONS}

    logger.info("Loading table preprocessors from configs/table_preprocessors.yaml...")
    preprocessors = OmegaConf.load("configs/table_preprocessors.yaml")
    for table_name, preprocessor_cfg in preprocessors.items():
        logger.info(f"  Adding preprocessor for {table_name}:\n{OmegaConf.to_yaml(preprocessor_cfg)}")
        functions[table_name] = PreProcessor(
            join_and_get_pseudotime_fntr(table_name=table_name, **preprocessor_cfg),
            PATIENT_DEPENDENCY,
        )

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

        if pfx not in functions:
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

            fn, need_df = functions[pfx]
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
            fn, _ = functions[pfx]

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
