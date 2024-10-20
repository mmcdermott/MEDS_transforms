#!/usr/bin/env python

"""Performs pre-MEDS data wrangling for AUMCdb.

See the docstring of `main` for more information.
"""
import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_transforms.utils import get_shard_prefix, hydra_loguru_init, write_lazyframe

ADMISSION_ID = "admissionid"
PATIENT_ID = "patientid"

def load_raw_aumc_file(fp: Path, **kwargs) -> pl.LazyFrame:
    """Load a raw AUMCdb file into a Polars DataFrame.

    Args:
        fp: The path to the AUMCdb file.

    Returns:
        The Polars DataFrame containing the AUMCdb data.
    """

    return pl.scan_csv(fp, infer_schema_length=10000000, encoding="utf8-lossy", **kwargs)


def process_patient_and_admissions(df: pl.LazyFrame) -> pl.LazyFrame:
    """Takes the admissions table and converts it to a form that includes timestamps.

    As AUMCdb stores only offset times, note here that we add a CONSTANT TIME ACROSS ALL PATIENTS for the true
    timestamp of their health system admission. This is acceptable because in AUMCdb ONLY RELATIVE TIME
    DIFFERENCES ARE MEANINGFUL, NOT ABSOLUTE TIMES.

    The output of this process is ultimately converted to events via the `patient` key in the
    `configs/event_configs.yaml` file.
    """

    origin_pseudotime = pl.datetime(
        year = pl.col("admissionyeargroup").str.extract(r"(2003|2010)").cast(pl.Int32),
        month = 1, day = 1
    )

    # TODO: consider using better logic to infer date of birth for patients 
    #       with more than one admission.
    age_in_years = ((
        pl.col("agegroup").str.extract("(\\d{2}).?$").cast(pl.Int32) + 
        pl.col("agegroup").str.extract("^(\\d{2})").cast(pl.Int32)
    ) / 2).ceil()
    age_in_days = age_in_years * 365.25
    # We assume that the patient was born at the midpoint of the year as we don't know the actual birthdate
    pseudo_date_of_birth = origin_pseudotime - pl.duration(days=(age_in_days - 365.25 / 2))
    pseudo_date_of_death = origin_pseudotime + pl.duration(milliseconds=pl.col("dateofdeath"))


    return df.filter(pl.col("admissioncount") == 1).select(
        PATIENT_ID, 
        pseudo_date_of_birth.alias("dateofbirth"),
        "gender",
        origin_pseudotime.alias("firstadmittedattime"),
        pseudo_date_of_death.alias("dateofdeath")
    ), df.select(PATIENT_ID, ADMISSION_ID)


def join_and_get_pseudotime_fntr(
    table_name: str,
    offset_col: str | list[str],
    pseudotime_col: str | list[str],
    output_data_cols: list[str] | None = None,
    warning_items: list[str] | None = None,
) -> Callable[[pl.LazyFrame, pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that joins a dataframe to the `patient` table and adds pseudotimes.

    Also raises specified warning strings via the logger for uncertain columns.

    TODO
    """

    if output_data_cols is None:
        output_data_cols = []

    if isinstance(offset_col, str):
        offset_col = [offset_col]
    if isinstance(pseudotime_col, str):
        pseudotime_col = [pseudotime_col]

    if len(offset_col) != len(pseudotime_col):
        raise ValueError(
            "There must be the same number of `offset_col`s and `pseudotime_col`s specified. Got "
            f"{len(offset_col)} and {len(pseudotime_col)}, respectively."
        )

    def fn(df: pl.LazyFrame, patient_df: pl.LazyFrame) -> pl.LazyFrame:
        f"""Takes the {table_name} table and converts it to a form that includes pseudo-timestamps.

        The output of this process is ultimately converted to events via the `{table_name}` key in the
        `configs/event_configs.yaml` file.
        """
        pseudotimes = [
            (pl.col("firstadmittedattime") + pl.duration(milliseconds=pl.col(offset))).alias(pseudotime)
            for pseudotime, offset in zip(pseudotime_col, offset_col)
        ]

        if warning_items:
            warning_lines = [
                f"NOT SURE ABOUT THE FOLLOWING for {table_name} table. Check with the AUMCdb team:",
                *(f"  - {item}" for item in warning_items),
            ]
            logger.warning("\n".join(warning_lines))

        return df.join(patient_df, on=ADMISSION_ID, how="inner").select(
            PATIENT_ID,
            ADMISSION_ID,
            *pseudotimes,
            *output_data_cols,
        )

    return fn


@hydra.main(version_base=None, config_path="configs", config_name="pre_MEDS")
def main(cfg: DictConfig):
    """Performs pre-MEDS data wrangling for AUMCdb.
    """

    hydra_loguru_init()

    table_preprocessors_config_fp = Path("./AUMC_Example/configs/table_preprocessors.yaml")
    logger.info(f"Loading table preprocessors from {str(table_preprocessors_config_fp.resolve())}...")
    preprocessors = OmegaConf.load(table_preprocessors_config_fp)
    functions = {}
    for table_name, preprocessor_cfg in preprocessors.items():
        logger.info(f"  Adding preprocessor for {table_name}:\n{OmegaConf.to_yaml(preprocessor_cfg)}")
        functions[table_name] = join_and_get_pseudotime_fntr(table_name=table_name, **preprocessor_cfg)

    raw_cohort_dir = Path(cfg.input_dir)
    MEDS_input_dir = Path(cfg.cohort_dir)

    patient_out_fp = MEDS_input_dir / "patient.parquet"
    link_out_fp = MEDS_input_dir / "link_patient_to_admission.parquet"

    if patient_out_fp.is_file():
        logger.info(f"Reloading processed patient df from {str(patient_out_fp.resolve())}")
        patient_df = pl.read_parquet(patient_out_fp, use_pyarrow=True).lazy()
        link_df = pl.read_parquet(link_out_fp, use_pyarrow=True).lazy()
    else:
        logger.info("Processing patient table first...")

        admissions_fp = raw_cohort_dir / "admissions.csv"
        logger.info(f"Loading {str(admissions_fp.resolve())}...")
        raw_admissions_df = load_raw_aumc_file(admissions_fp)

        logger.info("Processing patient table...")
        patient_df, link_df = process_patient_and_admissions(raw_admissions_df)
        write_lazyframe(patient_df, patient_out_fp)
        write_lazyframe(link_df, link_out_fp)

    patient_df = patient_df.join(link_df, on=PATIENT_ID)

    all_fps = [fp for fp in raw_cohort_dir.glob("*.csv")]

    unused_tables = {}

    for in_fp in all_fps:
        pfx = get_shard_prefix(raw_cohort_dir, in_fp)
        if pfx in unused_tables:
            logger.warning(f"Skipping {pfx} as it is not supported in this pipeline.")
            continue
        elif pfx not in functions:
            logger.warning(f"No function needed for {pfx}. For AUMCdb, THIS IS UNEXPECTED")
            continue

        out_fp = MEDS_input_dir / f"{pfx}.parquet"

        if out_fp.is_file():
            print(f"Done with {pfx}. Continuing")
            continue

        out_fp.parent.mkdir(parents=True, exist_ok=True)

        fn = functions[pfx]

        st = datetime.now()
        logger.info(f"Processing {pfx}...")
        df = load_raw_aumc_file(in_fp)
        processed_df = fn(df, patient_df)
        processed_df.sink_parquet(out_fp)
        logger.info(f"  * Processed and wrote to {str(out_fp.resolve())} in {datetime.now() - st}")

    logger.info(f"Done! All dataframes processed and written to {str(MEDS_input_dir.resolve())}")


if __name__ == "__main__":
    main()
