"""Launches end-to-end MEDS extraction process."""
import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import os
import shutil
import subprocess
from pathlib import Path
from typing import Union

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm.auto import tqdm

from MEDS_polars_functions.utils import is_col_field

pl.enable_string_cache()


def extract_codes(cfg: DictConfig):
    """Extracts all codes from the event conversion configuration."""
    event_conversion_cfg_fp = Path(cfg["event_conversion_config_fp"])
    event_conversion_cfg = OmegaConf.load(event_conversion_cfg_fp)
    event_conversion_cfg.pop("patient_id_col")

    codes = []
    for file_cfg in event_conversion_cfg.values():
        for event_cfg in file_cfg.values():
            # If the config has a 'code' key and it contains column fields, parse and add them.
            code_fields = event_cfg["code"]
            if isinstance(code_fields, list) or isinstance(code_fields, ListConfig):
                for field in code_fields:
                    if not is_col_field(field):
                        codes.append(field)
            else:
                field = code_fields
                if not is_col_field(field):
                    codes.append(field)
    return codes


def validate(df: pl.DataFrame, cfg: DictConfig):
    # extracts all codes from the event conversion configuration
    codes = extract_codes(cfg)
    # check that all codes are prefixes for at least one row for the "code" column in the df
    for code in tqdm(codes, desc="Validating all config codes are in the final dataset"):
        if not df.select(pl.col("code").cast(pl.String).str.starts_with(code).any()).collect().item():
            logger.warning(f"Code {code} not found in dataframe")


def dict_to_hydra_args(d, prefix=""):
    """Recursively converts a nested dictionary into a list of Hydra-compatible command-line arguments."""
    args = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, Union[dict, DictConfig]):
            args.extend(dict_to_hydra_args(v, prefix=key))
        elif v is None:
            pass
        else:
            args.append(f"{key}={v}")
    return args


def run_command(script: Path, hydra_kwargs: dict[str, str], command_name: str):
    script = str(script.resolve())
    # Convert dictionary to Hydra command line arguments
    command_parts = ["python", script] + dict_to_hydra_args(hydra_kwargs)
    command = " ".join(command_parts)
    logger.info(f"Launching command: {command}")
    command_out = subprocess.run(command, shell=True, capture_output=True)
    stderr = command_out.stderr.decode()
    stdout = command_out.stdout.decode()
    if command_out.returncode != 0:
        raise AssertionError(f"{command_name} failed!\nstderr:\n{stderr}\nstdout:\n{stdout}")
    return stderr, stdout


@hydra.main(version_base=None, config_path="./configs", config_name="extraction")
def extraction(cfg: DictConfig):
    MEDS_cohort_dir = Path(cfg.MEDS_cohort_dir)

    # Create the directories
    if cfg.do_overwrite:
        shutil.rmtree(MEDS_cohort_dir, ignore_errors=True)
    os.makedirs(MEDS_cohort_dir, exist_ok=True)

    # Run the extraction script
    #   1. Sub-shard the data (this will be a null operation in this case, but it is worth doing just in
    #      case.
    #   2. Collect the patient splits.
    #   3. Extract the events and sub-shard by patient.
    #   4. Merge to the final output.

    extraction_root = root / "scripts" / "extraction"

    all_stderrs = []
    all_stdouts = []

    # Step 1: Sub-shard the data
    stderr, stdout = run_command(extraction_root / "shard_events.py", cfg, "shard_events")

    all_stderrs.append(stderr)
    all_stdouts.append(stdout)

    subsharded_dir = MEDS_cohort_dir / "sub_sharded"

    out_files = list(subsharded_dir.glob("*/*.parquet"))
    logger.info(f"Step 1 Complete: Generated files {[f.stem for f in out_files]} files in {subsharded_dir}.")

    # Step 2: Collect the patient splits
    stderr, stdout = run_command(
        extraction_root / "split_and_shard_patients.py",
        cfg,
        "split_and_shard_patients",
    )

    all_stderrs.append(stderr)
    all_stdouts.append(stdout)

    splits_fp = MEDS_cohort_dir / "splits.json"
    assert splits_fp.is_file(), f"Expected splits @ {str(splits_fp.resolve())} to exist."

    logger.info(f"Step 2 Complete: Generated splits file at {splits_fp}.")

    # Step 3: Extract the events and sub-shard by patient
    stderr, stdout = run_command(
        extraction_root / "convert_to_sharded_events.py",
        cfg,
        "convert_events",
    )
    all_stderrs.append(stderr)
    all_stdouts.append(stdout)

    patient_subsharded_folder = MEDS_cohort_dir / "patient_sub_sharded_events"
    assert patient_subsharded_folder.is_dir(), f"Expected {patient_subsharded_folder} to be a directory."

    logger.info(f"Step 3 Complete: Generated patient sub-sharded files in {patient_subsharded_folder}.")

    # Step 4: Merge to the final output
    stderr, stdout = run_command(
        extraction_root / "merge_to_MEDS_cohort.py",
        cfg,
        "merge_sharded_events",
    )
    all_stderrs.append(stderr)
    all_stdouts.append(stdout)

    # Check the final output
    output_folder = MEDS_cohort_dir / "final_cohort"
    assert output_folder.is_dir(), f"Expected {output_folder} to be a directory."
    out_files = list(output_folder.glob("*/*.parquet"))
    assert len(out_files) > 0, f"Expected at least one file in {output_folder}."
    logger.info(f"Step 4 Complete: Generated MEDS format files {[str(f.resolve()) for f in out_files]}")

    # Validate the final output
    validate(pl.scan_parquet(out_files[0]), cfg)


if __name__ == "__main__":
    extraction()
