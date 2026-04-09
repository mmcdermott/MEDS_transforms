"""Example stage that converts raw CSV files into MEDS format.

This demonstrates how a downstream package can define a stage that takes non-MEDS input (raw CSV
files) and produces MEDS-formatted output. Because the input is not in MEDS format, the stage is
registered as a ``MAIN`` stage (the function is named ``main``), giving it full control over I/O.

The example's ``in.yaml`` contains raw CSV files and a JSON shard map, which triggers the
``yaml_to_disk`` fallback in the ``StageExample`` framework for testing and documentation.
"""

from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig

from MEDS_transforms.stages import Stage


@Stage.register(is_metadata=False)
def main(cfg: DictConfig):
    """Reads raw CSV files from input_dir and writes MEDS-formatted parquet shards to output_dir.

    Expects the input directory to contain:
    - ``raw/*.csv``: Raw data files with a ``subject_id`` column and measurement columns.
    - ``metadata/.shards.json``: A JSON file mapping shard names to lists of subject IDs.

    Each CSV row becomes a MEDS measurement. Non-subject_id columns are pivoted into
    ``(code, numeric_value)`` pairs. The output is written as MEDS parquet shards.
    """

    import json

    input_dir = Path(cfg.input_dir)
    data_output_dir = Path(cfg.stage_cfg.output_dir)
    cohort_dir = Path(cfg.output_dir)

    # Read shard assignments
    shards_fp = input_dir / "metadata" / ".shards.json"
    with open(shards_fp) as f:
        shard_map = json.load(f)

    # Read and combine all CSVs
    csv_files = list((input_dir / "raw").glob("*.csv"))
    all_rows = []
    for csv_fp in sorted(csv_files):
        df = pl.read_csv(csv_fp)

        if "subject_id" not in df.columns:
            continue

        value_cols = [c for c in df.columns if c != "subject_id"]
        for col in value_cols:
            col_type = df.schema[col]
            rows = df.select(
                pl.col("subject_id"),
                pl.lit(None).cast(pl.Datetime("us")).alias("time"),
                pl.lit(col).alias("code"),
                (
                    pl.col(col).cast(pl.Float32).alias("numeric_value")
                    if col_type.is_numeric()
                    else pl.lit(None).cast(pl.Float32).alias("numeric_value")
                ),
            )
            all_rows.append(rows)

    if not all_rows:
        return

    combined = pl.concat(all_rows)

    # Write shards to the data output directory (stage_cfg.output_dir resolves to cohort/data)
    for shard_name, subject_ids in shard_map.items():
        shard_df = combined.filter(pl.col("subject_id").is_in(subject_ids))
        shard_fp = data_output_dir / f"{shard_name}.parquet"
        shard_fp.parent.mkdir(parents=True, exist_ok=True)
        table = shard_df.to_arrow()
        pq.write_table(table, shard_fp)

    # Write minimal metadata to cohort/metadata
    metadata_dir = cohort_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    codes = combined.select("code").unique().sort("code")
    codes_table = pa.table(
        {
            "code": codes["code"].to_arrow(),
            "description": pa.nulls(len(codes), type=pa.string()),
            "parent_codes": pa.nulls(len(codes), type=pa.list_(pa.string())),
        }
    )
    pq.write_table(codes_table, metadata_dir / "codes.parquet")
