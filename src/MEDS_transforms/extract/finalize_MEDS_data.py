#!/usr/bin/env python
"""Sets the MEDS data files to the right schema."""

import hydra
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from meds import data_schema
from omegaconf import DictConfig

from MEDS_transforms.extract import CONFIG_YAML, MEDS_DATA_MANDATORY_TYPES
from MEDS_transforms.mapreduce.mapper import map_over


def get_and_validate_data_schema(df: pl.LazyFrame, stage_cfg: DictConfig) -> pa.Table:
    """Validates the schema of a MEDS data DataFrame.

    This function validates the schema of a MEDS data DataFrame, ensuring that it has the correct columns
    and, if `do_retype` is True, that the columns are of the correct type. If `do_retype` is True, then this
    function will:
      1. Re-type any of the mandator MEDS column to the appropriate type.
      2. Attempt to add the ``numeric_value`` or ``time`` columns if either are missing, and set it to `None`.
         It will not attempt to add any other missing columns even if ``do_retype`` is `True` as the other
         columns cannot be set to `None`.

    Args:
        df: The MEDS data DataFrame to validate.
        stage_cfg: The stage configuration object.

    Returns:
        pa.Table: The validated MEDS data DataFrame, with columns re-typed as needed.

    Raises:
        ValueError: if do_retype is False and the MEDS data DataFrame is not schema compliant.

    Examples:
        >>> df = pl.DataFrame({})
        >>> get_and_validate_data_schema(df.lazy(), dict(do_retype=False)) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: MEDS Data DataFrame must have a 'subject_id' column of type Int64.
                    MEDS Data DataFrame must have a 'time' column of type
                        Datetime(time_unit='us', time_zone=None).
                    MEDS Data DataFrame must have a 'code' column of type String.
                    MEDS Data DataFrame must have a 'numeric_value' column of type Float32.
        >>> get_and_validate_data_schema(df.lazy(), {}) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: MEDS Data DataFrame must have a 'subject_id' column of type Int64.
                    MEDS Data DataFrame must have a 'code' column of type String.
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": pl.Series([1, 2], dtype=pl.UInt32),
        ...     "time": [datetime(2021, 1, 1), datetime(2021, 1, 2)],
        ...     "code": ["A", "B"], "text_value": ["1", None], "numeric_value": [None, 34.2]
        ... })
        >>> get_and_validate_data_schema(df.lazy(), dict(do_retype=False)) # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        ValueError: MEDS Data 'subject_id' column must be of type Int64. Got UInt32.
                    MEDS Data 'numeric_value' column must be of type Float32. Got Float64.
        >>> get_and_validate_data_schema(df.lazy(), {})
        pyarrow.Table
        subject_id: int64
        time: timestamp[us]
        code: string
        numeric_value: float
        text_value: large_string
        ----
        subject_id: [[1,2]]
        time: [[2021-01-01 00:00:00.000000,2021-01-02 00:00:00.000000]]
        code: [["A","B"]]
        numeric_value: [[null,34.2]]
        text_value: [["1",null]]
    """

    do_retype = stage_cfg.get("do_retype", True)
    schema = df.collect_schema()
    errors = []
    for col, dtype in MEDS_DATA_MANDATORY_TYPES.items():
        if col in schema and schema[col] != dtype:
            if do_retype:
                df = df.with_columns(pl.col(col).cast(dtype, strict=False))
            else:
                errors.append(f"MEDS Data '{col}' column must be of type {dtype}. Got {schema[col]}.")
        elif col not in schema:
            if col in ("numeric_value", "time") and do_retype:
                df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
            else:
                errors.append(f"MEDS Data DataFrame must have a '{col}' column of type {dtype}.")

    if errors:
        raise ValueError("\n".join(errors))

    additional_cols = [col for col in schema if col not in MEDS_DATA_MANDATORY_TYPES]

    if additional_cols:
        extra_schema = df.head(0).select(additional_cols).collect().to_arrow().schema
        measurement_properties = list(zip(extra_schema.names, extra_schema.types))
        df = df.select(*MEDS_DATA_MANDATORY_TYPES.keys(), *additional_cols)
    else:
        df = df.select(*MEDS_DATA_MANDATORY_TYPES.keys())
        measurement_properties = []

    validated_schema = data_schema(measurement_properties)
    return df.collect().to_arrow().cast(validated_schema)


@hydra.main(version_base=None, config_path=str(CONFIG_YAML.parent), config_name=CONFIG_YAML.stem)
def main(cfg: DictConfig):
    """Writes out schema compliant MEDS data files for the extracted dataset.

    In particular, this script ensures that all shard files are MEDS compliant with the mandatory columns
      - `subject_id` (Int64)
      - `time` (DateTime)
      - `code` (String)
      - `numeric_value` (Float32)

    This stage *_should almost always be the last data stage in an extraction pipeline._*

    All arguments are specified through the command line into the `cfg` object through Hydra.

    The `cfg.stage_cfg` object is a special key that is imputed by OmegaConf to contain the stage-specific
    configuration arguments based on the global, pipeline-level configuration file. It cannot be overwritten
    directly on the command line, but can be overwritten implicitly by overwriting components of the
    `stage_configs.extract_code_metadata` key.

    Args:
        stage_configs.finalize_MEDS_data.do_retype: Whether the script should throw an error or attempt to
            cast columns to the correct type if they are not already of the correct type. Defaults to `True`.
            May not work properly with other default aspects of the MEDS_Extract pipeline if set to `False`.
    """

    map_over(cfg, compute_fn=get_and_validate_data_schema, write_fn=pq.write_table)


if __name__ == "__main__":
    main()
