from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files

import polars as pl

__package_name__ = "MEDS_transforms"
try:
    __version__ = version(__package_name__)
except PackageNotFoundError:
    __version__ = "unknown"

PREPROCESS_CONFIG_YAML = files(__package_name__).joinpath("configs/preprocess.yaml")
EXTRACT_CONFIG_YAML = files(__package_name__).joinpath("configs/extract.yaml")

MANDATORY_COLUMNS = ["patient_id", "time", "code", "numeric_value"]

MANDATORY_TYPES = {
    "patient_id": pl.Int64,
    "time": pl.Datetime("us"),
    "code": pl.String,
    "numeric_value": pl.Float32,
    "categorical_value": pl.String,
    "text_value": pl.String,
}

DEPRECATED_NAMES = {
    "numerical_value": "numeric_value",
    "categoric_value": "categoric_value",
    "category_value": "categoric_value",
    "textual_value": "text_value",
    "timestamp": "time",
    "subject_id": "patient_id",
}

INFERRED_STAGE_KEYS = {
    "is_metadata",
    "data_input_dir",
    "metadata_input_dir",
    "output_dir",
    "reducer_output_dir",
}
