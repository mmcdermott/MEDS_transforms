import polars as pl

from MEDS_transforms import EXTRACT_CONFIG_YAML

# We set this equality explicitly here so linting does not remove an apparently "unused" import if we just
# rename with "as" during the import.
CONFIG_YAML = EXTRACT_CONFIG_YAML

# TODO(mmd): This should really somehow be pulled from MEDS.
MEDS_METADATA_MANDATORY_TYPES = {
    "code": pl.Utf8,
    "description": pl.Utf8,
    "parent_codes": pl.List(pl.Utf8),
}
