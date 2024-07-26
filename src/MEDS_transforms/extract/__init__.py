from MEDS_transforms import EXTRACT_CONFIG_YAML

# We set this equality explicitly here so linting does not remove an apparently "unused" import if we just
# rename with "as" during the import.
CONFIG_YAML = EXTRACT_CONFIG_YAML
