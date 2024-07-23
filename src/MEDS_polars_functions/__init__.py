from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files

__package_name__ = "MEDS_polars_functions"
try:
    __version__ = version(__package_name__)
except PackageNotFoundError:
    __version__ = "unknown"

PREPROCESS_CONFIG_YAML = files(__package_name__).joinpath("configs/preprocess.yaml")
EXTRACT_CONFIG_YAML = files(__package_name__).joinpath("configs/extract.yaml")
