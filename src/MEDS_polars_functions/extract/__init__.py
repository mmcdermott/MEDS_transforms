from importlib.resources import files

from MEDS_polars_functions import __package_name__

CONFIG_YAML = files(__package_name__).joinpath("configs/extract.yaml")
