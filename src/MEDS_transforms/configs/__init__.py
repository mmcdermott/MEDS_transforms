from importlib.resources import files

from .. import __package_name__

PREPROCESS_CONFIG_YAML = files(__package_name__).joinpath("configs/_preprocess.yaml")
RUNNER_CONFIG_YAML = files(__package_name__).joinpath("configs/_runner.yaml")
