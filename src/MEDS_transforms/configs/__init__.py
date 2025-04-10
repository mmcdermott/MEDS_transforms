from importlib.resources import files

from .. import __package_name__

RUNNER_CONFIG_YAML = files(__package_name__).joinpath("configs/_runner.yaml")

from .dataset import DatasetConfig
from .pipeline import PipelineConfig
from .stage import StageConfig

__all__ = ["DatasetConfig", "PipelineConfig", "StageConfig"]
