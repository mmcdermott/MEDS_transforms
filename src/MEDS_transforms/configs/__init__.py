from importlib.resources import files

from .. import __package_name__
from .dataset import DatasetConfig
from .pipeline import PipelineConfig
from .stage import StageConfig

RUNNER_CONFIG_YAML = files(__package_name__).joinpath("configs/_runner.yaml")

__all__ = ["DatasetConfig", "PipelineConfig", "StageConfig"]
