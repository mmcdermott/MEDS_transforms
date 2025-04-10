from importlib.resources import files

from .. import __package_name__

RUNNER_CONFIG_YAML = files(__package_name__).joinpath("configs/_runner.yaml")

from .dataset import DatasetConfig  # noqa: F401
from .pipeline import PipelineConfig  # noqa: F401
from .stage import StageConfig  # noqa: F401

__all__ = ["DatasetConfig", "PipelineConfig", "StageConfig"]
