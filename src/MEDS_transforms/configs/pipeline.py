"""This file defines the structured base classes for the various configs used in MEDS-Transforms."""

from __future__ import annotations

import dataclasses
import logging
import os
from importlib.resources import files
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ..stages.base import Stage

logger = logging.getLogger(__name__)

NULL_STR = "__null__"
PKG_PFX = "pkg://"
YAML_EXTENSIONS = {".yaml", ".yml"}


def resolve_pkg_path(pkg_path: str) -> Path:
    """Parse a package path into a package name and a relative path.

    Args:
        pkg_path (str): The package path to parse.

    Returns:
        tuple[str, str]: The package name and the relative path.
    """
    parts = pkg_path[len(PKG_PFX) :].split(".")
    pkg_name = parts[0]
    suffix = parts[-1]
    relative_path = Path(os.path.join(*parts[1:-1])).with_suffix(f".{suffix}")
    try:
        return files(pkg_name) / relative_path
    except ImportError:
        raise ValueError(f"Package '{pkg_name}' not found. Please check the package name.")


@dataclasses.dataclass
class PipelineConfig:
    """A base configuration class for MEDS-transforms pipelines."""

    stages: list[str] | None = None
    stage_configs: dict[str, dict] = dataclasses.field(default_factory=dict)
    additional_params: DictConfig | None = None

    @property
    def structured_config(self) -> DictConfig:
        """Return the structured config for this pipeline."""

        merged = {}
        if self.stages is not None:
            merged["stages"] = self.stages
        if self.stage_configs is not None:
            merged["stage_configs"] = self.stage_configs
        if self.additional_params is not None:
            merged.update(self.additional_params)
        return OmegaConf.create(merged)

    @classmethod
    def from_arg(cls, pipeline_yaml: str | Path) -> PipelineConfig:
        match pipeline_yaml:
            case str() if pipeline_yaml == NULL_STR:
                return cls()
            case str() as pkg_path if pipeline_yaml.startswith(PKG_PFX):
                pipeline_fp = resolve_pkg_path(pkg_path)
            case str() | Path() as path:
                pipeline_fp = Path(path)
            case _:
                raise TypeError(
                    f"Invalid pipeline YAML path type {type(pipeline_yaml)}. Expected str or Path."
                )

        if pipeline_fp.suffix not in YAML_EXTENSIONS:
            raise ValueError(
                f"Invalid pipeline YAML path '{pipeline_fp}'. "
                f"Expected a file with one of the following extensions: {YAML_EXTENSIONS}"
            )
        elif not pipeline_fp.is_file():
            raise FileNotFoundError(f"Pipeline YAML file '{pipeline_yaml}' does not exist.")

        as_dict_config = OmegaConf.load(pipeline_yaml)

        stages = as_dict_config.pop("stages", None)
        stage_configs = as_dict_config.pop("stage_configs", None)
        return cls(stages=stages, stage_configs=stage_configs, additional_params=as_dict_config)

    def resolve_stages(self, all_stages: dict[str, Stage]) -> dict[str, DictConfig]:
        stage_objects = []
        last_data_stage = None
        last_metadata_stage = None
        for s in self.stages:
            if s in self.stage_configs:
                config = self.stage_configs[s]
            else:
                config = {}

            load_name = config.get("_base_stage", s)
            if load_name not in all_stages:
                raise ValueError(
                    f"Stage '{s}' not found in the registered stages. Please check the pipeline config."
                )

            stage = all_stages[load_name]

            if stage.is_metadata:
                last_metadata_stage = s
            else:
                last_data_stage = s
            stage_objects.append((s, stage, config))

        prior_data_stage = None
        prior_metadata_stage = None

        resolved_stage_configs = {}

        input_dir = Path("${input_dir}")
        cohort_dir = Path("${cohort_dir}")

        for s, stage, config_overwrites in stage_objects:
            if stage.default_config:
                config = {**stage.default_config[stage.stage_name]}
            else:
                config = {}

            if prior_data_stage is None:
                config["data_input_dir"] = str(input_dir / "data")
            else:
                config["data_input_dir"] = prior_data_stage["output_dir"]

            if prior_metadata_stage is None:
                config["metadata_input_dir"] = str(input_dir / "metadata")
            else:
                config["metadata_input_dir"] = prior_metadata_stage["reducer_output_dir"]

            if stage.is_metadata:
                config["is_metadata"] = True
                config["output_dir"] = str(cohort_dir / s)
                config["train_only"] = True
                if s == last_metadata_stage:
                    config["reducer_output_dir"] = str(cohort_dir / "metadata")
                else:
                    config["reducer_output_dir"] = str(cohort_dir / s)
            else:
                config["is_metadata"] = False
                config["reducer_output_dir"] = None
                config["train_only"] = False
                if s == last_data_stage:
                    config["output_dir"] = str(cohort_dir / "data")
                else:
                    config["output_dir"] = str(cohort_dir / s)

            config.update(config_overwrites)
            resolved_stage_configs[s] = OmegaConf.create(config)

            if stage.is_metadata:
                prior_metadata_stage = config
            else:
                prior_data_stage = config

        return resolved_stage_configs
