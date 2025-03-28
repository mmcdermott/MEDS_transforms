"""Functions for registering and defining MEDS-transforms stages."""

import copy
import json
import logging
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from meds import dataset_metadata_filepath
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

from .. import __package_name__
from ..stages import get_all_registered_stages

MAIN_YAML = files(__package_name__).joinpath("configs/_main.yaml")
RUNNER_CONFIG_YAML = files(__package_name__).joinpath("configs/_runner.yaml")

logger = logging.getLogger(__name__)

NAME_KEY = "_name"
BASE_STAGE_KEY = "_base_stage"
SCRIPT_KEY = "_script"


@dataclass
class BasePipelineConfig:
    """Structured configuration object for a MEDS-Transforms pipeline."""

    name: str = MISSING
    version: str = MISSING
    description: str = MISSING

    stages: list[dict[str, dict]] = None


def get_dataset_metadata_from_root(root: str) -> dict[str, str]:
    fp = Path(root) / dataset_metadata_filepath
    if not fp.exists():
        raise FileNotFoundError(f"Dataset metadata file not found at {fp}")
    return json.loads(fp.read_text())


def get_dataset_name_from_root(root: str, default: str = "Unknown") -> str:
    """Get the dataset name from the root directory."""
    try:
        metadata = get_dataset_metadata_from_root(root)
    except FileNotFoundError:
        logger.warning(f"Dataset metadata file not found at {root}")
        return default
    return metadata.get("dataset_name", default)


def get_dataset_version_from_root(root: str, default: str = "Unknown") -> str:
    """Get the dataset name from the root directory."""
    try:
        metadata = get_dataset_metadata_from_root(root)
    except FileNotFoundError:
        logger.warning(f"Dataset metadata file not found at {root}")
        return default
    return metadata.get("dataset_version", default)


@dataclass
class DatasetConfig:
    root_dir: str
    name: str
    version: str
    code_modifiers: list[str] = field(default_factory=list)


def is_metadata_stage(stage: dict[str, Any] | DictConfig) -> bool:
    """Determines if a stage is a metadata stage either by explicit argument or via the "aggregations" key.

    Args:
        stage: The stage configuration dictionary.

    Returns:
        bool: True if the stage is a metadata stage, False otherwise.

    Raises:
        TypeError: If the "is_metadata" key is present but not a boolean or `None`.

    Examples:
        >>> is_metadata_stage({"aggregations": ["foo"]})
        True
        >>> is_metadata_stage({"is_metadata": True})
        True
        >>> is_metadata_stage({"is_metadata": False})
        False
        >>> is_metadata_stage({"is_metadata": None})
        False
        >>> is_metadata_stage({"output_dir": "/a/b/metadata"})
        False
        >>> is_metadata_stage({})
        False
        >>> is_metadata_stage(DictConfig({"aggregations": ["foo"]}))
        True
        >>> is_metadata_stage(DictConfig({"is_metadata": 32}))
        Traceback (most recent call last):
            ...
        TypeError: If specified manually, is_metadata must be a boolean. Got 32
    """
    if "is_metadata" in stage:
        if not isinstance(stage["is_metadata"], (bool, type(None))):
            raise TypeError(
                f"If specified manually, is_metadata must be a boolean. Got {stage['is_metadata']}"
            )
        return bool(stage["is_metadata"])
    else:
        return "aggregations" in stage


def populate_stage(
    stage_name: str,
    input_dir: str,
    cohort_dir: str,
    stages: list[str],
    stage_configs: dict[str, dict],
    pre_parsed_stages: dict[str, dict] | None = None,
) -> dict:
    """Populates a stage in the stages configuration with inferred stage parameters.

    Infers and adds (unless already present, in which case the provided value is used) the following
    parameters to the stage configuration:
      - `is_metadata`: Whether the stage is a metadata stage, which is determined to be `False` if the stage
        does not have an `aggregations` parameter.
      - `data_input_dir`: The input directory for the stage (either the global input directory or the previous
        data stage's output directory).
      - `metadata_input_dir`: The input directory for the stage (either the global input directory or the
        previous metadata stage's output directory).
      - `output_dir`: The output directory for the stage (the cohort directory with the stage name appended).

    Args:
        stage_name: The name of the stage to populate.
        input_dir: The global input directory.
        cohort_dir: The cohort directory into which this overall pipeline is writing data.
        stages: The names of the stages processed by this pipeline in order.
        stage_configs: The raw, unresolved stage configuration dictionaries for any stages with specific
            arguments, keyed by stage name.
        pre_parsed_stages: The stages configuration dictionaries (resolved), keyed by stage name. If
            specified, the function will not re-resolve the stages in this list.

    Returns:
        dict: The populated stage configuration.

    Raises:
        ValueError: If the stage is not present in the stages configuration.

    Examples:
        >>> from omegaconf import DictConfig
        >>> root_config = DictConfig({
        ...     "input_dir": "/a/b",
        ...     "cohort_dir": "/c/d",
        ...     "stages": ["stage1", "stage2", "stage3", "stage4", "stage5", "stage6"],
        ...     "stage_configs": {
        ...         "stage2": {"is_metadata": True},
        ...         "stage3": {"is_metadata": None, "output_dir": "/g/h"},
        ...         "stage4": {"data_input_dir": "/e/f"},
        ...         "stage5": {"aggregations": ["foo"], "train_only": None},
        ...     },
        ... })
        >>> args = [root_config[k] for k in ["input_dir", "cohort_dir", "stages", "stage_configs"]]
        >>> populate_stage("stage1", *args)
        {'is_metadata': False, 'data_input_dir': '/a/b/data', 'metadata_input_dir': '/a/b/metadata',
         'output_dir': '/c/d/stage1', 'reducer_output_dir': None}
        >>> populate_stage("stage2", *args)
        {'is_metadata': True, 'data_input_dir': '/c/d/stage1', 'metadata_input_dir': '/a/b/metadata',
         'output_dir': '/c/d/stage2', 'reducer_output_dir': '/c/d/stage2', 'train_only': True}
        >>> populate_stage("stage3", *args)
        {'is_metadata': None, 'output_dir': '/g/h', 'data_input_dir': '/c/d/stage1',
         'metadata_input_dir': '/c/d/stage2', 'reducer_output_dir': None}
        >>> populate_stage("stage4", *args)
        {'data_input_dir': '/e/f', 'is_metadata': False,
         'metadata_input_dir': '/c/d/stage2', 'output_dir': '/c/d/stage4', 'reducer_output_dir': None}
        >>> populate_stage("stage5", *args)
        {'aggregations': ['foo'], 'train_only': None, 'is_metadata': True, 'data_input_dir': '/c/d/stage4',
         'metadata_input_dir': '/c/d/stage2', 'output_dir': '/c/d/stage5',
         'reducer_output_dir': '/c/d/metadata'}
        >>> populate_stage("stage6", *args)
        {'is_metadata': False, 'data_input_dir': '/c/d/stage4',
         'metadata_input_dir': '/c/d/metadata', 'output_dir': '/c/d/data', 'reducer_output_dir': None}
        >>> populate_stage("stage7", *args)
        Traceback (most recent call last):
            ...
        ValueError: 'stage7' is not a valid stage name. Options are: stage1, stage2, stage3, stage4, stage5,
            stage6
        >>> root_config = DictConfig({
        ...     "input_dir": "/a/b",
        ...     "cohort_dir": "/c/d",
        ...     "stages": ["stage1", "stage2", "stage3", "stage4", "stage5", "stage6"],
        ...     "stage_configs": {"stage2": {"is_metadata": 34}},
        ... })
        >>> args = [root_config[k] for k in ["input_dir", "cohort_dir", "stages", "stage_configs"]]
        >>> populate_stage("stage2", *args)
        Traceback (most recent call last):
            ...
        TypeError: If specified manually, is_metadata must be a boolean. Got 34
    """

    if stage_name not in stages:
        raise ValueError(f"'{stage_name}' is not a valid stage name. Options are: {', '.join(stages)}")

    if pre_parsed_stages is None:
        pre_parsed_stages = {}

    stage = None
    prior_data_stage = None
    prior_metadata_stage = None
    for s in stages:
        if s == stage_name:
            stage = stage_configs.get(s, {})
            break
        elif s in pre_parsed_stages:
            s_resolved = pre_parsed_stages[s]
        else:
            s_resolved = populate_stage(s, input_dir, cohort_dir, stages, stage_configs, pre_parsed_stages)

        pre_parsed_stages[s] = s_resolved
        if s_resolved["is_metadata"]:
            prior_metadata_stage = s_resolved
        else:
            prior_data_stage = s_resolved

    # First, we set the stage's input directories. It needs to be able to read both data shards and reduced
    # metadata files. These will either be pulled from the overall pipeline input directories or from the
    # relevant preceding stage's output directories, as appropriate.
    is_first_data_stage = prior_data_stage is None
    is_first_metadata_stage = prior_metadata_stage is None

    pipeline_input_data_dir = str(Path(input_dir) / "data")
    pipeline_input_metadata_dir = str(Path(input_dir) / "metadata")

    if is_first_data_stage:
        default_data_input_dir = pipeline_input_data_dir
    else:
        default_data_input_dir = prior_data_stage["output_dir"]

    if is_first_metadata_stage:
        default_metadata_input_dir = pipeline_input_metadata_dir
    else:
        default_metadata_input_dir = prior_metadata_stage["reducer_output_dir"]

    # Now, we need to set output directories. The output directory for the stage will either be a stage
    # specific output directory, or, for the last data or metadata stages, respectively, will be the global
    # pipeline output data shard or metadata directories, as appropriate.
    is_metadata = is_metadata_stage(stage)

    stage_index = stages.index(stage_name)

    is_later_data_stage = False
    is_later_metadata_stage = False
    for i in range(stage_index + 1, len(stages)):
        stage_i = stage_configs.get(stages[i], {})
        if is_metadata_stage(stage_i):
            is_later_metadata_stage = True
        else:
            is_later_data_stage = True

    is_last_data_stage = (not is_later_data_stage) and (not is_metadata)
    is_last_metadata_stage = (not is_later_metadata_stage) and is_metadata

    cohort_dir = Path(cohort_dir)

    if is_last_data_stage:
        default_mapper_output_dir = str(cohort_dir / "data")
        default_reducer_output_dir = None
    elif is_last_metadata_stage:
        default_mapper_output_dir = str(cohort_dir / stage_name)
        default_reducer_output_dir = str(cohort_dir / "metadata")
    elif is_metadata:
        default_mapper_output_dir = str(cohort_dir / stage_name)
        default_reducer_output_dir = str(cohort_dir / stage_name)
    else:
        default_mapper_output_dir = str(cohort_dir / stage_name)
        default_reducer_output_dir = None

    inferred_keys = {
        "is_metadata": is_metadata,
        "data_input_dir": default_data_input_dir,
        "metadata_input_dir": default_metadata_input_dir,
        "output_dir": default_mapper_output_dir,
        "reducer_output_dir": default_reducer_output_dir,
    }

    if is_metadata:
        inferred_keys["train_only"] = True

    out = {**stage}
    for key, val in inferred_keys.items():
        if key not in out:
            out[key] = val

    return out


def register_structured_config(pipeline_config_path: Path, stage_name: str, stage_docstring: str):
    if pipeline_config_path.is_dir():
        raise ValueError(f"Pipeline config path '{pipeline_config_path}' is a directory, not a file.")

    all_stages = get_all_registered_stages()

    if not pipeline_config_path.exists():
        logger.warning(
            f"Pipeline config file '{pipeline_config_path}' does not exist. Creating a single-stage pipeline"
        )

        if stage_name not in all_stages:
            raise ValueError(f"Stage '{stage_name}' not registered...")
        stage_info = all_stages[stage_name]

        pipeline = DictConfig(
            {
                "name": f"{stage_info['package_name']}/{stage_name}",
                "version": stage_info["package_version"],
                "description": f"Single stage pipeline: {stage_name}",
                "stages": [stage_name],
            }
        )
    else:
        pipeline = OmegaConf.load(pipeline_config_path)

    new_stages = []
    for i, stage_cfg in enumerate(pipeline.stages):
        stage_cfg = copy.deepcopy(stage_cfg)
        match stage_cfg:
            case str():
                stage = stage_cfg
                stage_options = DictConfig()
            case dict() | DictConfig():
                if NAME_KEY in stage_cfg:
                    if any(k not in {NAME_KEY, BASE_STAGE_KEY, SCRIPT_KEY} for k in stage_cfg):
                        raise ValueError(f"Invalid stage configuration: {stage_cfg}")
                    stage = stage_cfg.pop(NAME_KEY)
                    stage_options = DictConfig(stage_cfg)
                elif all(k in {BASE_STAGE_KEY, SCRIPT_KEY} for k in stage_cfg):
                    stage = f"_stage_{str(i)}"
                    stage_options = DictConfig(stage_cfg)
                elif len(set(stage_cfg.keys()) - {BASE_STAGE_KEY, SCRIPT_KEY}) == 1:
                    stage = list(set(stage_cfg.keys()) - {BASE_STAGE_KEY, SCRIPT_KEY})[0]
                    stage_options = DictConfig({**stage_cfg.pop(stage), **stage_cfg})
            case _:
                raise ValueError(f"Invalid stage configuration: {stage_cfg}")

        logger.debug(f"Stage '{stage}' found in pipeline '{pipeline.name}'")

        has_base_stage = BASE_STAGE_KEY in stage_cfg
        has_script = SCRIPT_KEY in stage_cfg

        base_stage = stage_cfg.get(BASE_STAGE_KEY, None)

        if has_base_stage and not (base_stage in all_stages):
            raise ValueError(f"Base stage '{base_stage}' not registered...")

        has_registered_name = stage in all_stages

        can_run = has_registered_name or has_base_stage or has_script

        if not can_run:
            raise ValueError(f"Stage '{stage}' not registered...")
        if has_registered_name and has_base_stage and (stage != base_stage):
            raise ValueError(f"Stage '{stage}' is registered but has base stage '{base_stage}'")

        defaults = {}
        if has_registered_name:
            defaults.update(all_stages[stage]["default_config"])
        elif has_base_stage:
            defaults.update(all_stages[stage_cfg[BASE_STAGE_KEY]]["default_config"])

        stage_opts = DictConfig({**defaults, **stage_options})
        stage_opts[NAME_KEY] = stage
        new_stages.append(stage_opts)

    pipeline.stages = ListConfig(new_stages)

    stage_cfg = DictConfig(
        populate_stage(
            stage_name,
            "${input_dir}",
            "${cohort_dir}",
            [stage[NAME_KEY] for stage in pipeline.stages],
            {stage[NAME_KEY]: stage for stage in pipeline.stages},
        )
    )

    OmegaConf.register_new_resolver("get_dataset_name_from_root", get_dataset_name_from_root)
    OmegaConf.register_new_resolver("get_dataset_version_from_root", get_dataset_version_from_root)

    cs = ConfigStore.instance()
    cs.store(group="dataset", name="_base_dataset", node=DatasetConfig)
    cs.store(group="pipeline", name="pipeline", node=pipeline)
    cs.store(group="stage", name="stage", node=stage_cfg)
