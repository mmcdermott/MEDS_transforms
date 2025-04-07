"""This file defines the structured base classes for the various configs used in MEDS-Transforms."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from meds import DatasetMetadata, dataset_metadata_filepath
from omegaconf import DictConfig

from .utils import OmegaConfResolver, hydra_registered_dataclass

logger = logging.getLogger(__name__)


def get_dataset_metadata_from_root(root: str) -> DatasetMetadata:
    """Get the dataset metadata from the MEDS root directory.

    Args:
        root (str): The root directory of the dataset.

    Returns:
        dict[str, str]: The dataset metadata.

    Raises:
        FileNotFoundError: If the dataset metadata file is not found.

    Examples:
        >>> metadata = DatasetMetadata(
        ...     dataset_name="example_dataset",
        ...     dataset_version="1.0.0",
        ... )
        >>> with tempfile.TemporaryDirectory() as MEDS_root:
        ...     metadata_fp = Path(MEDS_root) / dataset_metadata_filepath
        ...     metadata_fp.parent.mkdir(parents=True, exist_ok=False)
        ...     _ = metadata_fp.write_text(json.dumps(metadata))
        ...     get_dataset_metadata_from_root(MEDS_root)
        {'dataset_name': 'example_dataset', 'dataset_version': '1.0.0'}

        Errors will be raised if the dataset metadata file is not found:

        >>> with tempfile.TemporaryDirectory() as MEDS_root:
        ...     get_dataset_metadata_from_root(MEDS_root)
        Traceback (most recent call last):
            ...
        FileNotFoundError: Dataset metadata file not found at /tmp/tmp.../metadata/dataset.json
    """
    fp = Path(root) / dataset_metadata_filepath
    if not fp.exists():
        raise FileNotFoundError(f"Dataset metadata file not found at {fp}")
    return DatasetMetadata(**json.loads(fp.read_text()))


@OmegaConfResolver
def get_dataset_name_from_root(root: str, default: str = "Unknown") -> str:
    """Get the dataset name from the root directory.

    Args:
        root (str): The root directory of the dataset.
        default (str): The default value to return if the dataset name is not found.

    Returns:
        str: The dataset name, or the default value if not found.

    Raises:
        DatasetMetadataNotFoundWarning: If the dataset metadata file is not found or is invalid.

    Examples:
        >>> metadata = DatasetMetadata(
        ...     dataset_name="example_dataset",
        ...     dataset_version="1.0.0",
        ... )
        >>> with tempfile.TemporaryDirectory() as MEDS_root:
        ...     metadata_fp = Path(MEDS_root) / dataset_metadata_filepath
        ...     metadata_fp.parent.mkdir(parents=True, exist_ok=False)
        ...     _ = metadata_fp.write_text(json.dumps(metadata))
        ...     get_dataset_name_from_root(MEDS_root)
        'example_dataset'

        If the dataset metadata file is not found or can't be parsed, the default is returned and a warning is
        logged, which we can catch and print with the `print_warnings` context manager (defined in our
        `conftest.py`):

        >>> metadata = DatasetMetadata(
        ...     dataset_name="example_dataset",
        ...     dataset_version="1.0.0",
        ... )
        >>> with print_warnings(), tempfile.TemporaryDirectory() as MEDS_root:
        ...     get_dataset_name_from_root(MEDS_root, default="Foo")
        'Foo'
        Warning: Valid dataset metadata file not found in /tmp/tmp...:
                 Dataset metadata file not found at /tmp/tmp.../metadata/dataset.json
        >>> with print_warnings(), tempfile.TemporaryDirectory() as MEDS_root:
        ...     metadata_fp = Path(MEDS_root) / dataset_metadata_filepath
        ...     metadata_fp.parent.mkdir(parents=True, exist_ok=False)
        ...     _ = metadata_fp.write_text("def foo(): return 42") # Invalid JSON
        ...     get_dataset_name_from_root(MEDS_root, default="Foo")
        'Foo'
        Warning: Valid dataset metadata file not found in /tmp/tmp...:
                 Expecting value: line 1 column 1 (char 0)
    """
    try:
        return get_dataset_metadata_from_root(root).get("dataset_name", default)
    except Exception as e:
        logger.warning(f"Valid dataset metadata file not found in {root}: {e}")
        return default


@OmegaConfResolver
def get_dataset_version_from_root(root: str, default: str = "Unknown") -> str:
    """Get the dataset version from the root directory.

    Args:
        root (str): The root directory of the dataset.
        default (str): The default value to return if the dataset name is not found.

    Returns:
        str: The dataset version, or the default value if not found.

    Raises:
        DatasetMetadataNotFoundWarning: If the dataset metadata file is not found or is invalid.

    Examples:
        >>> metadata = DatasetMetadata(
        ...     dataset_name="example_dataset",
        ...     dataset_version="1.0.0",
        ... )
        >>> with tempfile.TemporaryDirectory() as MEDS_root:
        ...     metadata_fp = Path(MEDS_root) / dataset_metadata_filepath
        ...     metadata_fp.parent.mkdir(parents=True, exist_ok=False)
        ...     _ = metadata_fp.write_text(json.dumps(metadata))
        ...     get_dataset_version_from_root(MEDS_root)
        '1.0.0'

        If the dataset metadata file is not found or can't be parsed, the default is returned and a warning is
        logged, which we can catch and print with the `print_warnings` context manager (defined in our
        `conftest.py`):

        >>> metadata = DatasetMetadata(
        ...     dataset_name="example_dataset",
        ...     dataset_version="1.0.0",
        ... )
        >>> with print_warnings(), tempfile.TemporaryDirectory() as MEDS_root:
        ...     metadata_fp = Path(MEDS_root) / dataset_metadata_filepath
        ...     metadata_fp.parent.mkdir(parents=True, exist_ok=False)
        ...     _ = metadata_fp.write_text("Hello world!") # Invalid JSON
        ...     get_dataset_version_from_root(MEDS_root, default="Bar")
        'Bar'
        Warning: Valid dataset metadata file not found in /tmp/tmp...:
                 Expecting value: line 1 column 1 (char 0)
        >>> with print_warnings(), tempfile.TemporaryDirectory() as MEDS_root:
        ...     get_dataset_version_from_root(MEDS_root, default="Bar")
        'Bar'
        Warning: Valid dataset metadata file not found in /tmp/tmp...:
                 Dataset metadata file not found at /tmp/tmp.../metadata/dataset.json
    """
    try:
        return get_dataset_metadata_from_root(root).get("dataset_version", default)
    except Exception as e:
        logger.warning(f"Valid dataset metadata file not found in {root}: {e}")
        return default


@hydra_registered_dataclass(group="dataset", name="_base_dataset")
class DatasetConfig:
    """A base configuration class for MEDS dataset inputs.

    This class is used to define the base configuration for a dataset (largely for type safety purposes). It
    includes the root directory of the dataset, and the name and version of the dataset. This is merely a base
    class used for type safety in the Hydra configs. In hydra configuration usage, the resolvers defined below
    populate the name and version parameters given the root dir automatically, if possible.

    Attributes:
        root_dir: The root directory of the dataset.
        name: The name of the dataset.
        version: The version of the dataset.
        code_modifiers: A list of code modifiers for use when processing to the dataset.
    """

    root_dir: str
    name: str
    version: str
    code_modifiers: list[str] = dataclasses.field(default_factory=list)


UNRESOLVED_STAGE_CONFIG_T = str | dict[str, Any] | DictConfig


@dataclasses.dataclass
class StageConfig:
    _name: str
    _base_stage: str | None = None
    _cfg: dict[str, Any] | DictConfig = dataclasses.field(default_factory=dict)

    @classmethod
    def parse(cls, raw: UNRESOLVED_STAGE_CONFIG_T) -> StageConfig:
        match raw:
            case str() as name:
                return cls(_name=name)
            case dict():
                raise NotImplementedError
            case DictConfig():
                raise NotImplementedError
            case _:
                raise TypeError(
                    f"Invalid stage configuration: {raw}. Expected a string, dictionary, or DictConfig."
                )


@dataclasses.dataclass
class PipelineConfig:
    """A base configuration class for MEDS-transforms pipelines.

    This class is used to define the base configuration class for a pipeline (largely for type safety
    purposes). It contains a name, a version, a description, and a list of stages (alongside their
    configuration objects). This is intended to be used largely in the context of the MEDS-Transforms
    configuration stack, not as a standalone Hydra configuration object (as in that context, it will lack
    sufficient information to infer stage-specific default arguments.

    The primary information in this configuration is contained in the `stages` key, which contains the list of
    all stages in this pipeline. It is a list, such that each element in the list has one of the following
    forms:
      1. A plain string that is the name of the (registered) target stage. This indicates that the stage
         should be run with no changes to the default arguments (unless further changes are added on the
         command line).
      2. A dictionary with only one non-meta key that is the name of the target stage. The value of this
         non-meta key points to a dictionary of configuration options to pass to the stage. Meta-keys
         correspond to additional information aobut how the stage should be run, and currently consist solely
         of the `_base_stage` option, which points to the string name of the registered stage that should be
         run when the stage of the target name is called, if it is not the target name. This is useful for
         stages that can be used repeatedly in the same pipeline with different arguments, such as aggregation
         of code metadata.

    See examples below for more information.

    Attributes:
        name: The name of the pipeline.
        version: The version of the pipeline.
        description: A description of the pipeline.
        stages: The list of stages. See above for a description of their organization.
    """

    name: str
    version: str
    description: str
    stages: list[UNRESOLVED_STAGE_CONFIG_T] = dataclasses.field(default_factory=list)
