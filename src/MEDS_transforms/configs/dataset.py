"""This file defines the structured base classes for the various configs used in MEDS-Transforms."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

from meds import DatasetMetadataSchema, dataset_metadata_filepath

from .utils import OmegaConfResolver, hydra_registered_dataclass

logger = logging.getLogger(__name__)


def get_dataset_metadata_from_root(root: str) -> DatasetMetadataSchema:
    """Get the dataset metadata from the MEDS root directory.

    Args:
        root (str): The root directory of the dataset.

    Returns:
        dict[str, str]: The dataset metadata.

    Raises:
        FileNotFoundError: If the dataset metadata file is not found.

    Examples:
        >>> metadata = DatasetMetadataSchema(
        ...     dataset_name="example_dataset",
        ...     dataset_version="1.0.0",
        ... )
        >>> with tempfile.TemporaryDirectory() as MEDS_root:
        ...     metadata_fp = Path(MEDS_root) / dataset_metadata_filepath
        ...     metadata_fp.parent.mkdir(parents=True, exist_ok=False)
        ...     _ = metadata_fp.write_text(json.dumps(metadata.to_dict()))
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
    return DatasetMetadataSchema(**json.loads(fp.read_text())).to_dict()


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
        >>> metadata = DatasetMetadataSchema(
        ...     dataset_name="example_dataset",
        ...     dataset_version="1.0.0",
        ... )
        >>> with tempfile.TemporaryDirectory() as MEDS_root:
        ...     metadata_fp = Path(MEDS_root) / dataset_metadata_filepath
        ...     metadata_fp.parent.mkdir(parents=True, exist_ok=False)
        ...     _ = metadata_fp.write_text(json.dumps(metadata.to_dict()))
        ...     get_dataset_name_from_root(MEDS_root)
        'example_dataset'

        If the dataset metadata file is not found or can't be parsed, the default is returned and a warning is
        logged, which we can catch and print with the `print_warnings` context manager (defined in our
        `conftest.py`):

        >>> metadata = DatasetMetadataSchema(
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
        ...     _ = metadata_fp.write_text("def foo(): return 42")  # Invalid JSON
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
        >>> metadata = DatasetMetadataSchema(
        ...     dataset_name="example_dataset",
        ...     dataset_version="1.0.0",
        ... )
        >>> with tempfile.TemporaryDirectory() as MEDS_root:
        ...     metadata_fp = Path(MEDS_root) / dataset_metadata_filepath
        ...     metadata_fp.parent.mkdir(parents=True, exist_ok=False)
        ...     _ = metadata_fp.write_text(json.dumps(metadata.to_dict()))
        ...     get_dataset_version_from_root(MEDS_root)
        '1.0.0'

        If the dataset metadata file is not found or can't be parsed, the default is returned and a warning is
        logged, which we can catch and print with the `print_warnings` context manager (defined in our
        `conftest.py`):

        >>> metadata = DatasetMetadataSchema(
        ...     dataset_name="example_dataset",
        ...     dataset_version="1.0.0",
        ... )
        >>> with print_warnings(), tempfile.TemporaryDirectory() as MEDS_root:
        ...     metadata_fp = Path(MEDS_root) / dataset_metadata_filepath
        ...     metadata_fp.parent.mkdir(parents=True, exist_ok=False)
        ...     _ = metadata_fp.write_text("Hello world!")  # Invalid JSON
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
    name: str = "${oc:get_dataset_name_from_root:root_dir}"
    version: str = "${oc:get_dataset_version_from_root:root_dir}"
    code_modifiers: list[str] = dataclasses.field(default_factory=list)
