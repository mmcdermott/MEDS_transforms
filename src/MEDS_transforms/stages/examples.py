"""This file contains code to aid in defining and programmatically loading stage examples.

This code is not an example of how to use the package, but rather code to help operationalize packaging and
exposing examples for stages within this and derived packages programmatically for testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from meds import code_metadata_filepath
from meds_testing_helpers.dataset import MEDSDataset
from omegaconf import OmegaConf
from yaml import load as load_yaml

try:
    from yaml import CLoader as Loader
except ImportError:  # pragma: no cover
    from yaml import Loader


def read_metadata_only(fp: Path, **schema_updates) -> pl.DataFrame:
    data = load_yaml(fp.read_text(), Loader=Loader)
    assert len(data) == 1
    key = list(data.keys())[0]
    assert key == code_metadata_filepath

    val = data[key]
    if isinstance(val, str):
        return MEDSDataset.parse_csv(data[key], **schema_updates)
    elif isinstance(val, dict):
        return pl.from_dict(val, schema_overrides=schema_updates)
    elif isinstance(val, list):
        return pl.from_dicts(val, schema_overrides=schema_updates)
    else:
        raise ValueError(f"Unsupported data type for metadata: {type(val)}")


@dataclass
class StageExample:
    """A dataclass to encapsulate an example of a stage being used."""

    stage_cfg: dict
    want_data: MEDSDataset | None
    want_metadata: pl.DataFrame | None
    in_data: MEDSDataset | None
    test_kwargs: dict

    def __post_init__(self):
        if self.want_data is None and self.want_metadata is None:
            raise ValueError("Either want_data or want_metadata must be provided.")

    @classmethod
    def is_example_dir(cls, path: Path) -> bool:
        """Check if the given path is a valid example directory."""
        want_data_fp = path / "out_data.yaml"
        want_metadata_fp = path / "out_metadata.yaml"
        return want_data_fp.is_file() or want_metadata_fp.is_file()

    @classmethod
    def from_dir(cls, example_dir: Path, **schema_updates) -> StageExample:
        """Parse the example directory and return a StageExample object, or raise an error if invalid."""

        stage_cfg_fp = example_dir / "cfg.yaml"
        in_fp = example_dir / "in.yaml"
        want_data_fp = example_dir / "out_data.yaml"
        want_metadata_fp = example_dir / "out_metadata.yaml"
        test_cfg_fp = example_dir / "_test_cfg.yaml"

        if want_data_fp.is_file() and want_metadata_fp.is_file():
            raise ValueError(
                f"Both want_data and want_metadata files found in {example_dir}. "
                "Please provide only one of them."
            )
        elif not want_data_fp.is_file() and not want_metadata_fp.is_file():
            raise FileNotFoundError(f"Neither {want_data_fp} nor {want_metadata_fp} files found.")
        elif want_data_fp.is_file():
            want_data = MEDSDataset.from_yaml(want_data_fp, **schema_updates)
            want_metadata = None
        else:
            want_data = None
            want_metadata = read_metadata_only(want_metadata_fp, **schema_updates)

        in_data = MEDSDataset.from_yaml(in_fp) if in_fp.is_file() else None
        stage_cfg = OmegaConf.to_container(OmegaConf.load(stage_cfg_fp)) if stage_cfg_fp.is_file() else {}
        test_kwargs = OmegaConf.to_container(OmegaConf.load(test_cfg_fp)) if test_cfg_fp.is_file() else {}

        return cls(
            want_data=want_data,
            want_metadata=want_metadata,
            stage_cfg=stage_cfg,
            in_data=in_data,
            test_kwargs=test_kwargs,
        )
