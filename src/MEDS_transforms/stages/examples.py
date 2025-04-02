"""This file contains code to aid in defining and programmatically loading stage examples.

This code is not an example of how to use the package, but rather code to help operationalize packaging and
exposing examples for stages within this and derived packages programmatically for testing.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from meds import code_metadata_filepath
from meds_testing_helpers.dataset import MEDSDataset
from meds_testing_helpers.static_sample_data import SIMPLE_STATIC_SHARDED_BY_SPLIT
from omegaconf import DictConfig, OmegaConf
from polars.testing import assert_frame_equal
from yaml import load as load_yaml

try:
    from yaml import CLoader as Loader
except ImportError:  # pragma: no cover
    from yaml import Loader


def pretty_list_directory(path: Path, prefix: str | None = None) -> list[str]:
    space = "    "
    branch = "│   "
    tee = "├── "
    last = "└── "

    if prefix is None:
        prefix = ""

    lines = []

    if not path.is_dir():
        return lines

    children = list(path.iterdir())

    for i, child in enumerate(children):
        is_last = i == len(children) - 1

        node_prefix = last if is_last else tee
        subdir_prefix = space if is_last else branch

        if child.is_file():
            lines.append(f"{prefix}{node_prefix}{child.name}")
        elif child.is_dir():
            lines.append(f"{prefix}{node_prefix}{child.name}")
            lines.extend(pretty_list_directory(child, prefix=prefix + subdir_prefix))
        else:
            raise ValueError(f"Unsupported file type: {child}")
    return lines


def dict_to_hydra_kwargs(d: dict[str, str]) -> str:
    """Converts a dictionary to a hydra kwargs string for testing purposes.

    Args:
        d: The dictionary to convert.

    Returns:
        A string representation of the dictionary in hydra kwargs (dot-list) format.

    Raises:
        ValueError: If a key in the dictionary is not dot-list compatible.

    Examples:
        >>> print(" ".join(dict_to_hydra_kwargs({"a": 1, "b": "foo", "c": {"d": 2, "f": ["foo", "bar"]}})))
        a=1 b=foo c.d=2 'c.f=["foo", "bar"]'
        >>> from datetime import datetime
        >>> dict_to_hydra_kwargs({"a": 1, 2: "foo"})
        Traceback (most recent call last):
            ...
        ValueError: Expected all keys to be strings, got 2
        >>> dict_to_hydra_kwargs({"a": datetime(2021, 11, 1)})
        Traceback (most recent call last):
            ...
        ValueError: Unexpected type for value for key a: <class 'datetime.datetime'>: 2021-11-01 00:00:00
    """

    modifier_chars = ["~", "'", "++", "+"]

    out = []
    for k, v in d.items():
        if not isinstance(k, str):
            raise ValueError(f"Expected all keys to be strings, got {k}")
        match v:
            case bool() if v is True:
                out.append(f"{k}=true")
            case bool() if v is False:
                out.append(f"{k}=false")
            case None:
                out.append(f"~{k}")
            case str() | int() | float():
                out.append(f"{k}={v}")
            case dict():
                inner_kwargs = dict_to_hydra_kwargs(v)
                for inner_kv in inner_kwargs:
                    handled = False
                    for mod in modifier_chars:
                        if inner_kv.startswith(mod):
                            out.append(f"{mod}{k}.{inner_kv[len(mod):]}")
                            handled = True
                            break
                    if not handled:
                        out.append(f"{k}.{inner_kv}")
            case list() | tuple():
                v = list(v)
                v_str_inner = ", ".join(f'"{x}"' for x in v)
                out.append(f"'{k}=[{v_str_inner}]'")
            case _:
                raise ValueError(f"Unexpected type for value for key {k}: {type(v)}: {v}")

    return out


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

    stage_name: str
    scenario_name: str | None
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
    def from_dir(
        cls, stage_name: str, scenario_name: str, example_dir: Path, **schema_updates
    ) -> StageExample:
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
            stage_name=stage_name,
            scenario_name=scenario_name,
            want_data=want_data,
            want_metadata=want_metadata,
            stage_cfg=stage_cfg,
            in_data=in_data,
            test_kwargs=test_kwargs,
        )

    def write_for_test(self, input_dir: Path) -> None:
        if self.in_data is None:
            in_data = MEDSDataset.from_yaml(SIMPLE_STATIC_SHARDED_BY_SPLIT)
            in_data.write(input_dir)
            return
        else:
            self.in_data.write(input_dir)
            self.in_data._pl_code_metadata.write_parquet(input_dir / code_metadata_filepath)
            for k, v in self.in_data._pl_shards.items():
                fp = input_dir / "data" / f"{k}.parquet"
                fp.parent.mkdir(parents=True, exist_ok=True)
                v.write_parquet(fp)

    def __data_files(self, cohort_dir: Path) -> list[Path]:
        return list((cohort_dir / "data").rglob("*.parquet"))

    def __data_shards(self, cohort_dir: Path) -> dict[str, pl.DataFrame]:
        shards = {}
        for fp in self.__data_files(cohort_dir):
            shard_name = fp.relative_to(cohort_dir / "data").with_suffix("").as_posix()
            shards[shard_name] = pl.read_parquet(fp)
        return shards

    def check_files(self, cohort_dir: Path):
        all_files_str = f"{cohort_dir.name}\n" + "\n".join(pretty_list_directory(cohort_dir))

        if self.want_data is not None:
            if not self.__data_files(cohort_dir):
                raise AssertionError(
                    f"Expected data files in {cohort_dir}/data/**.parquet, but none were found. Got:\n"
                    f"{all_files_str}"
                )

        if self.want_metadata is not None:
            metadata_fp = cohort_dir / code_metadata_filepath
            if not metadata_fp.is_file():
                raise AssertionError(
                    f"Expected metadata file {code_metadata_filepath} in {cohort_dir}. Got:\n"
                    f"{all_files_str}"
                )

    def check_outputs(self, cohort_dir: Path):
        self.check_files(cohort_dir)

        if self.want_data is not None:
            got_data = MEDSDataset(data_shards=self.__data_shards(cohort_dir), dataset_metadata={})

            try:
                assert got_data._pl_shards.keys() == self.want_data._pl_shards.keys()
                for shard_name, got_df in got_data._pl_shards.items():
                    want_df = self.want_data._pl_shards[shard_name]
                    assert_frame_equal(got_df, want_df, rtol=1e-3, atol=1e-5)
            except AssertionError as e:
                pl.Config.set_tbl_rows(-1)
                raise AssertionError(f"Want data:\n{self.want_data}\nGot data:\n{got_data}\n{e}")

        if self.want_metadata is not None:
            got_metadata = pl.read_parquet(cohort_dir / code_metadata_filepath)
            try:
                assert_frame_equal(self.want_metadata, got_metadata, rtol=1e-3, atol=1e-5)
            except AssertionError as e:
                pl.Config.set_tbl_rows(-1)
                raise AssertionError(
                    f"Want metadata:\n{self.want_metadata}\nGot metadata:\n{got_metadata}\n{e}"
                )

    @property
    def do_use_config_yaml(self) -> bool:
        """Check if the test should use a config YAML file."""
        return self.test_kwargs.get("do_use_config_yaml", False)

    @property
    def _pipeline_kwargs(self) -> dict:
        return {"stages": [self.stage_name], "stage_configs": {self.stage_name: self.stage_cfg}}

    @property
    def cmd_pipeline_cfg(self) -> DictConfig | None:
        if not self.do_use_config_yaml:
            return None

        return OmegaConf.create(
            {
                "defaults": ["_preprocess"],
                **self._pipeline_kwargs,
                "hydra": {"searchpath": ["pkg://MEDS_transforms.configs"]},
            }
        )

    @property
    def cmd_args(self) -> list[str]:
        return [] if self.do_use_config_yaml else dict_to_hydra_kwargs(self._pipeline_kwargs)

    def get_test_run_command(self, test_dir: Path) -> tuple[str, Path]:
        if not test_dir.is_dir():
            raise FileNotFoundError(f"Test directory {test_dir} does not exist.")

        input_dir = test_dir / "input"
        input_dir.mkdir()
        self.write_for_test(input_dir)

        cohort_dir = test_dir / "cohort"

        if self.do_use_config_yaml:
            cfg_yaml_fp = test_dir / "config.yaml"
            OmegaConf.save(self.cmd_pipeline_cfg, cfg_yaml_fp)
            pipeline_cfg_yaml = str(cfg_yaml_fp.resolve())
        else:
            pipeline_cfg_yaml = "pkg://MEDS_transforms.configs._preprocess.yaml"

        script = (
            f"MEDS_transform-stage {pipeline_cfg_yaml} {self.stage_name} "
            f"{' '.join(self.cmd_args)} input_dir={input_dir} cohort_dir={cohort_dir}"
        )

        return script, cohort_dir

    @property
    def _err_prefix(self) -> str:
        lines = [f"Stage example {self.stage_name}/{self.scenario_name} Failed:"]
        if self.do_use_config_yaml:
            lines.append(f"Config:\n{self.cmd_pipeline_cfg}")
        return "\n".join(lines)

    def test(self) -> None:
        """Run a test for this example and assert correctness."""

        with tempfile.TemporaryDirectory() as test_dir:
            script, cohort_dir = self.get_test_run_command(Path(test_dir))

            command_out = subprocess.run(script, shell=True, capture_output=True)

            err_lines = [self._err_prefix, f"Script: {' '.join(script)}"]
            err_lines.append(f"Stdout:\n{command_out.stdout.decode()}")
            err_lines.append(f"Stderr:\n{command_out.stderr.decode()}")

            if command_out.returncode != 0:
                err_lines.append(f"Command errored with {command_out.returncode}")
                raise AssertionError("\n".join(err_lines))

            try:
                self.check_outputs(cohort_dir)
            except AssertionError as e:
                err_lines.append(str(e))
                raise AssertionError("\n".join(err_lines))
