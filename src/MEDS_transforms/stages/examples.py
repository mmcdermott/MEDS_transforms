"""This file contains code to aid in defining and programmatically loading stage examples.

This code is not an example of how to use the package, but rather code to help operationalize packaging and
exposing examples for stages within this and derived packages programmatically for testing.
"""

from __future__ import annotations

import subprocess
import tempfile
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

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

_SPACE = "    "
_BRANCH = "│   "
_TEE = "├── "
_LAST = "└── "


def pretty_list_directory(path: Path, prefix: str | None = None) -> list[str]:
    """Pretty prints the contents of a directory.

    Args:
        path: The path to the directory to list.
        prefix: Used for the recursive prefixing of subdirectories. Defaults to None.

    Returns:
        A list of strings representing the contents of the directory. To be printed with newlines separating
        them.

    Raises:
        ValueError: If the path is not a directory.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir)
        ...     (path / "file1.txt").touch()
        ...     (path / "foo").mkdir()
        ...     (path / "bar").mkdir()
        ...     (path / "bar" / "baz.csv").touch()
        ...     for l in pretty_list_directory(path): print(l) # This is just used as newlines break doctests
        ├── bar
        │   └── baz.csv
        ├── file1.txt
        └── foo
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir)
        ...     pretty_list_directory(path / "foo")
        Traceback (most recent call last):
            ...
        ValueError: Path /tmp/tmp.../foo does not exist.
        >>> with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        ...     path = Path(tmp.name)
        ...     pretty_list_directory(path)
        Traceback (most recent call last):
            ...
        ValueError: Path /tmp/tmp....txt is not a directory.
        >>> pretty_list_directory("foo")
        Traceback (most recent call last):
            ...
        ValueError: Expected a Path object, got <class 'str'>: foo
    """

    if not isinstance(path, Path):
        raise ValueError(f"Expected a Path object, got {type(path)}: {path}")

    if not path.exists():
        raise ValueError(f"Path {path} does not exist.")

    if not path.is_dir():
        raise ValueError(f"Path {path} is not a directory.")

    if prefix is None:
        prefix = ""

    lines = []

    children = sorted(list(path.iterdir()))

    for i, child in enumerate(children):
        is_last = i == len(children) - 1

        node_prefix = _LAST if is_last else _TEE
        subdir_prefix = _SPACE if is_last else _BRANCH

        if child.is_file():
            lines.append(f"{prefix}{node_prefix}{child.name}")
        elif child.is_dir():
            lines.append(f"{prefix}{node_prefix}{child.name}")
            lines.extend(pretty_list_directory(child, prefix=prefix + subdir_prefix))
    return lines


def dict_to_hydra_kwargs(d: dict[str, str]) -> list[str]:
    """Converts a dictionary to a hydra kwargs string for testing purposes.

    Args:
        d: The dictionary to convert.

    Returns:
        A string representation of the dictionary in hydra kwargs (dot-list) format.

    Raises:
        ValueError: If a key in the dictionary is not dot-list compatible.

    Examples:
        >>> args = dict_to_hydra_kwargs({
        ...     "a": 1, "b": "foo", "c": {"d": True, "e": False, "f": ["foo", "bar"], "g": None}
        ... })
        >>> for arg in args:
        ...     print(arg)
        a=1
        b=foo
        c.d=true
        c.e=false
        'c.f=["foo", "bar"]'
        ~c.g
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
    """Reads a code metadata file from a yaml representation.

    Eventually, this should be replaced by functionality in `meds_testing_helpers`. TODO(mmd): File an issue
    there to track.

    Args:
        fp: The path to the yaml file.
        schema_updates: Optional schema updates to apply.

    Returns:
        A Polars DataFrame containing the metadata.

    Raises:
        ValueError: If the yaml file does not contain the expected structure or the metadata encoding type is
            not supported.

    Examples:
        >>> metadata_df = pl.DataFrame({
        ...     "code": ["foo", "bar"],
        ...     "description": ["Foo", "Bar"],
        ...     "fake_number": [1, 2],
        ... })
        >>> csv_rep = {"metadata/codes.parquet": metadata_df.write_csv()}
        >>> rows_rep = {"metadata/codes.parquet": metadata_df.to_dicts()}
        >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
        ...     yaml_fp = Path(tmp.name)
        ...     OmegaConf.save(csv_rep, yaml_fp)
        ...     read_metadata_only(Path(tmp.name))
        shape: (2, 3)
        ┌──────┬─────────────┬─────────────┐
        │ code ┆ description ┆ fake_number │
        │ ---  ┆ ---         ┆ ---         │
        │ str  ┆ str         ┆ i64         │
        ╞══════╪═════════════╪═════════════╡
        │ foo  ┆ Foo         ┆ 1           │
        │ bar  ┆ Bar         ┆ 2           │
        └──────┴─────────────┴─────────────┘
        >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
        ...     yaml_fp = Path(tmp.name)
        ...     OmegaConf.save(csv_rep, yaml_fp)
        ...     read_metadata_only(Path(tmp.name), fake_number=pl.Int32)
        shape: (2, 3)
        ┌──────┬─────────────┬─────────────┐
        │ code ┆ description ┆ fake_number │
        │ ---  ┆ ---         ┆ ---         │
        │ str  ┆ str         ┆ i32         │
        ╞══════╪═════════════╪═════════════╡
        │ foo  ┆ Foo         ┆ 1           │
        │ bar  ┆ Bar         ┆ 2           │
        └──────┴─────────────┴─────────────┘
        >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
        ...     yaml_fp = Path(tmp.name)
        ...     OmegaConf.save(rows_rep, yaml_fp)
        ...     read_metadata_only(Path(tmp.name))
        shape: (2, 3)
        ┌──────┬─────────────┬─────────────┐
        │ code ┆ description ┆ fake_number │
        │ ---  ┆ ---         ┆ ---         │
        │ str  ┆ str         ┆ i64         │
        ╞══════╪═════════════╪═════════════╡
        │ foo  ┆ Foo         ┆ 1           │
        │ bar  ┆ Bar         ┆ 2           │
        └──────┴─────────────┴─────────────┘
        >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
        ...     yaml_fp = Path(tmp.name)
        ...     OmegaConf.save(rows_rep, yaml_fp)
        ...     read_metadata_only(Path(tmp.name), fake_number=pl.String)
        shape: (2, 3)
        ┌──────┬─────────────┬─────────────┐
        │ code ┆ description ┆ fake_number │
        │ ---  ┆ ---         ┆ ---         │
        │ str  ┆ str         ┆ str         │
        ╞══════╪═════════════╪═════════════╡
        │ foo  ┆ Foo         ┆ 1           │
        │ bar  ┆ Bar         ┆ 2           │
        └──────┴─────────────┴─────────────┘

        Errors are raised if the yaml file is not a filepath:

        >>> read_metadata_only("foo")
        Traceback (most recent call last):
            ...
        TypeError: Expected a Path object, got <class 'str'>: foo

        Or if the file does not exist:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     read_metadata_only(Path(tmpdir) / "not_real.yaml")
        Traceback (most recent call last):
            ...
        FileNotFoundError: File /tmp/tmp.../not_real.yaml does not exist.

        Or if the yaml file isn't in the right structure

        >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
        ...     yaml_fp = Path(tmp.name)
        ...     OmegaConf.save(["foo", "bar", "baz"], yaml_fp)
        ...     read_metadata_only(Path(tmp.name))
        Traceback (most recent call last):
            ...
        ValueError: Expected YAML file to contain 'metadata/codes.parquet: ' pointing to contents, but got:
            ['foo', 'bar', 'baz']

        Or the data in the metadata key isn't in the right format

        >>> cols_rep = {"metadata/codes.parquet": metadata_df.to_dict(as_series=False)}
        >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
        ...     yaml_fp = Path(tmp.name)
        ...     OmegaConf.save(cols_rep, yaml_fp)
        ...     read_metadata_only(Path(tmp.name))
        Traceback (most recent call last):
            ...
        ValueError: Unsupported data type for metadata: <class 'dict'>
    """

    if not isinstance(fp, Path):
        raise TypeError(f"Expected a Path object, got {type(fp)}: {fp}")
    if not fp.is_file():
        raise FileNotFoundError(f"File {fp} does not exist.")

    data = load_yaml(fp.read_text(), Loader=Loader)

    if not isinstance(data, dict) or len(data) != 1 or list(data.keys())[0] != code_metadata_filepath:
        raise ValueError(
            f"Expected YAML file to contain '{code_metadata_filepath}: ' pointing to contents, "
            f"but got:\n{data}"
        )

    key = list(data.keys())[0]

    val = data[key]

    match data[key]:
        case str() as csv:
            return MEDSDataset.parse_csv(csv, **schema_updates)
        case list() as rows:
            return pl.from_dicts(rows, schema_overrides=schema_updates)
        case _:
            raise ValueError(f"Unsupported data type for metadata: {type(val)}")


@dataclass
class TestEnv:
    """A dataclass to encapsulate the test environment for a stage.

    This is largely just useful for type safety and for pretty printing during debugging and testing.
    """

    script: str
    cohort_dir: Path
    test_dir: Path
    input_dir: Path
    config_yaml_fp: Path | None = None

    def __str__(self) -> str:
        lines = [f"Test Environment in {self.test_dir}", "  - Files:"]
        lines.extend(pretty_list_directory(self.test_dir, prefix=_SPACE))
        lines.append(f"  - Input sub-directory: {self.input_dir.relative_to(self.test_dir)}")
        lines.append(f"  - Cohort sub-directory: {self.cohort_dir.relative_to(self.test_dir)}")

        if self.config_yaml_fp:
            lines.append(f"  - Config yaml file: {self.config_yaml_fp.relative_to(self.test_dir)}")
            cfg_yaml_contents = self.config_yaml_fp.read_text().strip()
            lines.extend(textwrap.indent(cfg_yaml_contents, _SPACE + _BRANCH).splitlines())
        lines.append(f"  - Script: {self.script}")
        return "\n".join(lines)


@dataclass
class StageExample:
    """A dataclass to encapsulate an example of a stage being used.

    This is used in the automated stage testing infrastructure both for testing the built-in MEDS transforms
    stages and for downstream package stages.

    Eventually, it may be used for documenting stages as well -- if this would be useful to you, please file a
    GitHub issue to request it so we can track interest.

    Attributes:
        stage_name: The name of the stage.
        scenario_name: The name of the scenario. If the string "." is used, it is replaced with `None`.
        stage_cfg: The configuration options for this example -- e.g., if the stage is run, with these
            options, on the specified in_data, the output should match the want_data or want_metadata.
        want_data: The expected output data for the stage. If `None`, then want_metadata must be provided and
            the stage is a metadata stage.
        want_metadata: The expected output metadata for the stage. If `None`, then want_data must be provided
            and the stage is a data stage.
        in_data: The input data for the stage. If `None`, then the default static data
            (`meds_testing_helpers.static_sample_data.SIMPLE_STATIC_SHARDED_BY_SPLIT`) is used.
        do_use_config_yaml: Whether to use a config.yaml file for the stage. If `True`, then the stage is run
            only with the parameters set via a file, not the command line. Defaults to `False`.

    Raises:
        ValueError: If neither want_data nor want_metadata is provided, or if both are provided.

    Examples:

    At its simplest, a stage example can be created with just the stage name and the expected output data:

        >>> metadata_df = pl.DataFrame({"code": ["foo", "bar"], "description": ["Foo", "Bar"]})
        >>> example = StageExample(stage_name="example_stage", want_metadata=metadata_df)
        >>> print(example)
        StageExample [example_stage]
          stage_cfg: {}
          want_metadata:
            shape: (2, 2)
            ┌──────┬─────────────┐
            │ code ┆ description │
            │ ---  ┆ ---         │
            │ str  ┆ str         │
            ╞══════╪═════════════╡
            │ foo  ┆ Foo         │
            │ bar  ┆ Bar         │
            └──────┴─────────────┘

    It has some helpful properties that you can access too, for its name, test kwargs, and testing usage:

        >>> print(example.full_name)
        example_stage
        >>> print(example.do_use_config_yaml)
        False
        >>> print(example._pipeline_kwargs)
        {'stages': ['example_stage'], 'stage_configs': {'example_stage': {}}}
        >>> print(example.cmd_pipeline_cfg)
        None
        >>> for arg in example.cmd_args:
        ...     print(arg)
        'stages=["example_stage"]'
        >>> print(example._err_prefix)
        Stage example example_stage Failed:

    Note that the scenario name is ignored if it is set to "." as well as if it is not set at all. This is to
    reflect the fact that a singleton example directory will have a relative path to root of "." and should be
    inferred to have a null scenario name.

    >>> example = StageExample(stage_name="example_stage", scenario_name=".", want_metadata=metadata_df)
    >>> print(example.scenario_name)
    None

    You can also create an example with a scenario name, stage configuration arguments, test kwargs, and
    output (want) data instead of metadata:

        >>> data_df = pl.DataFrame({"subject_id": [1], "time": [1], "code": ["A"], "numeric_value": [None]})
        >>> data = MEDSDataset(data_shards={"0": data_df}, dataset_metadata={})
        >>> example_data = StageExample(
        ...     stage_name="with_scenario",
        ...     scenario_name="example_scenario",
        ...     stage_cfg={"arg1": "value1", "arg2": "value2"},
        ...     want_data=data,
        ...     do_use_config_yaml=True,
        ... )
        >>> print(example_data)
        StageExample [with_scenario/example_scenario]
          stage_cfg: {'arg1': 'value1', 'arg2': 'value2'}
          do_use_config_yaml: True
          want_data:
            MEDSDataset:
            dataset_metadata:
            data_shards:
              - 0:
                pyarrow.Table
                subject_id: int64
                time: timestamp[us]
                code: string
                numeric_value: float
                ----
                subject_id: [[1]]
                time: [[1970-01-01 00:00:00.000001]]
                code: [["A"]]
                numeric_value: [[null]]
            code_metadata:
              pyarrow.Table
              code: string
              description: string
              parent_codes: list<item: string>
                child 0, item: string
              ----
              code: []
              description: []
              parent_codes: []
            subject_splits: None

    You can also set an input dataset for the stage example manually as well; if it is unset (as above), it
    defaults to `meds_testing_helpers.static_sample_data.SIMPLE_STATIC_SHARDED_BY_SPLIT` in automated tests:

        >>> example_with_in_data = StageExample(
        ...     stage_name="with_scenario",
        ...     scenario_name="example_scenario",
        ...     stage_cfg={"arg1": "value1", "arg2": "value2"},
        ...     want_metadata=metadata_df,
        ...     in_data=data,
        ...     do_use_config_yaml=True,
        ... )
        >>> print(example_with_in_data)
        StageExample [with_scenario/example_scenario]
          stage_cfg: {'arg1': 'value1', 'arg2': 'value2'}
          do_use_config_yaml: True
          in_data:
            MEDSDataset:
            dataset_metadata:
            data_shards:
              - 0:
                pyarrow.Table
                subject_id: int64
                time: timestamp[us]
                code: string
                numeric_value: float
                ----
                subject_id: [[1]]
                time: [[1970-01-01 00:00:00.000001]]
                code: [["A"]]
                numeric_value: [[null]]
            code_metadata:
              pyarrow.Table
              code: string
              description: string
              parent_codes: list<item: string>
                child 0, item: string
              ----
              code: []
              description: []
              parent_codes: []
            subject_splits: None
          want_metadata:
            shape: (2, 2)
            ┌──────┬─────────────┐
            │ code ┆ description │
            │ ---  ┆ ---         │
            │ str  ┆ str         │
            ╞══════╪═════════════╡
            │ foo  ┆ Foo         │
            │ bar  ┆ Bar         │
            └──────┴─────────────┘

    On the testing front, there are a number of methods to help with testing the stage example. At the highest
    level of abstraction, you can call the `test` method to construct a test environment, run the stage
    through the MEDS-Transform CLI, and assert that the outputs are as expected. This, combined with the
    `stage_example` pytest fixtures in `../pytest_plugin.py`, gives an easy, accessible API to test stages
    that are registered with examples. If you need more flexibility in your usage, please file a GitHub Issue.

    To see the test code in action, though, we'll look at some lower layers of abstraction. First, we can set
    up and inspect the test environment with the `test_env` `contextmanager` property. This yields a `TestEnv`
    object, which has a nice readable `__str__` method, so we can print it to inspect it. Note that, upon
    creation, the `cohort_dir` part of the test environment is non-existent, but input data and config yaml
    files (if `do_use_config_yaml` is set to `True`) are created. The test environment is cleaned up when the
    context manager exits.

        >>> example = StageExample(stage_name="example_stage", want_metadata=metadata_df)
        >>> with example.test_env as test_env:
        ...     print(test_env)
        Test Environment in /tmp/tmp...
          - Files:
            └── input
                ├── data
                │   ├── held_out
                │   │   └── 0.parquet
                │   ├── train
                │   │   ├── 0.parquet
                │   │   └── 1.parquet
                │   └── tuning
                │       └── 0.parquet
                └── metadata
                    ├── codes.parquet
                    ├── dataset.json
                    └── subject_splits.parquet
          - Input sub-directory: input
          - Cohort sub-directory: cohort
          - Script: MEDS_transform-stage pkg://MEDS_transforms.configs._preprocess.yaml example_stage
                    'stages=["example_stage"]' input_dir=/tmp/tmp.../input cohort_dir=/tmp/tmp.../cohort
        >>> test_env.test_dir.is_dir()
        False
        >>> example = StageExample(
        ...     stage_name="example_stage", want_metadata=metadata_df,
        ...     stage_cfg={"arg1": "value1", "arg2": {"arg3": ["value3A", "value3B"]}},
        ...     in_data=data,
        ...     do_use_config_yaml=True,
        ... )
        >>> with example.test_env as test_env:
        ...     print(test_env)
        Test Environment in /tmp/tmp...
          - Files:
            ├── config.yaml
            └── input
                ├── data
                │   └── 0.parquet
                └── metadata
                    ├── codes.parquet
                    └── dataset.json
          - Input sub-directory: input
          - Cohort sub-directory: cohort
          - Config yaml file: config.yaml
            │   defaults:
            │   - _preprocess
            │   stages:
            │   - example_stage
            │   stage_configs:
            │     example_stage:
            │       arg1: value1
            │       arg2:
            │         arg3:
            │         - value3A
            │         - value3B
            │   hydra:
            │     searchpath:
            │     - pkg://MEDS_transforms.configs
          - Script: MEDS_transform-stage /tmp/tmp.../config.yaml example_stage
                    input_dir=/tmp/tmp.../input cohort_dir=/tmp/tmp.../cohort

    This test environment is used when we call the `test` method, which runs the stage, then checks to ensure
    that the stage ran successfully and the outputs in the `cohort_dir` are as expected. If the test fails, an
    error message including the string representation of the test environment and the command output is
    printed. To explore this, we'll make a fake run function that just returns a controllable return code and
    output.

        >>> def fake_run_success(script, shell, capture_output):
        ...     return subprocess.CompletedProcess([], returncode=0, stdout=b"Success", stderr=b"")
        >>> def fake_run_failure(script, shell, capture_output):
        ...     return subprocess.CompletedProcess([], returncode=1, stdout=b"", stderr=b"Failure")

    If we run and return a failing status code, the example will error as it is not expecting the command to
    fail:

        >>> example = StageExample(stage_name="example_stage", want_metadata=metadata_df)
        >>> example.test(run_fn=fake_run_failure)
        Traceback (most recent call last):
            ...
        AssertionError: Stage example example_stage Failed:
        Test Environment in /tmp/tmp...
          - Files:
            └── input
                ├── data
                │   ├── held_out
                │   │   └── 0.parquet
                │   ├── train
                │   │   ├── 0.parquet
                │   │   └── 1.parquet
                │   └── tuning
                │       └── 0.parquet
                └── metadata
                    ├── codes.parquet
                    ├── dataset.json
                    └── subject_splits.parquet
          - Input sub-directory: input
          - Cohort sub-directory: cohort
          - Script: MEDS_transform-stage pkg://MEDS_transforms.configs._preprocess.yaml example_stage
                    'stages=["example_stage"]' input_dir=/tmp/tmp.../input cohort_dir=/tmp/tmp.../cohort
        Stdout:
        <BLANKLINE>
        Stderr:
        Failure
        Command errored with return code 1

    If we have it succeed, it will still fail, as it won't find any of the output files it is expecting in the
    (non-existent) cohort directory:

        >>> example = StageExample(stage_name="example_stage", want_metadata=metadata_df)
        >>> example.test(run_fn=fake_run_success)
        Traceback (most recent call last):
            ...
        AssertionError: Stage example example_stage Failed:
        Test Environment in /tmp/tmp...
          - Files:
            └── input
                ├── data
                │   ├── held_out
                │   │   └── 0.parquet
                │   ├── train
                │   │   ├── 0.parquet
                │   │   └── 1.parquet
                │   └── tuning
                │       └── 0.parquet
                └── metadata
                    ├── codes.parquet
                    ├── dataset.json
                    └── subject_splits.parquet
          - Input sub-directory: input
          - Cohort sub-directory: cohort
          - Script: MEDS_transform-stage pkg://MEDS_transforms.configs._preprocess.yaml example_stage
                    'stages=["example_stage"]' input_dir=/tmp/tmp.../input cohort_dir=/tmp/tmp.../cohort
        Stdout:
        Success
        Stderr:
        <BLANKLINE>
        Expected cohort directory /tmp/tmp.../cohort to exist, but it does not.

    To explore test failures in more detail, we can use the `check_outputs` helper, which checks the output of
    the passed cohort directory is as expected. For example, if we add the expected metadata to a directory
    and validate that, we won't see any errors:

        >>> example = StageExample(stage_name="example_stage", want_metadata=metadata_df)
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cohort_dir = Path(tmpdir)
        ...     (cohort_dir / 'metadata').mkdir()
        ...     example.want_metadata.write_parquet(cohort_dir / code_metadata_filepath)
        ...     example.check_outputs(cohort_dir)
        ...     print("No error was raised!")
        No error was raised!

    This function will error if we don't have the expected metadata file in the cohort directory or if its
    contents are wrong. Note this function also returns a more minimal error, though the broader `test` helper
    wraps it in additional context.

        >>> example = StageExample(stage_name="example_stage", want_metadata=metadata_df)
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cohort_dir = Path(tmpdir)
        ...     (cohort_dir / 'metadata').mkdir()
        ...     example.check_outputs(cohort_dir)
        Traceback (most recent call last):
            ...
        AssertionError: Expected metadata file metadata/codes.parquet in /tmp/tmp.... Got:
        tmp...
        └── metadata
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cohort_dir = Path(tmpdir)
        ...     (cohort_dir / 'metadata').mkdir()
        ...     wrong_metadata = pl.DataFrame({"code": ["f"], "description": [None]})
        ...     wrong_metadata.write_parquet(cohort_dir / code_metadata_filepath)
        ...     example.check_outputs(cohort_dir)
        Traceback (most recent call last):
            ...
        AssertionError: Want metadata:
        shape: (2, 2)
        ┌──────┬─────────────┐
        │ code ┆ description │
        │ ---  ┆ ---         │
        │ str  ┆ str         │
        ╞══════╪═════════════╡
        │ foo  ┆ Foo         │
        │ bar  ┆ Bar         │
        └──────┴─────────────┘
        Got metadata:
        shape: (1, 2)
        ┌──────┬─────────────┐
        │ code ┆ description │
        │ ---  ┆ ---         │
        │ str  ┆ null        │
        ╞══════╪═════════════╡
        │ f    ┆ null        │
        └──────┴─────────────┘
        DataFrames are different (dtypes do not match)
        [left]:  {'code': String, 'description': String}
        [right]: {'code': String, 'description': Null}

    Similar assertion cases are used for data comparisons

        >>> data_df = pl.DataFrame(
        ...     {"subject_id": [1], "time": [datetime(2012, 12, 1)], "code": ["A"], "numeric_value": [None]},
        ...     schema_overrides={"numeric_value": pl.Float32},
        ... )
        >>> data = MEDSDataset(data_shards={"0": data_df}, dataset_metadata={})
        >>> example = StageExample(stage_name="with_scenario", want_data=data)
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cohort_dir = Path(tmpdir)
        ...     _ = example.want_data.write(cohort_dir)
        ...     example.check_outputs(cohort_dir)
        ...     print("No error was raised!")
        No error was raised!
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cohort_dir = Path(tmpdir)
        ...     (cohort_dir / 'data').mkdir()
        ...     (cohort_dir / 'data' / 'foo.json').write_text('{"foo": "bar"}')
        ...     example.check_outputs(cohort_dir)
        Traceback (most recent call last):
            ...
        AssertionError: Expected data files in /tmp/tmp.../data/**.parquet, but none were found. Got:
            tmp...
            └── data
                └── foo.json

    We also have an option to tell the system that we've already resolved the cohort dir into the appropriate
    data or metadata subdirectories; this is useful in pipeline tests. To do that, the `is_resolved_dir`
    parameter can be passed to `check_outputs` (it isn't passable to `test` as that is only used for
    single-stage tests which don't need this capability).

        >>> example = StageExample(stage_name="with_scenario", want_data=data)
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cohort_dir = Path(tmpdir)
        ...     _ = example.want_data.write(cohort_dir)
        ...     example.check_outputs(cohort_dir / "data", is_resolved_dir=True)
        ...     print("No error was raised!")
        No error was raised!
        >>> example = StageExample(stage_name="example_stage", want_metadata=metadata_df)
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cohort_dir = Path(tmpdir)
        ...     (cohort_dir / 'metadata').mkdir()
        ...     example.want_metadata.write_parquet(cohort_dir / code_metadata_filepath)
        ...     example.check_outputs(cohort_dir / "metadata", is_resolved_dir=True)
        ...     print("No error was raised!")
        No error was raised!

    Different errors are raised if the shards differ...

        >>> example = StageExample(stage_name="with_scenario", want_data=data)
        >>> wrong_data = MEDSDataset(data_shards={"1": data_df}, dataset_metadata={})
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cohort_dir = Path(tmpdir)
        ...     _ = wrong_data.write(cohort_dir)
        ...     example.check_outputs(cohort_dir)
        Traceback (most recent call last):
            ...
        AssertionError: Want data:
            MEDSDataset:
            dataset_metadata:
            data_shards:
              - 0:
                pyarrow.Table
                subject_id: int64
                time: timestamp[us]
                code: string
                numeric_value: float
                ----
                subject_id: [[1]]
                time: [[2012-12-01 00:00:00.000000]]
                code: [["A"]]
                numeric_value: [[null]]
            code_metadata:
              pyarrow.Table
              code: string
              description: string
              parent_codes: list<item: string>
                child 0, item: string
              ----
              code: []
              description: []
              parent_codes: []
            subject_splits: None
            Got data:
            MEDSDataset:
            dataset_metadata:
            data_shards:
              - 1:
                pyarrow.Table
                subject_id: int64
                time: timestamp[us]
                code: string
                numeric_value: float
                ----
                subject_id: [[1]]
                time: [[2012-12-01 00:00:00.000000]]
                code: [["A"]]
                numeric_value: [[null]]
            code_metadata:
              pyarrow.Table
              code: string
              description: string
              parent_codes: list<item: string>
                child 0, item: string
              ----
              code: []
              description: []
              parent_codes: []
            subject_splits: None
            Shards differ: dict_keys(['1']) vs dict_keys(['0'])

    ...or if the contents of the data are different:

        >>> wrong_data_df = pl.DataFrame(
        ...     {"subject_id": [1], "time": [datetime(2015, 12, 1)], "code": ["A"], "numeric_value": [None]},
        ... )
        >>> wrong_data = MEDSDataset(data_shards={"0": wrong_data_df}, dataset_metadata={})
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     cohort_dir = Path(tmpdir)
        ...     _ = wrong_data.write(cohort_dir)
        ...     example.check_outputs(cohort_dir)
        Traceback (most recent call last):
            ...
        AssertionError: Want data:
        MEDSDataset:
        dataset_metadata:
        data_shards:
          - 0:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[1]]
            time: [[2012-12-01 00:00:00.000000]]
            code: [["A"]]
            numeric_value: [[null]]
        code_metadata:
          pyarrow.Table
          code: string
          description: string
          parent_codes: list<item: string>
            child 0, item: string
          ----
          code: []
          description: []
          parent_codes: []
        subject_splits: None
        Got data:
        MEDSDataset:
        dataset_metadata:
        data_shards:
          - 0:
            pyarrow.Table
            subject_id: int64
            time: timestamp[us]
            code: string
            numeric_value: float
            ----
            subject_id: [[1]]
            time: [[2015-12-01 00:00:00.000000]]
            code: [["A"]]
            numeric_value: [[null]]
        code_metadata:
          pyarrow.Table
          code: string
          description: string
          parent_codes: list<item: string>
            child 0, item: string
          ----
          code: []
          description: []
          parent_codes: []
        subject_splits: None
        Data differs in (at least) shard 0: DataFrames are different (value mismatch for column 'time')
        [left]:  [datetime.datetime(2015, 12, 1, 0, 0)]
        [right]: [datetime.datetime(2012, 12, 1, 0, 0)]
    """

    stage_name: str
    scenario_name: str | None = None
    stage_cfg: dict = field(default_factory=dict)
    want_data: MEDSDataset | None = None
    want_metadata: pl.DataFrame | None = None
    in_data: MEDSDataset | None = None
    do_use_config_yaml: bool = False
    df_check_kwargs: dict | None = None

    BASE_PACKAGE: ClassVar[str] = "pkg://MEDS_transforms.configs._preprocess.yaml"

    def __post_init__(self):
        if self.want_data is None and self.want_metadata is None:
            raise ValueError("Either want_data or want_metadata must be provided.")

        if self.scenario_name == ".":
            self.scenario_name = None

        if self.df_check_kwargs is None:
            self.df_check_kwargs = dict(rtol=1e-3, atol=1e-5)

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
            **test_kwargs,
        )

    @property
    def full_name(self) -> str:
        if self.scenario_name:
            return f"{self.stage_name}/{self.scenario_name}"
        return self.stage_name

    def __str__(self) -> str:
        lines = [
            f"StageExample [{self.full_name}]",
            f"  stage_cfg: {self.stage_cfg}",
        ]

        if self.do_use_config_yaml:
            lines.append(f"  do_use_config_yaml: {self.do_use_config_yaml}")

        def add_nested_line(k: str):
            val = getattr(self, k)
            if val is not None:
                lines.append(f"  {k}:")
                lines.append(textwrap.indent(str(val), _SPACE))

        add_nested_line("in_data")
        add_nested_line("want_data")
        add_nested_line("want_metadata")

        return "\n".join(lines)

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

    def __data_files(self, data_dir: Path) -> list[Path]:
        return list((data_dir).rglob("*.parquet"))

    def __data_shards(self, data_dir: Path) -> dict[str, pl.DataFrame]:
        shards = {}
        for fp in self.__data_files(data_dir):
            shard_name = fp.relative_to(data_dir).with_suffix("").as_posix()
            shards[shard_name] = pl.read_parquet(fp)
        return shards

    def __check_files(self, cohort_dir: Path, is_resolved_dir: bool = False) -> None:
        if not cohort_dir.exists():
            raise AssertionError(f"Expected cohort directory {cohort_dir} to exist, but it does not.")

        all_files_str = f"{cohort_dir.name}\n" + "\n".join(pretty_list_directory(cohort_dir))

        if self.want_data is not None:
            data_dir = cohort_dir if is_resolved_dir else cohort_dir / "data"
            if not self.__data_files(data_dir):
                raise AssertionError(
                    f"Expected data files in {cohort_dir}/data/**.parquet, but none were found. Got:\n"
                    f"{all_files_str}"
                )

        if self.want_metadata is not None:
            metadata_dir = cohort_dir if is_resolved_dir else cohort_dir / "metadata"
            metadata_fp = metadata_dir / "codes.parquet"
            if not metadata_fp.is_file():
                raise AssertionError(
                    f"Expected metadata file {code_metadata_filepath} in {cohort_dir}. Got:\n"
                    f"{all_files_str}"
                )

    def check_outputs(self, cohort_dir: Path, is_resolved_dir: bool = False) -> None:
        self.__check_files(cohort_dir, is_resolved_dir)

        if self.want_data is not None:
            data_dir = cohort_dir if is_resolved_dir else cohort_dir / "data"
            got_data = MEDSDataset(data_shards=self.__data_shards(data_dir), dataset_metadata={})

            try:
                assert (
                    got_data._pl_shards.keys() == self.want_data._pl_shards.keys()
                ), f"Shards differ: {got_data._pl_shards.keys()} vs {self.want_data._pl_shards.keys()}"
                for shard_name, got_df in got_data._pl_shards.items():
                    want_df = self.want_data._pl_shards[shard_name]
                    try:
                        assert_frame_equal(got_df, want_df, **self.df_check_kwargs)
                    except AssertionError as e:
                        raise AssertionError(f"Data differs in (at least) shard {shard_name}: {e}")
            except AssertionError as e:
                pl.Config.set_tbl_rows(-1)
                raise AssertionError(f"Want data:\n{self.want_data}\nGot data:\n{got_data}\n{e}")

        if self.want_metadata is not None:
            metadata_dir = cohort_dir if is_resolved_dir else cohort_dir / "metadata"
            metadata_fp = metadata_dir / "codes.parquet"
            got_metadata = pl.read_parquet(metadata_fp)
            try:
                assert_frame_equal(self.want_metadata, got_metadata, **self.df_check_kwargs)
            except AssertionError as e:
                pl.Config.set_tbl_rows(-1)
                raise AssertionError(
                    f"Want metadata:\n{self.want_metadata}\nGot metadata:\n{got_metadata}\n{e}"
                )

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

    @property
    @contextmanager
    def test_env(self) -> TestEnv:
        """Context manager to set up the test environment.

        This sets up the test directory and cleans it up after use. It yields the script to run this stage and
        the cohort directory to check the outputs.
        """
        with tempfile.TemporaryDirectory() as test_dir:
            test_dir = Path(test_dir)

            input_dir = test_dir / "input"
            input_dir.mkdir()

            self.write_for_test(input_dir)

            cohort_dir = test_dir / "cohort"

            if self.do_use_config_yaml:
                cfg_yaml_fp = test_dir / "config.yaml"
                OmegaConf.save(self.cmd_pipeline_cfg, cfg_yaml_fp)
                pipeline_cfg_yaml = cfg_yaml_fp.resolve()
            else:
                pipeline_cfg_yaml = self.BASE_PACKAGE

            script = (
                f"MEDS_transform-stage {pipeline_cfg_yaml} {self.stage_name} "
                f"{' '.join(self.cmd_args)} input_dir={input_dir} cohort_dir={cohort_dir}"
            )

            yield TestEnv(
                script=script,
                cohort_dir=cohort_dir,
                test_dir=test_dir,
                input_dir=input_dir,
                config_yaml_fp=pipeline_cfg_yaml if self.do_use_config_yaml else None,
            )

    @property
    def _err_prefix(self) -> str:
        lines = [f"Stage example {self.full_name} Failed:"]
        if self.do_use_config_yaml:
            lines.append(f"Config:\n{self.cmd_pipeline_cfg}")
        return "\n".join(lines)

    def test(self, run_fn: callable | None = subprocess.run) -> None:
        """Run a test for this example and assert correctness.

        Args:
            run_fn: The function to use to run the command. Defaults to subprocess.run. This is useful for
                dependency injection or customizing the test set-up.

        Raises:
            AssertionError: If the test fails.
        """

        with self.test_env as test_env:
            command_out = run_fn(test_env.script, shell=True, capture_output=True)
            err_lines = [self._err_prefix, *str(test_env).splitlines()]

            err_lines.append(f"Stdout:\n{command_out.stdout.decode()}")
            err_lines.append(f"Stderr:\n{command_out.stderr.decode()}")

            if command_out.returncode != 0:
                err_lines.append(f"Command errored with return code {command_out.returncode}")
                raise AssertionError("\n".join(err_lines))

            try:
                self.check_outputs(test_env.cohort_dir)
            except AssertionError as e:
                err_lines.append(str(e))
                raise AssertionError("\n".join(err_lines))


class StageExampleDict(dict):
    """A dictionary subclass to hold stage examples.

    The only purpose to this is to display the examples in a more readable format when printed. It does not
    validate that the values are `StageExample` objects,

    Examples:
        >>> print(StageExampleDict())
        {}

    Singleton example dictionaries with an empty scenario name just prints the stage:

        >>> print(StageExampleDict(**{".": "foo"}))
        foo

    With a scenario name or multiple scenarios, it prints the stage and scenario, nesting with a "│" indicator

        >>> print(StageExampleDict(**{"foo": "bar"}))
        foo:
        │   bar

    Note that if the empty scenario name (".") is included in a multi-scenario dict, it is printed, unlike in
    the singleton case.

        >>> print(StageExampleDict(**{"foo": "bar", "baz": "qux", ".": "quux"}))
        foo:
        │   bar
        baz:
        │   qux
        .:
        │   quux
    """

    def __str__(self) -> str:
        if not self:
            return "{}"

        if len(self) == 1 and list(self.keys())[0] == ".":
            return str(list(self.values())[0])

        lines = []
        for k, v in self.items():
            lines.append(f"{k}:")
            v_str = textwrap.indent(str(v), _BRANCH)

            lines.append(v_str)

        return "\n".join(lines)
