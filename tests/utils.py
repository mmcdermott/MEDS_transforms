import json
import re
import subprocess
import tempfile
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any

import polars as pl
from meds import parent_codes_field, time_field
from omegaconf import OmegaConf
from polars.testing import assert_frame_equal
from yaml import load as load_yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

DEFAULT_CSV_TS_FORMAT = "%m/%d/%Y, %H:%M:%S"

# TODO: Make use meds library
MEDS_PL_SCHEMA = {
    "subject_id": pl.Int64,
    "time": pl.Datetime("us"),
    "code": pl.String,
    "numeric_value": pl.Float32,
    "numeric_value/is_inlier": pl.Boolean,
    "text_value": pl.String,
}


def parse_meds_csvs(
    csvs: str | dict[str, str], schema: dict[str, pl.DataType] = MEDS_PL_SCHEMA
) -> pl.DataFrame | dict[str, pl.DataFrame]:
    """Converts a string or dict of named strings to a MEDS DataFrame by interpreting them as CSVs."""

    default_read_schema = {**schema}
    default_read_schema["time"] = pl.Utf8

    def reader(csv_str: str) -> pl.DataFrame:
        cols = csv_str.strip().split("\n")[0].split(",")
        read_schema = {k: v for k, v in default_read_schema.items() if k in cols}

        df = pl.read_csv(StringIO(csv_str), schema=read_schema)

        if time_field in cols:
            df = df.with_columns(
                pl.col(time_field).str.strptime(MEDS_PL_SCHEMA[time_field], DEFAULT_CSV_TS_FORMAT)
            )
        if parent_codes_field in cols:
            df = df.with_columns(pl.col(parent_codes_field).str.split(", "))

        return df.select(cols)

    if isinstance(csvs, str):
        return reader(csvs)
    else:
        return {k: reader(v) for k, v in csvs.items()}


def parse_shards_yaml(yaml_str: str, **schema_updates) -> pl.DataFrame:
    schema = {**MEDS_PL_SCHEMA, **schema_updates}
    return parse_meds_csvs(load_yaml(yaml_str.strip(), Loader=Loader), schema=schema)


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


def run_command(
    script: str,
    hydra_kwargs: dict[str, str],
    test_name: str,
    config_name: str | None = None,
    should_error: bool = False,
    do_use_config_yaml: bool = False,
):
    command_parts = [script]

    err_cmd_lines = []

    if do_use_config_yaml:
        if config_name is None:
            raise ValueError("config_name must be provided if do_use_config_yaml is True.")

        conf = OmegaConf.create(
            {
                "defaults": [config_name],
                **hydra_kwargs,
            }
        )

        conf_dir = tempfile.TemporaryDirectory()
        conf_path = Path(conf_dir.name) / "config.yaml"
        OmegaConf.save(conf, conf_path)

        command_parts.extend(
            [
                f"--config-path={str(conf_path.parent.resolve())}",
                "--config-name=config",
                "'hydra.searchpath=[pkg://MEDS_transforms.configs]'",
            ]
        )
        err_cmd_lines.append(f"Using config yaml:\n{OmegaConf.to_yaml(conf)}")
    else:
        command_parts.append(" ".join(dict_to_hydra_kwargs(hydra_kwargs)))

    full_cmd = " ".join(command_parts)
    err_cmd_lines.append(f"Running command: {full_cmd}")
    command_out = subprocess.run(full_cmd, shell=True, capture_output=True)

    command_errored = command_out.returncode != 0

    stderr = command_out.stderr.decode()
    err_cmd_lines.append(f"stderr:\n{stderr}")
    stdout = command_out.stdout.decode()
    err_cmd_lines.append(f"stdout:\n{stdout}")

    if should_error and not command_errored:
        if do_use_config_yaml:
            conf_dir.cleanup()
        raise AssertionError(
            f"{test_name} failed as command did not error when expected!\n" + "\n".join(err_cmd_lines)
        )
    elif not should_error and command_errored:
        if do_use_config_yaml:
            conf_dir.cleanup()
        raise AssertionError(
            f"{test_name} failed as command errored when not expected!\n" + "\n".join(err_cmd_lines)
        )
    if do_use_config_yaml:
        conf_dir.cleanup()
    return stderr, stdout


def assert_df_equal(want: pl.DataFrame, got: pl.DataFrame, msg: str = None, **kwargs):
    try:
        update_exprs = {}
        for k, v in want.schema.items():
            assert k in got.schema, f"missing column {k}."
            if kwargs.get("check_dtypes", False):
                assert v == got.schema[k], f"column {k} has different types."
            if v == pl.List(pl.String) and got.schema[k] == pl.List(pl.String):
                update_exprs[k] = pl.col(k).list.join("||")
        if update_exprs:
            want_cols = want.columns
            got_cols = got.columns

            want = want.with_columns(**update_exprs).select(want_cols)
            got = got.with_columns(**update_exprs).select(got_cols)

        assert_frame_equal(want, got, **kwargs, rtol=1e-3, atol=1e-5)
    except AssertionError as e:
        pl.Config.set_tbl_rows(-1)
        raise AssertionError(f"{msg}:\nWant:\n{want}\nGot:\n{got}\n{e}") from e


FILE_T = pl.DataFrame | dict[str, Any] | str


@contextmanager
def input_dataset(input_dir: Path | None = None, input_files: dict[str, FILE_T] | None = None):
    with tempfile.TemporaryDirectory() as d:
        cohort_dir = Path(d) / "output_cohort"

        if input_dir is not None:
            assert not input_files
        else:
            input_dir = Path(d) / "input_cohort"
            for filename, data in input_files.items():
                fp = input_dir / filename
                fp.parent.mkdir(parents=True, exist_ok=True)

                match data:
                    case pl.DataFrame() if fp.suffix == "":
                        data.write_parquet(fp.with_suffix(".parquet"), use_pyarrow=True)
                    case pl.DataFrame() if fp.suffix in {".parquet", ".par"}:
                        data.write_parquet(fp, use_pyarrow=True)
                    case pl.DataFrame() if fp.suffix == ".csv":
                        data.write_csv(fp)
                    case dict() if fp.suffix == "":
                        fp.with_suffix(".json").write_text(json.dumps(data))
                    case dict() if fp.suffix.endswith(".json"):
                        fp.write_text(json.dumps(data))
                    case str():
                        fp.write_text(data.strip())
                    case _ if callable(data):
                        data_str = data(
                            input_dir=str(input_dir.resolve()),
                            cohort_dir=str(cohort_dir.resolve()),
                        )
                        fp.write_text(data_str)
                    case _:
                        raise ValueError(
                            f"Unknown data type {type(data)} for file {fp.relative_to(input_dir)}"
                        )

        yield input_dir, cohort_dir


def check_outputs(
    cohort_dir: Path,
    want_outputs: dict[str, pl.DataFrame],
    assert_no_other_outputs: bool = True,
    **df_check_kwargs,
):
    all_file_suffixes = set()

    for output_name, want in want_outputs.items():
        if Path(output_name).suffix == "":
            output_name = f"{output_name}.parquet"

        file_suffix = Path(output_name).suffix
        all_file_suffixes.add(file_suffix)

        output_fp = cohort_dir / output_name

        files_found = [str(fp.relative_to(cohort_dir)) for fp in cohort_dir.glob("**/*{file_suffix}")]
        all_files_found = [str(fp.relative_to(cohort_dir)) for fp in cohort_dir.rglob("*")]

        if not output_fp.is_file():
            raise AssertionError(
                f"Wanted {output_fp.relative_to(cohort_dir)} to exist. "
                f"{len(files_found)} {file_suffix} files found with suffix: {', '.join(files_found)}. "
                f"{len(all_files_found)} generic files found: {', '.join(all_files_found)}."
            )

        msg = f"Expected {output_fp.relative_to(cohort_dir)} to be equal to the target"

        match file_suffix:
            case ".parquet":
                got_df = pl.read_parquet(output_fp, glob=False)
                assert_df_equal(want, got_df, msg=msg, **df_check_kwargs)
            case _:
                raise ValueError(f"Unknown file suffix: {file_suffix}")

    if assert_no_other_outputs:
        all_outputs = []
        for suffix in all_file_suffixes:
            all_outputs.extend(list(cohort_dir.glob(f"**/*{suffix}")))
        assert len(want_outputs) == len(all_outputs), (
            f"Want {len(want_outputs)} outputs, but found {len(all_outputs)}.\n"
            f"Found outputs: {[fp.relative_to(cohort_dir) for fp in all_outputs]}\n"
        )


def MEDS_transforms_pipeline_tester(
    *,
    script: str,
    test_name: str,
    do_use_config_yaml: bool = False,
    want_outputs: dict[str, pl.DataFrame] | None = None,
    assert_no_other_outputs: bool = True,
    should_error: bool = False,
    config_name: str = "_preprocess",
    input_files: dict[str, FILE_T] | None = None,
    input_dir: Path | None = None,
    df_check_kwargs: dict | None = None,
    do_include_dirs: bool = True,
    stdout_regex: str | None = None,
    **pipeline_kwargs,
):
    if df_check_kwargs is None:
        df_check_kwargs = {}

    with input_dataset(input_dir, input_files) as (input_dir, cohort_dir):
        for k, v in pipeline_kwargs.items():
            if type(v) is str and "{input_dir}" in v:
                pipeline_kwargs[k] = v.format(input_dir=str(input_dir.resolve()))

        pipeline_config_kwargs = {**pipeline_kwargs}

        if do_include_dirs:
            pipeline_config_kwargs["input_dir"] = str(input_dir.resolve())
            pipeline_config_kwargs["cohort_dir"] = str(cohort_dir.resolve())

        # Run the transform
        stderr, stdout = run_command(
            script=script,
            hydra_kwargs=pipeline_config_kwargs,
            test_name=test_name,
            config_name=config_name,
            should_error=should_error,
            do_use_config_yaml=do_use_config_yaml,
        )
        if should_error:
            return

        if stdout_regex is not None:
            regex = re.compile(stdout_regex)
            assert regex.search(stdout) is not None, (
                f"Expected stdout to match regex:\n{stdout_regex}\n" f"Got:\n{stdout}"
            )

        try:
            check_outputs(
                cohort_dir,
                want_outputs=want_outputs,
                assert_no_other_outputs=assert_no_other_outputs,
                **df_check_kwargs,
            )
        except Exception as e:
            raise AssertionError(
                f"{test_name} failed -- {e}:\n" f"Script stdout:\n{stdout}\n" f"Script stderr:\n{stderr}\n"
            ) from e
