import subprocess
import tempfile
from io import StringIO
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf
from polars.testing import assert_frame_equal

DEFAULT_CSV_TS_FORMAT = "%m/%d/%Y, %H:%M:%S"

# TODO: Make use meds library
MEDS_PL_SCHEMA = {
    "patient_id": pl.UInt32,
    "time": pl.Datetime("us"),
    "code": pl.String,
    "numeric_value": pl.Float32,
    "numeric_value/is_inlier": pl.Boolean,
    "text_value": pl.String,
}


def parse_meds_csvs(
    csvs: str | dict[str, str], schema: dict[str, pl.DataType] = MEDS_PL_SCHEMA
) -> pl.DataFrame | dict[str, pl.DataFrame]:
    """Converts a string or dict of named strings to a MEDS DataFrame by interpreting them as CSVs.

    TODO: doctests.
    """

    default_read_schema = {**schema}
    default_read_schema["time"] = pl.Utf8

    def reader(csv_str: str) -> pl.DataFrame:
        cols = csv_str.strip().split("\n")[0].split(",")
        read_schema = {k: v for k, v in default_read_schema.items() if k in cols}
        return pl.read_csv(StringIO(csv_str), schema=read_schema).with_columns(
            pl.col("time").str.strptime(MEDS_PL_SCHEMA["time"], DEFAULT_CSV_TS_FORMAT)
        )

    if isinstance(csvs, str):
        return reader(csvs)
    else:
        return {k: reader(v) for k, v in csvs.items()}


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
                    if inner_kv.startswith("~"):
                        out.append(f"~{k}.{inner_kv[1:]}")
                    elif inner_kv.startswith("'"):
                        out.append(f"'{k}.{inner_kv[1:]}")
                    else:
                        out.append(f"{k}.{inner_kv}")
            case list() | tuple():
                v = list(v)
                v_str_inner = ", ".join(f'"{x}"' for x in v)
                out.append(f"'{k}=[{v_str_inner}]'")
            case _:
                raise ValueError(f"Unexpected type for value for key {k}: {type(v)}: {v}")

    return out


def run_command(
    script: Path | str,
    hydra_kwargs: dict[str, str],
    test_name: str,
    config_name: str | None = None,
    should_error: bool = False,
    do_use_config_yaml: bool = False,
    stage_name: str | None = None,
    do_pass_stage_name: bool = False,
):
    script = ["python", str(script.resolve())] if isinstance(script, Path) else [script]
    command_parts = script

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
        if config_name is not None:
            command_parts.append(f"--config-name={config_name}")
        command_parts.append(" ".join(dict_to_hydra_kwargs(hydra_kwargs)))

    if do_pass_stage_name:
        if stage_name is None:
            raise ValueError("stage_name must be provided if do_pass_stage_name is True.")
        command_parts.append(f"stage={stage_name}")

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
        assert_frame_equal(want, got, **kwargs)
    except AssertionError as e:
        pl.Config.set_tbl_rows(-1)
        print(f"DFs are not equal: {msg}\nwant:")
        print(want)
        print("got:")
        print(got)
        raise AssertionError(f"{msg}\n{e}") from e
