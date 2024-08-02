import subprocess
from io import StringIO
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

DEFAULT_CSV_TS_FORMAT = "%m/%d/%Y, %H:%M:%S"

# TODO: Make use meds library
MEDS_PL_SCHEMA = {
    "patient_id": pl.UInt32,
    "time": pl.Datetime("us"),
    "code": pl.Utf8,
    "numerical_value": pl.Float64,
}


def parse_meds_csvs(
    csvs: str | dict[str, str], schema: dict[str, pl.DataType] = MEDS_PL_SCHEMA
) -> pl.DataFrame | dict[str, pl.DataFrame]:
    """Converts a string or dict of named strings to a MEDS DataFrame by interpreting them as CSVs.

    TODO: doctests.
    """

    read_schema = {**schema}
    read_schema["time"] = pl.Utf8

    def reader(csv_str: str) -> pl.DataFrame:
        return pl.read_csv(StringIO(csv_str), schema=read_schema).with_columns(
            pl.col("time").str.strptime(MEDS_PL_SCHEMA["time"], DEFAULT_CSV_TS_FORMAT)
        )

    if isinstance(csvs, str):
        return reader(csvs)
    else:
        return {k: reader(v) for k, v in csvs.items()}


def dict_to_hydra_kwargs(d: dict[str, str]) -> str:
    """Converts a dictionary to a hydra kwargs string for testing purposes.

    TODO: doctests.
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
            case str() | int() | float() | list():
                out.append(f"{k}={v}")
            case dict():
                inner_kwargs = dict_to_hydra_kwargs(v)
                for inner_kv in inner_kwargs.split():
                    if inner_kv.startswith("~"):
                        out.append(f"~{k}.{inner_kv[1:]}")
                    else:
                        out.append(f"{k}.{inner_kv}")
            case _:
                raise ValueError(f"Unexpected type for value for key {k}: {type(v)}: {v}")

    return " ".join(out)


def run_command(
    script: Path | str,
    hydra_kwargs: dict[str, str],
    test_name: str,
    config_name: str | None = None,
    should_error: bool = False,
):
    script = ["python", str(script.resolve())] if isinstance(script, Path) else [script]
    command_parts = script
    if config_name is not None:
        command_parts.append(f"--config-name={config_name}")
    command_parts.append(dict_to_hydra_kwargs(hydra_kwargs))

    full_cmd = " ".join(command_parts)
    command_out = subprocess.run(full_cmd, shell=True, capture_output=True)

    command_errored = command_out.returncode != 0

    stderr = command_out.stderr.decode()
    stdout = command_out.stdout.decode()

    if should_error and not command_errored:
        raise AssertionError(
            f"{test_name} failed as command did not error when expected!\n"
            f"command:{full_cmd}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    elif not should_error and command_errored:
        raise AssertionError(
            f"{test_name} failed as command errored when not expected!"
            f"\ncommand:{full_cmd}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
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
