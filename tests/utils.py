import json
import re
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import polars as pl
from meds_testing_helpers.dataset import MEDSDataset
from omegaconf import OmegaConf
from polars.testing import assert_frame_equal
from yaml import load as load_yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from MEDS_transforms.stages.examples import dict_to_hydra_kwargs


def parse_shards_yaml(yaml_str: str, **schema_updates) -> pl.DataFrame:
    data = load_yaml(yaml_str.strip(), Loader=Loader)

    schema_updates = {"numeric_value/is_inlier": pl.Boolean, **schema_updates}

    return {k: MEDSDataset.parse_csv(v, **schema_updates) for k, v in data.items()}


def run_command(
    script: str,
    hydra_kwargs: dict[str, str],
    test_name: str,
    should_error: bool = False,
    do_use_config_yaml: bool = False,
):
    command_parts = [script]

    err_cmd_lines = []

    if do_use_config_yaml:
        conf = OmegaConf.create(
            {
                "defaults": ["_main", "_self_"],
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
