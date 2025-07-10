import importlib
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal
from pretty_print_directory import print_directory


def test_simple_example_pipeline():
    examples_dir = Path(__file__).resolve().parents[1] / "example"
    pkg_dir = examples_dir / "simple_example_pkg"
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(pkg_dir)], check=True)

    sys.path.insert(0, str(pkg_dir / "src"))
    importlib.invalidate_caches()

    input_dir = examples_dir / "data"
    regression_target = examples_dir / "output_data"

    want_data_files = list((regression_target / "data").rglob("*.parquet"))
    if len(want_data_files) == 0:
        raise ValueError("No regression target data files found.")

    want_metadata_files = list((regression_target / "metadata").rglob("*.*"))
    if len(want_metadata_files) == 0:
        raise ValueError("No regression target metadata files found.")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        cmd = [
            "MEDS_transform-pipeline",
            "pkg://simple_example_pkg.pipelines.example_pipeline.yaml",
            "--overrides",
            f"input_dir={input_dir!s}",
            f"output_dir={output_dir!s}",
        ]

        out = subprocess.run(cmd, check=False, capture_output=True)

        stdout = out.stdout.decode("utf-8")
        stderr = out.stderr.decode("utf-8")

        sio = StringIO()
        print_directory(output_dir, file=sio)
        dir_contents = sio.getvalue()

        pipeline_log_fp = output_dir / ".logs" / "pipeline.log"
        pipeline_log = pipeline_log_fp.read_text() if pipeline_log_fp.exists() else "DOES NOT EXIST"

        msg_parts = {
            "Directory contents": dir_contents,
            "Pipeline log": pipeline_log,
            "stdout": stdout,
            "stderr": stderr,
        }
        msg_parts_str = "\n".join(f"{k}:\n{v}" for k, v in msg_parts.items())

        assert out.returncode == 0, f"Error running pipeline:\n{msg_parts_str}"

        got_data_files = list((output_dir / "data").rglob("*.parquet"))
        assert len(got_data_files) == len(want_data_files), (
            f"Expected {len(want_data_files)} data files, but got {len(got_data_files)}.\n{msg_parts_str}"
        )

        for fp in got_data_files:
            fn = fp.relative_to(output_dir)
            want_fp = regression_target / fn
            assert want_fp.exists(), f"Expected output file {want_fp} does not exist."

            want_df = pl.read_parquet(want_fp)
            got_df = pl.read_parquet(fp)
            assert_frame_equal(want_df, got_df)

        got_metadata_files = list((output_dir / "metadata").rglob("*.*"))
        assert len(got_metadata_files) == len(want_metadata_files), (
            f"Expected {len(want_metadata_files)} metadata files, but got {len(got_metadata_files)}.\n"
            f"{msg_parts_str}"
        )

        for fp in got_metadata_files:
            fn = fp.relative_to(output_dir)
            want_fp = regression_target / fn
            assert want_fp.exists(), f"Expected metadata file {want_fp} does not exist."

            if fp.suffix == ".parquet":
                want_df = pl.read_parquet(want_fp)
                got_df = pl.read_parquet(fp)
                assert_frame_equal(want_df, got_df)
            elif fp.suffix == ".json":
                assert want_fp.read_text() == fp.read_text(), (
                    f"Expected metadata file {want_fp} does not match got {fp}."
                )
