import importlib
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path

from pretty_print_directory import print_directory


def test_simple_example_pipeline():
    examples_dir = Path(__file__).resolve().parents[1] / "example"
    pkg_dir = examples_dir / "simple_example_pkg"
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(pkg_dir)], check=True)

    sys.path.insert(0, str(pkg_dir / "src"))
    importlib.invalidate_caches()

    pipeline = "pkg://simple_example_pkg.pipelines.example_pipeline.yaml"
    input_dir = examples_dir / "data"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        cmd = [
            "MEDS_transform-pipeline",
            pipeline,
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
