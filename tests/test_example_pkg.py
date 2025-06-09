import importlib
import subprocess
import sys
from importlib import resources
from pathlib import Path

from MEDS_transforms.pytest_plugin import pipeline_tester


def test_simple_example_pipeline():
    pkg_dir = Path(__file__).resolve().parents[1] / "examples" / "simple_example_pkg"
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(pkg_dir)], check=True)

    sys.path.insert(0, str(pkg_dir / "src"))
    importlib.invalidate_caches()

    pipeline_fp = resources.files("simple_example_pkg.pipelines").joinpath("identity_pipeline.yaml")
    pipeline_yaml = pipeline_fp.read_text()

    pipeline_tester(pipeline_yaml, None, ["identity_stage"])
