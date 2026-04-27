"""End-to-end tests for ``MEDS_transform-pipeline --in_memory``.

These tests exercise the runner-level integration of in-memory mode (Phase 1 of #56). They
boot a real pipeline configuration, run all stages in-process under a shared
``FrameRegistry``, and verify:

- A multi-stage pipeline (MAP → MAPREDUCE) completes with no parquet round-trips between stages.
- The output is bit-equal to the disk-mode subprocess runner's output for the same pipeline.
- Incompatible flags (``--stage_runner_fp``, ``--do_profile``) raise ``ValueError``, surfacing
  the limitation rather than silently degrading.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from MEDS_transforms.runner import main as runner_main
from MEDS_transforms.stages.discovery import get_all_registered_stages

if TYPE_CHECKING:
    from pathlib import Path

PIPELINE_YAML = """
input_dir: {input_dir}
output_dir: {output_dir}

stages:
  - filter_subjects:
      min_events_per_subject: 5
  - aggregate_code_metadata:
      aggregations:
        - "code/n_occurrences"
        - "code/n_subjects"
"""


def _seed_input(input_dir: Path) -> None:
    stages = get_all_registered_stages()
    fs = stages["filter_subjects"].load()
    ex = fs.test_cases["."]
    ex.write_for_test(input_dir)


def test_in_memory_runner_writes_no_data_parquet(tmp_path: Path):
    """A two-stage pipeline run with --in_memory leaves no data/metadata parquet on disk."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _seed_input(input_dir)

    pipeline_fp = tmp_path / "pipeline.yaml"
    pipeline_fp.write_text(PIPELINE_YAML.format(input_dir=input_dir, output_dir=output_dir))

    rc = runner_main([str(pipeline_fp), "--in_memory"])
    assert rc == 0

    # Only logs/done-files should land on disk.
    on_disk = sorted(p.relative_to(output_dir) for p in output_dir.rglob("*") if p.is_file())
    assert all(str(p).startswith(".logs") for p in on_disk), (
        f"in-memory mode wrote unexpected files: {on_disk}"
    )

    # Each stage's done-file is present so resumes work.
    assert (output_dir / ".logs" / "filter_subjects.done").exists()
    assert (output_dir / ".logs" / "aggregate_code_metadata.done").exists()
    assert (output_dir / ".logs" / "_all_stages.done").exists()


def test_in_memory_matches_disk_pipeline(tmp_path: Path):
    """Same pipeline yields bit-equal codes.parquet under --in_memory and disk-mode runner."""
    input_dir = tmp_path / "input"
    _seed_input(input_dir)

    # 1. Disk-mode reference: subprocess invocation through the installed CLI.
    disk_dir = tmp_path / "disk_out"
    disk_pipeline = tmp_path / "pipeline_disk.yaml"
    disk_pipeline.write_text(PIPELINE_YAML.format(input_dir=input_dir, output_dir=disk_dir))
    out = subprocess.run(["MEDS_transform-pipeline", str(disk_pipeline)], capture_output=True, check=False)
    assert out.returncode == 0, out.stderr.decode()
    disk_codes = pl.read_parquet(disk_dir / "metadata" / "codes.parquet").sort("code")

    # 2. In-memory: drive the same pipeline through ``runner_main`` while capturing the registry
    # by entering ``in_memory_mode`` ourselves and calling the in-process loop directly. We can't
    # round-trip through ``runner_main(["--in_memory"])`` because that writes nothing to disk and
    # the registry is dropped on exit.
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from MEDS_transforms import __package_name__, __version__
    from MEDS_transforms.__main__ import MAIN_CFG_PATH
    from MEDS_transforms.compute_modes import FrameRegistry, in_memory_mode
    from MEDS_transforms.configs import PipelineConfig

    inmem_dir = tmp_path / "inmem_out"
    inmem_pipeline = tmp_path / "pipeline_inmem.yaml"
    inmem_pipeline.write_text(PIPELINE_YAML.format(input_dir=input_dir, output_dir=inmem_dir))

    pipeline_cfg = PipelineConfig.from_arg(str(inmem_pipeline))
    OmegaConf.register_new_resolver("get_package_version", lambda: __version__, replace=True)
    OmegaConf.register_new_resolver("get_package_name", lambda: __package_name__, replace=True)

    registry = FrameRegistry()
    with in_memory_mode(registry):
        for stage_name in [s.name for s in pipeline_cfg.parsed_stages]:
            stage_obj = pipeline_cfg.register_for(stage_name)
            OmegaConf.register_new_resolver("stage_name", lambda sn=stage_name: sn, replace=True)
            OmegaConf.register_new_resolver(
                "stage_docstring",
                lambda s=stage_obj: (s.stage_docstring or "").replace("$", "$$"),
                replace=True,
            )
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            with initialize_config_dir(version_base=None, config_dir=str(MAIN_CFG_PATH.parent.absolute())):
                cfg = compose(config_name="_main", overrides=[f"stage={stage_name}"])
            stage_obj.main(cfg)

    inmem_codes = registry.get(inmem_dir / "metadata" / "codes.parquet").collect().sort("code")

    assert_frame_equal(disk_codes, inmem_codes)


@pytest.mark.parametrize("incompatible_flag", [["--do_profile"]])
def test_in_memory_rejects_incompatible_flags(tmp_path: Path, incompatible_flag: list[str]):
    """``--in_memory`` plus subprocess-only options must error rather than silently misbehave."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _seed_input(input_dir)
    pipeline_fp = tmp_path / "pipeline.yaml"
    pipeline_fp.write_text(PIPELINE_YAML.format(input_dir=input_dir, output_dir=output_dir))

    with pytest.raises(ValueError, match="--in_memory is incompatible"):
        runner_main([str(pipeline_fp), "--in_memory", *incompatible_flag])


def test_in_memory_rejects_stage_runner_fp(tmp_path: Path):
    """``--stage_runner_fp`` is also incompatible — stage-runner subprocesses can't share state."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _seed_input(input_dir)
    pipeline_fp = tmp_path / "pipeline.yaml"
    pipeline_fp.write_text(PIPELINE_YAML.format(input_dir=input_dir, output_dir=output_dir))
    runner_fp = tmp_path / "runner.yaml"
    runner_fp.write_text("parallelize: {}\n")

    with pytest.raises(ValueError, match="--in_memory is incompatible"):
        runner_main([str(pipeline_fp), "--in_memory", "--stage_runner_fp", str(runner_fp)])
