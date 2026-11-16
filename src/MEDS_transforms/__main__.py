import logging
import sys
from uuid import uuid4

import hydra

from .configs import PipelineConfig
from .stage_launch import register_stage, stamp_run_id
from .stage_outputs import CaptureStageConfig, carry_forward_unchanged_outputs
from .stages.discovery import get_all_registered_stages
from .utils import invocation_name

logger = logging.getLogger(__name__)

HELP_STRS = {"--help", "-h", "help", "h"}


def print_help_stage():
    """Print help for all stages."""

    from ._cli_help import pipeline_keys_help_block

    all_stage_names = list(get_all_registered_stages().keys())

    print(f"Usage: {invocation_name()} <pipeline_yaml> <stage_name> [args]")
    print(
        "  * pipeline_yaml: Path to the pipeline YAML file on disk or in the "
        "'pkg://<pkg_name>.<relative_path>' format."
    )
    print("  * stage_name: Name of the stage to run.")
    print()
    print("Available stages:")
    for name in sorted(all_stage_names):
        print(f"  - {name}")
    print()
    print(pipeline_keys_help_block())


def run_stage():  # pragma: no cover
    """Run a stage based on command line arguments."""

    if len(sys.argv) < 2:
        print_help_stage()
        sys.exit(1)
    elif sys.argv[1] in HELP_STRS:
        print_help_stage()
        sys.exit(0)
    elif len(sys.argv) < 3:
        print_help_stage()
        sys.exit(1)

    pipeline_cfg = PipelineConfig.from_arg(sys.argv[1])
    stage_name = sys.argv[2]

    sys.argv = sys.argv[2:]  # remove dispatcher arguments

    # Stamped here, in the single dispatcher process, so that a `--multirun` sweep hands the same value
    # to every worker it launches.
    sys.argv = stamp_run_id(sys.argv, uuid4().hex)

    stage = register_stage(pipeline_cfg, stage_name)
    hydra_wrapper = hydra.main(version_base=None, config_name="_main")

    CaptureStageConfig.config = None
    hydra_wrapper(stage.main)()

    # Hydra only returns once every worker it launched has finished, and only if all of them succeeded, so
    # this runs once per invocation of this command. It is idempotent, so concurrent invocations of the same
    # stage (which share its work between them) may each run it.
    if CaptureStageConfig.config is not None:
        carry_forward_unchanged_outputs(
            CaptureStageConfig.config, stage.stage_type, stage.is_metadata, stage.carry_input_data
        )


if __name__ == "__main__":
    run_stage()
