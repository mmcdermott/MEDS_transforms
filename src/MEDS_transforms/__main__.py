import sys

from .stages import get_all_registered_stages


def run_stage():
    """Run a stage based on command line arguments."""

    all_stages = get_all_registered_stages()

    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h"):
        print(f"Usage: {sys.argv[0]} <stage_name> [args]")
        print("Available stages:")
        for name in sorted(all_stages):
            print(f"  - {name}")
        if len(sys.argv) < 2:
            sys.exit(1)
        else:
            sys.exit(0)

    stage_name = sys.argv[1]
    sys.argv = sys.argv[1:]  # remove dispatcher argument

    if stage_name not in all_stages:
        raise ValueError(f"Stage '{stage_name}' not found.")

    main_fn = all_stages[stage_name]["entry_point"].load()
    main_fn()
