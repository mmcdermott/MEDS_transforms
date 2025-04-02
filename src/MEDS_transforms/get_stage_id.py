import os
import sys

os.environ["DISABLE_STAGE_VALIDATION"] = "1"

from .stages.discovery import get_all_registered_stages


def main():
    if len(sys.argv) < 2:
        print("Usage: python get_stage_id.py <stage_name>")
        sys.exit(1)

    stage_name = sys.argv[1]

    stages = get_all_registered_stages()

    if stage_name not in stages:
        raise ValueError(f"Stage '{stage_name}' not found. Available stages: {list(stages.keys())}")

    stage = stages[stage_name].load()
    print(stage._ID)
    sys.exit(0)
