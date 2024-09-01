import os

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

code_root = root / "src" / "MEDS_transforms"

USE_LOCAL_SCRIPTS = os.environ.get("DO_USE_LOCAL_SCRIPTS", "0") == "1"

if USE_LOCAL_SCRIPTS:
    RUNNER_SCRIPT = code_root / "runner.py"
else:
    RUNNER_SCRIPT = "MEDS_transform-runner"
