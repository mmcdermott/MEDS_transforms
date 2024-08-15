import json
from datetime import datetime
from pathlib import Path

import hydra
from hydra.experimental.callback import Callback
from hydra.types import TaskFunction
from memray import Tracker
from omegaconf import DictConfig


class ProfilerCallback(Callback):
    def __init__(self):
        self.memray_tracker = None

    def on_job_start(self, task_function: TaskFunction, config: DictConfig) -> None:
        hydra_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        job_name = hydra.core.hydra_config.HydraConfig.get().job.name
        memray_fp = hydra_path / f"{job_name}.memray"
        self.memray_tracker = Tracker(memray_fp)
        self.memray_tracker.__enter__()

        st = datetime.now()
        timing_fp = hydra_path / f"{job_name}.timing.json"
        timings = {"start": st.isoformat()}
        print(f"START: Writing {timings} to {timing_fp}")
        timing_fp.write_text(json.dumps(timings))

    def on_job_end(self, config: DictConfig, job_return: hydra.core.utils.JobReturn) -> None:
        hydra_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        job_name = hydra.core.hydra_config.HydraConfig.get().job.name
        if self.memray_tracker is None:
            print("WARNING: Tracker is None")
        else:
            self.memray_tracker.__exit__(None, None, None)

        end = datetime.now()
        timing_fp = hydra_path / f"{job_name}.timing.json"
        print(f"END: Loading timings from {timing_fp}")
        timings = json.loads(timing_fp.read_text())
        timings["end"] = end.isoformat()
        timings["duration_seconds"] = (end - datetime.fromisoformat(timings["start"])).total_seconds()
        print(f"END: Writing {timings} to {timing_fp}")
        timing_fp.write_text(json.dumps(timings))
