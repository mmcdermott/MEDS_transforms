"""Example stage demonstrating non-MEDS input data.

This stage serves as a reference for how downstream packages (like MEDS-Extract) can define stages
that take raw input files (CSVs, configs, etc.) rather than MEDS-formatted data. Such stages use the
``yaml_to_disk`` fallback in the ``StageExample`` framework for testing and documentation.

Because the standard ``MEDS_transform-stage`` runner expects MEDS-formatted input, stages like this
typically have their own custom runner. The example here is intentionally minimal -- it registers a
no-op stage to demonstrate the example/documentation infrastructure, not the stage logic itself.
"""

from collections.abc import Callable

import polars as pl
from omegaconf import DictConfig

from MEDS_transforms.compute_modes.compute_fn import identity_fn
from MEDS_transforms.stages import Stage


@Stage.register
def csv_to_meds(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Converts raw CSV files into MEDS format.

    This is a placeholder stage demonstrating how to define a stage with non-MEDS input. In a real
    downstream package like MEDS-Extract, this would read raw CSV files and convert them into the MEDS
    schema. Here it simply returns an identity function.

    The example for this stage uses ``in.yaml`` with raw file paths (CSVs, JSON, YAML) instead of
    MEDS-formatted data, which triggers the ``yaml_to_disk`` fallback in the ``StageExample`` framework.
    """

    return identity_fn
