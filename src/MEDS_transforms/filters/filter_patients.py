#!/usr/bin/env python
"""A polars-to-polars transformation function for filtering patients by sequence length."""
from collections.abc import Callable
from functools import partial

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def filter_patients_by_num_measurements(df: pl.LazyFrame, min_measurements_per_patient: int) -> pl.LazyFrame:
    """Filters patients by the number of dynamic (timestamp non-null) measurements they have.

    Args:
        df: The input DataFrame.
        min_measurements_per_patient: The minimum number of measurements a patient must have to be included.

    Returns:
        The filtered DataFrame.

    Examples:
        >>> df = pl.DataFrame({
        ...     "patient_id": [1, 1, 1, 2, 2, 3, 3,    4],
        ...     "time":       [1, 2, 1, 1, 2, 1, None, None],
        ... })
        >>> filter_patients_by_num_measurements(df, 1)
        shape: (7, 2)
        ┌────────────┬──────┐
        │ patient_id ┆ time │
        │ ---        ┆ ---  │
        │ i64        ┆ i64  │
        ╞════════════╪══════╡
        │ 1          ┆ 1    │
        │ 1          ┆ 2    │
        │ 1          ┆ 1    │
        │ 2          ┆ 1    │
        │ 2          ┆ 2    │
        │ 3          ┆ 1    │
        │ 3          ┆ null │
        └────────────┴──────┘
        >>> filter_patients_by_num_measurements(df, 2)
        shape: (5, 2)
        ┌────────────┬──────┐
        │ patient_id ┆ time │
        │ ---        ┆ ---  │
        │ i64        ┆ i64  │
        ╞════════════╪══════╡
        │ 1          ┆ 1    │
        │ 1          ┆ 2    │
        │ 1          ┆ 1    │
        │ 2          ┆ 1    │
        │ 2          ┆ 2    │
        └────────────┴──────┘
        >>> filter_patients_by_num_measurements(df, 3)
        shape: (3, 2)
        ┌────────────┬──────┐
        │ patient_id ┆ time │
        │ ---        ┆ ---  │
        │ i64        ┆ i64  │
        ╞════════════╪══════╡
        │ 1          ┆ 1    │
        │ 1          ┆ 2    │
        │ 1          ┆ 1    │
        └────────────┴──────┘
        >>> filter_patients_by_num_measurements(df, 4)
        shape: (0, 2)
        ┌────────────┬──────┐
        │ patient_id ┆ time │
        │ ---        ┆ ---  │
        │ i64        ┆ i64  │
        ╞════════════╪══════╡
        └────────────┴──────┘
        >>> filter_patients_by_num_measurements(df, 2.2)
        Traceback (most recent call last):
            ...
        TypeError: min_measurements_per_patient must be an integer; got <class 'float'> 2.2
    """
    if not isinstance(min_measurements_per_patient, int):
        raise TypeError(
            f"min_measurements_per_patient must be an integer; got {type(min_measurements_per_patient)} "
            f"{min_measurements_per_patient}"
        )

    return df.filter(pl.col("time").count().over("patient_id") >= min_measurements_per_patient)


def filter_patients_by_num_events(df: pl.LazyFrame, min_events_per_patient: int) -> pl.LazyFrame:
    """Filters patients by the number of events (unique timepoints) they have.

    Args:
        df: The input DataFrame.
        min_events_per_patient: The minimum number of events a patient must have to be included.

    Returns:
        The filtered DataFrame.

    Examples:
        >>> df = pl.DataFrame({
        ...     "patient_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4],
        ...     "time": [1, 1, 1, 1, 2, 1, 1, 2, 3, None, None, 1, 2, 3],
        ... })
        >>> with pl.Config(tbl_rows=15):
        ...     filter_patients_by_num_events(df, 1)
        shape: (14, 2)
        ┌────────────┬──────┐
        │ patient_id ┆ time │
        │ ---        ┆ ---  │
        │ i64        ┆ i64  │
        ╞════════════╪══════╡
        │ 1          ┆ 1    │
        │ 1          ┆ 1    │
        │ 1          ┆ 1    │
        │ 2          ┆ 1    │
        │ 2          ┆ 2    │
        │ 2          ┆ 1    │
        │ 3          ┆ 1    │
        │ 3          ┆ 2    │
        │ 3          ┆ 3    │
        │ 4          ┆ null │
        │ 4          ┆ null │
        │ 4          ┆ 1    │
        │ 4          ┆ 2    │
        │ 4          ┆ 3    │
        └────────────┴──────┘
        >>> with pl.Config(tbl_rows=15):
        ...     filter_patients_by_num_events(df, 2)
        shape: (11, 2)
        ┌────────────┬──────┐
        │ patient_id ┆ time │
        │ ---        ┆ ---  │
        │ i64        ┆ i64  │
        ╞════════════╪══════╡
        │ 2          ┆ 1    │
        │ 2          ┆ 2    │
        │ 2          ┆ 1    │
        │ 3          ┆ 1    │
        │ 3          ┆ 2    │
        │ 3          ┆ 3    │
        │ 4          ┆ null │
        │ 4          ┆ null │
        │ 4          ┆ 1    │
        │ 4          ┆ 2    │
        │ 4          ┆ 3    │
        └────────────┴──────┘
        >>> filter_patients_by_num_events(df, 3)
        shape: (8, 2)
        ┌────────────┬──────┐
        │ patient_id ┆ time │
        │ ---        ┆ ---  │
        │ i64        ┆ i64  │
        ╞════════════╪══════╡
        │ 3          ┆ 1    │
        │ 3          ┆ 2    │
        │ 3          ┆ 3    │
        │ 4          ┆ null │
        │ 4          ┆ null │
        │ 4          ┆ 1    │
        │ 4          ┆ 2    │
        │ 4          ┆ 3    │
        └────────────┴──────┘
        >>> filter_patients_by_num_events(df, 4)
        shape: (5, 2)
        ┌────────────┬──────┐
        │ patient_id ┆ time │
        │ ---        ┆ ---  │
        │ i64        ┆ i64  │
        ╞════════════╪══════╡
        │ 4          ┆ null │
        │ 4          ┆ null │
        │ 4          ┆ 1    │
        │ 4          ┆ 2    │
        │ 4          ┆ 3    │
        └────────────┴──────┘
        >>> filter_patients_by_num_events(df, 5)
        shape: (0, 2)
        ┌────────────┬──────┐
        │ patient_id ┆ time │
        │ ---        ┆ ---  │
        │ i64        ┆ i64  │
        ╞════════════╪══════╡
        └────────────┴──────┘
        >>> filter_patients_by_num_events(df, 2.2)
        Traceback (most recent call last):
            ...
        TypeError: min_events_per_patient must be an integer; got <class 'float'> 2.2
    """
    if not isinstance(min_events_per_patient, int):
        raise TypeError(
            f"min_events_per_patient must be an integer; got {type(min_events_per_patient)} "
            f"{min_events_per_patient}"
        )

    return df.filter(pl.col("time").n_unique().over("patient_id") >= min_events_per_patient)


def filter_patients_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that filters patients by the number of measurements and events they have.

    Args:
        stage_cfg: The stage configuration. Arguments include: min_measurements_per_patient,
            min_events_per_patient, both of which should be integers or None which specify the minimum number
            of measurements and events a patient must have to be included, respectively.

    Returns:
        The function that filters patients by the number of measurements and/or events they have.

    Examples:
        >>> df = pl.DataFrame({
        ...     "patient_id": [1, 1, 1, 1, 1, 2, 2, 2, 3,    3,    4, 4, 4, 4,    5, 5, 5, 5],
        ...     "time":       [1, 1, 1, 1, 1, 1, 2, 3, None, None, 1, 2, 2, None, 1, 2, 3, 1],
        ... })
        >>> stage_cfg = DictConfig({"min_measurements_per_patient": 4, "min_events_per_patient": 2})
        >>> filter_patients_fntr(stage_cfg)(df)
        shape: (4, 2)
        ┌────────────┬──────┐
        │ patient_id ┆ time │
        │ ---        ┆ ---  │
        │ i64        ┆ i64  │
        ╞════════════╪══════╡
        │ 5          ┆ 1    │
        │ 5          ┆ 2    │
        │ 5          ┆ 3    │
        │ 5          ┆ 1    │
        └────────────┴──────┘
    """

    compute_fns = []
    if stage_cfg.min_measurements_per_patient:
        logger.info(
            f"Filtering patients with fewer than {stage_cfg.min_measurements_per_patient} measurements "
            "(observations of any kind)."
        )
        compute_fns.append(
            partial(
                filter_patients_by_num_measurements,
                min_measurements_per_patient=stage_cfg.min_measurements_per_patient,
            )
        )
    if stage_cfg.min_events_per_patient:
        logger.info(
            f"Filtering patients with fewer than {stage_cfg.min_events_per_patient} events "
            "(unique timepoints)."
        )
        compute_fns.append(
            partial(filter_patients_by_num_events, min_events_per_patient=stage_cfg.min_events_per_patient)
        )

    def fn(data: pl.LazyFrame) -> pl.LazyFrame:
        for compute_fn in compute_fns:
            data = compute_fn(data)
        return data

    return fn


@hydra.main(
    version_base=None, config_path=str(PREPROCESS_CONFIG_YAML.parent), config_name=PREPROCESS_CONFIG_YAML.stem
)
def main(cfg: DictConfig):
    """TODO."""

    map_over(cfg, compute_fn=filter_patients_fntr)


if __name__ == "__main__":  # pragma: no cover
    main()
