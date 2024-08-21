#!/usr/bin/env python
"""A polars-to-polars transformation function for filtering subjects by sequence length."""
from collections.abc import Callable
from functools import partial

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from MEDS_transforms import PREPROCESS_CONFIG_YAML
from MEDS_transforms.mapreduce.mapper import map_over


def filter_subjects_by_num_measurements(df: pl.LazyFrame, min_measurements_per_subject: int) -> pl.LazyFrame:
    """Filters subjects by the number of measurements they have.

    Args:
        df: The input DataFrame.
        min_measurements_per_subject: The minimum number of measurements a subject must have to be included.

    Returns:
        The filtered DataFrame.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2, 3],
        ...     "time": [1, 2, 1, 1, 2, 1],
        ... })
        >>> filter_subjects_by_num_measurements(df, 1)
        shape: (6, 2)
        ┌────────────┬──────┐
        │ subject_id ┆ time │
        │ ---        ┆ ---  │
        │ i64        ┆ i64  │
        ╞════════════╪══════╡
        │ 1          ┆ 1    │
        │ 1          ┆ 2    │
        │ 1          ┆ 1    │
        │ 2          ┆ 1    │
        │ 2          ┆ 2    │
        │ 3          ┆ 1    │
        └────────────┴──────┘
        >>> filter_subjects_by_num_measurements(df, 2)
        shape: (5, 2)
        ┌────────────┬──────┐
        │ subject_id ┆ time │
        │ ---        ┆ ---  │
        │ i64        ┆ i64  │
        ╞════════════╪══════╡
        │ 1          ┆ 1    │
        │ 1          ┆ 2    │
        │ 1          ┆ 1    │
        │ 2          ┆ 1    │
        │ 2          ┆ 2    │
        └────────────┴──────┘
        >>> filter_subjects_by_num_measurements(df, 3)
        shape: (3, 2)
        ┌────────────┬──────┐
        │ subject_id ┆ time │
        │ ---        ┆ ---  │
        │ i64        ┆ i64  │
        ╞════════════╪══════╡
        │ 1          ┆ 1    │
        │ 1          ┆ 2    │
        │ 1          ┆ 1    │
        └────────────┴──────┘
        >>> filter_subjects_by_num_measurements(df, 4)
        shape: (0, 2)
        ┌────────────┬──────┐
        │ subject_id ┆ time │
        │ ---        ┆ ---  │
        │ i64        ┆ i64  │
        ╞════════════╪══════╡
        └────────────┴──────┘
        >>> filter_subjects_by_num_measurements(df, 2.2)
        Traceback (most recent call last):
            ...
        TypeError: min_measurements_per_subject must be an integer; got <class 'float'> 2.2
    """
    if not isinstance(min_measurements_per_subject, int):
        raise TypeError(
            f"min_measurements_per_subject must be an integer; got {type(min_measurements_per_subject)} "
            f"{min_measurements_per_subject}"
        )

    return df.filter(pl.col("time").count().over("subject_id") >= min_measurements_per_subject)


def filter_subjects_by_num_events(df: pl.LazyFrame, min_events_per_subject: int) -> pl.LazyFrame:
    """Filters subjects by the number of events (unique timepoints) they have.

    Args:
        df: The input DataFrame.
        min_events_per_subject: The minimum number of events a subject must have to be included.

    Returns:
        The filtered DataFrame.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4],
        ...     "time": [1, 1, 1, 1, 2, 1, 1, 2, 3, None, None, 1, 2, 3],
        ... })
        >>> filter_subjects_by_num_events(df, 1)
        shape: (14, 2)
        ┌────────────┬──────┐
        │ subject_id ┆ time │
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
        >>> filter_subjects_by_num_events(df, 2)
        shape: (11, 2)
        ┌────────────┬──────┐
        │ subject_id ┆ time │
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
        >>> filter_subjects_by_num_events(df, 3)
        shape: (8, 2)
        ┌────────────┬──────┐
        │ subject_id ┆ time │
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
        >>> filter_subjects_by_num_events(df, 4)
        shape: (5, 2)
        ┌────────────┬──────┐
        │ subject_id ┆ time │
        │ ---        ┆ ---  │
        │ i64        ┆ i64  │
        ╞════════════╪══════╡
        │ 4          ┆ null │
        │ 4          ┆ null │
        │ 4          ┆ 1    │
        │ 4          ┆ 2    │
        │ 4          ┆ 3    │
        └────────────┴──────┘
        >>> filter_subjects_by_num_events(df, 5)
        shape: (0, 2)
        ┌────────────┬──────┐
        │ subject_id ┆ time │
        │ ---        ┆ ---  │
        │ i64        ┆ i64  │
        ╞════════════╪══════╡
        └────────────┴──────┘
        >>> filter_subjects_by_num_events(df, 2.2)
        Traceback (most recent call last):
            ...
        TypeError: min_events_per_subject must be an integer; got <class 'float'> 2.2
    """
    if not isinstance(min_events_per_subject, int):
        raise TypeError(
            f"min_events_per_subject must be an integer; got {type(min_events_per_subject)} "
            f"{min_events_per_subject}"
        )

    return df.filter(pl.col("time").n_unique().over("subject_id") >= min_events_per_subject)


def filter_subjects_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    compute_fns = []
    if stage_cfg.min_measurements_per_subject:
        logger.info(
            f"Filtering subjects with fewer than {stage_cfg.min_measurements_per_subject} measurements "
            "(observations of any kind)."
        )
        compute_fns.append(
            partial(
                filter_subjects_by_num_measurements,
                min_measurements_per_subject=stage_cfg.min_measurements_per_subject,
            )
        )
    if stage_cfg.min_events_per_subject:
        logger.info(
            f"Filtering subjects with fewer than {stage_cfg.min_events_per_subject} events "
            "(unique timepoints)."
        )
        compute_fns.append(
            partial(filter_subjects_by_num_events, min_events_per_subject=stage_cfg.min_events_per_subject)
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

    map_over(cfg, compute_fn=filter_subjects_fntr)


if __name__ == "__main__":
    main()
