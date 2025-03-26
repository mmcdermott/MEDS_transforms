"""A polars-to-polars transformation function for filtering subjects by sequence length."""

import logging
from collections.abc import Callable
from functools import partial

import polars as pl
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


from ..stages import registered_stage


def filter_subjects_by_num_measurements(df: pl.LazyFrame, min_measurements_per_subject: int) -> pl.LazyFrame:
    """Filters subjects by the number of dynamic (timestamp non-null) measurements they have.

    Args:
        df: The input DataFrame.
        min_measurements_per_subject: The minimum number of measurements a subject must have to be included.

    Returns:
        The filtered DataFrame.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2, 3, 3,    4],
        ...     "time":       [1, 2, 1, 1, 2, 1, None, None],
        ... })
        >>> filter_subjects_by_num_measurements(df, 1)
        shape: (7, 2)
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
        │ 3          ┆ null │
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
        >>> with pl.Config(tbl_rows=15):
        ...     filter_subjects_by_num_events(df, 1)
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
        >>> with pl.Config(tbl_rows=15):
        ...     filter_subjects_by_num_events(df, 2)
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


@registered_stage
def main(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that filters subjects by the number of measurements and events they have.

    Args:
        stage_cfg: The stage configuration. Arguments include: min_measurements_per_subject,
            min_events_per_subject, both of which should be integers or None which specify the minimum number
            of measurements and events a subject must have to be included, respectively.

    Returns:
        The function that filters subjects by the number of measurements and/or events they have.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1, 1, 2, 2, 2, 3,    3,    4, 4, 4, 4,    5, 5, 5, 5],
        ...     "time":       [1, 1, 1, 1, 1, 1, 2, 3, None, None, 1, 2, 2, None, 1, 2, 3, 1],
        ... })
        >>> stage_cfg = DictConfig({"min_measurements_per_subject": 4, "min_events_per_subject": 2})
        >>> filter_subjects_fntr(stage_cfg)(df)
        shape: (4, 2)
        ┌────────────┬──────┐
        │ subject_id ┆ time │
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
