"""A polars-to-polars transformation function for filtering patients by sequence length."""

import polars as pl


def filter_patients_by_num_measurements(df: pl.LazyFrame, min_measurements_per_patient: int) -> pl.LazyFrame:
    """Filters patients by the number of measurements they have.

    Args:
        df: The input DataFrame.
        min_measurements_per_patient: The minimum number of measurements a patient must have to be included.

    Returns:
        The filtered DataFrame.

    Examples:
        >>> df = pl.DataFrame({
        ...     "patient_id": [1, 1, 1, 2, 2, 3],
        ...     "timestamp": [1, 2, 1, 1, 2, 1],
        ... })
        >>> filter_patients_by_num_measurements(df, 1)
        shape: (6, 2)
        ┌────────────┬───────────┐
        │ patient_id ┆ timestamp │
        │ ---        ┆ ---       │
        │ i64        ┆ i64       │
        ╞════════════╪═══════════╡
        │ 1          ┆ 1         │
        │ 1          ┆ 2         │
        │ 1          ┆ 1         │
        │ 2          ┆ 1         │
        │ 2          ┆ 2         │
        │ 3          ┆ 1         │
        └────────────┴───────────┘
        >>> filter_patients_by_num_measurements(df, 2)
        shape: (5, 2)
        ┌────────────┬───────────┐
        │ patient_id ┆ timestamp │
        │ ---        ┆ ---       │
        │ i64        ┆ i64       │
        ╞════════════╪═══════════╡
        │ 1          ┆ 1         │
        │ 1          ┆ 2         │
        │ 1          ┆ 1         │
        │ 2          ┆ 1         │
        │ 2          ┆ 2         │
        └────────────┴───────────┘
        >>> filter_patients_by_num_measurements(df, 3)
        shape: (3, 2)
        ┌────────────┬───────────┐
        │ patient_id ┆ timestamp │
        │ ---        ┆ ---       │
        │ i64        ┆ i64       │
        ╞════════════╪═══════════╡
        │ 1          ┆ 1         │
        │ 1          ┆ 2         │
        │ 1          ┆ 1         │
        └────────────┴───────────┘
        >>> filter_patients_by_num_measurements(df, 4)
        shape: (0, 2)
        ┌────────────┬───────────┐
        │ patient_id ┆ timestamp │
        │ ---        ┆ ---       │
        │ i64        ┆ i64       │
        ╞════════════╪═══════════╡
        └────────────┴───────────┘
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

    return df.filter(pl.col("timestamp").count().over("patient_id") >= min_measurements_per_patient)


def filter_patients_by_num_events(df: pl.LazyFrame, min_events_per_patient: int) -> pl.LazyFrame:
    """Filters patients by the number of events (unique timepoints) they have.

    Args:
        df: The input DataFrame.
        min_events_per_patient: The minimum number of events a patient must have to be included.

    Returns:
        The filtered DataFrame.

    Examples:
        >>> df = pl.DataFrame({
        ...     "patient_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        ...     "timestamp": [1, 1, 1, 1, 2, 1, 1, 2, 3],
        ... })
        >>> filter_patients_by_num_events(df, 1)
        shape: (9, 2)
        ┌────────────┬───────────┐
        │ patient_id ┆ timestamp │
        │ ---        ┆ ---       │
        │ i64        ┆ i64       │
        ╞════════════╪═══════════╡
        │ 1          ┆ 1         │
        │ 1          ┆ 1         │
        │ 1          ┆ 1         │
        │ 2          ┆ 1         │
        │ 2          ┆ 2         │
        │ 2          ┆ 1         │
        │ 3          ┆ 1         │
        │ 3          ┆ 2         │
        │ 3          ┆ 3         │
        └────────────┴───────────┘
        >>> filter_patients_by_num_events(df, 2)
        shape: (6, 2)
        ┌────────────┬───────────┐
        │ patient_id ┆ timestamp │
        │ ---        ┆ ---       │
        │ i64        ┆ i64       │
        ╞════════════╪═══════════╡
        │ 2          ┆ 1         │
        │ 2          ┆ 2         │
        │ 2          ┆ 1         │
        │ 3          ┆ 1         │
        │ 3          ┆ 2         │
        │ 3          ┆ 3         │
        └────────────┴───────────┘
        >>> filter_patients_by_num_events(df, 3)
        shape: (3, 2)
        ┌────────────┬───────────┐
        │ patient_id ┆ timestamp │
        │ ---        ┆ ---       │
        │ i64        ┆ i64       │
        ╞════════════╪═══════════╡
        │ 3          ┆ 1         │
        │ 3          ┆ 2         │
        │ 3          ┆ 3         │
        └────────────┴───────────┘
        >>> filter_patients_by_num_events(df, 4)
        shape: (0, 2)
        ┌────────────┬───────────┐
        │ patient_id ┆ timestamp │
        │ ---        ┆ ---       │
        │ i64        ┆ i64       │
        ╞════════════╪═══════════╡
        └────────────┴───────────┘
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

    return df.filter(pl.col("timestamp").n_unique().over("patient_id") >= min_events_per_patient)
