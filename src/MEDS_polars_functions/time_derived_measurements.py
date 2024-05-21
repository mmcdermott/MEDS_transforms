"""Transformations for adding time-derived measurements (e.g., a patient's age) to a MEDS dataset."""

from collections.abc import Callable

import polars as pl
from omegaconf import DictConfig


def add_new_events_fntr(fn: Callable[[pl.DataFrame], pl.DataFrame]) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Creates a "meta" functor that computes the input functor on a MEDS shard then combines both dataframes.

    Args:
        fn: The function that computes the new events.

    Returns:
        A function that computes the new events and combines them with the original DataFrame, returning a
        result in proper MEDS sorted order.

    Examples: TODO
    """
    raise NotImplementedError("TODO")


def age_fntr(cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Create a function that adds a patient's age to a DataFrame.

    Args:
        cfg: The configuration for the age function. This must contain the following mandatory keys:
            - "DOB_code": The code for the date of birth event in the raw data.
            - "age_code": The code for the age event in the output data.
            - "age_unit": The unit for the age event when converted to a numeric value in the output data.

    Returns:
        A function that returns the to-be-added "age" events with the patient's age for all input events with
        unique, non-null timestamps in the data, for all patients who have an observed date of birth.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "patient_id": [1, 1, 1, 1, 2, 2, 3, 3],
        ...         "timestamp": [
        ...             None,
        ...             datetime(1990, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(2021, 1, 1),
        ...             datetime(1988, 1, 2),
        ...             datetime(2023, 1, 3),
        ...             datetime(2022, 1, 1),
        ...             datetime(2022, 1, 1),
        ...         ],
        ...         "code": ["static", "DOB", "lab//A", "lab//B", "DOB", "lab//A", "lab//B"],
        ...     },
        ...     schema={"patient_id": pl.UInt32, "timestamp": pl.Date32, "code": pl.Categorical},
        ... )
        >>> age_cfg = DictConfig({"DOB_code": "DOB", "age_code": "AGE", "age_unit": "years"})
        >>> age_fn = age_fntr(age_cfg)
        >>> age_fn(df)
    """
    raise NotImplementedError("TODO")


def time_of_day_fntr(cfg: DictConfig) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Create a function that adds the time of day to a DataFrame.

    Args:
        cfg: The configuration for the time of day function. This must contain an "endpoints" key with the
            endpoints (in hours of the day in 24-hour time) for each time of day category and a
            "time_of_day_code" key for the code for the time of day event in the output data.

    Returns:
        A function that adds the time of day to a DataFrame.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame(
        ...     {
        ...         "patient_id": [1, 1, 1, 1, 2, 2, 3, 3],
        ...         "timestamp": [
        ...             None,
        ...             datetime(1990, 1, 1, 1, 0),
        ...             datetime(2021, 1, 1, 12, 0),
        ...             datetime(2021, 1, 2, 23, 59),
        ...             datetime(1988, 1, 2, 6, 0),
        ...             datetime(2023, 1, 3, 12, 0),
        ...             datetime(2022, 1, 1, 18, 0),
        ...             datetime(2022, 1, 2, 0, 0),
        ...         ],
        ...         "code": ["static", "DOB", "lab//A", "lab//B", "DOB", "lab//A", "lab//B"],
        ...     },
        ...     schema={"patient_id": pl.UInt32, "timestamp": pl.Date32, "code": pl.Categorical},
        ... )
        >>> time_of_day_cfg = DictConfig({"time_of_day_code": "time_of_day", "endpoints": [6, 12, 18]})
        >>> time_of_day_fn = time_of_day_fntr(time_of_day_cfg)
        >>> time_of_day_fn(df)
    """
    raise NotImplementedError("TODO")
