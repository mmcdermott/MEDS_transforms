"""Tests for the bin index computed by `bin_numeric_values`.

The bin index of a value is its right-side insertion point into the row's sorted bin endpoints, i.e., the
number of endpoints `<=` the value. `NaN` values sort after every endpoint, and a null value or null bin
endpoints yield a null index. These tests check `_get_val_bin_idx` against that definition directly, via
Python's `bisect.bisect_right`, on property-based inputs, plus a table of hand-checked edge cases.
"""

import math
from bisect import bisect_right

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from MEDS_transforms.stages.bin_numeric_values.bin_numeric_values import _get_val_bin_idx

ENDPOINTS_DTYPE = pl.List(pl.Float64)

# A small pool shared by endpoints and values, so duplicated endpoints and exact endpoint hits are common.
SHARED_POOL = [-1.0, 0.0, 0.5, 1.0, 2.0]

Row = tuple[float | None, list[float] | None]


def expected_bin_idx(value: float | None, endpoints: list[float] | None) -> int | None:
    """The bin index, by definition: the right-side insertion point of `value` into `endpoints`."""
    if value is None or endpoints is None:
        return None
    if math.isnan(value):
        return len(endpoints)
    return bisect_right(endpoints, value)


def bin_idx(df: pl.DataFrame) -> pl.Series:
    return df.select(_get_val_bin_idx(pl.col("bin_endpoints")).alias("idx"))["idx"]


@st.composite
def rows(draw, width: int) -> Row:
    """One `(numeric_value, bin_endpoints)` row.

    Floats are drawn at the given bit `width`, so Float32 values are exactly representable and can tie with
    the endpoints.
    """
    float_or_inf = st.floats(allow_nan=False, width=width)
    endpoint = st.one_of(float_or_inf, st.sampled_from(SHARED_POOL))
    endpoints = sorted(draw(st.lists(endpoint, min_size=1, max_size=12)))

    value = draw(
        st.one_of(
            st.none(),
            st.just(math.nan),
            st.sampled_from(endpoints),  # exactly on an endpoint; ties go right
            st.sampled_from(SHARED_POOL),
            float_or_inf,
        )
    )
    if draw(st.integers(0, 9)) == 0:
        return value, None  # no bin endpoints for this row's code
    return value, endpoints


def check_against_bisect(table: list[Row], value_dtype: pl.DataType):
    values, endpoints = zip(*table, strict=True)
    df = pl.DataFrame(
        {"numeric_value": list(values), "bin_endpoints": list(endpoints)},
        schema={"numeric_value": value_dtype, "bin_endpoints": ENDPOINTS_DTYPE},
    )
    # Read the values back from the frame, so the oracle sees exactly what polars stored for this dtype.
    stored_values = df["numeric_value"].to_list()
    want = [expected_bin_idx(v, e) for v, e in zip(stored_values, endpoints, strict=True)]

    got = bin_idx(df)
    assert got.dtype == pl.UInt32
    assert got.to_list() == want


@settings(deadline=None)
@given(st.lists(rows(width=64), min_size=1, max_size=50))
def test_bin_idx_matches_bisect_float64(table: list[Row]):
    check_against_bisect(table, pl.Float64)


@settings(deadline=None)
@given(st.lists(rows(width=32), min_size=1, max_size=50))
def test_bin_idx_matches_bisect_float32(table: list[Row]):
    check_against_bisect(table, pl.Float32)


def test_bin_idx_edge_cases():
    nan = float("nan")
    df = pl.DataFrame(
        {
            "numeric_value": [nan, -1.0, 0.0, 0.0, 2.0, 5.0, None, 1.0, nan],
            "bin_endpoints": [
                [0.0, 1.0, 2.0],  # NaN sorts after every endpoint
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],  # on the first endpoint
                [0.0, 0.0, 1.0],  # duplicated endpoints
                [0.0, 1.0, 2.0],  # on the last endpoint
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                None,
                [0.0],
            ],
        },
        schema={"numeric_value": pl.Float64, "bin_endpoints": ENDPOINTS_DTYPE},
    )
    assert bin_idx(df).to_list() == [3, 0, 1, 2, 3, 3, None, None, 1]
