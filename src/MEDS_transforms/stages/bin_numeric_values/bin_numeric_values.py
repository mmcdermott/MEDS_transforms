"""Convert numeric values into categorical code modifiers through pre-computed or specified bins."""

import logging
import re
from collections.abc import Callable
from pathlib import Path

import polars as pl
from meds import CodeMetadataSchema, DataSchema
from omegaconf import DictConfig, OmegaConf

from ...utils import PKG_PFX, resolve_pkg_path
from .. import Stage

logger = logging.getLogger(__name__)


def _get_and_strip_format_fields(s: str) -> tuple[str, list[str]]:
    """Extracts the fields from a format string, in order.

    Args:
        s: The format string to extract fields from.

    Returns:
        A list of field names extracted from the format string.

    Examples:
        >>> _get_and_strip_format_fields("{code}//value_[{left},{right})")
        ('{}//value_[{},{})', ['code', 'left', 'right'])
        >>> _get_and_strip_format_fields("{code}")
        ('{}', ['code'])
        >>> _get_and_strip_format_fields("{code}{{{bin}}}")
        ('{}{{{}}}', ['code', 'bin'])
        >>> _get_and_strip_format_fields("code/bin")
        ('code/bin', [])
        >>> _get_and_strip_format_fields("")
        ('', [])
    """

    pattern = r"{([^{}:]*)(?::[^{}]*)?}"
    return re.sub(pattern, "{}", s), re.findall(pattern, s)


def _check_and_get_bin_endpoints(schema: pl.Schema, col: str) -> pl.Expr:
    """Checks and ensures the bin column is valid given the schema and returns the bin endpoints if so.

    Args:
        schema: The schema of the dataframe.
        col: The name of the bin column.

    Returns:
        The bin endpoints as a polars list expression.

    Raises:
        ValueError: If the bin column is not found in the schema or if it is not a struct type or if any field
            within the struct does not match the numeric value type.

    Examples:
        >>> schema = {
        ...     "numeric_value": pl.Float32,
        ...     "A": pl.Struct([pl.Field("bin1", pl.Float32), pl.Field("bin2", pl.Float32)]),
        ...     "B": pl.String,
        ...     "C": pl.Struct([pl.Field("bin1", pl.Float32), pl.Field("bin2", pl.Int64)]),
        ... }
        >>> print(_check_and_get_bin_endpoints(schema, "A"))
        .when(col("A").is_not_null()).then(col("A").multiple_fields().list.concat()).otherwise(null)
        >>> _check_and_get_bin_endpoints(schema, "B")
        Traceback (most recent call last):
            ...
        ValueError: bin_with_columns entry 'B' is not a struct type; got String.
        >>> _check_and_get_bin_endpoints(schema, "C")
        Traceback (most recent call last):
            ...
        ValueError: bin_with_columns entry 'C' has field 'bin2' with dtype Int64, which does not match the
            numeric_value dtype Float32.
        >>> _check_and_get_bin_endpoints(schema, "non_existent_col")
        Traceback (most recent call last):
            ...
        ValueError: bin_with_columns entry 'non_existent_col' not found in the dataframe schema.
    """

    if col not in schema:
        raise ValueError(f"bin_with_columns entry '{col}' not found in the dataframe schema.")

    col_dtype = schema[col]

    if not isinstance(col_dtype, pl.Struct):
        raise ValueError(f"bin_with_columns entry '{col}' is not a struct type; got {col_dtype}.")

    numeric_dtype = schema[DataSchema.numeric_value_name]

    for field in col_dtype.fields:
        if field.dtype != numeric_dtype:
            raise ValueError(
                f"bin_with_columns entry '{col}' has field '{field.name}' with dtype {field.dtype}, "
                f"which does not match the numeric_value dtype {numeric_dtype}."
            )

    return pl.when(pl.col(col).is_not_null()).then(pl.concat_list(pl.col(col).struct.unnest()))


def _get_val_bin_idx(bin_endpoints: pl.Expr) -> pl.Expr:
    """Get the bin index for the numeric value, if both it and the bin endpoints are not null.

    This is mostly separated out to enable more direct testing.

    > [!WARNING]
    > This will only work on a dataframe with a row index column named `__idx` and a numeric values column
    > named `numeric_value`.

    Args:
        bin_endpoints: The bin endpoints as a polars list expression.

    Returns:
        A polars expression that returns the bin index for the numeric value, or null if either the numeric
        value or the bin endpoints are null.

    Examples:
        >>> df = pl.DataFrame({
        ...     "numeric_value": [
        ...         -1.0, # Left of all endpoints
        ...         3.0, # Right of all endpoints
        ...         0.5, # In the middle of the first bin
        ...         1.0, # A bin endpoint directly.
        ...         None, # Null value
        ...         0.25, # Different scale, different number of bins.
        ...         1.0, # Null bin endpoints
        ...         -1.0, # Before singleton endpoint
        ...         1.0, # After singleton endpoint
        ...         0.0, # Equals singleton endpoint
        ...     ],
        ...     "bin_endpoints": [
        ...         [0.0, 1.0, 2.0],
        ...         [0.0, 1.0, 2.0],
        ...         [0.0, 1.0, 2.0],
        ...         [0.0, 1.0, 2.0],
        ...         [0.0, 1.0, 2.0],
        ...         [0.0, 0.1, 0.2, 0.3],
        ...         None,
        ...         [0.0],
        ...         [0.0],
        ...         [0.0],
        ...     ],
        ... }).with_row_index("__idx")
        >>> df
        shape: (10, 3)
        ┌───────┬───────────────┬───────────────────┐
        │ __idx ┆ numeric_value ┆ bin_endpoints     │
        │ ---   ┆ ---           ┆ ---               │
        │ u32   ┆ f64           ┆ list[f64]         │
        ╞═══════╪═══════════════╪═══════════════════╡
        │ 0     ┆ -1.0          ┆ [0.0, 1.0, 2.0]   │
        │ 1     ┆ 3.0           ┆ [0.0, 1.0, 2.0]   │
        │ 2     ┆ 0.5           ┆ [0.0, 1.0, 2.0]   │
        │ 3     ┆ 1.0           ┆ [0.0, 1.0, 2.0]   │
        │ 4     ┆ null          ┆ [0.0, 1.0, 2.0]   │
        │ 5     ┆ 0.25          ┆ [0.0, 0.1, … 0.3] │
        │ 6     ┆ 1.0           ┆ null              │
        │ 7     ┆ -1.0          ┆ [0.0]             │
        │ 8     ┆ 1.0           ┆ [0.0]             │
        │ 9     ┆ 0.0           ┆ [0.0]             │
        └───────┴───────────────┴───────────────────┘
        >>> df.with_columns(_get_val_bin_idx(pl.col("bin_endpoints")).alias("idx"))
        shape: (10, 4)
        ┌───────┬───────────────┬───────────────────┬──────┐
        │ __idx ┆ numeric_value ┆ bin_endpoints     ┆ idx  │
        │ ---   ┆ ---           ┆ ---               ┆ ---  │
        │ u32   ┆ f64           ┆ list[f64]         ┆ u32  │
        ╞═══════╪═══════════════╪═══════════════════╪══════╡
        │ 0     ┆ -1.0          ┆ [0.0, 1.0, 2.0]   ┆ 0    │
        │ 1     ┆ 3.0           ┆ [0.0, 1.0, 2.0]   ┆ 3    │
        │ 2     ┆ 0.5           ┆ [0.0, 1.0, 2.0]   ┆ 1    │
        │ 3     ┆ 1.0           ┆ [0.0, 1.0, 2.0]   ┆ 2    │
        │ 4     ┆ null          ┆ [0.0, 1.0, 2.0]   ┆ null │
        │ 5     ┆ 0.25          ┆ [0.0, 0.1, … 0.3] ┆ 3    │
        │ 6     ┆ 1.0           ┆ null              ┆ null │
        │ 7     ┆ -1.0          ┆ [0.0]             ┆ 0    │
        │ 8     ┆ 1.0           ┆ [0.0]             ┆ 1    │
        │ 9     ┆ 0.0           ┆ [0.0]             ┆ 1    │
        └───────┴───────────────┴───────────────────┴──────┘
    """

    val_col = pl.col(DataSchema.numeric_value_name)
    do_bin = bin_endpoints.is_not_null() & val_col.is_not_null()
    idx_expr = bin_endpoints.list.explode().search_sorted(val_col, side="right").over("__idx")
    return pl.when(do_bin).then(idx_expr)


def get_code(bin_endpoints: pl.Expr, val_bin_idx: pl.Expr) -> pl.Expr:
    return pl.col(DataSchema.code_name)


def get_bin(bin_endpoints: pl.Expr, val_bin_idx: pl.Expr) -> pl.Expr:
    return val_bin_idx.cast(pl.String)


def get_left(bin_endpoints: pl.Expr, val_bin_idx: pl.Expr) -> pl.Expr:
    return bin_endpoints.list.get(val_bin_idx - 1, null_on_oob=True).cast(pl.String).fill_null("-inf")


def get_right(bin_endpoints: pl.Expr, val_bin_idx: pl.Expr) -> pl.Expr:
    return bin_endpoints.list.get(val_bin_idx, null_on_oob=True).cast(pl.String).fill_null("inf")


BIN_NAME_FMT_EXPRS: dict[str, Callable[[pl.Expr, pl.Expr], pl.Expr]] = {
    "code": get_code,
    "bin": get_bin,
    "left": get_left,
    "right": get_right,
}


def add_bin_to_code(
    df: pl.LazyFrame | pl.DataFrame,
    bin_with_columns: list[str],
    code_with_bin_name: str,
    do_drop_numeric_value: bool = False,
) -> pl.LazyFrame | pl.DataFrame:
    """Converts numeric values into categorical code modifiers through a joined column of bin endpoints.

    Args:
        df: The dataframe to process. It should contain the following columns:
            - `code`: The code to be modified.
            - `numeric_value`: The numeric value to be binned.
            - All columns specified in `bin_with_columns`, which contain the linked bin endpoints to use for
              that particular measurement. Each column in `bin_with_columns` should contain a struct whose
              values are all of the same dtype and contain the increasing list of bin endpoints.
        code_with_bin_name: A string template for the code name with bin information. The template can
            leverage the keys `{code}`, `{left}`, `{right}`, and `{bin}` to capture the original code name,
            the left endpoint of the bin, the right endpoint of the bin, and the bin name.
        do_drop_numeric_value: A boolean flag indicating whether to drop the numeric_value column after
            binning. Default is `False`.

    Returns:
        A new dataframe with the codes modified to include bin information, according to the format specified.
        Codes without numeric values or with no bin endpoints included in either `code_metadata` or
        `custom_bins` are left unchanged. If `do_drop_numeric_value` is set to `True`, the returned DataFrame
        will have all cases where numeric values are converted to bins dropped. The returned dataframe will
        not have the columns contained in `bin_with_columns` in it.

    Raises:
        ValueError: If the dataframe does not contain the required columns or if the bin columns are not
            present in the schema.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2, 2, 3],
        ...     "code": ["lab//A", "lab//B", "lab//C", "lab//A", "lab//C", "dx//1", "lab//D"],
        ...     "numeric_value": [-1.0, 2.0, None, 1.0, 1.0, None, 1.2],
        ...     "values/quantiles": [
        ...         {"values/quantile/0.25": 0., "values/quantile/0.5": 1., "values/quantile/0.75": 2.},
        ...         {"values/quantile/0.25": -2., "values/quantile/0.5": 3., "values/quantile/0.75": 100.},
        ...         {"values/quantile/0.25": 0.01, "values/quantile/0.5": 0.4, "values/quantile/0.75": 0.6},
        ...         {"values/quantile/0.25": 0., "values/quantile/0.5": 1., "values/quantile/0.75": 2.},
        ...         {"values/quantile/0.25": 0.01, "values/quantile/0.5": 0.4, "values/quantile/0.75": 0.6},
        ...         None,
        ...         None,
        ...     ],
        ... })
        >>> df
        shape: (7, 4)
        ┌────────────┬────────┬───────────────┬──────────────────┐
        │ subject_id ┆ code   ┆ numeric_value ┆ values/quantiles │
        │ ---        ┆ ---    ┆ ---           ┆ ---              │
        │ i64        ┆ str    ┆ f64           ┆ struct[3]        │
        ╞════════════╪════════╪═══════════════╪══════════════════╡
        │ 1          ┆ lab//A ┆ -1.0          ┆ {0.0,1.0,2.0}    │
        │ 1          ┆ lab//B ┆ 2.0           ┆ {-2.0,3.0,100.0} │
        │ 1          ┆ lab//C ┆ null          ┆ {0.01,0.4,0.6}   │
        │ 2          ┆ lab//A ┆ 1.0           ┆ {0.0,1.0,2.0}    │
        │ 2          ┆ lab//C ┆ 1.0           ┆ {0.01,0.4,0.6}   │
        │ 2          ┆ dx//1  ┆ null          ┆ null             │
        │ 3          ┆ lab//D ┆ 1.2           ┆ null             │
        └────────────┴────────┴───────────────┴──────────────────┘
        >>> add_bin_to_code(df, ["values/quantiles"], "{code}//value_[{left},{right})")
        shape: (7, 3)
        ┌────────────┬──────────────────────────┬───────────────┐
        │ subject_id ┆ code                     ┆ numeric_value │
        │ ---        ┆ ---                      ┆ ---           │
        │ i64        ┆ str                      ┆ f64           │
        ╞════════════╪══════════════════════════╪═══════════════╡
        │ 1          ┆ lab//A//value_[-inf,0.0) ┆ -1.0          │
        │ 1          ┆ lab//B//value_[-2.0,3.0) ┆ 2.0           │
        │ 1          ┆ lab//C                   ┆ null          │
        │ 2          ┆ lab//A//value_[1.0,2.0)  ┆ 1.0           │
        │ 2          ┆ lab//C//value_[0.6,inf)  ┆ 1.0           │
        │ 2          ┆ dx//1                    ┆ null          │
        │ 3          ┆ lab//D                   ┆ 1.2           │
        └────────────┴──────────────────────────┴───────────────┘

    You can change the format to use the bin index ('{bin}') as well:

        >>> add_bin_to_code(df, ["values/quantiles"], "{code}//{bin}", do_drop_numeric_value=True)
        shape: (7, 3)
        ┌────────────┬───────────┬───────────────┐
        │ subject_id ┆ code      ┆ numeric_value │
        │ ---        ┆ ---       ┆ ---           │
        │ i64        ┆ str       ┆ f64           │
        ╞════════════╪═══════════╪═══════════════╡
        │ 1          ┆ lab//A//0 ┆ null          │
        │ 1          ┆ lab//B//1 ┆ null          │
        │ 1          ┆ lab//C    ┆ null          │
        │ 2          ┆ lab//A//2 ┆ null          │
        │ 2          ┆ lab//C//3 ┆ null          │
        │ 2          ┆ dx//1     ┆ null          │
        │ 3          ┆ lab//D    ┆ 1.2           │
        └────────────┴───────────┴───────────────┘

    You can also have multiple bin endpoint columns; in this case, the first non-null column is used. Note
    that the names of the struct keys are ignored and the structs need not match in anything but value type.

        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 1],
        ...     "code": ["A", "B", "C", "D"],
        ...     "numeric_value": [1.5, 2.5, 0.5, 0.6],
        ...     "bins_1": [
        ...         {"1": 0., "2": 1.}, None, None, {"1": 0., "2": 1.},
        ...     ],
        ...     "bins_2": [
        ...         None, None, {"a": 0.01, "b": 0.4, "c": 0.6}, {"a": -0.5, "b": 0.5, "c": 1.5},
        ...     ],
        ... })
        >>> df
        shape: (4, 5)
        ┌────────────┬──────┬───────────────┬───────────┬────────────────┐
        │ subject_id ┆ code ┆ numeric_value ┆ bins_1    ┆ bins_2         │
        │ ---        ┆ ---  ┆ ---           ┆ ---       ┆ ---            │
        │ i64        ┆ str  ┆ f64           ┆ struct[2] ┆ struct[3]      │
        ╞════════════╪══════╪═══════════════╪═══════════╪════════════════╡
        │ 1          ┆ A    ┆ 1.5           ┆ {0.0,1.0} ┆ null           │
        │ 1          ┆ B    ┆ 2.5           ┆ null      ┆ null           │
        │ 1          ┆ C    ┆ 0.5           ┆ null      ┆ {0.01,0.4,0.6} │
        │ 1          ┆ D    ┆ 0.6           ┆ {0.0,1.0} ┆ {-0.5,0.5,1.5} │
        └────────────┴──────┴───────────────┴───────────┴────────────────┘
        >>> add_bin_to_code(df, ["bins_1", "bins_2"], "[{left},{right})")
        shape: (4, 3)
        ┌────────────┬───────────┬───────────────┐
        │ subject_id ┆ code      ┆ numeric_value │
        │ ---        ┆ ---       ┆ ---           │
        │ i64        ┆ str       ┆ f64           │
        ╞════════════╪═══════════╪═══════════════╡
        │ 1          ┆ [1.0,inf) ┆ 1.5           │
        │ 1          ┆ B         ┆ 2.5           │
        │ 1          ┆ [0.4,0.6) ┆ 0.5           │
        │ 1          ┆ [0.0,1.0) ┆ 0.6           │
        └────────────┴───────────┴───────────────┘

    Errors will be raised if the dataframe is missing the numeric value field or if an invalid code name
    format string is provided (e.g., if it contains a field that is not in the list of valid fields).

        >>> add_bin_to_code(df.drop("numeric_value"), ["bins_1", "bins_2"], "{code}//{invalid_field}")
        Traceback (most recent call last):
            ...
        ValueError: Dataframe does not contain the required column 'numeric_value'.
        >>> add_bin_to_code(df, ["bins_1", "bins_2"], "{code}//{invalid_field}")
        Traceback (most recent call last):
            ...
        ValueError: Invalid bin name format field 'invalid_field' in '{code}//{invalid_field}'.
    """

    schema = df.collect_schema()

    df = df.with_row_index("__idx")

    if DataSchema.numeric_value_name not in schema:
        raise ValueError(f"Dataframe does not contain the required column '{DataSchema.numeric_value_name}'.")

    bin_endpoints = pl.coalesce([_check_and_get_bin_endpoints(schema, col) for col in bin_with_columns])
    val_bin_idx = _get_val_bin_idx(bin_endpoints)

    stripped_code_name, bin_name_fmt_cols = _get_and_strip_format_fields(code_with_bin_name)
    bin_name_fmt_exprs = []
    for n in bin_name_fmt_cols:
        if n not in BIN_NAME_FMT_EXPRS:
            raise ValueError(f"Invalid bin name format field '{n}' in '{code_with_bin_name}'.")
        bin_name_fmt_exprs.append(BIN_NAME_FMT_EXPRS[n](bin_endpoints, val_bin_idx))

    code_with_bin = pl.format(stripped_code_name, *bin_name_fmt_exprs)

    do_bin = bin_endpoints.is_not_null() & pl.col(DataSchema.numeric_value_name).is_not_null()
    df = df.with_columns(
        pl.when(do_bin).then(code_with_bin).otherwise(DataSchema.code_name).alias(DataSchema.code_name)
    )

    if do_drop_numeric_value:
        df = df.with_columns(
            pl.when(~do_bin).then(DataSchema.numeric_value_name).alias(DataSchema.numeric_value_name)
        )

    return df.drop(*bin_with_columns, "__idx")


@Stage.register
def bin_numeric_values_fntr(
    stage_cfg: DictConfig,
    code_metadata: pl.DataFrame,
    code_modifiers: list[str] | None = None,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Uses pre-computed value bin endpoints to add value range modifiers to code names.

    Args:
        stage_cfg: Configuration for the stage, including custom_bins and drop_numeric_value. The following
            keys are used:
                - `bin_with_columns`: A list of column names in `metadata/codes.parquet` to use for binning.
                  These columns should contain a struct whose key names correspond to bin names and whose
                  values capture the bin endpoints. The bin endpoints should correspond to the "right"
                  endpoint (exclusive) of the bin. In this way, a struct with three key-value pairs implicitly
                  creates four bins. E.g., the struct
                  `{values/quantile/0.25: 99.9, values/quantile/0.5: 105.1, values/quantile/0.75: 113.4}`
                  captures the bins `[-inf, 99.9)`, `[99.9, 105.1)`, `[105.1, 113.4)`, and `[113.4, inf)`. The
                  struct should be in sorted order of bin endpoints from least to greatest. Default is a list
                  with a single entry of `"values/quantiles"`, which captures the quantiles computed via the
                  `aggregate_code_metadata` stage. If multiple columns are specified in this list, the first
                  column that is non-null is used for each code.
                - code_with_bin_name: A string template for the code name with bin information. The template
                  can leverage the keys `{code}`, `{left}`, `{right}`, and `{bin}` to capture the original
                  code name, the left endpoint of the bin, the right endpoint of the bin, and the bin name.
                  Default is `{code}//value_[{left},{right})`.
                - do_drop_numeric_value: A boolean flag indicating whether to drop the numeric_value column
                  after binning. Default is `False`.
                - custom_bins: A dictionary mapping code names to custom bin endpoints. The keys should be the
                  code names and the values should be dictionaries with the same structure as the
                  bin_with_columns. This allows for custom binning for specific codes. Default is an empty
                  dictionary. Custom bins are used preferentially over entries in `bin_with_columns`.
                - custom_bins_filepath: Optional path to a YAML file containing custom bin endpoints. If
                  provided, the file should define the same dictionary structure as ``custom_bins``.
                  Entries loaded from this file are merged with any directly specified in ``custom_bins``.
        code_metadata: A DataFrame containing the metadata for the codes, including the bin endpoints and
            custom bins.
        code_modifiers: A list of additional columns to use for joining against codes. These columns should be
            present in both the raw data and the code metadata.

    > [!WARNING]
    > If `code_modifiers` are used, note that they do not change the structure of `custom_bins` -- that
    > strategy still only joins by `code`. File a GitHub Issue if this does not work for you.

    Returns:
        A function that takes a dataframe and returns a new dataFrame with the codes modified to include bin
        information, according to the format specified. Codes without numeric values or with no bin endpoints
        included in either `code_metadata` or `custom_bins` are left unchanged. If `do_drop_numeric_value` is
        set to `True`, the returned DataFrame will have all cases where numeric values are converted to bins
        dropped.

    Examples:
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2, 2, 3],
        ...     "code": ["lab//A", "lab//B", "lab//C", "lab//A", "lab//C", "dx//1", "lab//D"],
        ...     "numeric_value": [-1.0, 2.0, None, 1.0, 1.0, None, 1.2],
        ... })
        >>> df
        shape: (7, 3)
        ┌────────────┬────────┬───────────────┐
        │ subject_id ┆ code   ┆ numeric_value │
        │ ---        ┆ ---    ┆ ---           │
        │ i64        ┆ str    ┆ f64           │
        ╞════════════╪════════╪═══════════════╡
        │ 1          ┆ lab//A ┆ -1.0          │
        │ 1          ┆ lab//B ┆ 2.0           │
        │ 1          ┆ lab//C ┆ null          │
        │ 2          ┆ lab//A ┆ 1.0           │
        │ 2          ┆ lab//C ┆ 1.0           │
        │ 2          ┆ dx//1  ┆ null          │
        │ 3          ┆ lab//D ┆ 1.2           │
        └────────────┴────────┴───────────────┘
        >>> code_metadata = pl.DataFrame({
        ...     "code": ["lab//A", "lab//B", "lab//C", "dx//1", "lab//D"],
        ...     "values/quantiles": [
        ...         {"values/quantile/0.25": 0., "values/quantile/0.5": 1., "values/quantile/0.75": 2.},
        ...         {"values/quantile/0.25": -2., "values/quantile/0.5": 3., "values/quantile/0.75": 100.},
        ...         {"values/quantile/0.25": 0.01, "values/quantile/0.5": 0.4, "values/quantile/0.75": 0.6},
        ...         None,
        ...         None,
        ...     ],
        ... })

    If we run the functor from a default (empty) stage configuration, we get the following:

        >>> fn = bin_numeric_values_fntr(DictConfig({}), code_metadata)
        >>> fn(df)
        shape: (7, 3)
        ┌────────────┬──────────────────────────┬───────────────┐
        │ subject_id ┆ code                     ┆ numeric_value │
        │ ---        ┆ ---                      ┆ ---           │
        │ i64        ┆ str                      ┆ f64           │
        ╞════════════╪══════════════════════════╪═══════════════╡
        │ 1          ┆ lab//A//value_[-inf,0.0) ┆ -1.0          │
        │ 1          ┆ lab//B//value_[-2.0,3.0) ┆ 2.0           │
        │ 1          ┆ lab//C                   ┆ null          │
        │ 2          ┆ lab//A//value_[1.0,2.0)  ┆ 1.0           │
        │ 2          ┆ lab//C//value_[0.6,inf)  ┆ 1.0           │
        │ 2          ┆ dx//1                    ┆ null          │
        │ 3          ┆ lab//D                   ┆ 1.2           │
        └────────────┴──────────────────────────┴───────────────┘

    We can use the stage configuration to change the format of the bin name...

        >>> fn = bin_numeric_values_fntr(DictConfig({"code_with_bin_name": "{code}//{bin}"}), code_metadata)
        >>> fn(df)
        shape: (7, 3)
        ┌────────────┬───────────┬───────────────┐
        │ subject_id ┆ code      ┆ numeric_value │
        │ ---        ┆ ---       ┆ ---           │
        │ i64        ┆ str       ┆ f64           │
        ╞════════════╪═══════════╪═══════════════╡
        │ 1          ┆ lab//A//0 ┆ -1.0          │
        │ 1          ┆ lab//B//1 ┆ 2.0           │
        │ 1          ┆ lab//C    ┆ null          │
        │ 2          ┆ lab//A//2 ┆ 1.0           │
        │ 2          ┆ lab//C//3 ┆ 1.0           │
        │ 2          ┆ dx//1     ┆ null          │
        │ 3          ┆ lab//D    ┆ 1.2           │
        └────────────┴───────────┴───────────────┘

    Drop the numeric values we bin:

        >>> fn = bin_numeric_values_fntr(DictConfig({"drop_numeric_value": True}), code_metadata)
        >>> fn(df)
        shape: (7, 3)
        ┌────────────┬──────────────────────────┬───────────────┐
        │ subject_id ┆ code                     ┆ numeric_value │
        │ ---        ┆ ---                      ┆ ---           │
        │ i64        ┆ str                      ┆ f64           │
        ╞════════════╪══════════════════════════╪═══════════════╡
        │ 1          ┆ lab//A//value_[-inf,0.0) ┆ null          │
        │ 1          ┆ lab//B//value_[-2.0,3.0) ┆ null          │
        │ 1          ┆ lab//C                   ┆ null          │
        │ 2          ┆ lab//A//value_[1.0,2.0)  ┆ null          │
        │ 2          ┆ lab//C//value_[0.6,inf)  ┆ null          │
        │ 2          ┆ dx//1                    ┆ null          │
        │ 3          ┆ lab//D                   ┆ 1.2           │
        └────────────┴──────────────────────────┴───────────────┘

    Add custom bin endpoints:

        >>> fn = bin_numeric_values_fntr(DictConfig({"custom_bins": {"lab//D": {"foo": 1.0}}}), code_metadata)
        >>> fn(df)
        shape: (7, 3)
        ┌────────────┬──────────────────────────┬───────────────┐
        │ subject_id ┆ code                     ┆ numeric_value │
        │ ---        ┆ ---                      ┆ ---           │
        │ i64        ┆ str                      ┆ f64           │
        ╞════════════╪══════════════════════════╪═══════════════╡
        │ 1          ┆ lab//A//value_[-inf,0.0) ┆ -1.0          │
        │ 1          ┆ lab//B//value_[-2.0,3.0) ┆ 2.0           │
        │ 1          ┆ lab//C                   ┆ null          │
        │ 2          ┆ lab//A//value_[1.0,2.0)  ┆ 1.0           │
        │ 2          ┆ lab//C//value_[0.6,inf)  ┆ 1.0           │
        │ 2          ┆ dx//1                    ┆ null          │
        │ 3          ┆ lab//D//value_[1.0,inf)  ┆ 1.2           │
        └────────────┴──────────────────────────┴───────────────┘

    Load custom bins from a YAML file on disk:

        >>> import tempfile, yaml
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     bins_fp = Path(tmpdir) / "bins.yaml"
        ...     yaml.safe_dump({"lab//D": {"foo": 1.0}}, bins_fp.open("w"))
        ...     fn = bin_numeric_values_fntr(
        ...         DictConfig({"custom_bins_filepath": str(bins_fp)}),
        ...         code_metadata,
        ...     )
        ...     fn(df)
        shape: (7, 3)
        ┌────────────┬──────────────────────────┬───────────────┐
        │ subject_id ┆ code                     ┆ numeric_value │
        │ ---        ┆ ---                      ┆ ---           │
        │ i64        ┆ str                      ┆ f64           │
        ╞════════════╪══════════════════════════╪═══════════════╡
        │ 1          ┆ lab//A//value_[-inf,0.0) ┆ -1.0          │
        │ 1          ┆ lab//B//value_[-2.0,3.0) ┆ 2.0           │
        │ 1          ┆ lab//C                   ┆ null          │
        │ 2          ┆ lab//A//value_[1.0,2.0)  ┆ 1.0           │
        │ 2          ┆ lab//C//value_[0.6,inf)  ┆ 1.0           │
        │ 2          ┆ dx//1                    ┆ null          │
        │ 3          ┆ lab//D//value_[1.0,inf)  ┆ 1.2           │
        └────────────┴──────────────────────────┴───────────────┘

    The path may also reference a resource using the ``pkg://`` scheme:

        >>> fn = bin_numeric_values_fntr(
        ...     DictConfig({
        ...         "custom_bins_filepath": "pkg://MEDS_transforms.stages.bin_numeric_values.examples.custom_bins_fp.custom_bins.yaml"
        ...     }),
        ...     code_metadata,
        ... )

    Use different bin columns (sourced from the code metadata)

        >>> code_metadata = pl.DataFrame({
        ...     "code": ["lab//A", "lab//B", "lab//C", "dx//1", "lab//D"],
        ...     "bins_1": [
        ...         {"values/quantile/0.25": 0., "values/quantile/0.5": 1., "values/quantile/0.75": 2.},
        ...         None,
        ...         {"values/quantile/0.25": 0.01, "values/quantile/0.5": 0.4, "values/quantile/0.75": 0.6},
        ...         None,
        ...         None,
        ...     ],
        ...     "bins_2": [
        ...         {"values/quantile/0.25": -2., "values/quantile/0.5": 3., "values/quantile/0.75": 100.},
        ...         {"values/quantile/0.25": -2., "values/quantile/0.5": 3., "values/quantile/0.75": 100.},
        ...         None,
        ...         None,
        ...         None,
        ...     ],
        ... })
        >>> fn = bin_numeric_values_fntr(
        ...     DictConfig({"bin_with_columns": ["bins_1", "bins_2"]}), code_metadata
        ... )
        >>> fn(df)
        shape: (7, 3)
        ┌────────────┬──────────────────────────┬───────────────┐
        │ subject_id ┆ code                     ┆ numeric_value │
        │ ---        ┆ ---                      ┆ ---           │
        │ i64        ┆ str                      ┆ f64           │
        ╞════════════╪══════════════════════════╪═══════════════╡
        │ 1          ┆ lab//A//value_[-inf,0.0) ┆ -1.0          │
        │ 1          ┆ lab//B//value_[-2.0,3.0) ┆ 2.0           │
        │ 1          ┆ lab//C                   ┆ null          │
        │ 2          ┆ lab//A//value_[1.0,2.0)  ┆ 1.0           │
        │ 2          ┆ lab//C//value_[0.6,inf)  ┆ 1.0           │
        │ 2          ┆ dx//1                    ┆ null          │
        │ 3          ┆ lab//D                   ┆ 1.2           │
        └────────────┴──────────────────────────┴───────────────┘

    We can also use code modifiers to join in a richer manner against the code metadata:

        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 1, 2, 2, 2, 3],
        ...     "code": ["lab", "lab", "lab", "lab", "lab", "dx//1", "lab"],
        ...     "unit": ["A", "B", "C", "A", "C", None, "D"],
        ...     "numeric_value": [-1.0, 2.0, None, 1.0, 1.0, None, 1.2],
        ... })
        >>> df
        shape: (7, 4)
        ┌────────────┬───────┬──────┬───────────────┐
        │ subject_id ┆ code  ┆ unit ┆ numeric_value │
        │ ---        ┆ ---   ┆ ---  ┆ ---           │
        │ i64        ┆ str   ┆ str  ┆ f64           │
        ╞════════════╪═══════╪══════╪═══════════════╡
        │ 1          ┆ lab   ┆ A    ┆ -1.0          │
        │ 1          ┆ lab   ┆ B    ┆ 2.0           │
        │ 1          ┆ lab   ┆ C    ┆ null          │
        │ 2          ┆ lab   ┆ A    ┆ 1.0           │
        │ 2          ┆ lab   ┆ C    ┆ 1.0           │
        │ 2          ┆ dx//1 ┆ null ┆ null          │
        │ 3          ┆ lab   ┆ D    ┆ 1.2           │
        └────────────┴───────┴──────┴───────────────┘
        >>> code_metadata = pl.DataFrame({
        ...     "code": ["lab", "lab", "lab", "dx//1", "lab"],
        ...     "unit": ["A", "B", "C", None, "D"],
        ...     "values/quantiles": [
        ...         {"values/quantile/0.25": 0., "values/quantile/0.5": 1., "values/quantile/0.75": 2.},
        ...         {"values/quantile/0.25": -2., "values/quantile/0.5": 3., "values/quantile/0.75": 100.},
        ...         {"values/quantile/0.25": 0.01, "values/quantile/0.5": 0.4, "values/quantile/0.75": 0.6},
        ...         None,
        ...         None,
        ...     ],
        ... })
        >>> fn = bin_numeric_values_fntr(DictConfig({}), code_metadata, ["unit"])
        >>> fn(df)
        shape: (7, 4)
        ┌────────────┬───────────────────────┬──────┬───────────────┐
        │ subject_id ┆ code                  ┆ unit ┆ numeric_value │
        │ ---        ┆ ---                   ┆ ---  ┆ ---           │
        │ i64        ┆ str                   ┆ str  ┆ f64           │
        ╞════════════╪═══════════════════════╪══════╪═══════════════╡
        │ 1          ┆ lab//value_[-inf,0.0) ┆ A    ┆ -1.0          │
        │ 1          ┆ lab//value_[-2.0,3.0) ┆ B    ┆ 2.0           │
        │ 1          ┆ lab                   ┆ C    ┆ null          │
        │ 2          ┆ lab//value_[1.0,2.0)  ┆ A    ┆ 1.0           │
        │ 2          ┆ lab//value_[0.6,inf)  ┆ C    ┆ 1.0           │
        │ 2          ┆ dx//1                 ┆ null ┆ null          │
        │ 3          ┆ lab                   ┆ D    ┆ 1.2           │
        └────────────┴───────────────────────┴──────┴───────────────┘
    """
    if code_modifiers is None:
        code_modifiers = []

    # Step 1: Add custom_bins column to code_metadata
    custom_bins = stage_cfg.get("custom_bins", {})
    custom_bins_fp = stage_cfg.get("custom_bins_filepath")

    if isinstance(custom_bins, DictConfig):
        custom_bins = OmegaConf.to_container(custom_bins)

    if custom_bins_fp:
        fp = Path(custom_bins_fp)
        if custom_bins_fp.startswith(PKG_PFX):
            fp = resolve_pkg_path(custom_bins_fp)
        if not fp.is_file():
            raise FileNotFoundError(f"custom_bins_filepath '{custom_bins_fp}' does not exist.")

        file_bins = OmegaConf.load(fp)
        if isinstance(file_bins, DictConfig):
            file_bins = OmegaConf.to_container(file_bins)
        if not isinstance(file_bins, dict):
            raise TypeError("custom_bins_filepath must point to a YAML file with a dictionary")
        custom_bins = {**file_bins, **custom_bins}

    do_use_custom_bins = bool(custom_bins)
    do_drop_numeric_value = stage_cfg.get("drop_numeric_value", False)
    bin_with_columns = stage_cfg.get("bin_with_columns", ["values/quantiles"])
    code_with_bin_name = stage_cfg.get("code_with_bin_name", "{code}//value_[{left},{right})")

    if do_use_custom_bins:
        if isinstance(custom_bins, DictConfig):
            custom_bins = OmegaConf.to_container(custom_bins)

        struct_dtype = pl.Struct(dict.fromkeys(next(iter(custom_bins.values())).keys(), pl.Float32))
        custom_bins_series = pl.Series(
            [custom_bins.get(c, None) for c in code_metadata[CodeMetadataSchema.code_name]],
            dtype=struct_dtype,
        )
        code_metadata = code_metadata.with_columns(custom_bins_series.alias("__custom_bins"))
        bin_with_columns = ["__custom_bins", *bin_with_columns]

    join_cols = [CodeMetadataSchema.code_name, *code_modifiers]

    code_metadata = code_metadata.select(*join_cols, *bin_with_columns)

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        numeric_dtype = df.collect_schema()[DataSchema.numeric_value_name]

        cast_exprs = []
        for col in bin_with_columns:
            new_struct_dtype = pl.Struct({f.name: numeric_dtype for f in code_metadata.schema[col].fields})
            cast_exprs.append(pl.col(col).cast(new_struct_dtype))

        local_metadata = code_metadata.with_columns(*cast_exprs)

        if isinstance(df, pl.LazyFrame):
            local_metadata = local_metadata.lazy()

        df = df.join(local_metadata, on=join_cols, how="left", maintain_order="left")
        return add_bin_to_code(df, bin_with_columns, code_with_bin_name, do_drop_numeric_value)

    return fn
