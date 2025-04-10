"""Simple helper functions to define a consistent code vocabulary for normalizing a MEDS dataset."""

from collections.abc import Callable
from enum import StrEnum
import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import polars as pl

from .. import Stage

logger = logging.getLogger(__name__)


class VocabularyOrdering(StrEnum):
    """Enumeration of different ways a vocabulary order can be selected.

    These are stored as a `StrEnum` so that they can be easily specified by the user in a configuration file
    or on the command line.

    Currently, only one ordering method is supported, but others can be added, such as a frequency-based
    ordering so that the most frequent codes have the smallest indices.

    Args:
        "lexicographic": Assigns vocabulary indices to codes and code modifiers via a lexicographic order.
    """

    LEXICOGRAPHIC = "lexicographic"


INDEX_ASSIGNMENT_FN = Callable[[pl.DataFrame, list[str]], pl.DataFrame]


def validate_code_metadata(code_metadata: pl.DataFrame, code_modifiers: list[str]):
    """Validate the code metadata has the requisite columns and is unique.

    Args:
        code_metadata: Metadata about the codes in the MEDS dataset, with a column `code` and a collection
            of code modifier columns.
        code_modifiers: The names of the code modifier columns in the `code_metadata` dataset.

    Raises:
        KeyError: If the `code_metadata` dataset does not contain the specified `code_modifiers` or `code`
            columns.
        ValueError: If the `code_metadata` dataset is not unique on the `code` and `code_modifiers` columns.

    Examples:
        >>> code_metadata = pl.DataFrame(
        ...     {
        ...         "code": ["A", "B", "A", "A"],
        ...         "modifier1": ["X", "D", "Z", "Z"],
        ...         "modifier2": [None, None, None, 3],
        ...     }
        ... )
        >>> validate_code_metadata(code_metadata, ["modifier1", "modifier2"])
        >>> # This returns None in the absence of an exception.
        >>> code_metadata = pl.DataFrame(
        ...     {
        ...         "code": ["A", "B", "A", "A"],
        ...         "modifier1": ["X", "D", "Z", "Z"],
        ...         "modifier2": [None, None, None, 3],
        ...     }
        ... )
        >>> validate_code_metadata(code_metadata, ["modifier1", "modifier2", "missing_modifier"])
        Traceback (most recent call last):
            ...
        KeyError: "The following columns are not present in the code metadata: 'missing_modifier'."
        >>> code_metadata = pl.DataFrame(
        ...     {
        ...         "code": ["A", "B", "A", "A", "B", "B"],
        ...         "modifier1": ["X", "D", "Z", "Z", "Y", "Y"],
        ...         "modifier2": [None, None, None, None, 2, 1],
        ...     }
        ... )
        >>> validate_code_metadata(code_metadata, ["modifier1", "modifier2"])
        Traceback (most recent call last):
            ...
        ValueError: The code and code modifiers are not unique:
        shape: (1, 4)
        ┌──────┬───────────┬───────────┬───────┐
        │ code ┆ modifier1 ┆ modifier2 ┆ count │
        │ ---  ┆ ---       ┆ ---       ┆ ---   │
        │ str  ┆ str       ┆ i64       ┆ u32   │
        ╞══════╪═══════════╪═══════════╪═══════╡
        │ A    ┆ Z         ┆ null      ┆ 2     │
        └──────┴───────────┴───────────┴───────┘
    """

    cols = ["code", *code_modifiers]

    # Check that the code and code modifiers are present in the code metadata
    if not set(cols).issubset(code_metadata.columns):
        missing_cols = set(cols) - set(code_metadata.columns)
        missing_cols_str = "', '".join(missing_cols)
        raise KeyError(f"The following columns are not present in the code metadata: '{missing_cols_str}'.")

    # Check that the code and code modifiers are unique
    n_unique_codes = code_metadata.n_unique(cols)
    n_total_rows = len(code_metadata)

    if n_unique_codes != n_total_rows:
        code_counts = code_metadata.group_by(cols).agg(pl.len().alias("count")).sort("count", descending=True)
        extra_codes = code_counts.filter(pl.col("count") > 1)
        raise ValueError(f"The code and code modifiers are not unique:\n{extra_codes.head(100)}")


def lexicographic_indices(code_metadata: pl.DataFrame, code_modifiers: list[str]) -> pl.DataFrame:
    """Assign vocabulary indices to codes and code modifiers via a lexicographic order.

    Args:
        code_metadata: Metadata about the codes in the MEDS dataset, with a column `code` and a collection
            of code modifier columns.
        code_modifiers: The names of the code modifier columns in the `code_metadata` dataset. Each of these
            columns should be lexicographically orderable.

    Returns:
        The code metadata dataframe with an additional column added that hasvocabulary token indices to the
        code + modifier unique combinations in the `code_metadata` dataset. The expression will be aliased to
        "code/vocab_index", and will be of the smallest unsigned dtype possible given the number of included
        vocabulary elements. The given vocabulary indices will correspond to the following order:
          - The index `0` will be assigned to a sentinel, `"UNK"` code for codes/modifiers not present in the
            vocabulary.
          - The remaining indices will be assigned to the unique code + modifier combinations such that
            sorting the code + modifier combinations by the assigned index will result in a lexicographically
            ordered list of codes + modifiers, sorting first by code and subsequently by each modifier column,
            in order of specification. The sort will go from smallest lexiographic value to largest (e.g., be
            an ascending sort). `null` values in the modifier columns (`null`s are disallowed in the code
            columns) will be treated as the smallest possible value in the lexicographic order.

    Raises:
        KeyError: If the `code_metadata` dataset does not contain the specified `code_modifiers` or `code`
            columns.
        ValueError: If the `code_metadata` dataset is not unique on the `code` and `code_modifiers` columns.
        ValueError: If the `code` and `code_modifier` columns are not all lexicographically orderable.

    Examples:
        >>> code_metadata = pl.DataFrame(
        ...     {
        ...         "code": ["A", "B", "A", "A", "B", "B"],
        ...         "modifier1": ["X", "D", None, "Z", "Y", "Y"],
        ...         "modifier2": [None, None, None, None, 2, 1],
        ...     }
        ... )
        >>> code_modifiers = ["modifier1", "modifier2"]
        >>> lexicographic_indices(code_metadata, code_modifiers)
        shape: (6, 4)
        ┌──────┬───────────┬───────────┬──────────────────┐
        │ code ┆ modifier1 ┆ modifier2 ┆ code/vocab_index │
        │ ---  ┆ ---       ┆ ---       ┆ ---              │
        │ str  ┆ str       ┆ i64       ┆ u8               │
        ╞══════╪═══════════╪═══════════╪══════════════════╡
        │ A    ┆ X         ┆ null      ┆ 2                │
        │ B    ┆ D         ┆ null      ┆ 4                │
        │ A    ┆ null      ┆ null      ┆ 1                │
        │ A    ┆ Z         ┆ null      ┆ 3                │
        │ B    ┆ Y         ┆ 2         ┆ 6                │
        │ B    ┆ Y         ┆ 1         ┆ 5                │
        └──────┴───────────┴───────────┴──────────────────┘
    """

    validate_code_metadata(code_metadata, code_modifiers)

    # We'll perform this sort in three steps. To guide this, consider as an example that our set of codes is
    # ["B", "D", "A", "C"]. For this, we want to produce a set of indices that correspond to the lexicographic
    # order of the codes, starting at 1; namely, [2, 4, 1, 3]. This is because "B" is the 2nd letter, "D" the
    # fourth, and so on.
    # 1. First, we'll use an `pl.arg_sort_by` to produce a set of indices that, were we to select the rows in
    #    that order, would give us the codes in lexicographic order. This is _not_ the final order -- it tells
    #    us what row we'd need to put in each position to _get_ the codes in sorted order.
    #    E.g., if our codes were ["B", "D", "A", "C"], the result of this step would be [2, 0, 3, 1], because
    #    we'd need to get the 2nd row to have the first lexicographically ordered code, the 0th row to have
    #    the second, and so on.
    # 2. Second, we'll use _another_ `pl.arg_sort_by` to identify the row indices that would sort the very
    #    sort indices we just produced. This works because the index of the destination each row would have in
    #    the final sorted array is exactly the position that that row's index appears in the sort indices
    #    produced in step 1, by definition. And, in the second arg-sort, when we ask "which row do we need to
    #    grab to fill slot $j$ in this array with the sorted element that would belong at position $j$ of this
    #    set of numbers between $0$ and $N-1$, we are really asking which row has $j$ in it now, which is
    #    exactly the lexicographically ordered index.
    # 3. Finally, third, we will add one (to start at one) and shrink the dtype.
    #
    # Note that we use this algorithm over something like just sorting the whole dataframe once then assigning
    # integer indices is that this approach merely assigns indices, and does not change the order of the
    # dataframe, and similarly does not require actually touching any of the memory of the dataframe. Though,
    # admittedly, it is not clear how significant this choice is in practice.

    sort_cols = ["code", *code_modifiers]

    return code_metadata.with_columns(
        (pl.arg_sort_by(pl.arg_sort_by(sort_cols, descending=False, nulls_last=False)) + 1)
        .shrink_dtype()
        .alias("code/vocab_index")
    )


VocabularyOrdering_METHODS: dict[VocabularyOrdering, INDEX_ASSIGNMENT_FN] = {
    VocabularyOrdering.LEXICOGRAPHIC: lexicographic_indices,
}

VOCABULARY_SCHEMA_UPDATES = {"code/vocab_index": pl.UInt8}


@Stage.register(output_schema_updates=VOCABULARY_SCHEMA_UPDATES, is_metadata=True)
def main(cfg: DictConfig):
    """Assigns integral vocabulary IDs to codes in the metadata file, for use in tokenizing the dataset.

    This stage merely modifies the specified input code metadata file, by assigning a `code/vocab_index`
    column to the input dataframe whose entries correspond to the ordered vocabulary index realized by the
    specified ordering method over all unique codes.

    Args:
        stage_configs.fit_vocabulary_indices.ordering_method: How the code vocabulary should be ordered when
            vocab indices are assigned. Currently, only `lexographic` is supported. File a GitHub issue if
            you'd like other ordering methods supported.

    Raises:
        ValueError: If the ordering method is not one of the supported methods.

    Examples:

        To show this in action, we'll use this example code metadata file:

        >>> code_metadata = pl.DataFrame(
        ...     {
        ...         "code": ["A", "B", "A", "A", "B", "B"],
        ...         "modifier1": ["X", "D", None, "Z", "Y", "Y"],
        ...         "modifier2": [None, None, None, None, 2, 1],
        ...     }
        ... )

        We'll also use the following global configuration file:

        >>> cfg = DictConfig(
        ...     {
        ...         "stage": "fit_vocabulary_indices",
        ...         "code_modifier_columns": ["modifier1", "modifier2"],
        ...         "stage_cfg": {
        ...             "metadata_input_dir": "???",  # Will be assigned later in the test.
        ...             "reducer_output_dir": "???",  # Will be assigned later in the test.
        ...             "ordering_method": "lexicographic",
        ...         },
        ...     }
        ... )

        Now, we'll run the stage with the above code metadata and configuration:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     # Set up the input and output directories.
        ...     metadata_input_dir = Path(tmpdir) / "input"
        ...     metadata_input_dir.mkdir(parents=True, exist_ok=True)
        ...     code_metadata.write_parquet(metadata_input_dir / "codes.parquet")
        ...     output_dir = Path(tmpdir) / "output"
        ...     cfg.stage_cfg.metadata_input_dir = str(metadata_input_dir)
        ...     cfg.stage_cfg.reducer_output_dir = str(output_dir)
        ...     # Run the stage
        ...     main(cfg)
        ...     # Read the output file and print it.
        ...     pl.read_parquet(output_dir / "codes.parquet")
        shape: (6, 4)
        ┌──────┬───────────┬───────────┬──────────────────┐
        │ code ┆ modifier1 ┆ modifier2 ┆ code/vocab_index │
        │ ---  ┆ ---       ┆ ---       ┆ ---              │
        │ str  ┆ str       ┆ i64       ┆ u8               │
        ╞══════╪═══════════╪═══════════╪══════════════════╡
        │ A    ┆ X         ┆ null      ┆ 2                │
        │ B    ┆ D         ┆ null      ┆ 4                │
        │ A    ┆ null      ┆ null      ┆ 1                │
        │ A    ┆ Z         ┆ null      ┆ 3                │
        │ B    ┆ Y         ┆ 2         ┆ 6                │
        │ B    ┆ Y         ┆ 1         ┆ 5                │
        └──────┴───────────┴───────────┴──────────────────┘

        If the `ordering_method` were set to another parameter, an error would be thrown (as right now only
        lexicographic ordering is supported):

        >>> cfg.stage_cfg.ordering_method = "other"
        >>> main(cfg)
        Traceback (most recent call last):
            ...
        ValueError: Invalid ordering method: other. Expected one of lexicographic

        Note that if the code and code modifiers in the metadata file are not unique, an error will be thrown
        as well -- this is because without unique assignments between codes/modifiers and indices, you would
        not be able to convert between codes and indices in the raw data. We can see that in action here by
        removing the (necessary) code modifiers from our configuration file:

        >>> cfg.stage_cfg.ordering_method = "lexicographic"
        >>> cfg.code_modifier_columns = None
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     # Set up the input and output directories.
        ...     metadata_input_dir = Path(tmpdir) / "input"
        ...     metadata_input_dir.mkdir(parents=True, exist_ok=True)
        ...     code_metadata.write_parquet(metadata_input_dir / "codes.parquet")
        ...     output_dir = Path(tmpdir) / "output"
        ...     cfg.stage_cfg.metadata_input_dir = str(metadata_input_dir)
        ...     cfg.stage_cfg.reducer_output_dir = str(output_dir)
        ...     # Run the stage
        ...     main(cfg)
        Traceback (most recent call last):
            ...
        ValueError: The code and code modifiers are not unique: ...
    """

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    ordering_method = cfg.stage_cfg.get("ordering_method", VocabularyOrdering.LEXICOGRAPHIC)

    if ordering_method not in VocabularyOrdering_METHODS:
        raise ValueError(
            f"Invalid ordering method: {ordering_method}. "
            f"Expected one of {', '.join(VocabularyOrdering_METHODS.keys())}"
        )

    metadata_input_dir = Path(cfg.stage_cfg.metadata_input_dir)
    output_dir = Path(cfg.stage_cfg.reducer_output_dir)

    code_metadata = pl.read_parquet(metadata_input_dir / "codes.parquet", use_pyarrow=True)

    logger.info(f"Assigning code vocabulary indices via a {ordering_method} order.")
    ordering_fn = VocabularyOrdering_METHODS[ordering_method]

    code_modifiers = cfg.get("code_modifier_columns", None)
    if code_modifiers is None:
        code_modifiers = []

    code_metadata = ordering_fn(code_metadata, code_modifiers)

    output_fp = output_dir / "codes.parquet"
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Indices assigned. Writing to {output_fp}")

    code_metadata.write_parquet(output_fp, use_pyarrow=True)

    logger.info(f"Done with {cfg.stage}")
