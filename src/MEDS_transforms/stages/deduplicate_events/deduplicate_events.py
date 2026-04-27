"""Drop duplicate event rows under a configurable conflict policy.

See ``config.yaml`` for the parameters; see :func:`deduplicate_events` for the
behavior, the ``on_conflict`` policies, and end-to-end doctests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from meds import DataSchema

from .. import Stage

if TYPE_CHECKING:
    from collections.abc import Callable

    from omegaconf import DictConfig


# Default ``by`` set: the four mandatory MEDS event-row columns. ``text_value`` is
# part of the schema but optional, so callers who want it as part of the uniqueness
# key must list it explicitly via ``by``.
_DEFAULT_BY: tuple[str, ...] = (
    DataSchema.subject_id_name,
    DataSchema.time_name,
    DataSchema.code_name,
    DataSchema.numeric_value_name,
)

_VALID_POLICIES: frozenset[str] = frozenset({"error", "first", "last", "drop"})


class DuplicateEventConflictError(ValueError):
    """Rows sharing the same ``by`` columns disagree on a non-``by`` column.

    Raised by :func:`deduplicate_events` under ``on_conflict="error"`` (the default).
    The message names the offending ``by`` group and, when feasible, the disagreeing
    column(s) and example values — diagnosing the data-quality issue is the whole
    point of the strict default.
    """


@Stage.register(output_schema_updates={})
def deduplicate_events(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Returns a function that drops duplicate event rows.

    The output schema is identical to the input — this is a pure row-filter, declared
    via the empty ``output_schema_updates={}``. Once the flexible-schema mechanism
    (#379) lands, that explicit declaration helps validation tooling see this stage
    is intentionally a no-op on schema.

    Two rows are "in the same ``by`` group" when their ``by``-column values are equal.
    For each such group:

    * If every non-``by`` column agrees (true duplicate) → collapse to one row in
      every ``on_conflict`` policy.
    * If any non-``by`` column disagrees → behavior is controlled by ``on_conflict``:

      - ``"error"`` (default): raise :class:`DuplicateEventConflictError` with an
        example. Conservative — surfaces a real data-quality problem rather than
        silently masking it. Trade-off: the error path collects the disagreement
        groups to materialize a useful error message, so a pipeline running with
        millions of small disagreement groups in the dataset will pay an extra
        full scan just to fail. The clear-failure-over-fast-path bias is the right
        one for a data-cleaning stage; switch to ``drop`` / ``first`` / ``last``
        if you'd rather lose that diagnostic in exchange for speed.
      - ``"first"`` / ``"last"``: keep the first / last row in source order.
      - ``"drop"``: remove **all** rows in the disagreement group.

    The default ``by`` set is the four mandatory MEDS event-row columns
    (``subject_id``, ``time``, ``code``, ``numeric_value``). ``by`` columns that
    aren't present in the input are silently dropped from the key — datasets that
    don't carry e.g. ``numeric_value`` still dedupe correctly on the columns they do
    have. Pipeline configs that carry ``text_value`` (or any other extra schema
    column) should override ``by`` to include it; otherwise rows that share the
    default key but disagree on ``text_value`` will be silently collapsed.

    Args:
        stage_cfg: Hydra-supplied stage config. Recognized keys:

            - ``by`` (``list[str] | None``): see above. Default ``None`` → use the
              MEDS mandatory columns.
            - ``on_conflict`` (``str``): one of ``"error"``, ``"first"``,
              ``"last"``, ``"drop"``. Default ``"error"``.

    Returns:
        A function ``df → df`` that applies the dedup policy.

    Raises:
        ValueError: If ``on_conflict`` is not one of the four supported policies.

    Examples:
        Pure duplicates collapse under every policy — try it on the strictest
        (``error``) first to confirm:

        >>> stage_cfg = DictConfig({"on_conflict": "error"})
        >>> fn = deduplicate_events(stage_cfg)
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 2],
        ...     "time": [None, None, None],
        ...     "code": ["A", "A", "B"],
        ...     "numeric_value": [1.0, 1.0, 2.0],
        ... }).lazy()
        >>> fn(df).collect().sort("subject_id", "code")
        shape: (2, 4)
        ┌────────────┬──────┬──────┬───────────────┐
        │ subject_id ┆ time ┆ code ┆ numeric_value │
        │ ---        ┆ ---  ┆ ---  ┆ ---           │
        │ i64        ┆ null ┆ str  ┆ f64           │
        ╞════════════╪══════╪══════╪═══════════════╡
        │ 1          ┆ null ┆ A    ┆ 1.0           │
        │ 2          ┆ null ┆ B    ┆ 2.0           │
        └────────────┴──────┴──────┴───────────────┘

        Rows that share the ``by`` columns but disagree on something outside
        ``by`` raise under the default ``error`` policy. The error message names
        the offending group so the user can find it in the source data:

        >>> df_disagree = pl.DataFrame({
        ...     "subject_id": [1, 1],
        ...     "time": [None, None],
        ...     "code": ["MEDS_DEATH", "MEDS_DEATH"],
        ...     "numeric_value": [None, None],
        ...     "text_value": ["2020-03-01", "2020-03-05"],   # disagrees!
        ... }).lazy()
        >>> deduplicate_events(DictConfig({}))(df_disagree).collect()
        Traceback (most recent call last):
            ...
        MEDS_transforms.stages.deduplicate_events.deduplicate_events.DuplicateEventConflictError: ...

        Switch to ``first`` to keep the earliest row in source order, ``last`` for
        the latest:

        >>> deduplicate_events(DictConfig({"on_conflict": "first"}))(df_disagree).collect()
        shape: (1, 5)
        ┌────────────┬──────┬────────────┬───────────────┬────────────┐
        │ subject_id ┆ time ┆ code       ┆ numeric_value ┆ text_value │
        │ ---        ┆ ---  ┆ ---        ┆ ---           ┆ ---        │
        │ i64        ┆ null ┆ str        ┆ null          ┆ str        │
        ╞════════════╪══════╪════════════╪═══════════════╪════════════╡
        │ 1          ┆ null ┆ MEDS_DEATH ┆ null          ┆ 2020-03-01 │
        └────────────┴──────┴────────────┴───────────────┴────────────┘
        >>> deduplicate_events(DictConfig({"on_conflict": "last"}))(df_disagree).collect()
        shape: (1, 5)
        ┌────────────┬──────┬────────────┬───────────────┬────────────┐
        │ subject_id ┆ time ┆ code       ┆ numeric_value ┆ text_value │
        │ ---        ┆ ---  ┆ ---        ┆ ---           ┆ ---        │
        │ i64        ┆ null ┆ str        ┆ null          ┆ str        │
        ╞════════════╪══════╪════════════╪═══════════════╪════════════╡
        │ 1          ┆ null ┆ MEDS_DEATH ┆ null          ┆ 2020-03-05 │
        └────────────┴──────┴────────────┴───────────────┴────────────┘

        ``drop`` removes every row in the disagreement group. Useful when "if I
        can't tell which value is right, throw the observation away" matches the
        downstream contract better than picking one:

        >>> df_mixed = pl.DataFrame({
        ...     "subject_id": [1, 1, 2, 3],
        ...     "time": [None, None, None, None],
        ...     "code": ["A", "A", "B", "C"],
        ...     "numeric_value": [None, None, None, None],
        ...     "text_value": ["x", "y", "ok", "fine"],   # subject 1 disagrees
        ... }).lazy()
        >>> deduplicate_events(DictConfig({"on_conflict": "drop"}))(df_mixed).collect().sort("subject_id")
        shape: (2, 5)
        ┌────────────┬──────┬──────┬───────────────┬────────────┐
        │ subject_id ┆ time ┆ code ┆ numeric_value ┆ text_value │
        │ ---        ┆ ---  ┆ ---  ┆ ---           ┆ ---        │
        │ i64        ┆ null ┆ str  ┆ null          ┆ str        │
        ╞════════════╪══════╪══════╪═══════════════╪════════════╡
        │ 2          ┆ null ┆ B    ┆ null          ┆ ok         │
        │ 3          ┆ null ┆ C    ┆ null          ┆ fine       │
        └────────────┴──────┴──────┴───────────────┴────────────┘

        Custom ``by`` controls what counts as "the same row". Including
        ``text_value`` here makes the disagreement above NOT a conflict — the two
        rows now have different keys and both survive:

        >>> stage_cfg = DictConfig({
        ...     "by": ["subject_id", "time", "code", "numeric_value", "text_value"],
        ...     "on_conflict": "error",
        ... })
        >>> deduplicate_events(stage_cfg)(df_disagree).collect().sort("text_value")
        shape: (2, 5)
        ┌────────────┬──────┬────────────┬───────────────┬────────────┐
        │ subject_id ┆ time ┆ code       ┆ numeric_value ┆ text_value │
        │ ---        ┆ ---  ┆ ---        ┆ ---           ┆ ---        │
        │ i64        ┆ null ┆ str        ┆ null          ┆ str        │
        ╞════════════╪══════╪════════════╪═══════════════╪════════════╡
        │ 1          ┆ null ┆ MEDS_DEATH ┆ null          ┆ 2020-03-01 │
        │ 1          ┆ null ┆ MEDS_DEATH ┆ null          ┆ 2020-03-05 │
        └────────────┴──────┴────────────┴───────────────┴────────────┘

        ``by`` columns missing from the input are silently dropped from the key
        — datasets without ``numeric_value`` still dedupe on the columns they
        carry:

        >>> df_no_num = pl.DataFrame({
        ...     "subject_id": [1, 1],
        ...     "time": [None, None],
        ...     "code": ["A", "A"],
        ... }).lazy()
        >>> deduplicate_events(DictConfig({}))(df_no_num).collect()
        shape: (1, 3)
        ┌────────────┬──────┬──────┐
        │ subject_id ┆ time ┆ code │
        │ ---        ┆ ---  ┆ ---  │
        │ i64        ┆ null ┆ str  │
        ╞════════════╪══════╪══════╡
        │ 1          ┆ null ┆ A    │
        └────────────┴──────┴──────┘

        If the user-supplied ``by`` shares NO columns with the input schema, the
        stage degrades to a no-op rather than collapsing the entire frame to one
        row — better to do nothing than to silently destroy data:

        >>> df_no_overlap = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]}).lazy()
        >>> deduplicate_events(DictConfig({"by": ["subject_id", "time"]}))(df_no_overlap).collect()
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ str │
        ╞═════╪═════╡
        │ 1   ┆ x   │
        │ 1   ┆ x   │
        │ 2   ┆ y   │
        └─────┴─────┘

        Unknown ``on_conflict`` raises immediately — at stage-construction time,
        before any data is touched:

        >>> deduplicate_events(DictConfig({"on_conflict": "median"}))
        Traceback (most recent call last):
            ...
        ValueError: deduplicate_events: unknown on_conflict 'median'. ...
    """
    on_conflict = stage_cfg.get("on_conflict", "error")
    if on_conflict not in _VALID_POLICIES:
        supported = ", ".join(sorted(_VALID_POLICIES))
        raise ValueError(f"deduplicate_events: unknown on_conflict {on_conflict!r}. Supported: {supported}.")
    by_cfg = stage_cfg.get("by") or list(_DEFAULT_BY)

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        cols = df.collect_schema().names()
        by_cols = [c for c in by_cfg if c in cols]
        if not by_cols:
            # No usable key — every row is its own group, dedup is a no-op.
            return df

        if on_conflict == "first":
            return df.unique(subset=by_cols, keep="first", maintain_order=True)
        if on_conflict == "last":
            return df.unique(subset=by_cols, keep="last", maintain_order=True)

        # ``error`` and ``drop`` both need to identify disagreement groups. Strategy:
        # collapse exact duplicates first (they're fine in every policy), then count
        # remaining rows per ``by`` group — anything left with count > 1 is a group
        # whose rows agree on ``by`` but disagree on some other column.
        deduped = df.unique(maintain_order=True)
        if on_conflict == "drop":
            return deduped.filter(pl.len().over(by_cols) == 1)

        # on_conflict == "error": materialize the conflicts so the message can name them.
        conflicts = (
            deduped.with_columns(_n=pl.len().over(by_cols)).filter(pl.col("_n") > 1).drop("_n").collect()
        )
        if conflicts.is_empty():
            return deduped
        # Show the first conflicting group in the error to make diagnosis easy. We
        # join back on ``by`` to grab every row that shares the offending key, not
        # just the first arbitrarily. ``None`` values in the key need ``.is_null()``
        # rather than ``== None`` (which would silently produce null and match nothing).
        first_key = conflicts.select(by_cols).row(0, named=True)
        offending = conflicts.filter(
            pl.all_horizontal(
                *[pl.col(c).is_null() if v is None else pl.col(c) == v for c, v in first_key.items()]
            )
        )
        raise DuplicateEventConflictError(
            f"deduplicate_events: {len(conflicts)} rows in {conflicts.select(by_cols).n_unique()} "
            f"group(s) share their `by` columns ({by_cols!r}) but disagree on a non-`by` column. "
            f"First offending group: by={first_key!r}, rows=\n{offending}\n"
            "Fix the source data, or rerun this stage with on_conflict in "
            "{'first', 'last', 'drop'} to apply a silent-collapse policy."
        )

    return fn
