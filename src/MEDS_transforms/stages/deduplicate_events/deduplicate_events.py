"""Drop duplicate event rows under a configurable conflict policy.

See ``config.yaml`` for the parameters; see :func:`deduplicate_events` for the
behavior, the ``on_conflict`` policies, and end-to-end doctests.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING

import polars as pl
from meds import DataSchema

from .. import Stage

if TYPE_CHECKING:
    from collections.abc import Callable

    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _default_by_cols() -> tuple[str, ...]:
    """Return the full set of MEDS ``DataSchema`` field names to use as the default ``by``."""
    return tuple(f.name for f in DataSchema.schema())


class OnConflictPolicy(StrEnum):
    """How :func:`deduplicate_events` resolves rows that agree on ``by`` but disagree elsewhere.

    Stored as a ``StrEnum`` so users can specify the policy as a plain string in the Hydra
    stage config (``on_conflict: error`` etc.) and the same constant powers the dispatch
    inside the function body.

    Attributes:
        ERROR: Raise :class:`DuplicateEventConflictError` naming the offending group.
        FIRST: Keep the first row in source order.
        LAST: Keep the last row in source order.
        DROP: Remove every row in the disagreement group, with a logger warning naming the
            dropped groups so the data loss isn't silent.
        CAPTURE: Keep every row in the disagreement group and add a ``_conflicts`` struct
            column annotating each conflict-group row with the disagreeing values across the
            group. Useful for downstream auditability — preserves the data the other policies
            collapse or drop, at the cost of (a) a struct column the rest of the pipeline must
            tolerate, and (b) materializing the per-group aggregates.
    """

    ERROR = "error"
    FIRST = "first"
    LAST = "last"
    DROP = "drop"
    CAPTURE = "capture"


# Column name for the per-row conflicts struct produced under ``OnConflictPolicy.CAPTURE``.
# The struct shape is ``{n_collisions: u32, <col>_values: list[<dtype>] for each non-by col}``;
# rows that aren't part of any disagreement group carry ``null``.
CONFLICTS_COL: str = "_conflicts"


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
      - ``"drop"``: remove **all** rows in the disagreement group. Always logs the
        dropped keys at WARNING — destructive, never silent.
      - ``"capture"``: keep every row in the disagreement group and add a
        ``_conflicts`` struct column carrying ``n_collisions`` and per-non-``by``-column
        value lists. Best for ETL pipelines that want to retain the raw disagreement
        for downstream audit/QA.

    Independent of ``on_conflict``, ``log_conflicts: bool = False`` opts every
    non-``error`` policy into the same WARNING-level logging that ``drop`` uses
    unconditionally. Useful when ``first`` / ``last`` / ``capture`` shouldn't run
    silently.

    The default ``by`` set is the full MEDS ``DataSchema`` field set (``subject_id``,
    ``time``, ``code``, ``numeric_value``, ``text_value``) read at runtime via
    :meth:`meds.DataSchema.schema`. ``by`` columns that aren't present in the input are
    silently dropped from the key — datasets that don't carry e.g. ``text_value`` still
    dedupe correctly on the columns they do have. To dedupe on a strict subset (or to
    include extra non-schema columns), override ``by`` explicitly.

    Args:
        stage_cfg: Hydra-supplied stage config. Recognized keys:

            - ``by`` (``list[str] | None``): see above. Default ``None`` → use the full
              ``DataSchema`` field set.
            - ``on_conflict`` (``str``): one of the values of
              :class:`OnConflictPolicy`: ``"error"``, ``"first"``, ``"last"``,
              ``"drop"``, ``"capture"``. Default ``"error"``.
            - ``log_conflicts`` (``bool``): when True, log offending ``by`` keys at
              WARNING for every non-``error`` policy. Default ``False``. ``drop``
              always logs regardless of this flag — destructiveness is never silent.

    Returns:
        A function ``df → df`` that applies the dedup policy.

    Raises:
        ValueError: If ``on_conflict`` is not one of the supported :class:`OnConflictPolicy`
            values.

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

        Rows that share every ``DataSchema`` column but disagree on a non-schema
        column raise under the default ``error`` policy. The error message names
        the offending group so the user can find it in the source data. Here
        ``annotation`` is outside ``DataSchema`` and therefore not in the default
        ``by``, so the two MEDS_DEATH rows look like the same event with two
        different annotations attached:

        >>> df_disagree = pl.DataFrame({
        ...     "subject_id": [1, 1],
        ...     "time": [None, None],
        ...     "code": ["MEDS_DEATH", "MEDS_DEATH"],
        ...     "numeric_value": [None, None],
        ...     "text_value": [None, None],
        ...     "annotation": ["from-EHR", "from-claims"],   # disagrees!
        ... }).lazy()
        >>> deduplicate_events(DictConfig({}))(df_disagree).collect()
        ... # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
            ...
        MEDS_transforms.stages.deduplicate_events.deduplicate_events.DuplicateEventConflictError:
        deduplicate_events: 2 rows in 1 group(s) share their `by` columns
        (['subject_id', 'time', 'code', 'numeric_value', 'text_value']) but disagree on a
        non-`by` column. First offending group: by={'subject_id': 1, 'time': None,
        'code': 'MEDS_DEATH', 'numeric_value': None, 'text_value': None}, rows=
        shape: (2, 6)
        ┌────────────┬──────┬────────────┬───────────────┬────────────┬─────────────┐
        │ subject_id ┆ time ┆ code       ┆ numeric_value ┆ text_value ┆ annotation  │
        │ ---        ┆ ---  ┆ ---        ┆ ---           ┆ ---        ┆ ---         │
        │ i64        ┆ null ┆ str        ┆ null          ┆ null       ┆ str         │
        ╞════════════╪══════╪════════════╪═══════════════╪════════════╪═════════════╡
        │ 1          ┆ null ┆ MEDS_DEATH ┆ null          ┆ null       ┆ from-EHR    │
        │ 1          ┆ null ┆ MEDS_DEATH ┆ null          ┆ null       ┆ from-claims │
        └────────────┴──────┴────────────┴───────────────┴────────────┴─────────────┘
        Fix the source data, or rerun this stage with on_conflict in {'first', 'last',
        'drop', 'capture'} to apply a silent-collapse / capture policy.

        Switch to ``first`` to keep the earliest row in source order, ``last`` for
        the latest:

        >>> deduplicate_events(DictConfig({"on_conflict": "first"}))(df_disagree).collect()
        shape: (1, 6)
        ┌────────────┬──────┬────────────┬───────────────┬────────────┬────────────┐
        │ subject_id ┆ time ┆ code       ┆ numeric_value ┆ text_value ┆ annotation │
        │ ---        ┆ ---  ┆ ---        ┆ ---           ┆ ---        ┆ ---        │
        │ i64        ┆ null ┆ str        ┆ null          ┆ null       ┆ str        │
        ╞════════════╪══════╪════════════╪═══════════════╪════════════╪════════════╡
        │ 1          ┆ null ┆ MEDS_DEATH ┆ null          ┆ null       ┆ from-EHR   │
        └────────────┴──────┴────────────┴───────────────┴────────────┴────────────┘
        >>> deduplicate_events(DictConfig({"on_conflict": "last"}))(df_disagree).collect()
        shape: (1, 6)
        ┌────────────┬──────┬────────────┬───────────────┬────────────┬─────────────┐
        │ subject_id ┆ time ┆ code       ┆ numeric_value ┆ text_value ┆ annotation  │
        │ ---        ┆ ---  ┆ ---        ┆ ---           ┆ ---        ┆ ---         │
        │ i64        ┆ null ┆ str        ┆ null          ┆ null       ┆ str         │
        ╞════════════╪══════╪════════════╪═══════════════╪════════════╪═════════════╡
        │ 1          ┆ null ┆ MEDS_DEATH ┆ null          ┆ null       ┆ from-claims │
        └────────────┴──────┴────────────┴───────────────┴────────────┴─────────────┘

        ``drop`` removes every row in the disagreement group. Useful when "if I
        can't tell which value is right, throw the observation away" matches the
        downstream contract better than picking one. The dropped keys are emitted
        via ``logger.warning`` so the data loss isn't silent:

        >>> df_mixed = pl.DataFrame({
        ...     "subject_id": [1, 1, 2, 3],
        ...     "time": [None, None, None, None],
        ...     "code": ["A", "A", "B", "C"],
        ...     "numeric_value": [None, None, None, None],
        ...     "annotation": ["x", "y", "ok", "fine"],   # subject 1 disagrees
        ... }).lazy()
        >>> deduplicate_events(DictConfig({"on_conflict": "drop"}))(df_mixed).collect().sort("subject_id")
        shape: (2, 5)
        ┌────────────┬──────┬──────┬───────────────┬────────────┐
        │ subject_id ┆ time ┆ code ┆ numeric_value ┆ annotation │
        │ ---        ┆ ---  ┆ ---  ┆ ---           ┆ ---        │
        │ i64        ┆ null ┆ str  ┆ null          ┆ str        │
        ╞════════════╪══════╪══════╪═══════════════╪════════════╡
        │ 2          ┆ null ┆ B    ┆ null          ┆ ok         │
        │ 3          ┆ null ┆ C    ┆ null          ┆ fine       │
        └────────────┴──────┴──────┴───────────────┴────────────┘

        Custom ``by`` controls what counts as "the same row". Including
        ``annotation`` in ``by`` makes the disagreement above NOT a conflict — the
        two rows now have different keys and both survive:

        >>> stage_cfg = DictConfig({
        ...     "by": ["subject_id", "time", "code", "numeric_value", "text_value", "annotation"],
        ...     "on_conflict": "error",
        ... })
        >>> deduplicate_events(stage_cfg)(df_disagree).collect().sort("annotation")
        shape: (2, 6)
        ┌────────────┬──────┬────────────┬───────────────┬────────────┬─────────────┐
        │ subject_id ┆ time ┆ code       ┆ numeric_value ┆ text_value ┆ annotation  │
        │ ---        ┆ ---  ┆ ---        ┆ ---           ┆ ---        ┆ ---         │
        │ i64        ┆ null ┆ str        ┆ null          ┆ null       ┆ str         │
        ╞════════════╪══════╪════════════╪═══════════════╪════════════╪═════════════╡
        │ 1          ┆ null ┆ MEDS_DEATH ┆ null          ┆ null       ┆ from-EHR    │
        │ 1          ┆ null ┆ MEDS_DEATH ┆ null          ┆ null       ┆ from-claims │
        └────────────┴──────┴────────────┴───────────────┴────────────┴─────────────┘

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

        ``capture`` keeps every row in the disagreement group and adds a
        ``_conflicts`` struct column with ``n_collisions`` plus per-non-``by``-column
        value lists. Rows outside any disagreement group carry ``null``:

        >>> capture_df = deduplicate_events(DictConfig({"on_conflict": "capture"}))(
        ...     df_disagree
        ... ).collect()
        >>> capture_df["_conflicts"].struct.field("n_collisions").to_list()
        [2, 2]
        >>> capture_df["_conflicts"].struct.field("_conflict_annotation_values").to_list()
        [['from-EHR', 'from-claims'], ['from-EHR', 'from-claims']]

        ``log_conflicts: True`` opts every non-``error`` policy into the same
        WARNING-level offending-key log that ``drop`` uses unconditionally. Useful
        when ``first`` / ``last`` / ``capture`` shouldn't run silently. Captured
        here via the ``print_warnings`` doctest helper from ``conftest.py``:

        >>> with print_warnings():
        ...     _ = deduplicate_events(
        ...         DictConfig({"on_conflict": "first", "log_conflicts": True})
        ...     )(df_disagree).collect()
        ... # doctest: +NORMALIZE_WHITESPACE
        Warning: deduplicate_events(on_conflict='first'): collapsing 2 row(s) across 1
        `by` group(s) that disagree on a non-`by` column. Offending keys (first 10):
        shape: (1, 5)
        ┌────────────┬──────┬────────────┬───────────────┬────────────┐
        │ subject_id ┆ time ┆ code       ┆ numeric_value ┆ text_value │
        │ ---        ┆ ---  ┆ ---        ┆ ---           ┆ ---        │
        │ i64        ┆ null ┆ str        ┆ null          ┆ null       │
        ╞════════════╪══════╪════════════╪═══════════════╪════════════╡
        │ 1          ┆ null ┆ MEDS_DEATH ┆ null          ┆ null       │
        └────────────┴──────┴────────────┴───────────────┴────────────┘

        Unknown ``on_conflict`` raises immediately — at stage-construction time,
        before any data is touched:

        >>> deduplicate_events(DictConfig({"on_conflict": "median"}))
        Traceback (most recent call last):
            ...
        ValueError: deduplicate_events: unknown on_conflict 'median'. ...
    """
    raw_policy = stage_cfg.get("on_conflict", OnConflictPolicy.ERROR)
    try:
        on_conflict = OnConflictPolicy(raw_policy)
    except ValueError as e:
        supported = ", ".join(p.value for p in OnConflictPolicy)
        raise ValueError(
            f"deduplicate_events: unknown on_conflict {raw_policy!r}. Supported: {supported}."
        ) from e
    by_cfg = stage_cfg.get("by") or list(_default_by_cols())
    log_conflicts = bool(stage_cfg.get("log_conflicts", False))

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        cols = df.collect_schema().names()
        by_cols = [c for c in by_cfg if c in cols]
        if not by_cols:
            # No usable key — every row is its own group, dedup is a no-op.
            return df

        # Collapse exact duplicates first — they're fine in every policy, and removing them up
        # front means the conflict detection below sees only the rows that genuinely disagree.
        deduped = df.unique(maintain_order=True)

        if on_conflict is OnConflictPolicy.FIRST or on_conflict is OnConflictPolicy.LAST:
            keep = "first" if on_conflict is OnConflictPolicy.FIRST else "last"
            if log_conflicts:
                _log_offending(deduped, by_cols, on_conflict.value, "collapsing")
            return deduped.unique(subset=by_cols, keep=keep, maintain_order=True)

        if on_conflict is OnConflictPolicy.DROP:
            # ``drop`` always logs (data loss is destructive), regardless of ``log_conflicts``.
            _log_offending(deduped, by_cols, "drop", "dropping")
            return deduped.filter(pl.len().over(by_cols) == 1)

        if on_conflict is OnConflictPolicy.CAPTURE:
            if log_conflicts:
                _log_offending(deduped, by_cols, "capture", "annotating")
            return _attach_conflicts_struct(deduped, by_cols, cols)

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
            "{'first', 'last', 'drop', 'capture'} to apply a silent-collapse / capture policy."
        )

    return fn


def _log_offending(deduped: pl.LazyFrame, by_cols: list[str], policy_name: str, action_verb: str) -> None:
    """Materialize the conflict groups in ``deduped`` and emit a WARNING naming them.

    Shared by the policies that need to surface conflicts at run time (``drop`` always; the
    rest only when ``log_conflicts=True``). Costs an extra ``.collect()`` of the conflict
    rows; the trade-off matches the ``error`` policy's diagnostic collect.
    """
    conflicts = deduped.with_columns(_n=pl.len().over(by_cols)).filter(pl.col("_n") > 1).drop("_n").collect()
    if conflicts.is_empty():
        return
    offending_keys = conflicts.select(by_cols).unique(maintain_order=True)
    logger.warning(
        "deduplicate_events(on_conflict=%r): %s %d row(s) across %d `by` group(s) that disagree "
        "on a non-`by` column. Offending keys (first 10):\n%s",
        policy_name,
        action_verb,
        len(conflicts),
        offending_keys.height,
        offending_keys.head(10),
    )


def _attach_conflicts_struct(deduped: pl.LazyFrame, by_cols: list[str], cols: list[str]) -> pl.LazyFrame:
    """Add a ``_conflicts`` struct column annotating disagreement groups.

    Non-conflict rows carry ``null`` in ``_conflicts``. Each conflict-group row carries the
    same struct: ``{n_collisions, <col>_values}`` for each non-``by`` column. The list-valued
    fields are typed as ``list[<original dtype>]`` because polars structs require a static
    schema, and the column dtypes are read off the input.

    Implementation note: we ``group_by(by_cols).agg(pl.col(c).implode())`` to get the per-group
    value lists and ``pl.len()`` for the count, then join back to ``deduped`` so the struct
    attaches per-row. Rows whose group is size 1 get ``null`` via the ``len > 1`` check.
    """
    non_by_cols = [c for c in cols if c not in by_cols and c != CONFLICTS_COL]
    if not non_by_cols:
        # Nothing to capture — no non-by columns means no possible disagreement.
        return deduped

    value_aliases = [f"_conflict_{c}_values" for c in non_by_cols]
    agg_exprs: list[pl.Expr] = [pl.len().cast(pl.UInt32).alias("n_collisions")]
    agg_exprs.extend(pl.col(c).alias(alias) for c, alias in zip(non_by_cols, value_aliases, strict=True))

    grouped = deduped.group_by(by_cols, maintain_order=True).agg(*agg_exprs)
    # ``nulls_equal=True`` is required: MEDS event rows commonly have null ``time`` (static
    # facts) and null ``text_value`` (numeric measurements). With the default null-not-equal
    # semantics, rows that share those null keys would join as miss → ``null`` ``_conflicts``,
    # silently losing the annotation we just computed.
    return (
        deduped.join(grouped, on=by_cols, how="left", coalesce=True, nulls_equal=True)
        .with_columns(
            pl.when(pl.col("n_collisions") > 1)
            .then(pl.struct(["n_collisions", *value_aliases]))
            .otherwise(None)
            .alias(CONFLICTS_COL)
        )
        .drop(["n_collisions", *value_aliases])
    )
