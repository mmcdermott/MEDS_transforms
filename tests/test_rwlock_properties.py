"""Property tests over how parallel workers divide a shard list (issue #399).

The `do_overwrite` bug was a concurrency bug, but not a *scheduling* one: what broke it was simply a
worker reaching an entry another worker had already finished. That is an ordering over visits, which is
exactly the kind of thing Hypothesis generates well — so rather than racing threads and hoping to catch
a rare interleaving (the issue's own repro rate was 1 in 12), these drive `rwlock_wrap` through
generated visit orders single-threaded and assert the invariants hold for every one of them.

The property that matters is that **work division does not depend on `do_overwrite`**. Whichever value
it takes, N workers walking the same list should compute each output exactly once between them, and
should never make an output that already exists disappear.

Not covered here: two workers genuinely overlapping inside `compute_fn`, since a visit is atomic in
this model. `test_concurrent_reader_never_sees_a_missing_output` in `test_rwlock_run_scoping.py`
exercises that with real threads.
"""

import tempfile
from pathlib import Path

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from MEDS_transforms.dataframe import read_df, write_df
from MEDS_transforms.mapreduce.rwlock import rwlock_wrap

MAX_SHARDS = 4
MAX_WORKERS = 3


@st.composite
def visit_orders(draw) -> tuple[int, int, list[tuple[int, int]]]:
    """Generate a shard count, a worker count, and one interleaving of the workers' passes.

    Every worker walks the whole `(input -> output)` list — that is how the mapreduce machinery divides
    work — so the visits are exactly the `(worker, shard)` pairs. Any permutation of them is a possible
    execution: it induces some shard order per worker (which is shuffled per worker anyway, by
    `shuffle_shards`) and some interleaving between them.
    """

    n_shards = draw(st.integers(min_value=1, max_value=MAX_SHARDS))
    n_workers = draw(st.integers(min_value=1, max_value=MAX_WORKERS))
    visits = [(w, s) for w in range(n_workers) for s in range(n_shards)]
    return n_shards, n_workers, draw(st.permutations(visits))


def double_value(df):
    """The map function under test; returns the shard it was called for so calls can be counted."""

    return df.with_columns(pl.col("value") * 2)


# Each example builds its own temp dir rather than taking `tmp_path`: a function-scoped fixture is
# created once and then shared across every example, which leaks state between them.
@settings(max_examples=150, deadline=None)  # Parquet IO is too jittery for a per-example deadline.
@given(order=visit_orders(), do_overwrite=st.booleans())
def test_each_output_is_computed_once_and_never_vanishes(order, do_overwrite):
    """For any interleaving, and for either `do_overwrite`, the workers share the work exactly once."""

    n_shards, _, visits = order

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        in_dir, out_dir = root / "in", root / "out"
        in_dir.mkdir()
        marker_dir = root / "logs" / ".run_markers" / "run-1"

        for shard in range(n_shards):
            pl.DataFrame({"subject_id": [shard], "value": [shard + 1]}).write_parquet(
                in_dir / f"{shard}.parquet"
            )

        computed: list[int] = []

        def counting_map_fn(df, _computed=computed):
            frame = df.collect() if isinstance(df, pl.LazyFrame) else df
            _computed.append(frame["subject_id"][0])
            return double_value(frame)

        produced: set[Path] = set()
        for _worker, shard in visits:
            out_fp = out_dir / f"{shard}.parquet"
            rwlock_wrap(
                in_dir / f"{shard}.parquet",
                out_fp,
                read_df,
                write_df,
                counting_map_fn,
                do_overwrite=do_overwrite,
                marker_dir=marker_dir,
            )
            produced.add(out_fp)

            # Nothing already written may have gone missing. This is the reducer's view of the world:
            # it waits for the map outputs and then reads them, and the bug was that a revisiting
            # worker deleted one out from under it.
            for fp in produced:
                assert fp.is_file(), f"{fp.name} vanished after a later visit (do_overwrite={do_overwrite})"

        assert sorted(computed) == list(range(n_shards)), (
            f"Each shard should be computed exactly once across all workers; got {sorted(computed)} "
            f"for {n_shards} shards over {len(visits)} visits (do_overwrite={do_overwrite})."
        )

        for shard in range(n_shards):
            out = pl.read_parquet(out_dir / f"{shard}.parquet")
            assert out["value"].to_list() == [(shard + 1) * 2], "Output contents are wrong."


@settings(max_examples=100, deadline=None)
@given(order=visit_orders())
def test_a_second_run_redoes_everything_exactly_once(order):
    """`do_overwrite` still means what it says: a *new* run replaces the previous run's outputs.

    Run scoping must not tip over into never overwriting. Whatever the interleaving, the second run recomputes
    each output once — not zero times (stale data kept) and not once per worker.
    """

    n_shards, _, visits = order

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        in_dir, out_dir = root / "in", root / "out"
        in_dir.mkdir()

        for shard in range(n_shards):
            pl.DataFrame({"subject_id": [shard], "value": [shard + 1]}).write_parquet(
                in_dir / f"{shard}.parquet"
            )

        def run(marker_dir: Path) -> list[int]:
            computed: list[int] = []

            def counting_map_fn(df, _computed=computed):
                frame = df.collect() if isinstance(df, pl.LazyFrame) else df
                _computed.append(frame["subject_id"][0])
                return double_value(frame)

            for _worker, shard in visits:
                rwlock_wrap(
                    in_dir / f"{shard}.parquet",
                    out_dir / f"{shard}.parquet",
                    read_df,
                    write_df,
                    counting_map_fn,
                    do_overwrite=True,
                    marker_dir=marker_dir,
                )
            return computed

        first = run(root / "logs" / ".run_markers" / "run-1")
        second = run(root / "logs" / ".run_markers" / "run-2")

        assert sorted(first) == list(range(n_shards))
        assert sorted(second) == list(range(n_shards)), (
            "A new run must redo every output exactly once, not skip them as if they were fresh."
        )
