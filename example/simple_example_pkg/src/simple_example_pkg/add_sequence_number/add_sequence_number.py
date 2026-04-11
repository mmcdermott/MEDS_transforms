"""Adds a sequence number column to each subject's events.

This stage demonstrates using a custom ``example_class`` on ``Stage.register`` so that downstream
packages can subclass ``StageExample`` with custom output validation logic.
"""

from collections.abc import Callable

import polars as pl
from omegaconf import DictConfig
from polars.testing import assert_frame_equal

from MEDS_transforms.stages import Stage
from MEDS_transforms.stages.examples import StageExample


class ExtraColumnsStageExample(StageExample):
    """A StageExample subclass that tolerates extra columns in the actual output.

    This is useful for stages that produce additional columns beyond what the expected output specifies (e.g.,
    computed indices or auxiliary metadata), allowing test examples to validate only a subset of the output.

    Examples:
        >>> import tempfile
        >>> from datetime import datetime
        >>> from pathlib import Path
        >>> from meds import DatasetMetadataSchema
        >>> from MEDS_transforms.stages.examples import MEDSDataset
        >>> want_df = pl.DataFrame({
        ...     "subject_id": [1, 2],
        ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        ...     "code": ["A", "B"],
        ... })
        >>> got_df = pl.DataFrame({
        ...     "subject_id": [1, 2],
        ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        ...     "code": ["A", "B"],
        ...     "extra_col": [10, 20],
        ... })
        >>> want_ds = MEDSDataset(data_shards={"0": want_df}, dataset_metadata=DatasetMetadataSchema())
        >>> got_ds = MEDSDataset(data_shards={"0": got_df}, dataset_metadata=DatasetMetadataSchema())
        >>> example = ExtraColumnsStageExample(stage_name="test", scenario_name="s", want_data=want_ds)
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     _ = got_ds.write(Path(tmpdir))
        ...     example.check_outputs(Path(tmpdir))
    """

    def check_outputs(self, output_dir, is_resolved_dir=False):
        """Check outputs, tolerating extra columns in the actual data."""
        self._StageExample__check_files(output_dir, is_resolved_dir)

        if self.want_data is not None:
            data_dir = output_dir if is_resolved_dir else output_dir / "data"
            got_shards = self._StageExample__data_shards(data_dir)

            assert got_shards.keys() == self.want_data._pl_shards.keys(), (
                f"Shards differ: {got_shards.keys()} vs {self.want_data._pl_shards.keys()}"
            )
            for shard_name, got_df in got_shards.items():
                want_df = self.want_data._pl_shards[shard_name]
                got_df = got_df.select(want_df.columns)
                assert_frame_equal(got_df, want_df, **self.df_check_kwargs)

        if self.want_metadata is not None:
            super().check_outputs(output_dir, is_resolved_dir)


@Stage.register(example_class=ExtraColumnsStageExample)
def add_sequence_number(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Adds a per-subject sequence number column to each event.

    The sequence number counts events per subject in order, starting from 1. This produces an extra column
    ``seq_num`` beyond the standard MEDS schema, demonstrating how ``example_class`` lets downstream packages
    validate only the columns they care about.

    Examples:
        >>> from datetime import datetime
        >>> df = pl.DataFrame({
        ...     "subject_id": [1, 1, 2],
        ...     "time": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 1)],
        ...     "code": ["A", "B", "C"],
        ... })
        >>> add_sequence_number(DictConfig({}))(df)
        shape: (3, 4)
        ┌────────────┬─────────────────────┬──────┬─────────┐
        │ subject_id ┆ time                ┆ code ┆ seq_num │
        │ ---        ┆ ---                 ┆ ---  ┆ ---     │
        │ i64        ┆ datetime[μs]        ┆ str  ┆ u32     │
        ╞════════════╪═════════════════════╪══════╪═════════╡
        │ 1          ┆ 2020-01-01 00:00:00 ┆ A    ┆ 1       │
        │ 1          ┆ 2020-01-02 00:00:00 ┆ B    ┆ 2       │
        │ 2          ┆ 2020-01-01 00:00:00 ┆ C    ┆ 1       │
        └────────────┴─────────────────────┴──────┴─────────┘
    """

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        return df.with_columns(
            pl.arange(0, pl.len()).over("subject_id").cast(pl.UInt32).add(1).alias("seq_num")
        )

    return fn
