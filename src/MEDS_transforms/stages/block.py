"""Stage blocks: a series of map stages run on each shard, one after another, by the same worker.

A stage block is written in the pipeline YAML as an entry with a ``_block`` list of stages:

```yaml
stages:
  - filter_subjects
  - preprocess:
      _block:
        - add_time_derived_measurements:
            age: {DOB_code: MEDS_BIRTH, age_code: AGE, age_unit: years}
        - occlude_outliers:
            stddev_cutoff: 1
  - normalization
```

To the rest of the pipeline, a block is a single stage: it has one output directory
(``<output_dir>/<block name>``), one ``.done`` marker, and one parallelization setting. Each of its workers
takes a shard, reads it once, applies every stage's transformation in turn, and writes the result once, so
the intermediate results never touch disk, and polars optimizes the whole chain as one query. Workers share
a block's shards exactly as they share a map stage's.

For now, a block may contain only map stages. They all see the block's input metadata, which is unchanged
throughout the block as map stages don't modify metadata. Only the first stage may use its own read function
and only the last its own write function, and the stages must agree on ``train_only``.
"""

from __future__ import annotations

import dataclasses
import textwrap
from typing import TYPE_CHECKING

from ..compute_modes.compute_fn import bind_compute_fn
from ..compute_modes.match_revise import is_match_revise, match_revise_fntr
from ..mapreduce import map_stage
from .base import StageType

if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl
    from omegaconf import DictConfig

    from .base import Stage


@dataclasses.dataclass
class StageBlock:
    """A series of map stages that runs as one stage of a pipeline. See the module docstring.

    It offers what the stage launcher uses of a `Stage`: ``main``, ``stage_type``, ``is_metadata``,
    ``carry_input_data`` and ``stage_docstring``.

    Attributes:
        stage_name: The block's name in the pipeline.
        stages: The block's stages, in order, as ``(name in the block, loaded stage)`` pairs.

    Raises:
        ValueError: If a stage isn't a map stage, or uses a custom read or write function anywhere but first
            or last.

    Examples:
        >>> from MEDS_transforms.stages import filter_subjects, occlude_outliers, aggregate_code_metadata
        >>> block = StageBlock("prep", [("filter_subjects", filter_subjects), ("occlude", occlude_outliers)])
        >>> block.stage_type, block.is_metadata, block.carry_input_data
        (<StageType.MAP: 'map'>, False, None)
        >>> print(block.stage_docstring.splitlines()[0])
        A block of map stages, run in turn on each shard: filter_subjects, occlude.
        >>> StageBlock("prep", [("fit", aggregate_code_metadata)])
        Traceback (most recent call last):
            ...
        ValueError: Stage block prep can only contain map stages for now, but fit is a mapreduce stage.

        Custom read and write functions are allowed only where the block itself reads and writes:

        >>> import polars as pl
        >>> from MEDS_transforms.stages import Stage
        >>> with Stage.suppress_validation():
        ...     reader = Stage(map_fn=lambda df: df, read_fn=pl.scan_csv)
        ...     writer = Stage(map_fn=lambda df: df, write_fn=pl.DataFrame.write_csv)
        >>> StageBlock("b", [("first", reader), ("second", reader)])
        Traceback (most recent call last):
            ...
        ValueError: Stage block b: only its first stage may use its own read function, but second does.
        >>> StageBlock("b", [("first", writer), ("second", reader)])
        Traceback (most recent call last):
            ...
        ValueError: Stage block b: only its last stage may use its own write function, but first does.
        >>> block = StageBlock("b", [("first", reader), ("second", writer)])
        >>> block.read_fn is pl.scan_csv, block.write_fn is pl.DataFrame.write_csv, block.default_config
        (True, True, None)
    """

    stage_name: str
    stages: list[tuple[str, Stage]]

    stage_type: StageType = StageType.MAP
    is_metadata: bool = False
    carry_input_data: None = None

    def __post_init__(self) -> None:
        last = len(self.stages) - 1
        for i, (name, stage) in enumerate(self.stages):
            if stage.stage_type is not StageType.MAP:
                raise ValueError(
                    f"Stage block {self.stage_name} can only contain map stages for now, but {name} is a "
                    f"{stage.stage_type} stage."
                )
            if stage.read_fn is not None and i != 0:
                raise ValueError(
                    f"Stage block {self.stage_name}: only its first stage may use its own read function, "
                    f"but {name} does."
                )
            if stage.write_fn is not None and i != last:
                raise ValueError(
                    f"Stage block {self.stage_name}: only its last stage may use its own write function, "
                    f"but {name} does."
                )

    @property
    def read_fn(self):
        return self.stages[0][1].read_fn

    @property
    def write_fn(self):
        return self.stages[-1][1].write_fn

    @property
    def default_config(self) -> None:
        return None

    @property
    def stage_docstring(self) -> str:
        names = ", ".join(name for name, _ in self.stages)
        lines = [f"A block of map stages, run in turn on each shard: {names}."]
        for name, stage in self.stages:
            lines.extend(["", f"{name}:", textwrap.indent(stage.stage_docstring or "", "  ")])
        return "\n".join(lines)

    def map_fntr(self, cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
        """Bind each stage's map function to its own config, and chain them.

        Each stage's config is its entry in ``cfg.stage_cfg._block``, resolved by `PipelineConfig`: the
        stage's defaults and pipeline options, with the block's directories.

        Examples:
            >>> import polars as pl
            >>> from MEDS_transforms.stages import Stage
            >>> def add(stage_cfg: DictConfig):
            ...     return lambda df: df.with_columns(pl.col("x") + stage_cfg.n)
            >>> def double(df):
            ...     return df.with_columns(pl.col("x") * 2)
            >>> with Stage.suppress_validation():
            ...     stages = [("add", Stage(map_fn=add)), ("double", Stage(map_fn=double))]
            >>> block = StageBlock("b", stages)
            >>> cfg = DictConfig({"stage_cfg": {"_block": [{"n": 1}, {}]}})
            >>> block.map_fntr(cfg)(pl.LazyFrame({"x": [1, 2]})).collect()["x"].to_list()
            [4, 6]

        A stage in a block can use match & revise, as on its own:

        >>> df = pl.LazyFrame({"subject_id": [1, 1], "time": [1, 2], "code": ["A", "B"], "x": [1, 2]})
        >>> add_cfg = {"_match_revise": [{"_matcher": {"code": "A"}, "n": 10}]}
        >>> cfg = DictConfig({"stage_cfg": {"_block": [add_cfg, {}]}})
        >>> block.map_fntr(cfg)(df).collect()["x"].to_list()
        [22, 4]
        """
        stage_cfgs = cfg.stage_cfg._block
        fns = []
        for (_, stage), stage_cfg in zip(self.stages, stage_cfgs, strict=True):
            if is_match_revise(stage_cfg):
                fns.append(match_revise_fntr(cfg, stage_cfg, stage.map_fn))
            else:
                fns.append(bind_compute_fn(cfg, stage_cfg, stage.map_fn))

        def block_fn(df: pl.LazyFrame) -> pl.LazyFrame:
            for fn in fns:
                df = fn(df)
            return df

        return block_fn

    @property
    def main(self) -> Callable[[DictConfig], None]:
        """Run the block like a map stage, with the chained map functions.

        Examples:
            >>> from pathlib import Path
            >>> from MEDS_transforms.stages import filter_subjects
            >>> block = StageBlock("b", [("filter", filter_subjects)])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     dirs = {
            ...         "data_input_dir": f"{simple_static_MEDS}/data",
            ...         "metadata_input_dir": f"{simple_static_MEDS}/metadata",
            ...         "data_output_dir": f"{tmpdir}/data",
            ...         "metadata_output_dir": f"{tmpdir}/metadata",
            ...         "output_dir": f"{tmpdir}/data",
            ...         "train_only": False,
            ...     }
            ...     filter_cfg = {**filter_subjects.default_config, **dirs, "min_events_per_subject": 1000}
            ...     cfg = DictConfig({
            ...         "do_overwrite": False, "worker": 0, "stage_cfg": {**dirs, "_block": [filter_cfg]}
            ...     })
            ...     block.main(cfg)
            ...     print(sorted(p.name for p in Path(tmpdir).iterdir()))
            ...     print(sum(pl.read_parquet(fp).height for fp in Path(tmpdir).rglob("*.parquet")))
            ['data', 'metadata']
            0
        """
        from ..stage_outputs import prepare_output_dirs

        def main_fn(cfg: DictConfig) -> None:
            prepare_output_dirs(cfg)
            map_stage(cfg, self.map_fntr, read_fn=self.read_fn, write_fn=self.write_fn)

        main_fn.__name__ = self.stage_name
        main_fn.__doc__ = self.stage_docstring
        return main_fn
