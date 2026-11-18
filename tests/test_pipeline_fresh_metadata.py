"""Tests for the fresh-metadata dependency contract on Stage/Pipeline (issues #117/116/118/200)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from MEDS_transforms.configs import PipelineConfig
from MEDS_transforms.stages.base import Stage
from MEDS_transforms.stages.discovery import get_all_registered_stages

if TYPE_CHECKING:
    import polars as pl
    from omegaconf import DictConfig


def test_warns_when_requires_fresh_metadata_has_no_preceding_metadata_stage(
    caplog,
) -> None:
    """Resolving a pipeline with filter_measurements and no preceding metadata stage warns."""
    caplog.set_level(logging.WARNING, logger="MEDS_transforms.configs.pipeline")
    pipeline = PipelineConfig(stages=["filter_measurements"])
    pipeline.register_for("filter_measurements")
    assert any("requires a fresh metadata/codes.parquet" in r.message for r in caplog.records), (
        f"Expected warning about fresh metadata dependency; got: {[r.message for r in caplog.records]}"
    )


def test_no_warning_when_metadata_stage_precedes_dependent_stage(caplog) -> None:
    """aggregate_code_metadata -> filter_measurements does not warn."""
    caplog.set_level(logging.WARNING, logger="MEDS_transforms.configs.pipeline")
    pipeline = PipelineConfig(
        stages=[
            {"aggregate_code_metadata": {"aggregations": ["code/n_occurrences"]}},
            "filter_measurements",
        ]
    )
    pipeline.register_for("filter_measurements")
    assert not any("requires a fresh metadata/codes.parquet" in r.message for r in caplog.records), (
        f"Unexpected warning; got: {[r.message for r in caplog.records]}"
    )


def test_warns_when_non_refreshing_metadata_stage_precedes_dependent_stage(caplog) -> None:
    """A metadata stage that doesn't refresh codes (e.g. fit_vocabulary_indices) still warns.

    fit_vocabulary_indices consumes an existing codes.parquet and writes a decorated copy; it does
    not aggregate codes from the data. Pipelines with ``fit_vocabulary_indices -> filter_measurements``
    should still surface the warning because ``codes.parquet`` may be stale relative to the data.
    """
    caplog.set_level(logging.WARNING, logger="MEDS_transforms.configs.pipeline")
    pipeline = PipelineConfig(stages=["fit_vocabulary_indices", "filter_measurements"])
    pipeline.register_for("filter_measurements")
    assert any("requires a fresh metadata/codes.parquet" in r.message for r in caplog.records), (
        f"Expected warning about fresh metadata dependency; got: {[r.message for r in caplog.records]}"
    )


# ----------------------------------------------------------------------------------------------
# Auto-inference of requires_fresh_metadata from compute-fn signatures
# ----------------------------------------------------------------------------------------------


def test_inference_sets_true_when_compute_fn_takes_code_metadata(caplog) -> None:
    """A stage whose registered fn takes ``code_metadata`` auto-infers requires_fresh_metadata=True.

    ``filter_measurements`` and ``reorder_measurements`` no longer pass the explicit flag — the
    inference path should still produce the same warning behavior.
    """
    stages = get_all_registered_stages()
    for name in ("filter_measurements", "reorder_measurements"):
        stage = stages[name].load()
        assert stage.requires_fresh_metadata is True, (
            f"{name} should infer requires_fresh_metadata=True from its `code_metadata` parameter"
        )


def test_explicit_true_still_respected_when_signature_doesnt_take_code_metadata() -> None:
    """fit_vocabulary_indices.main(cfg) doesn't take code_metadata, but explicit True wins."""
    stages = get_all_registered_stages()
    stage = stages["fit_vocabulary_indices"].load()
    assert stage.requires_fresh_metadata is True


def test_explicit_false_overrides_signature_inference() -> None:
    """Explicit ``requires_fresh_metadata=False`` is honored even when the fn takes ``code_metadata``.

    Escape hatch for stages that take ``code_metadata`` but tolerate stale input.
    """

    def map_fn(stage_cfg: DictConfig, code_metadata: pl.DataFrame, df: pl.LazyFrame) -> pl.LazyFrame:
        return df

    with Stage.suppress_validation():
        stage = Stage(map_fn=map_fn, stage_name="opt_out", requires_fresh_metadata=False)
    assert stage.requires_fresh_metadata is False


def test_no_signature_no_inference() -> None:
    """A stage whose fns don't declare ``code_metadata`` defaults to False without manual flag."""

    def map_fn(df: pl.LazyFrame) -> pl.LazyFrame:
        return df

    with Stage.suppress_validation():
        stage = Stage(map_fn=map_fn, stage_name="no_metadata_use")
    assert stage.requires_fresh_metadata is False


def test_inference_sees_through_functools_wraps() -> None:
    """``inspect.signature`` follows ``__wrapped__`` chains so wrapped fns are inferred correctly."""
    import functools

    def inner(stage_cfg: DictConfig, code_metadata: pl.DataFrame, df: pl.LazyFrame) -> pl.LazyFrame:
        return df

    @functools.wraps(inner)
    def wrapped(*args, **kwargs):
        return inner(*args, **kwargs)

    with Stage.suppress_validation():
        stage = Stage(map_fn=wrapped, stage_name="wrapped_stage")
    assert stage.requires_fresh_metadata is True


def test_takes_param_returns_false_when_signature_raises() -> None:
    """``takes_param`` swallows ``ValueError``/``TypeError`` from ``inspect.signature``.

    Some opaque callables (a few C builtins, objects whose ``__signature__`` descriptor raises)
    don't expose an inspectable signature. The defensive try/except keeps the inference path
    from blowing up Stage construction in those cases — it's what lets us safely walk
    ``map_fn``/``reduce_fn``/``main_fn`` without filtering in advance.
    """
    from MEDS_transforms.compute_modes import takes_param

    class FailingSignature:
        def __call__(self, x):  # pragma: no cover - never invoked
            return x

        @property
        def __signature__(self):
            raise ValueError("no signature for this thing")

    assert takes_param(FailingSignature(), "code_metadata") is False


def test_inference_skips_callables_with_unintrospectable_signatures() -> None:
    """A stage whose compute fn raises on signature inspection still constructs successfully.

    Belt-and-suspenders for ``test_takes_param_returns_false_when_signature_raises`` at the
    Stage layer: the inference path must not bring down construction when signature inspection
    fails.
    """

    class WeirdMap:
        def __call__(self, df: pl.LazyFrame) -> pl.LazyFrame:  # pragma: no cover
            return df

        @property
        def __signature__(self):
            raise TypeError("signature unavailable")

    with Stage.suppress_validation():
        stage = Stage(map_fn=WeirdMap(), stage_name="weird")
    assert stage.requires_fresh_metadata is False


@pytest.mark.parametrize(
    "stage_name", ["filter_measurements", "reorder_measurements", "bin_numeric_values", "normalization"]
)
def test_known_metadata_consumers_are_inferred(stage_name: str) -> None:
    """Stages whose registered fns take ``code_metadata`` should all auto-infer the flag."""
    stages = get_all_registered_stages()
    stage = stages[stage_name].load()
    assert stage.requires_fresh_metadata is True, (
        f"{stage_name}'s fn declares `code_metadata`; expected auto-inference to set the flag"
    )
