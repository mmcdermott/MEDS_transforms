"""Tests for the fresh-metadata dependency contract on Stage/Pipeline (issues #117/116/118/200)."""

from __future__ import annotations

import logging

from MEDS_transforms.configs import PipelineConfig


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
