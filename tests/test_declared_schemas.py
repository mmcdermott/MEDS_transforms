"""Tests for the ``Stage.declared_schemas`` property (refs #324)."""

from __future__ import annotations

import pytest
from meds import CodeMetadataSchema, DataSchema

from MEDS_transforms.stages.base import Stage
from MEDS_transforms.stages.discovery import get_all_registered_stages


def _compute(cfg):
    """Minimal map_fn so Stage(...) construction succeeds."""
    return 0


@pytest.mark.parametrize(
    ("input_s", "output_s", "metadata_input_s", "metadata_output_s"),
    [
        (None, None, None, None),
        (DataSchema, None, None, None),
        (None, DataSchema, None, None),
        (None, None, CodeMetadataSchema, None),
        (None, None, None, CodeMetadataSchema),
        (DataSchema, DataSchema, None, None),
        (DataSchema, None, None, CodeMetadataSchema),
        (DataSchema, DataSchema, CodeMetadataSchema, CodeMetadataSchema),
    ],
)
def test_declared_schemas_returns_all_four_roles(
    input_s, output_s, metadata_input_s, metadata_output_s
) -> None:
    """``declared_schemas`` always returns a 4-key dict; absent roles are ``None``.

    Parameterized truth-table covers each role individually populated, the typical MAP shape (data in/out),
    the typical MAPREDUCE-to-metadata shape (data in, metadata out), and the four-roles-populated case.
    Confirms the dict key set is stable regardless of which roles were declared at construction.
    """
    with Stage.suppress_validation():
        stage = Stage(
            map_fn=_compute,
            stage_name="declared_schemas_truth_table",
            input_schema=input_s,
            output_schema=output_s,
            metadata_input_schema=metadata_input_s,
            metadata_output_schema=metadata_output_s,
        )

    schemas = stage.declared_schemas
    assert sorted(schemas.keys()) == ["input", "metadata_input", "metadata_output", "output"]
    assert schemas["input"] is input_s
    assert schemas["output"] is output_s
    assert schemas["metadata_input"] is metadata_input_s
    assert schemas["metadata_output"] is metadata_output_s


def test_aggregate_code_metadata_declares_data_in_metadata_out() -> None:
    """The PR migrates ``aggregate_code_metadata`` to declare ``input_schema`` + ``metadata_output_schema``.

    Lock the migration in: a regression that drops or renames the declarations would silently
    bypass any schema-validation tooling that consumes ``declared_schemas``. The asymmetric
    shape (no ``output_schema``, no ``metadata_input_schema``) is the canonical
    "MAPREDUCE that reduces data into metadata" pattern documented on the property.
    """
    stages = get_all_registered_stages()
    stage = stages["aggregate_code_metadata"].load()

    schemas = stage.declared_schemas
    assert schemas["input"] is DataSchema
    assert schemas["metadata_output"] is CodeMetadataSchema
    # Asymmetric — the stage doesn't write data shards or read existing code metadata.
    assert schemas["output"] is None
    assert schemas["metadata_input"] is None
