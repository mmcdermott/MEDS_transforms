"""This file will use the automated stage example definitions to test each defined stage in this package."""

from pathlib import Path

from MEDS_transforms.stages import StageExample
from MEDS_transforms.stages.docgen import generate_stage_docs


def test_stage_scenario(stage_example: StageExample):
    stage_example.test()


def test_docgen_renders_custom_example_class():
    """``generate_stage_docs`` must walk stages whose ``StageExample`` subclass owns rendering.

    ``export_code_summary`` registers with a ``JsonOutputStageExample`` subclass whose
    ``want_data`` is a ``Path`` to a ``yaml_to_disk`` spec. The subclass overrides
    ``render_content`` to materialize the spec and render each JSON file as a Markdown table.
    This validates the hook end-to-end: if the base class's hard-coded ``_pl_shards`` assumption
    were still in place, this call would raise ``AttributeError``.
    """

    pkg_root = Path(__file__).resolve().parents[1]
    docs = {doc.stage_name: doc for doc in generate_stage_docs("simple_example_pkg", root=pkg_root)}

    export = docs["export_code_summary"]
    assert "**Expected `code_summary.json`:**" in export.content
    assert "| ADMISSION//CARDIAC | 2 |" in export.content
    # The base-class ``_pl_shards``-based renderer would have produced this heading.
    assert "**Expected output data:**" not in export.content
