"""Generate Markdown documentation for registered stages.

This helper builds a ``StageDoc`` for each registered stage so it can be
rendered in the documentation site.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from .discovery import get_all_registered_stages


@dataclass
class StageDoc:
    """Markdown documentation page for a stage.

    Attributes:
        stage_name: Name of the stage.
        path: Relative path of the Markdown file.
        content: Markdown contents to write.
        edit_path: Optional repository path for MkDocs edit links.
    """

    stage_name: str
    path: Path
    content: str
    edit_path: Path | None = None


def _format_example(example) -> str:
    """Format a single :class:`StageExample` as Markdown.

    Renders each part of the example (configuration, input data, expected output) as its own labelled, fenced
    block so the documentation is easy to scan.
    """

    lines: list[str] = []

    # Stage configuration
    if example.stage_cfg:
        cfg_str = OmegaConf.to_yaml(OmegaConf.create(example.stage_cfg)).strip()
        lines.extend(["**Stage configuration:**", "", "```yaml", cfg_str, "```", ""])

    if example.do_use_config_yaml:
        lines.extend(["> This example uses the stage's `config.yaml` file.", ""])

    # Input data
    if example.in_data is not None:
        lines.extend(["**Input data:**", "", "```", str(example.in_data), "```", ""])

    # Expected output data
    if example.want_data is not None:
        lines.extend(["**Expected output data:**", "", "```", str(example.want_data), "```", ""])

    # Expected output metadata
    if example.want_metadata is not None:
        lines.extend(["**Expected output metadata:**", "", "```", str(example.want_metadata), "```", ""])

    return "\n".join(lines)


def _build_stage_content(stage_name: str, stage) -> str:
    """Build Markdown content for a single stage.

    Args:
        stage_name: The registered name of the stage.
        stage: The loaded :class:`Stage` object.

    Returns:
        A Markdown string documenting the stage.
    """

    lines = [f"# `{stage_name}`"]

    # Description from docstring
    docstring = stage.stage_docstring
    if docstring:
        lines.append("")
        lines.append(textwrap.dedent(docstring).strip())

    # Stage metadata table
    lines.append("")
    lines.append("## Details")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| **Type** | `{stage.stage_type}` |")
    if stage.is_metadata is not None:
        lines.append(f"| **Metadata stage** | `{stage.is_metadata}` |")

    # Default configuration
    if stage.default_config:
        config_yaml = OmegaConf.to_yaml(stage.default_config).strip()
        lines.extend(["", "## Default Configuration", "", "```yaml", config_yaml, "```"])

    # Output schema updates
    if stage.output_schema_updates:
        lines.extend(["", "## Output Schema Updates", ""])
        lines.append("| Column | Type |")
        lines.append("| --- | --- |")
        for col, dtype in stage.output_schema_updates.items():
            lines.append(f"| `{col}` | `{dtype}` |")

    # Examples / test cases
    if stage.test_cases:
        lines.append("")
        lines.append("## Examples")
        for scenario, example in stage.test_cases.items():
            scenario_name = scenario or "default"
            lines.extend(["", f"### {scenario_name}", ""])
            lines.append(_format_example(example))

    return "\n".join(lines)


def generate_stage_docs(package: str, root: Path | None = None) -> list[StageDoc]:
    """Build documentation pages for all registered stages.

    Args:
        package: Package prefix used to filter stages from the registry.
        root: Repository root for computing edit links. Defaults to two
            directories above this file.

    Returns:
        A list of :class:`StageDoc` objects describing each page.

    Examples:
        >>> import textwrap
        >>> docs = generate_stage_docs("MEDS_transforms")
        >>> print(docs[0].stage_name)
        add_time_derived_measurements
        >>> print(docs[0].path)
        add_time_derived_measurements
        >>> print(textwrap.shorten(docs[0].content, width=50))
        # `add_time_derived_measurements` Adds all [...]

    It can also attempt to make docs for other packages; in this case, it will return None as there are no
    stages in this fake package:

        >>> generate_stage_docs("fake_package")
        []
    """

    root = Path(root) if root else Path(__file__).resolve().parents[1]

    docs: list[StageDoc] = []

    for stage_name, entry_point in sorted(get_all_registered_stages().items()):
        if not entry_point.module.startswith(package):
            continue

        stage = entry_point.load()
        content = _build_stage_content(stage_name, stage)
        edit_path = stage.stage_dir.relative_to(root) if stage.stage_dir else None

        docs.append(StageDoc(stage_name, stage_name, content, edit_path))

    return docs


__all__ = ["StageDoc", "generate_stage_docs"]
