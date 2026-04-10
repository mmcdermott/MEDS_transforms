"""Generate Markdown documentation for registered stages.

This helper builds a ``StageDoc`` for each registered stage so it can be
rendered in the documentation site.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

if TYPE_CHECKING:
    import polars as pl

from .discovery import get_all_registered_stages

_GOOGLE_DOCSTRING_SECTIONS = re.compile(
    r"^(Args|Returns|Raises|Yields|Examples|Attributes|Note|Notes|References|Todo|See Also"
    r"|Warnings?|\\.\\.)\s*:",
    re.MULTILINE,
)


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


def _read_readme(directory: Path | None) -> str | None:
    """Read a README.md from *directory*, stripping any leading ``#`` heading.

    The leading heading is removed because the README content is embedded under an existing section heading in
    the generated page.

    Examples:
        >>> _read_readme(None) is None
        True
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = (Path(d) / "README.md").write_text("# Title\\n\\nBody text.\\n")
        ...     _read_readme(Path(d))
        'Body text.'
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _read_readme(Path(d)) is None
        True
    """

    if directory is None:
        return None
    readme = directory / "README.md"
    if not readme.is_file():
        return None
    text = readme.read_text().strip()
    # Strip a leading ATX heading (e.g. "# Title\n\n...") to avoid duplicate/conflicting headings.
    text = re.sub(r"^#[^\n]*\n+", "", text).strip()
    return text or None


def _extract_description(docstring: str) -> str:
    """Return only the leading description from a Google-style docstring.

    Strips Args, Returns, Examples, etc. sections so the rendered page shows a clean prose description rather
    than raw docstring syntax.

    Examples:
        >>> _extract_description("Does something useful.\\n\\nArgs:\\n    x: an int")
        'Does something useful.'
        >>> _extract_description("No sections here.")
        'No sections here.'
    """

    text = textwrap.dedent(docstring).strip()
    match = _GOOGLE_DOCSTRING_SECTIONS.search(text)
    if match:
        text = text[: match.start()].strip()
    return text


def _df_to_markdown(df: pl.DataFrame, max_rows: int = 20) -> str:
    """Render a Polars DataFrame as a Markdown table.

    Truncates to *max_rows* to keep examples readable.

    Examples:
        >>> import polars as pl
        >>> print(_df_to_markdown(pl.DataFrame({"a": [1, None], "b": ["x", "y"]})))
        | **a** | **b** |
        | --- | --- |
        | 1 | x |
        | *null* | y |
        >>> print(_df_to_markdown(pl.DataFrame({"v": [1, 2, 3]}), max_rows=2))
        | **v** |
        | --- |
        | 1 |
        | 2 |
        <BLANKLINE>
        *... truncated to 2 rows*
    """

    if df.height > max_rows:
        df = df.head(max_rows)
        truncated = True
    else:
        truncated = False

    header = "| " + " | ".join(f"**{c}**" for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = []
    for row in df.iter_rows():
        cells = []
        for val in row:
            if val is None:
                cells.append("*null*")
            elif isinstance(val, list):
                cells.append(str(val))
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")

    lines = [header, sep, *rows]
    if truncated:
        lines.append("")
        lines.append(f"*... truncated to {max_rows} rows*")
    return "\n".join(lines)


def _format_dataset(dataset, label: str) -> list[str]:
    """Render a MEDSDataset as labelled Markdown sections with tables.

    Examples:
        >>> from types import SimpleNamespace
        >>> import polars as pl
        >>> ds = SimpleNamespace(_pl_shards={"0": pl.DataFrame({"x": [1]})})
        >>> lines = _format_dataset(ds, "Test")
        >>> print("\\n".join(lines))
        **Test:**
        <BLANKLINE>
        *Shard `0`:*
        <BLANKLINE>
        | **x** |
        | --- |
        | 1 |
        <BLANKLINE>
    """

    lines: list[str] = [f"**{label}:**", ""]

    shards = dataset._pl_shards
    if len(shards) == 1:
        shard_name, df = next(iter(shards.items()))
        lines.append(f"*Shard `{shard_name}`:*")
        lines.append("")
        lines.append(_df_to_markdown(df))
        lines.append("")
    else:
        for shard_name, df in shards.items():
            lines.append(
                f'<details markdown="1"><summary>Shard <code>{shard_name}</code> ({df.height} rows)</summary>'
            )
            lines.append("")
            lines.append(_df_to_markdown(df))
            lines.append("")
            lines.append("</details>")
            lines.append("")

    return lines


def _format_example(stage_name: str, example, example_dir: Path | None = None) -> str:
    """Format a single :class:`StageExample` as structured Markdown.

    Renders configuration as YAML, data as Markdown tables, and includes a sample CLI invocation.
    """

    lines: list[str] = []

    # Per-example description from README.md in the example directory
    example_readme = _read_readme(example_dir)
    if example_readme:
        lines.extend([example_readme, ""])

    # Stage configuration
    if example.stage_cfg:
        cfg_str = OmegaConf.to_yaml(OmegaConf.create(example.stage_cfg)).strip()
        lines.extend(["**Stage configuration:**", "", "```yaml", cfg_str, "```", ""])

    if example.do_use_config_yaml:
        lines.extend(["> This example uses the stage's `config.yaml` file.", ""])

    # Input data as tables
    if example.in_data is not None:
        if isinstance(example.in_data, Path):
            lines.extend(["**Input files:**", "", "```yaml", example.in_data.read_text().strip(), "```", ""])
        else:
            lines.extend(_format_dataset(example.in_data, "Input data"))

    # Expected output data as tables
    if example.want_data is not None:
        lines.extend(_format_dataset(example.want_data, "Expected output data"))

    # Expected output metadata as a table
    if example.want_metadata is not None:
        lines.extend(["**Expected output metadata:**", ""])
        lines.append(_df_to_markdown(example.want_metadata))
        lines.append("")

    # CLI usage hint
    cfg_parts = [
        f"stage_cfg.{k}={v}" for k, v in (example.stage_cfg or {}).items() if not isinstance(v, dict)
    ]
    cmd = f"MEDS_transform-stage <pipeline.yaml> {stage_name}"
    if cfg_parts:
        cmd += " " + " ".join(cfg_parts)
    cmd += " input_dir=<input> output_dir=<output>"
    lines.extend(["**Run this stage:**", "", "```bash", cmd, "```", ""])

    return "\n".join(lines)


def _build_stage_content(stage_name: str, stage) -> str:
    """Build Markdown content for a single stage."""

    lines = [f"# `{stage_name}`"]

    # Description from docstring (prose only, no Args/Returns/Examples)
    docstring = stage.stage_docstring
    if docstring:
        description = _extract_description(docstring)
        if description:
            lines.append("")
            lines.append(description)

    # Stage-level README (from the stage directory itself)
    stage_readme = _read_readme(stage.stage_dir)
    if stage_readme:
        lines.extend(["", stage_readme])

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

    # CLI usage
    lines.extend(
        [
            "",
            "## Usage",
            "",
            "```bash",
            f"MEDS_transform-stage <pipeline.yaml> {stage_name} input_dir=<input> output_dir=<output>",
            "```",
        ]
    )

    # Examples / test cases
    if stage.test_cases:
        lines.append("")
        lines.append("## Examples")

        # Examples-level README (overview of all examples)
        examples_readme = _read_readme(stage.examples_dir)
        if examples_readme:
            lines.extend(["", examples_readme])

        for scenario, example in stage.test_cases.items():
            scenario_name = scenario or "default"
            example_dir = stage.examples_dir / scenario if stage.examples_dir and scenario else None
            lines.extend(["", f"### {scenario_name}", ""])
            lines.append(_format_example(stage_name, example, example_dir))

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
