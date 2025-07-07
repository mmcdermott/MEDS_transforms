"""Generate Markdown documentation for registered stages.

This helper builds a ``StageDoc`` for each registered stage so it can be
rendered in the documentation site.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

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
        doc_path = Path("stages") / stage_name / "index.md"

        lines = [f"# `{stage_name}`"]

        doc = stage.stage_docstring
        if doc:
            lines.extend(["", inspect.cleandoc(doc)])

        if stage.stage_dir is not None:
            readme_path = stage.stage_dir / "README.md"
            if readme_path.is_file():
                rel = readme_path.relative_to(root).as_posix()
                lines.extend(["", f'--8<-- "{rel}"'])

        if stage.test_cases:
            lines.append("")
            lines.append("## Examples")
            for scenario, example in stage.test_cases.items():
                scenario_name = scenario or "default"
                lines.extend(["", f"### {scenario_name}", "```", str(example), "```"])

        edit_path = stage.stage_dir.relative_to(root) if stage.stage_dir else None

        docs.append(StageDoc(stage_name, doc_path, "\n".join(lines), edit_path))

    return docs


__all__ = ["StageDoc", "generate_stage_docs"]
