"""Generate Markdown documentation for registered stages."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

from .stages import get_all_registered_stages


@dataclass
class StageDoc:
    """A generated documentation page for a stage."""

    stage_name: str
    path: Path
    content: str
    edit_path: Path | None = None


def generate_stage_docs(package: str, root: Path | None = None) -> list[StageDoc]:
    """Return Markdown documentation pages for all stages in ``package``.

    Parameters
    ----------
    package:
        Package prefix used to filter stages from the registry.
    root:
        Repository root used for calculating relative edit paths. Defaults to
        two directories above this file.
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

        edit_path = stage.source_file.relative_to(root) if stage.source_file else None

        docs.append(StageDoc(stage_name, doc_path, "\n".join(lines), edit_path))

    return docs


__all__ = ["StageDoc", "generate_stage_docs"]

