"""Shared CLI help text for the pipeline runner and stage dispatcher.

Kept as a small, dependency-light module so ``runner.main()`` can format its
``argparse`` epilog without pulling Hydra through ``__main__``.
"""

from __future__ import annotations

#: Pipeline-level configuration keys consumed by the runner rather than individual stages. Surfaced
#: in CLI help so users know what is accepted via ``--overrides`` or at the top level of a pipeline
#: YAML.
PIPELINE_CONFIG_KEYS: dict[str, str] = {
    "input_dir": "Root MEDS directory to read from (contains `data/` and `metadata/`).",
    "output_dir": "Root directory to write stage outputs into.",
    "stages": "List of stage configs defining the pipeline (see each stage's docs).",
    "description": "Optional free-text pipeline description.",
    "parallelize.n_workers": "Number of parallel workers to launch per stage (default 1).",
    "parallelize.launcher": "Hydra launcher to use (e.g. 'joblib', 'slurm').",
    "parallelize.launcher_params": "Launcher-specific keyword arguments forwarded to Hydra.",
}


def pipeline_keys_help_block() -> str:
    """Return a stable help-text block listing pipeline-level config keys.

    Examples:
        >>> block = pipeline_keys_help_block()
        >>> block.startswith("Pipeline-level configuration keys")
        True
        >>> "parallelize.n_workers" in block
        True
    """
    lines = ["Pipeline-level configuration keys (settable via pipeline YAML or `--overrides`):"]
    width = max(len(k) for k in PIPELINE_CONFIG_KEYS)
    for key, desc in PIPELINE_CONFIG_KEYS.items():
        lines.append(f"  {key.ljust(width)}  {desc}")
    return "\n".join(lines)
