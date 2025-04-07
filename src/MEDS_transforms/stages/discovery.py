from collections import defaultdict
from importlib.metadata import EntryPoint, entry_points

from .. import __package_name__


class StageNotFoundError(Exception):
    """Custom error for when a stage is not found."""


class StageDiscoveryError(Exception):
    """Custom error for stage discovery."""


def get_all_registered_stages() -> dict[str, EntryPoint]:
    """Get all available stages."""
    eps = entry_points(group=f"{__package_name__}.stages")

    stages = defaultdict(list)
    for name in eps.names:
        stages[name].append(eps[name])

    errors = []
    for name, entries in stages.items():
        if len(entries) > 1:
            errors.append(f"Multiple entry points registered for stage '{name}': {entries}")
    if errors:
        raise StageDiscoveryError(
            "Multiple entry points registered for the same stage:\n" + "\n".join(errors)
        )

    return {n: entries[0] for n, entries in stages.items()}
