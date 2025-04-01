from importlib.metadata import EntryPoint, entry_points

from .. import __package_name__
from .base import Stage  # noqa: F401
from .examples import StageExample, get_nested_test_cases  # noqa: F401

# Here are all the stages that are registered in the entry points, imported here so they can be imported at a
# module level.
# isort: split
from .add_time_derived_measurements import stage as add_time_derived_measurements  # noqa: F401
from .aggregate_code_metadata import stage as aggregate_code_metadata  # noqa: F401
from .extract_values import stage as extract_values  # noqa: F401
from .filter_measurements import stage as filter_measurements  # noqa: F401
from .filter_subjects import stage as filter_subjects  # noqa: F401
from .fit_vocabulary_indices import stage as fit_vocabulary_indices  # noqa: F401
from .normalization import stage as normalization  # noqa: F401
from .occlude_outliers import stage as occlude_outliers  # noqa: F401
from .reorder_measurements import stage as reorder_measurements  # noqa: F401
from .reshard_to_split import stage as reshard_to_split  # noqa: F401


def get_all_stages() -> dict[str, EntryPoint]:
    """Get all available stages."""
    eps = entry_points(group=f"{__package_name__}.stages")
    return {name: eps[name] for name in eps.names}
