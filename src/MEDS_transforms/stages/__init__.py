from importlib.metadata import entry_points

from .. import __package_name__
from .base import MEDS_transforms_stage  # noqa: F401

# Here are all the stages that are registered in the entry points, imported here so they can be imported at a
# module level.
# isort: split
from .add_time_derived_measurements import main as add_time_derived_measurements  # noqa: F401
from .aggregate_code_metadata import main as aggregate_code_metadata  # noqa: F401
from .extract_values import main as extract_values  # noqa: F401
from .filter_measurements import main as filter_measurements  # noqa: F401
from .filter_subjects import main as filter_subjects  # noqa: F401
from .fit_vocabulary_indices import main as fit_vocabulary_indices  # noqa: F401
from .normalization import main as normalization  # noqa: F401
from .occlude_outliers import main as occlude_outliers  # noqa: F401
from .reorder_measurements import main as reorder_measurements  # noqa: F401
from .reshard_to_split import main as reshard_to_split  # noqa: F401


def get_all_stages():
    """Get all available stages."""
    eps = entry_points(group=f"{__package_name__}.stages")
    return {name: eps[name] for name in eps.names}
