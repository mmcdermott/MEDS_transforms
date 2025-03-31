import logging
from importlib.metadata import EntryPoint, entry_points
from importlib.resources import files
from typing import TypedDict

from omegaconf import DictConfig, OmegaConf

from .. import __package_name__
from .base import MEDS_transforms_stage  # noqa: F401
from .examples import StageExample, get_nested_test_cases  # noqa: F401

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

logger = logging.getLogger(__name__)


class StageInfo(TypedDict, total=True):
    entry_point: EntryPoint
    package_name: str
    package_version: str
    default_config: DictConfig


def get_all_registered_stages() -> dict[str, StageInfo]:
    """Get all available stages."""

    entry_point = f"{__package_name__}.stages"

    logger.debug(f'Scanning for registered stages under the "{entry_point}" entry point')
    eps = entry_points(group=entry_point)

    out = {}
    for ep in eps:
        if ep.name in out:
            raise ValueError(f"Duplicate entry point found: {ep.name}")

        logger.debug(f"Found stage {ep.name}: ")

        ep_package = ep.dist.metadata["Name"]
        ep_package_version = ep.dist.version

        logger.debug(f"  - package: {ep_package}")
        logger.debug(f"  - package version: {ep_package_version}")

        # Get the default stage configuration file, if present:
        config_filepath = files(ep_package).joinpath(f"configs/stages/{ep.name}.yaml")

        logger.debug(f"  - config file: {config_filepath}")

        if config_filepath.exists():
            logger.debug("    defaults found, loading...")
            default_config = OmegaConf.load(config_filepath)
            logger.debug("    defaults loaded!")
        else:
            logger.debug("  - config file not found, using empty config")
            default_config = DictConfig({})

        out[ep.name] = {
            "entry_point": ep,
            "package_name": ep_package,
            "package_version": ep_package_version,
            "default_config": default_config,
        }

    return out
