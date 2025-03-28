import logging
from importlib.metadata import EntryPoint, entry_points
from importlib.resources import files
from typing import TypedDict

from omegaconf import DictConfig, OmegaConf

from .. import __package_name__

logger = logging.getLogger(__name__)


class StageInfo(TypedDict, total=True):
    entry_point: EntryPoint
    package: str
    module: str
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
        ep_module = ep.module
        ep_package_version = ep.dist.version

        logger.debug(f"  - package: {ep_package}")
        logger.debug(f"  - module: {ep_module}")
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
            "package": ep_package,
            "module": ep_module,
            "package_version": ep_package_version,
            "default_config": default_config,
        }

    return out
