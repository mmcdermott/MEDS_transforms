from importlib.metadata import PackageNotFoundError, version

__package_name__ = __package__
try:
    __version__ = version(__package_name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"


INFERRED_STAGE_KEYS = {
    "is_metadata",
    "train_only",
    "data_input_dir",
    "metadata_input_dir",
    "output_dir",
    "reducer_output_dir",
    "_script",
}
