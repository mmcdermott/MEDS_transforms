from importlib.metadata import version

__package_name__ = __package__
__version__ = version(__package_name__)


INFERRED_STAGE_KEYS = {
    "is_metadata",
    "train_only",
    "data_input_dir",
    "metadata_input_dir",
    "output_dir",
    "reducer_output_dir",
    "_script",
}
