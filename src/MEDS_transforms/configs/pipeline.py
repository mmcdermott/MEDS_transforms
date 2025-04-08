"""This file defines the structured base classes for the various configs used in MEDS-Transforms."""

from __future__ import annotations

import dataclasses
import logging
import os
from importlib.resources import files
from pathlib import Path
from typing import ClassVar

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from ..stages.base import Stage
from ..stages.discovery import StageNotFoundError, get_all_registered_stages

logger = logging.getLogger(__name__)

NULL_STR = "__null__"
PKG_PFX = "pkg://"
YAML_EXTENSIONS = {".yaml", ".yml"}


def resolve_pkg_path(pkg_path: str) -> Path:
    """Parse a package path into a package name and a relative path.

    Args:
        pkg_path (str): The package path to parse.

    Returns:
        The file-path on disk to the package resource.

    Raises:
        ValueError: If the package path is not valid.

    Examples:
        >>> resolve_pkg_path("pkg://MEDS_transforms.configs.pipeline.py")
        PosixPath('...MEDS_transforms/configs/pipeline.py')

        Files need not exist to be returned:

        >>> resolve_pkg_path("pkg://MEDS_transforms.configs.pipeline.zip")
        PosixPath('...MEDS_transforms/configs/pipeline.zip')

        Note that this _returns something likely wrong_ for multi-suffix or no-suffix files!

        >>> resolve_pkg_path("pkg://MEDS_transforms.configs.pipeline") # likely should end in /pipeline
        PosixPath('...MEDS_transforms/configs.pipeline')
        >>> resolve_pkg_path("pkg://MEDS_transforms.configs.data.tar.gz") # likely should end in /data.tar.gz
        PosixPath('...MEDS_transforms/configs/data/tar.gz')

        Errors occur if the package is not importable:

        >>> resolve_pkg_path("pkg://non_existent_package.configs.pipeline.py")
        Traceback (most recent call last):
            ...
        ValueError: Package 'non_existent_package' not found. Please check the package name.
    """
    parts = pkg_path[len(PKG_PFX) :].split(".")
    pkg_name = parts[0]

    suffix = parts[-1]
    relative_path = Path(os.path.join(*parts[1:-1])).with_suffix(f".{suffix}")
    try:
        return files(pkg_name) / relative_path
    except ModuleNotFoundError as e:
        raise ValueError(f"Package '{pkg_name}' not found. Please check the package name.") from e


@dataclasses.dataclass
class StageConfig:
    """A configuration class for a stage in the MEDS-transforms pipeline.

    This class is used to define the structure of a stage configuration file. It is not intended to be
    instantiated directly; rather, it is used as a base class for specific stage configurations.

    Attributes:
        name: The name of the stage.
        config: The configuration for the stage.
    """

    name: str
    base_stage: str | None = None
    config: dict | DictConfig | None = None

    META_KEYS: ClassVar[set[str]] = {
        "_base_stage",
    }

    @classmethod
    def _is_meta_key(cls, key: str) -> bool:
        """Checks if a key is a meta-key for the configuration.

        Meta-keys are keys that re in the `cls.META_KEYS` set.

        Args:
            key: The key to check.

        Returns:
            True if the key is a meta-key, False otherwise.

        Examples:
            >>> StageConfig._is_meta_key("_base_stage")
            True
            >>> StageConfig._is_meta_key("base_stage")
            False
            >>> StageConfig._is_meta_key("_foo")
            False
        """
        return key in cls.META_KEYS

    @classmethod
    def _split_meta_keys(cls, config: dict | DictConfig) -> tuple[dict, dict]:
        """Splits the meta-keys from the rest of the configuration.

        Args:
            config: The configuration to split.

        Returns:
            A tuple containing the dictionary's meta-keys and the rest of the configuration.

        Examples:
            >>> StageConfig._split_meta_keys({"_base_stage": "base_stage", "param1": 1})
            ({'_base_stage': 'base_stage'}, {'param1': 1})
            >>> StageConfig._split_meta_keys({"param1": 1})
            ({}, {'param1': 1})
            >>> StageConfig._split_meta_keys({"_base_stage": "base_stage"})
            ({'_base_stage': 'base_stage'}, {})
        """
        meta_keys = {}
        non_meta_keys = {}

        for k, v in config.items():
            if StageConfig._is_meta_key(k):
                meta_keys[k] = v
            else:
                non_meta_keys[k] = v

        return meta_keys, non_meta_keys

    @classmethod
    def _dict_arg_error_str(cls, arg: dict | DictConfig) -> str | None:
        """Checks if the argument is a valid dictionary representation of a stage configuration.

        Args:
            arg: The argument to check.

        Returns:
            True if the argument is a valid dictionary representation, False otherwise. A valid dictionary has
                all keys as strings and has only one key that is not a meta-key, which points to a dictionary.

        Examples:
            >>> print(StageConfig._dict_arg_error_str({"stage": {"param1": 1}}))
            None
            >>> print(StageConfig._dict_arg_error_str({"stage": {"_base_stage": "base_stage", "param1": 1}}))
            None
            >>> print(StageConfig._dict_arg_error_str({"stage": {"foo": "foobar"}, "_base_stage": "base"}))
            None
            >>> print(StageConfig._dict_arg_error_str({123: {"param1": 1}}))
            All keys must be strings. Got key(s):
              - int: 123
            >>> print(StageConfig._dict_arg_error_str({"stage": {"param1": 1}, "param2": 2}))
            Expected a single non-meta key-value pair representing the stage name and configuration. Got 2
            non-meta keys: stage, param2.
            >>> print(StageConfig._dict_arg_error_str({"S": {"_base_stage": "foo"}, "_base_stage": "foo"}))
            You cannot specify meta keys in both the raw and stage configuration dictionaries. Got duplicate
            meta keys: _base_stage.
        """
        if any(not isinstance(k, str) for k in arg.keys()):
            non_str_keys = [k for k in arg.keys() if not isinstance(k, str)]
            err_strs = [f"  - {type(key).__name__}: {key}" for key in non_str_keys]
            err_str = "\n".join(err_strs)
            return f"All keys must be strings. Got key(s):\n{err_str}"

        meta_keys, non_meta_keys = cls._split_meta_keys(arg)
        if len(non_meta_keys) != 1:
            return (
                "Expected a single non-meta key-value pair representing the stage name and configuration. "
                f"Got {len(non_meta_keys)} non-meta keys: {', '.join(non_meta_keys.keys())}."
            )

        stage_config = list(non_meta_keys.values())[0]

        meta_key_duplicates = [k for k in meta_keys if k in stage_config]
        if meta_key_duplicates:
            return (
                "You cannot specify meta keys in both the raw and stage configuration dictionaries. "
                f"Got duplicate meta keys: {', '.join(meta_key_duplicates)}."
            )
        return None

    @classmethod
    def from_arg(cls, arg: str | dict | DictConfig) -> StageConfig:
        """Resolves a `StageConfig` object from a given argument (yaml file) representation.

        Args:
            arg: The argument to resolve. This can be a string representing the stage name, indicating no
                additional configuration arguments, a dictionary representing the stage configuration in
                resolved form, or a dictionary with a single key-value, non-meta-key pair representing the
                stage name and its configuration, with meta-keys (e.g., `_base_stage`) being pulled from the
                base dict or the stage configuration as appropriate.

        Returns:
            A `StageConfig` object representing the resolved stage configuration.

        Raises:
            ValueError: If the argument is not a valid stage configuration.
            TypeError: If the argument is not a string, dictionary, or `DictConfig`.

        Examples:
            >>> StageConfig.from_arg("stage_name")
            StageConfig(name='stage_name', base_stage=None, config=None)
            >>> StageConfig.from_arg({"stage_name": {"param1": 1}})
            StageConfig(name='stage_name', base_stage=None, config={'param1': 1})
            >>> StageConfig.from_arg({"stage_name": {"_base_stage": "base_stage", "param1": 1}})
            StageConfig(name='stage_name', base_stage='base_stage', config={'param1': 1})
            >>> StageConfig.from_arg({"foobar": {"param1": 1}, "_base_stage": "barfoo"})
            StageConfig(name='foobar', base_stage='barfoo', config={'param1': 1})

            Errors are thrown if the argument is the wrong type:

            >>> StageConfig.from_arg(3)
            Traceback (most recent call last):
                ...
            TypeError: Invalid argument type <class 'int'>. Expected str, dict, or DictConfig.

            Or if the argument is not a valid stage configuration. For example, if the argument is not a
            single (non-meta) key-value pair:

            >>> StageConfig.from_arg({"stage_name": {"param1": 1}, "param2": 2})
            Traceback (most recent call last):
                ...
            ValueError: Invalid stage config: Expected a single non-meta key-value pair representing the stage
            name and configuration. Got 2 non-meta keys: stage_name, param2.

            Or if the argument is not a valid stage name, or if meta-keys are used improperly:

            >>> StageConfig.from_arg({123: {"param1": 1}})
            Traceback (most recent call last):
                ...
            ValueError: Invalid stage config: All keys must be strings. Got key(s):
              - int: 123
            >>> StageConfig.from_arg({"stage_name": {"_base_stage": "foobar"}, "_base_stage": "barfoo"})
            Traceback (most recent call last):
                ...
            ValueError: Invalid stage config: You cannot specify meta keys in both the raw and stage
                configuration dictionaries. Got duplicate meta keys: _base_stage.
        """

        match arg:
            case str() as name:
                return cls(name=name)
            case dict() | DictConfig() as config:
                if cls._dict_arg_error_str(config):
                    raise ValueError(f"Invalid stage config: {cls._dict_arg_error_str(config)}")

                meta_keys, non_meta_keys = cls._split_meta_keys(config)
                stage_name = list(non_meta_keys.keys())[0]
                stage_config = non_meta_keys[stage_name]

                meta_keys = {k[1:]: v for k, v in meta_keys.items()}
                for k in cls.META_KEYS:
                    if k in stage_config:
                        meta_keys[k[1:]] = stage_config.pop(k)

                return cls(name=stage_name, config=stage_config, **meta_keys)
            case _:
                raise TypeError(f"Invalid argument type {type(arg)}. Expected str, dict, or DictConfig.")


@dataclasses.dataclass
class PipelineConfig:
    """A base configuration class for MEDS-transforms pipelines.

    This class is used to define the structure of a pipeline configuration file. It manually tracks the
    necessary parameters (`stages` and `stage_configs`) and stores all other parameters in an
    `additional_params` `DictConfig`.

    It's primary use is to abstract functionality for resolving stage specific parameters to form the stage
    configuration object for a stage and pipeline realization from arguments.

    Attributes:
        stages: A list of Stage configuration objects that are part of the pipeline.
        additional_params: A dictionary of additional parameters that are not stage-specific.

    Examples:

        >>> PipelineConfig()
        PipelineConfig(stages=None, stage_configs={}, additional_params=None)
        >>> stages = ["stage1", "stage2"]
        >>> stage_configs = {"stage1": {"param1": 1}}
        >>> additional_params = {"param2": 2}
        >>> PipelineConfig(stages=stages, stage_configs=stage_configs, additional_params=additional_params)
        PipelineConfig(stages=['stage1', 'stage2'],
                       stage_configs={'stage1': {'param1': 1}},
                       additional_params={'param2': 2})

    While you can create a PipelineConfig object directly, it is more often created via the `from_arg` method,
    which loads a pipeline configuration from a YAML file. This YAML file is not structured precisely as the
    object is, as the additional parameters are flattened in the file representation. See the `from_arg`
    method for more details.

        >>> file_representation = {"stages": stages, "stage_configs": stage_configs, **additional_params}
        >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as pipeline_yaml:
        ...     OmegaConf.save(file_representation, pipeline_yaml.name)
        ...     cfg = PipelineConfig.from_arg(pipeline_yaml.name)
        >>> cfg
        PipelineConfig(stages=['stage1', 'stage2'],
                       stage_configs={'stage1': {'param1': 1}},
                       additional_params={'param2': 2})

    Pipeline configurations can be resolved to a `DictConfig` representation suitable for Hydra registration
    via the `structured_config` property. This `DictConfig` representation has `additional_params` flattened
    into the output, much like the file representation. This representation will omit missing or empty keys
    automatically as well:

        >>> cfg.structured_config
        {'stages': ['stage1', 'stage2'], 'stage_configs': {'stage1': {'param1': 1}}, 'param2': 2}
        >>> PipelineConfig().structured_config
        {}

    We can also use the `PipelineConfig` for a pipeline to prepare to run a stage in that pipeline, by
    resolving the relevant input and output file paths given the prior pipeline stages, resolving stages to
    their associated base (runnable) stage names, and registering the necessary structured configuration nodes
    within the Hydra `ConfigStore` for runtime use. These behaviors all take place via the `register_for`
    method. This method both (a) identifies and loads the runnable stage specified by the argument (and
    returns it), and (b) constructs the Hydra nodes and adds them to the `ConfigStore`. Let's first inspect
    the returned stage:

        >>> pipeline_cfg = PipelineConfig(stages=["filter_subjects"])
        >>> runnable_stage = pipeline_cfg.register_for("filter_subjects")
        >>> print(runnable_stage)
        Stage filter_subjects:
          Type: map
          is_metadata: False
          Docstring:
            | Returns a function that filters subjects by the number of measurements and events they have.
            ...
          Default config:
            | filter_subjects:
            |   min_events_per_subject: null
            |   min_measurements_per_subject: null
          Map function: filter_subjects
          Reduce function: None
          Main function: None
          Mimic function: filter_subjects

    And to see the registered configuration node, we can use the `ConfigStore` to get the node:

        >>> cs = ConfigStore.instance()
        >>> cs.repo["_pipeline.yaml"].node
        {'stages': ['filter_subjects'],
         'stage_cfg': {'min_events_per_subject': None, 'min_measurements_per_subject': None,
                       'data_input_dir': '${input_dir}/data', 'metadata_input_dir': '${input_dir}/metadata',
                       'reducer_output_dir': None, 'train_only': False, 'output_dir': '${cohort_dir}/data'}}

    The simplest pipeline, a "null" pipeline, with `PipelineConfig()` or
    `PipelineConfig.from_arg("__null__")`, will not have any stages or stage configurations. When you call
    `register_for` on such a pipeline, it will automatically add the stage being registered to the pipeline;
    this reflects that "null" pipelines are created automatically when running singleton stages.

        >>> pipeline_cfg = PipelineConfig()
        >>> runnable_stage = pipeline_cfg.register_for("occlude_outliers")
        >>> pipeline_cfg.stages
        ['occlude_outliers']

    If the pipeline indicates that a stage actually is an instance of a base stage, that will be reflected in
    the resolution. Similarly, arguments specified in the stage configs will appear in the resolved
    `stage_cfg` node. In addition, `register_for` automatically takes into account other stages in the
    pipeline:

        >>> pipeline_cfg = PipelineConfig(
        ...     stages=["count_codes", "filter_measurements", "fit_outlier_detection", "occlude_outliers"],
        ...     stage_configs={
        ...         "count_codes": {
        ...             "_base_stage": "aggregate_code_metadata",
        ...             "aggregations": ["code/n_subjects", "code/n_occurrences"],
        ...         },
        ...         "filter_measurements": {"min_subjects_per_code": 4, "min_occurrences_per_code": 10},
        ...         "fit_outlier_detection": {
        ...             "_base_stage": "aggregate_code_metadata",
        ...             "aggregations": ["values/n_occurrences", "values/sum", "values/sum_sqd"],
        ...         },
        ...     },
        ... )
        >>> runnable_stage = pipeline_cfg.register_for("count_codes")
        >>> print(runnable_stage.stage_name)
        aggregate_code_metadata
        >>> cs.repo["_pipeline.yaml"].node
        {'stages': ['count_codes', 'filter_measurements', 'fit_outlier_detection', 'occlude_outliers'],
         'stage_configs': {'count_codes': {'_base_stage': 'aggregate_code_metadata',
                                           'aggregations': ['code/n_subjects', 'code/n_occurrences']},
                           'filter_measurements': {'min_subjects_per_code': 4,
                                                   'min_occurrences_per_code': 10},
                           'fit_outlier_detection': {'_base_stage': 'aggregate_code_metadata',
                                                     'aggregations': ['values/n_occurrences',
                                                                      'values/sum',
                                                                      'values/sum_sqd']}},
         'stage_cfg': {'data_input_dir': '${input_dir}/data',
                       'metadata_input_dir': '${input_dir}/metadata',
                       'output_dir': '${cohort_dir}/count_codes',
                       'train_only': True,
                       'reducer_output_dir': '${cohort_dir}/count_codes',
                       'aggregations': ['code/n_subjects', 'code/n_occurrences']}}
        >>> runnable_stage = pipeline_cfg.register_for("filter_measurements")
        >>> print(runnable_stage.stage_name)
        filter_measurements
        >>> cs.repo["_pipeline.yaml"].node
        {'stages': ['count_codes', 'filter_measurements', 'fit_outlier_detection', 'occlude_outliers'],
         'stage_configs': {'count_codes': {'_base_stage': 'aggregate_code_metadata',
                                           'aggregations': ['code/n_subjects', 'code/n_occurrences']},
                           'filter_measurements': {'min_subjects_per_code': 4,
                                                   'min_occurrences_per_code': 10},
                           'fit_outlier_detection': {'_base_stage': 'aggregate_code_metadata',
                                                     'aggregations': ['values/n_occurrences',
                                                                      'values/sum',
                                                                      'values/sum_sqd']}},
         'stage_cfg': {'min_subjects_per_code': 4,
                       'min_occurrences_per_code': 10,
                       'data_input_dir': '${input_dir}/data',
                       'metadata_input_dir': '${cohort_dir}/count_codes',
                       'reducer_output_dir': None,
                       'train_only': False,
                       'output_dir': '${cohort_dir}/filter_measurements'}}
    """

    stages: list[str] | None = None
    stage_configs: dict[str, dict] = dataclasses.field(default_factory=dict)
    additional_params: DictConfig | None = None

    @property
    def structured_config(self) -> DictConfig:
        """Return a `DictConfig` representation of the pipeline configuration suitable for Hydra registration.

        This `DictConfig` representation has `additional_params` flattened into the main configuration. This
        means that were you to store the `DictConfig` in a file, then re-load it as a `PipelineConfig` via
        `PipelineConfig.from_arg`, it would match the original `PipelineConfig` object.
        """

        merged = {}
        if self.stages is not None:
            merged["stages"] = self.stages
        if self.stage_configs:
            merged["stage_configs"] = self.stage_configs
        if self.additional_params is not None:
            merged.update(self.additional_params)
        return OmegaConf.create(merged)

    @classmethod
    def from_arg(cls, arg: str | Path) -> PipelineConfig:
        """Construct a pipeline configuration object from a specified pipeline YAML file.

        Args:
            arg: The path to the pipeline YAML file on disk or in the
                'pkg://<pkg_name>.<relative_path>' format. It can also be the sentinel `__null__` string,
                which will return an empty PipelineConfig object.

        Returns:
            A PipelineConfig object corresponding to the specified pipeline YAML file. Note that this object
            will not exactly match the passed file; rather, the stages and stage configurations will be pulled
            out separately and stored as direct attributes, and the rest of the parameters will be stored in
            the `additional_params` attribute.

        Raises:
            TypeError: If the pipeline YAML path is not a string or Path object.
            ValueError: If the pipeline YAML path does not have a valid file extension.
            FileNotFoundError: If the pipeline YAML file does not exist.

        Examples:
            >>> PipelineConfig.from_arg("__null__")
            PipelineConfig(stages=None, stage_configs={}, additional_params=None)
            >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as pipeline_yaml:
            ...     OmegaConf.save({"stages": ["stage1", "stage2"], "foobar": 3}, pipeline_yaml.name)
            ...     PipelineConfig.from_arg(pipeline_yaml.name)
            PipelineConfig(stages=['stage1', 'stage2'], stage_configs={}, additional_params={'foobar': 3})

            To show the package path resolution, we can use the `pkg://` format, but for this test, we need to
            mock the package structure:

            >>> from unittest.mock import patch
            >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as pipeline_yaml:
            ...     pipeline_fp = Path(pipeline_yaml.name)
            ...     OmegaConf.save({"stages": ["stage1", "stage2"], "qux": "a"}, pipeline_fp)
            ...     with patch("MEDS_transforms.configs.pipeline.resolve_pkg_path", return_value=pipeline_fp):
            ...         PipelineConfig.from_arg("pkg://fake_pkg.pipeline.yaml")
            PipelineConfig(stages=['stage1', 'stage2'], stage_configs={}, additional_params={'qux': 'a'})

            It will throw errors if the file has the wrong extension, does not exist, or is a directory, and
            if the passed parameter is neither a string nor a Path object:

            >>> PipelineConfig.from_arg(3)
            Traceback (most recent call last):
                ...
            TypeError: Invalid pipeline YAML path type <class 'int'>. Expected str or Path.
            >>> with tempfile.NamedTemporaryFile(suffix=".json") as pipeline_yaml:
            ...     OmegaConf.save({"stages": ["stage1", "stage2"], "foobar": 3}, pipeline_yaml.name)
            ...     PipelineConfig.from_arg(pipeline_yaml.name)
            Traceback (most recent call last):
                ...
            ValueError: Invalid pipeline YAML path '/tmp/tmp....json'. Expected a file with one of the
                following extensions: ['.yaml', '.yml']
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     pipeline_yaml = Path(tmpdir) / "pipeline.yaml"
            ...     PipelineConfig.from_arg(pipeline_yaml)
            Traceback (most recent call last):
                ...
            FileNotFoundError: Pipeline YAML file '/tmp/tmp.../pipeline.yaml' does not exist.
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     PipelineConfig.from_arg(tmpdir)
            Traceback (most recent call last):
                ...
            FileNotFoundError: Pipeline YAML file '/tmp/tmp...' is a directory, not a file!
        """

        match arg:
            case str() if arg == NULL_STR:
                return cls()
            case str() as pkg_path if arg.startswith(PKG_PFX):
                pipeline_fp = resolve_pkg_path(pkg_path)
            case str() | Path() as path:
                pipeline_fp = Path(path)
            case _:
                raise TypeError(f"Invalid pipeline YAML path type {type(arg)}. Expected str or Path.")

        if pipeline_fp.exists() and pipeline_fp.is_dir():
            raise FileNotFoundError(f"Pipeline YAML file '{pipeline_fp}' is a directory, not a file!")
        elif pipeline_fp.suffix not in YAML_EXTENSIONS:
            raise ValueError(
                f"Invalid pipeline YAML path '{pipeline_fp}'. "
                f"Expected a file with one of the following extensions: {sorted(YAML_EXTENSIONS)}"
            )
        elif not pipeline_fp.exists():
            raise FileNotFoundError(f"Pipeline YAML file '{pipeline_fp}' does not exist.")

        as_dict_config = OmegaConf.load(pipeline_fp)

        stages = as_dict_config.pop("stages", None)
        stage_configs = as_dict_config.pop("stage_configs", {})
        return cls(stages=stages, stage_configs=stage_configs, additional_params=as_dict_config)

    def _resolve_stages(self, all_stages: dict[str, Stage]) -> dict[str, DictConfig]:
        stage_objects = []
        last_data_stage = None
        last_metadata_stage = None
        for s in self.stages:
            if s in self.stage_configs:
                config = self.stage_configs[s]
            else:
                config = {}

            load_name = config.get("_base_stage", s)
            if load_name not in all_stages:
                raise ValueError(
                    f"Stage '{s}' not found in the registered stages. Please check the pipeline config."
                )

            stage = all_stages[load_name]

            if stage.is_metadata:
                last_metadata_stage = s
            else:
                last_data_stage = s
            stage_objects.append((s, stage, config))

        prior_data_stage = None
        prior_metadata_stage = None

        resolved_stage_configs = {}

        input_dir = Path("${input_dir}")
        cohort_dir = Path("${cohort_dir}")

        for s, stage, config_overwrites in stage_objects:
            if stage.default_config:
                config = {**stage.default_config[stage.stage_name]}
            else:
                config = {}

            if prior_data_stage is None:
                config["data_input_dir"] = str(input_dir / "data")
            else:
                config["data_input_dir"] = prior_data_stage["output_dir"]

            if prior_metadata_stage is None:
                config["metadata_input_dir"] = str(input_dir / "metadata")
            else:
                config["metadata_input_dir"] = prior_metadata_stage["reducer_output_dir"]

            if stage.is_metadata:
                config["output_dir"] = str(cohort_dir / s)
                config["train_only"] = True
                if s == last_metadata_stage:
                    config["reducer_output_dir"] = str(cohort_dir / "metadata")
                else:
                    config["reducer_output_dir"] = str(cohort_dir / s)
            else:
                config["reducer_output_dir"] = None
                config["train_only"] = False
                if s == last_data_stage:
                    config["output_dir"] = str(cohort_dir / "data")
                else:
                    config["output_dir"] = str(cohort_dir / s)

            config.update({k: v for k, v in config_overwrites.items() if k != "_base_stage"})
            resolved_stage_configs[s] = OmegaConf.create(config)

            if stage.is_metadata:
                prior_metadata_stage = config
            else:
                prior_data_stage = config

        return resolved_stage_configs

    def _resolve_stage_name(self, stage_name: str) -> str:
        """Return the registered stage corresponding to the specified stage for the given pipeline.

        Args:
            stage_name: The name of the stage to resolve.

        Returns: Either (a) the `_base_stage` specified in the pipeline config's `stage_configs` for this
            stage, if specified, or (b) the stage name itself, otherwise. In both cases, the stage name
            returned is validated to be a registered stage.

        Raises:
            StageNotFoundError: If the stage name is not a registered stage.

        Examples:

            >>> pipeline_cfg = PipelineConfig()
            >>> pipeline_cfg._resolve_stage_name("occlude_outliers")
            'occlude_outliers'
            >>> pipeline_cfg._resolve_stage_name("foobar_non_existent_stage")
            Traceback (most recent call last):
                ...
            MEDS_transforms.stages.discovery.StageNotFoundError: Stage 'foobar_non_existent_stage' not
                registered! Registered stages: ...

            Stage names can be resolved to a base stage name, if specified in the stage configs:

            >>> pipeline_cfg.stage_configs = {"count_codes": {"_base_stage": "aggregate_code_metadata"}}
            >>> pipeline_cfg._resolve_stage_name("count_codes")
            'aggregate_code_metadata'
        """

        resolved_stage_name = self.stage_configs.get(stage_name, {}).get("_base_stage", stage_name)

        all_stages = get_all_registered_stages()
        if resolved_stage_name not in all_stages:
            raise StageNotFoundError(
                f"Stage '{resolved_stage_name}' not registered! Registered stages: "
                f"{', '.join(sorted(all_stages.keys()))}"
            )

        return resolved_stage_name

    @Stage.suppress_validation()
    def register_for(self, stage_name: str) -> Stage:
        if not self.stages:
            logger.warning("No stages specified in the pipeline config. Adding the target stage alone.")
            self.stages = [stage_name]

        registered_stages = get_all_registered_stages()
        loaded_stages = {}
        for raw_stage in self.stages:
            s = self._resolve_stage_name(raw_stage)
            if s not in loaded_stages:
                loaded_stages[s] = registered_stages[s].load()

        resolved_stage_name = self._resolve_stage_name(stage_name)
        stage = loaded_stages[resolved_stage_name]

        if stage.stage_name != resolved_stage_name:
            raise ValueError(
                f"Registered stage name '{stage.stage_name}' does not match the provided name "
                f"'{resolved_stage_name}'!"
            )

        all_stage_configs = self._resolve_stages(loaded_stages)

        pipeline_node = self.structured_config
        pipeline_node["stage_cfg"] = all_stage_configs[stage_name]

        cs = ConfigStore.instance()
        cs.store(name="_pipeline", node=pipeline_node)

        return stage
