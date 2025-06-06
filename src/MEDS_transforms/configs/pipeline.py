"""This file defines an object-oriented backing for configuring pipelines in MEDS-Transforms."""

from __future__ import annotations

import copy
import dataclasses
import logging
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, ListConfig, OmegaConf

from ..stages.base import Stage
from ..stages.discovery import StageNotFoundError, get_all_registered_stages
from ..utils import PKG_PFX, resolve_pkg_path
from .stage import UNPARSED_STAGE_T, StageConfig

logger = logging.getLogger(__name__)

NULL_STR = "__null__"
YAML_EXTENSIONS = {".yaml", ".yml"}


@dataclasses.dataclass
class PipelineConfig:
    """A base configuration class for MEDS-transforms pipelines.

    This class is used to define the structure of a pipeline configuration file. It manually tracks the
    necessary parameters (`stages` and `stage_configs`) and stores all other parameters in an
    `additional_params` `DictConfig`.

    It's primary use is to abstract functionality for resolving stage specific parameters to form the stage
    configuration object for a stage and pipeline realization from arguments.

    Attributes:
        stages: A list of (raw) stage configuration specifications that are part of the pipeline.
        additional_params: A dictionary of additional parameters that are not stage-specific.

    Raises:
        TypeError: If the stages or additional_params are not of the expected type.
        ValueError: If the pipeline configuration is invalid or if there are duplicate stage names.

    Examples:
        >>> PipelineConfig()
        PipelineConfig(stages=None, additional_params={})
        >>> stages = [{"stage1": {"param1": 1}}, "stage2"]
        >>> additional_params = {"param2": 2}
        >>> PipelineConfig(stages=stages, additional_params=additional_params)
        PipelineConfig(stages=[{'stage1': {'param1': 1}}, 'stage2'],
                       additional_params={'param2': 2})

    Some validation of the pipeline configuration is performed immediately after construction:

        >>> PipelineConfig(stages="invalid_type")
        Traceback (most recent call last):
            ...
        TypeError: Invalid type for stages: <class 'str'>. Expected list or ListConfig.
        >>> PipelineConfig(stages=[{3: {"param1": 1}}, "stage2"])
        Traceback (most recent call last):
            ...
        ValueError: Failed to parse pipeline configuration. Please check the pipeline YAML file 'stages' key.
        >>> PipelineConfig(stages=[{"stage1": {"param1": 1}}, "stage1"])
        Traceback (most recent call last):
            ...
        ValueError: Duplicate stage name found: stage1
        >>> PipelineConfig(stages=[{"stage1": {"param1": 1}}, "stage2"], additional_params=3)
        Traceback (most recent call last):
            ...
        TypeError: Invalid type for additional_params: <class 'int'>. Expected dict or DictConfig.

    While you can create a PipelineConfig object directly, it is more often created via the `from_arg` method,
    which loads a pipeline configuration from a YAML file. This YAML file is not structured precisely as the
    object is, as the additional parameters are flattened in the file representation. See the `from_arg`
    method for more details.

        >>> file_representation = {"stages": stages, **additional_params}
        >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as pipeline_yaml:
        ...     OmegaConf.save(file_representation, pipeline_yaml.name)
        ...     cfg = PipelineConfig.from_arg(pipeline_yaml.name)
        >>> cfg
        PipelineConfig(stages=[{'stage1': {'param1': 1}}, 'stage2'],
                       additional_params={'param2': 2})

    Once defined, you can also access the `parsed_stages` property to get a list of the stages in the pipeline
    parsed as `StageConfig` objects. This is how the user API for simple format of stages is translated to a
    readable format.

        >>> cfg.parsed_stages
        [StageConfig(name='stage1', base_stage=None, config={'param1': 1}),
         StageConfig(name='stage2', base_stage=None, config={})]
        >>> PipelineConfig().parsed_stages
        []
        >>> PipelineConfig(stages=[{"stage1": {"_base_stage": "foo"}}]).parsed_stages
        [StageConfig(name='stage1', base_stage='foo', config={})]

    Pipeline configurations can be resolved to a `DictConfig` representation suitable for Hydra registration
    via the `structured_config` property. This `DictConfig` representation has `additional_params` flattened
    into the output, much like the file representation. This representation will omit missing or empty keys
    automatically as well:

        >>> cfg.structured_config
        {'stages': [{'stage1': {'param1': 1}}, 'stage2'], 'param2': 2}
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
            | min_events_per_subject: null
            | min_measurements_per_subject: null
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
                       'reducer_output_dir': None, 'train_only': False, 'output_dir': '${output_dir}/data'}}

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
        ...     stages=[
        ...         {
        ...             "count_codes": {"aggregations": ["code/n_subjects", "code/n_occurrences"]},
        ...             "_base_stage": "aggregate_code_metadata",
        ...         },
        ...         {"filter_measurements": {"min_subjects_per_code": 4, "min_occurrences_per_code": 10}},
        ...         {
        ...             "fit_outlier_detection": {
        ...                 "aggregations": ["values/n_occurrences", "values/sum", "values/sum_sqd"],
        ...             },
        ...             "_base_stage": "aggregate_code_metadata",
        ...         },
        ...         "occlude_outliers",
        ...     ],
        ... )
        >>> runnable_stage = pipeline_cfg.register_for("count_codes")
        >>> print(runnable_stage.stage_name)
        aggregate_code_metadata
        >>> cs.repo["_pipeline.yaml"].node
        {'stages': [{'count_codes': {'aggregations': ['code/n_subjects', 'code/n_occurrences']},
                     '_base_stage': 'aggregate_code_metadata'},
                    {'filter_measurements': {'min_subjects_per_code': 4, 'min_occurrences_per_code': 10}},
                    {'fit_outlier_detection': {'aggregations': ['values/n_occurrences',
                                                                'values/sum',
                                                                'values/sum_sqd']},
                     '_base_stage': 'aggregate_code_metadata'},
                    'occlude_outliers'],
         'stage_cfg': {'data_input_dir': '${input_dir}/data',
                       'metadata_input_dir': '${input_dir}/metadata',
                       'output_dir': '${output_dir}/count_codes',
                       'train_only': True,
                       'reducer_output_dir': '${output_dir}/count_codes',
                       'aggregations': ['code/n_subjects', 'code/n_occurrences']}}
        >>> runnable_stage = pipeline_cfg.register_for("filter_measurements")
        >>> print(runnable_stage.stage_name)
        filter_measurements
        >>> cs.repo["_pipeline.yaml"].node
        {'stages': [{'count_codes': {'aggregations': ['code/n_subjects', 'code/n_occurrences']},
                     '_base_stage': 'aggregate_code_metadata'},
                    {'filter_measurements': {'min_subjects_per_code': 4, 'min_occurrences_per_code': 10}},
                    {'fit_outlier_detection': {'aggregations': ['values/n_occurrences',
                                                                'values/sum',
                                                                'values/sum_sqd']},
                     '_base_stage': 'aggregate_code_metadata'},
                    'occlude_outliers'],
         'stage_cfg': {'min_subjects_per_code': 4,
                       'min_occurrences_per_code': 10,
                       'data_input_dir': '${input_dir}/data',
                       'metadata_input_dir': '${output_dir}/count_codes',
                       'reducer_output_dir': None,
                       'train_only': False,
                       'output_dir': '${output_dir}/filter_measurements'}}

    If you try to register a pipeline that includes an unregistered stage, an error will be raised:

        >>> pipeline_cfg = PipelineConfig(stages=["foobar"])
        >>> pipeline_cfg.register_for("foobar")
        Traceback (most recent call last):
            ...
        ValueError: Stage 'foobar' not registered! Registered stages: ...
    """

    stages: list[UNPARSED_STAGE_T] | ListConfig | None = None
    additional_params: dict[str, Any] | DictConfig = dataclasses.field(default_factory=dict)

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
            PipelineConfig(stages=None, additional_params={})
            >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as pipeline_yaml:
            ...     OmegaConf.save({"stages": ["stage1", "stage2"], "foobar": 3}, pipeline_yaml.name)
            ...     PipelineConfig.from_arg(pipeline_yaml.name)
            PipelineConfig(stages=['stage1', 'stage2'], additional_params={'foobar': 3})

            To show the package path resolution, we can use the `pkg://` format, but for this test, we need to
            mock the package structure with unittest.mock.patch:

            >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as pipeline_yaml:
            ...     pipeline_fp = Path(pipeline_yaml.name)
            ...     OmegaConf.save({"stages": ["stage1", "stage2"], "qux": "a"}, pipeline_fp)
            ...     with patch("MEDS_transforms.configs.pipeline.resolve_pkg_path", return_value=pipeline_fp):
            ...         PipelineConfig.from_arg("pkg://fake_pkg.pipeline.yaml")
            PipelineConfig(stages=['stage1', 'stage2'], additional_params={'qux': 'a'})

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
        return cls(stages=stages, additional_params=as_dict_config)

    def __post_init__(self):
        if self.stages is not None and not isinstance(self.stages, list | ListConfig):
            raise TypeError(f"Invalid type for stages: {type(self.stages)}. Expected list or ListConfig.")

        try:
            self.parsed_stages  # noqa: B018
        except Exception as e:
            raise ValueError(
                "Failed to parse pipeline configuration. Please check the pipeline YAML file 'stages' key."
            ) from e

        duplicate_stages = set()
        for s in self.parsed_stages:
            if s.name in duplicate_stages:
                raise ValueError(f"Duplicate stage name found: {s.name}")
            duplicate_stages.add(s.name)

        if not isinstance(self.additional_params, DictConfig | dict):
            raise TypeError(
                f"Invalid type for additional_params: {type(self.additional_params)}. "
                f"Expected dict or DictConfig."
            )

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
        if self.additional_params:
            merged.update(self.additional_params)
        return OmegaConf.create(merged)

    @property
    def parsed_stages(self) -> list[StageConfig]:
        """Return a list of `StageConfig` objects representing the stages in the pipeline.

        This property parses the `stages` attribute and returns a list of `StageConfig` objects, which
        represent the stages in the pipeline. The `StageConfig` objects are created by parsing the
        `stages` attribute and resolving any base stage names.
        """
        if self.stages is None:
            return []

        return [StageConfig.from_arg(s) for s in copy.deepcopy(self.stages)]

    @property
    def parsed_stages_by_name(self) -> dict[str, StageConfig]:
        return {s.name: s for s in self.parsed_stages}

    def _resolve_stages(self, all_stages: dict[str, Stage]) -> dict[str, DictConfig]:
        stage_objects = []
        last_data_stage_name = None
        last_metadata_stage_name = None
        for s in self.parsed_stages:
            stage = all_stages[s.resolved_name]

            if stage.is_metadata:
                last_metadata_stage_name = s.name
            else:
                last_data_stage_name = s.name
            stage_objects.append((s.name, stage, {**s.config}))

        prior_data_stage = None
        prior_metadata_stage = None

        resolved_stage_configs = {}

        input_dir = Path("${input_dir}")
        output_dir = Path("${output_dir}")

        for name, stage, config_overwrites in stage_objects:
            config = {**stage.default_config} if stage.default_config else {}

            if prior_data_stage is None:
                config["data_input_dir"] = str(input_dir / "data")
            else:
                config["data_input_dir"] = prior_data_stage["output_dir"]

            if prior_metadata_stage is None:
                config["metadata_input_dir"] = str(input_dir / "metadata")
            else:
                config["metadata_input_dir"] = prior_metadata_stage["reducer_output_dir"]

            if stage.is_metadata:
                config["output_dir"] = str(output_dir / name)
                config["train_only"] = True
                if name == last_metadata_stage_name:
                    config["reducer_output_dir"] = str(output_dir / "metadata")
                else:
                    config["reducer_output_dir"] = str(output_dir / name)
            else:
                config["reducer_output_dir"] = None
                config["train_only"] = False
                if name == last_data_stage_name:
                    config["output_dir"] = str(output_dir / "data")
                else:
                    config["output_dir"] = str(output_dir / name)

            config.update({k: v for k, v in config_overwrites.items() if k != "_base_stage"})
            resolved_stage_configs[name] = OmegaConf.create(config)

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
            ValueError: If the stage name is not in the pipeline configuration.
            StageNotFoundError: If the stage name is not a registered stage.

        Examples:
            >>> PipelineConfig(stages=["occlude_outliers"])._resolve_stage_name("occlude_outliers")
            'occlude_outliers'
            >>> PipelineConfig(stages=["occlude_outliers"])._resolve_stage_name("foobar")
            Traceback (most recent call last):
                ...
            ValueError: Stage foobar not in pipeline configuration!
            >>> PipelineConfig(stages=["foobar"])._resolve_stage_name("foobar")
            Traceback (most recent call last):
                ...
            MEDS_transforms.stages.discovery.StageNotFoundError: Stage 'foobar' not
                registered! Registered stages: ...

        Stage names can be resolved to a base stage name, if specified in the stage configs:

            >>> cfg = PipelineConfig(stages=[{"count_codes": {"_base_stage": "aggregate_code_metadata"}}])
            >>> cfg._resolve_stage_name("count_codes")
            'aggregate_code_metadata'
        """

        if stage_name not in self.parsed_stages_by_name:
            raise ValueError(f"Stage {stage_name} not in pipeline configuration!")

        resolved_stage_name = self.parsed_stages_by_name[stage_name].resolved_name

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
        for s in self.parsed_stages:
            if s.resolved_name not in registered_stages:
                raise ValueError(
                    f"Stage '{s.resolved_name}' not registered! Registered stages: "
                    f"{', '.join(sorted(registered_stages.keys()))}"
                )
            if s.resolved_name not in loaded_stages:
                loaded_stages[s.resolved_name] = registered_stages[s.resolved_name].load()

        resolved_stage_name = self._resolve_stage_name(stage_name)

        pipeline_node = self.structured_config
        pipeline_node["stage_cfg"] = self._resolve_stages(loaded_stages)[stage_name]

        cs = ConfigStore.instance()
        cs.store(name="_pipeline", node=pipeline_node)

        return loaded_stages[resolved_stage_name]
