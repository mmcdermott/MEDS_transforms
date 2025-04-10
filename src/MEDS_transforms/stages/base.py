"""Functions for registering and defining MEDS-transforms stages."""

from __future__ import annotations

import copy
import inspect
import logging
import os
import sys
import textwrap
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from enum import StrEnum
from functools import partial, wraps
from pathlib import Path
from typing import Any, ClassVar

import polars as pl
from omegaconf import DictConfig, OmegaConf

from ..compute_modes import ANY_COMPUTE_FN_T
from ..mapreduce import map_stage, mapreduce_stage
from .discovery import get_all_registered_stages
from .examples import StageExample, StageExampleDict

logger = logging.getLogger(__name__)

MAIN_FN_T = Callable[[DictConfig], None]
VALIDATION_ENV_VAR = "DISABLE_STAGE_VALIDATION"


class StageType(StrEnum):
    """The types of stages MEDS-Transforms supports.

    Attributes:
        MAP: A stage that applies a transformation to each data shard of a MEDS dataset, outputting new data
            shards.
        MAPREDUCE: A stage that applies a metadata extraction operation to each data shard of a MEDS dataset,
            then reduces the outputs of those transformations into an updated metadata/codes.parquet file.
        MAIN: A stage that does not fit into either of the above categories, and provides a direct main
            function.
    """

    MAP = "map"
    MAPREDUCE = "mapreduce"
    MAIN = "main"

    @classmethod
    def from_fns(
        cls, main_fn: MAIN_FN_T | None, map_fn: ANY_COMPUTE_FN_T | None, reduce_fn: ANY_COMPUTE_FN_T | None
    ) -> StageType:
        """Determines the stage type based on the provided functions.

        Args:
            main_fn: The main function for the stage. May be None.
            map_fn: The mapping function for the stage. May be None.
            reduce_fn: The reducing function for the stage. May be None.

        Returns:
            StageType: The type of stage determined from the provided functions. If the passed functions do
                not correspond to a valid stage type, a ValueError will be raised.

        Raises:
            ValueError: If the provided functions do not correspond to a valid stage type.

        Examples:
            >>> def main(cfg: DictConfig):
            ...     pass
            >>> def map_fn(cfg: DictConfig, stage_cfg: DictConfig):
            ...     pass
            >>> def reduce_fn(cfg: DictConfig, stage_cfg: DictConfig):
            ...     pass
            >>> StageType.from_fns(main_fn=main, map_fn=None, reduce_fn=None)
            <StageType.MAIN: 'main'>
            >>> StageType.from_fns(main_fn=None, map_fn=map_fn, reduce_fn=None)
            <StageType.MAP: 'map'>
            >>> StageType.from_fns(main_fn=None, map_fn=map_fn, reduce_fn=reduce_fn)
            <StageType.MAPREDUCE: 'mapreduce'>
            >>> StageType.from_fns(main_fn=None, map_fn=None, reduce_fn=None)
            Traceback (most recent call last):
                ...
            ValueError: Either main_fn or map_fn/reduce_fn must be provided.
            >>> StageType.from_fns(main_fn=main, map_fn=map_fn, reduce_fn=reduce_fn)
            Traceback (most recent call last):
                ...
            ValueError: Only one of main_fn or map_fn/reduce_fn should be provided.
        """

        if main_fn is not None:
            if map_fn is not None or reduce_fn is not None:
                raise ValueError("Only one of main_fn or map_fn/reduce_fn should be provided.")
            return StageType.MAIN
        elif map_fn is not None and reduce_fn is not None:
            return StageType.MAPREDUCE
        elif map_fn is not None:
            return StageType.MAP
        else:
            raise ValueError("Either main_fn or map_fn/reduce_fn must be provided.")


class StageRegistrationError(Exception):
    pass


class Stage:
    """The representation of a MEDS-Transforms stage, in object form.

    Largely speaking, this is just a container around the different component functions that are used to
    execute the stage, as well as stage specific metadata like the name, docstring, and type.

    When constructed through the decorator `Stage.register`, it will be set to mimic in usage the function
    being decorated, though will still be a Stage object. This is done through the use of the `__call__`
    operator, and ensures that things like doctests and imports of the function being decorated do not behave
    unexpectedly. When constructed through the `Stage` constructor directly or through non-decorator usage,
    instead, the `__call__` operator will point to the main function of the stage. This enables two styles of
    usage; one where a map or compute function is decorated, but if the resulting function is imported or used
    in doctests it will behave like the decorated function, and the main callable for the stage is only
    accessible in the `main` variable of the stage object, and the second where a stage is constructed
    manually and (typically) assigned to a `main` variable in a stage python script or equivalent.

    This class is usually created through the `Stage.register` method -- either as a decorator, or as a
    function directly called. See the examples below for more details.

    Calling `print` on a `Stage` object will display a simple method indicating the name of the stage, the
    global stage docstring, and the names of the main, map, reduce, and mimic functions, if set. This is
    largely for testing and debugging purposes, and is not intended to be used in production code.

    Static data examples can also be attached to the stage object, to enable automated testing and
    documentation of the stage. This is managed through the `examples_dir` attribute, which can be passed in
    upon construction. When constructed through the `Stage.register` function, which is the typical usage, if
    unspecified this will automatically be set in accordance with the following rules:
      1. If the location in which `Stage.register` is called lives in a directory of the same name as the
         stage being defined _and_ has a subdirectory called `examples`, then that directory will be used.
      2. Otherwise, it will remain unset and no examples will be attached.
    If you would like additional modes of default inference to be added (such as on the basis of the locations
    in which the passed functions are defined) please file a GitHub Issue.

    Attributes:
        stage_type: The type of stage this is. This is set automatically based on the provided functions. See
            `StageType` for more details.
        stage_name: The name of the stage. This is set automatically based on the provided functions or can be
            set manually.
        stage_docstring: The docstring of the stage. This is set automatically based on the provided functions
            or can be set manually.
        map_fn: The mapping function for the stage. This is set automatically based on the provided functions.
            May be None for a `StageType.MAIN` stage.
        reduce_fn: The reducing function for the stage. This is set automatically based on the provided
            functions. May be None for a `StageType.MAP` or `StageType.MAIN` stage.
        main_fn: The main function for the stage. This is set automatically based on the provided functions.
        examples_dir: A directory containing nested test cases for the stage.
            If not set, this is automatically inferred in the case that the stage name and registering file
            conform to the pattern mentioned above.
        output_schema_updates: A dictionary mapping column name to a Polars type for the output of the stage,
            with the base MEDS schema options as defaults for unspecified columns.
        default_config: A dictionary containing the default configuration options for the stage. This can be
            passed manually during registration or is set automatically based on the calling file location in
            a manner similar to the examples directory.
        is_metadata: A boolean indicating whether the stage is a metadata stage. This, in some cases, is set
            automatically based on the functions passed or can be manually overridden.

    Examples:

    Most of the examples shown here are focused on more internal aspects of how stages work; for documentation
    on how you will most likely use stages, see the documentation for the `Stage.register` function.

    Stages come with tracked test cases and default configuration arguments. If no special parameters are set
    on the command line, they won't be tracked:

        >>> def compute(cfg: DictConfig):
        ...     '''base fn docstring'''
        ...     return "compute"
        >>> stage = Stage(map_fn=compute)
        >>> print(stage.examples_dir)
        None
        >>> stage.test_cases
        {}
        >>> print(stage.default_config)
        {}

    The test cases are inferred through the `examples_dir` attribute. This can be set manually:

        >>> stage = Stage(map_fn=compute, examples_dir=Path("foo"))
        >>> print(stage.examples_dir)
        foo

    But the test cases will only exist if the directory is properly set up:

        >>> stage.test_cases
        {}

    If the example dir is not a `Path` and is not `None`, an error will be raised:

        >>> stage = Stage(map_fn=compute, examples_dir="foo")
        Traceback (most recent call last):
            ...
        TypeError: examples_dir must be a Path or None. Got <class 'str'>: foo

    Likewise, the default config can be set directly through the `default_config` attribute. When set
    manually, it can be set to a dictionary or DictConfig object:

        >>> stage = Stage(map_fn=compute, default_config={"foo": "bar"})
        >>> print(f"{type(stage.default_config).__name__}: {stage.default_config}")
        DictConfig: {'foo': 'bar'}
        >>> stage = Stage(map_fn=compute, default_config=DictConfig({"foo_2": "bar_2"}))
        >>> print(f"{type(stage.default_config).__name__}: {stage.default_config}")
        DictConfig: {'foo_2': 'bar_2'}

    Or it can be set to a Path or str pointing to a YAML file containing the configuration:

        >>> config = DictConfig({"A": [1, 2, 3], "B": {"foo": "bar"}})
        >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as tmpfile:
        ...     OmegaConf.save(config, tmpfile.name)
        ...     stage = Stage(map_fn=compute, default_config=tmpfile.name)
        >>> print(f"{type(stage.default_config).__name__}: {stage.default_config}")
        DictConfig: {'A': [1, 2, 3], 'B': {'foo': 'bar'}}

    If the corresponding file does not exist, an error will be raised:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     Stage(map_fn=compute, default_config=Path(tmpdir) / "foo.yaml")
        Traceback (most recent call last):
            ...
        FileNotFoundError: Default configuration file /tmp/tmp.../foo.yaml does not exist.

    Or if it is set to any other type, a TypeError will be raised:

        >>> Stage(map_fn=compute, default_config=42)
        Traceback (most recent call last):
            ...
        TypeError: Default configuration must be a dictionary, DictConfig, or path to a YAML file. Got
                   <class 'int'>: 42



    Proper set-up for test cases means the directory is or has any children that satisfy the
    `StageExample.is_example_dir` method.

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     example_dir = Path(tmpdir) / "examples"
        ...     example_dir.mkdir()
        ...     example_data = example_dir / "out_data.yaml"
        ...     _ = example_data.write_text("data/0.parquet: 'code,time,subject_id,numeric_value'")
        ...     stage = Stage(map_fn=compute, examples_dir=example_dir)
        ...     print(stage.test_cases)
        StageExample [base]
          stage_cfg: {}
          want_data:
            MEDSDataset:
            dataset_metadata:
            data_shards:
              - 0:
                pyarrow.Table
                subject_id: int64
                time: timestamp[us]
                code: string
                numeric_value: float
                ----
                subject_id: [[]]
                time: [[]]
                code: []
                numeric_value: [[]]
            code_metadata:
              pyarrow.Table
              code: string
              description: string
              parent_codes: list<item: string>
                child 0, item: string
              ----
              code: []
              description: []
              parent_codes: []
            subject_splits: None

    We can also have nested test cases:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     example_dir = Path(tmpdir) / "examples"
        ...     # Example # 1
        ...     ex_1 = example_dir / "example_1"
        ...     ex_1.mkdir(parents=True)
        ...     ex_1_data_fp = (ex_1 / "out_data.yaml")
        ...     _ = ex_1_data_fp.write_text("data/0.parquet: 'code,time,subject_id,numeric_value'")
        ...     # Example # 2
        ...     ex_2 = example_dir / "example_2_foo"
        ...     ex_2.mkdir()
        ...     ex_2_metadata_fp = (ex_2 / "out_metadata.yaml")
        ...     _ = ex_2_metadata_fp.write_text("metadata/codes.parquet: 'code,description,parent_codes'")
        ...     stage = Stage(map_fn=compute, examples_dir=example_dir)
        ...     print(stage.test_cases)
        example_2_foo:
        │   StageExample [base/example_2_foo]
        │     stage_cfg: {}
        │     want_metadata:
        │       shape: (0, 3)
        │       ┌──────┬─────────────┬──────────────┐
        │       │ code ┆ description ┆ parent_codes │
        │       │ ---  ┆ ---         ┆ ---          │
        │       │ str  ┆ str         ┆ list[str]    │
        │       ╞══════╪═════════════╪══════════════╡
        │       └──────┴─────────────┴──────────────┘
        example_1:
        │   StageExample [base/example_1]
        │     stage_cfg: {}
        │     want_data:
        │       MEDSDataset:
        │       dataset_metadata:
        │       data_shards:
        │         - 0:
        │           pyarrow.Table
        │           subject_id: int64
        │           time: timestamp[us]
        │          code: string
        │          numeric_value: float
        │          ----
        │          subject_id: [[]]
        │          time: [[]]
        │          code: []
        │          numeric_value: [[]]
        │       code_metadata:
        │         pyarrow.Table
        │         code: string
        │         description: string
        │         parent_codes: list<item: string>
        │           child 0, item: string
        │         ----
        │         code: []
        │         description: []
        │         parent_codes: []
        │       subject_splits: None

    The examples directory can also be inferred via a passed `_calling_file` argument, which is intended to be
    the file in which the `Stage.register` function was called. This looks to see if (1) the calling file
    lives in a parent directory of the same name as the defining stage and (2) if that directory has an
    `examples` subdirectory. If so, this subdirectory is et to the examples directory. Otherwise, nothing is
    set.

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     stage_dir = Path(tmpdir) / "stage_foo"
        ...     calling_file = stage_dir / "stage_foo.py"
        ...     examples_dir = stage_dir / "examples"
        ...     examples_dir.mkdir(parents=True)
        ...     calling_file.touch()
        ...     stage = Stage(map_fn=compute, stage_name="stage_foo", _calling_file=calling_file)
        ...     print(stage.examples_dir.relative_to(tmpdir))
        stage_foo/examples

    If the stage name doesn't match the directory name, we won't infer `self.examples_dir`:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     stage_dir = Path(tmpdir) / "stage_foo"
        ...     calling_file = stage_dir / "stage_foo.py"
        ...     examples_dir = stage_dir / "examples"
        ...     examples_dir.mkdir(parents=True)
        ...     calling_file.touch()
        ...     stage = Stage(map_fn=compute, stage_name="not_stage_foo", _calling_file=calling_file)
        ...     print(stage.examples_dir)
        None

    If the examples directory doesn't exist, we won't infer `self.examples_dir`:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     stage_dir = Path(tmpdir) / "stage_foo"
        ...     stage_dir.mkdir()
        ...     calling_file = stage_dir / "stage_foo.py"
        ...     calling_file.touch()
        ...     stage = Stage(map_fn=compute, stage_name="stage_foo", _calling_file=calling_file)
        ...     print(stage.examples_dir)
        None

    If the calling file doesn't exist, we won't infer `self.examples_dir`:

        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     stage_dir = Path(tmpdir) / "stage_foo"
        ...     calling_file = stage_dir / "stage_foo.py"
        ...     examples_dir = stage_dir / "examples"
        ...     examples_dir.mkdir(parents=True)
        ...     stage = Stage(map_fn=compute, stage_name="stage_foo", _calling_file=calling_file)
        ...     print(stage.examples_dir)
        None

    If the calling file is somehow misconfigured (e.g., it is a directory, not a file), a warning will be
    logged, and nothing will be inferred:

        >>> with tempfile.TemporaryDirectory() as tmpdir, print_warnings(), Stage.suppress_validation():
        ...     stage_dir = Path(tmpdir) / "stage_foo"
        ...     examples_dir = stage_dir / "examples"
        ...     examples_dir.mkdir(parents=True)
        ...     stage = Stage(map_fn=compute, stage_name="stage_foo", _calling_file=stage_dir)
        ...     print(stage.examples_dir)
        None
        Warning: Stage definition file /tmp/tmp.../stage_foo is not a file. Cannot infer examples
            directory.

    The stage, upon construction, attempts to validate that the stage is registered in the entry points. This
    can be disabled (as shown above) with the `Stage.suppress_validation` context manager, or by setting
    certain class variables. Had we disabled that, we would have seen

        >>> with print_warnings():
        ...     stage = Stage(map_fn=compute, stage_name="stage_foo")
        Warning: Stage 'stage_foo' is not registered in the entry points.
        This may be due to a missing or incorrectly configured entry point in your setup.py or pyproject.toml
        file. If this is during development, you may need to run `pip install -e .` to install your package
        properly in editable mode and ensure your stage registration is detected.
        You can disable this warning by setting the class variable `WARN_IF_NO_ENTRY_POINT_AT_NAME` to
        `False`, or filtering out `StageRegistrationWarning` warnings.
        You can disable all validation by setting the environment variable `DISABLE_STAGE_VALIDATION` to `1`.

    This warning indicates that there is no stage registered in the entry points with the name `stage_foo`. It
    is intended to flag to developers or users that they need to add their entry point to their package set-up
    to ensure things work correctly. We can also turn off this warning by directly setting the class variable
    `WARN_IF_NO_ENTRY_POINT_AT_NAME` to `False` on the `Stage` class or a derived class:

        >>> class StageNoWarn(Stage):
        ...     WARN_IF_NO_ENTRY_POINT_AT_NAME: ClassVar[bool] = False
        >>> with print_warnings():
        ...     stage = StageNoWarn(map_fn=compute, stage_name="stage_foo")

    Warnings aren't the end of the stage validation; if the stage is correctly configured, not only will
    `stage_foo` (or the stage name more generally) be registered, but it _will point to the Stage object being
    registered at this moment_. The best we can do for now to ensure this is to validate that if we try to
    load the Stage object being registered, we get an error due to attempting a circular import (as we would
    be attempting to reload the code that is running the stage construction). To show the limits of this,
    we'll patch out the stage registration object to return without errors and report a `stage_foo` being
    registered, and see what happens:

        >>> mock_stages = {"stage_foo": MagicMock()}
        >>> mock_stages["stage_foo"].load.side_effect = lambda: None
        >>> with patch("MEDS_transforms.stages.base.get_all_registered_stages", return_value=mock_stages):
        ...     with print_warnings(): # No warnings are printed as the stage is listed as being registered.
        ...         Stage(map_fn=compute, stage_name="stage_foo")
        Traceback (most recent call last):
            ...
        MEDS_transforms.stages.base.StageRegistrationError: Stage stage_foo is registered, but an attempted
        reload causes no issues. If this were the stage you are constructing, a reload would cause a circular
        import error, so this means you are overwriting an external, different stage, which is a problem!
        This may be due to a missing or incorrectly configured entry point in your setup.py or pyproject.toml
        file. If this is during development, you may need to run `pip install -e .` to install your package
        properly in editable mode and ensure your stage registration is detected.
        You can disable reload-should-cause-error checking by setting the class variable
        `ERR_IF_ENTRY_POINT_IMPORTABLE` to `False`.
        You can disable all validation by setting the environment variable `DISABLE_STAGE_VALIDATION` to `1`.

    These errors also occur if the loading of the stage raises an error that is not a circular import error.

        >>> def raise_unexpected_error():
        ...     raise AttributeError("unrelated")
        >>> mock_stages["stage_foo"].load.side_effect = raise_unexpected_error
        >>> with patch("MEDS_transforms.stages.base.get_all_registered_stages", return_value=mock_stages):
        ...     with print_warnings(): # No warnings are printed as the stage is listed as being registered.
        ...         Stage(map_fn=compute, stage_name="stage_foo")
        Traceback (most recent call last):
            ...
        ValueError: Failed to validate stage stage_foo for an unexpected reason; it is
        possible that a different stage is defined at this name.

    You can also turn off these errors by setting the class variable `ERR_IF_ENTRY_POINT_IMPORTABLE` to
    `False` on the `Stage` class or a derived class:

        >>> class StageNoErr(Stage):
        ...     ERR_IF_ENTRY_POINT_IMPORTABLE: ClassVar[bool] = False
        >>> with patch("MEDS_transforms.stages.base.get_all_registered_stages", return_value=mock_stages):
        ...     stage = StageNoErr(map_fn=compute, stage_name="stage_foo")
    """

    stage_type: StageType
    stage_name: str

    map_fn: ANY_COMPUTE_FN_T | None = None
    reduce_fn: ANY_COMPUTE_FN_T | None = None
    main_fn: MAIN_FN_T | None = None

    output_schema_updates: dict[str, pl.DataType] | None = None
    is_metadata: bool | None = None

    __mimic_fn: Callable | None = None
    __stage_docstring: str | None = None
    __stage_name: str | None = None
    __stage_dir: Path | None = None
    __examples_dir: Path | None = None
    __default_config: DictConfig | None = None

    WARN_IF_NO_ENTRY_POINT_AT_NAME: ClassVar[bool] = os.environ.get(VALIDATION_ENV_VAR, "0") != "1"
    ERR_IF_ENTRY_POINT_IMPORTABLE: ClassVar[bool] = os.environ.get(VALIDATION_ENV_VAR, "0") != "1"

    ENTRY_POINT_SETUP_STRING: ClassVar[str] = (
        "This may be due to a missing or incorrectly configured entry point in your setup.py or "
        "pyproject.toml file. If this is during development, you may need to run "
        "`pip install -e .` to install your package properly in editable mode and ensure your "
        "stage registration is detected. "
    )
    DISABLE_WARNING_STRING: ClassVar[str] = (
        "You can disable this warning by setting the class variable `WARN_IF_NO_ENTRY_POINT_AT_NAME` to "
        "`False`, or filtering out `StageRegistrationWarning` warnings. "
    )
    DISABLE_ERROR_STRING: ClassVar[str] = (
        "You can disable reload-should-cause-error checking by setting the class variable "
        "`ERR_IF_ENTRY_POINT_IMPORTABLE` to `False`."
    )
    DISABLE_ALL_STAGE_VALIDATION_STRING: ClassVar[str] = (
        "You can disable all validation by setting the environment variable "
        "`DISABLE_STAGE_VALIDATION` to `1`."
    )

    @staticmethod
    @contextmanager
    def suppress_validation():
        """Context manager to disable stage validation, to be used in testing and execution settings.

        Example:
            >>> with Stage.suppress_validation():
            ...     # Your code here that requires no stage validation.
            ...     pass
        """
        old_env_val = os.environ.get(VALIDATION_ENV_VAR, None)
        os.environ[VALIDATION_ENV_VAR] = "1"
        try:
            yield
        finally:
            if old_env_val is None:
                del os.environ[VALIDATION_ENV_VAR]
            else:
                os.environ[VALIDATION_ENV_VAR] = old_env_val

    def __init__(
        self,
        *,
        main_fn: MAIN_FN_T | None = None,
        map_fn: ANY_COMPUTE_FN_T | None = None,
        reduce_fn: ANY_COMPUTE_FN_T | None = None,
        stage_name: str | None = None,
        stage_docstring: str | None = None,
        output_schema_updates: dict[str, pl.DataType] | None = None,
        examples_dir: Path | None = None,
        default_config: dict[str, Any] | DictConfig | Path | str | None = None,
        is_metadata: bool | None = None,
        _calling_file: Path | None = None,
    ) -> MAIN_FN_T:
        """Wraps or returns a function that can serve as the main function for a stage."""

        self.stage_type = StageType.from_fns(main_fn, map_fn, reduce_fn)
        self.stage_name = stage_name
        self.stage_docstring = stage_docstring

        self.main_fn = main_fn
        self.map_fn = map_fn
        self.reduce_fn = reduce_fn

        if is_metadata is None:
            if self.stage_type is StageType.MAPREDUCE:
                self.is_metadata = True
            elif self.stage_type is StageType.MAP:
                self.is_metadata = False
            else:
                raise ValueError(
                    "Stage type is not set to MAP or MAPREDUCE, but is_metadata is not set. Please set "
                    "is_metadata manually."
                )

            logger.debug(
                f"Automatically setting is_metadata={self.is_metadata} for {self.stage_name} as it is a "
                f"{self.stage_type} stage."
            )
        else:
            self.is_metadata = is_metadata

        self.__infer_stage_dir(_calling_file)

        self.examples_dir = examples_dir
        self.default_config = default_config

        do_skip_validation = (os.environ.get(VALIDATION_ENV_VAR, "0") == "1") or not (
            self.WARN_IF_NO_ENTRY_POINT_AT_NAME or self.ERR_IF_ENTRY_POINT_IMPORTABLE
        )

        if do_skip_validation:
            logger.debug(
                "Skipping stage validation at constructor time due to DISABLE_STAGE_VALIDATION environment "
                "variable being set. This is normal during execution of a stage via the MEDS-Transforms CLI, "
                "as validation happens manually in the main function in that context, but is typically not "
                "normal during testing, for example."
            )
        else:
            self.__validate_stage_entry_point_registration()

        if output_schema_updates is None:
            self.output_schema_updates = {}
        else:
            self.output_schema_updates = copy.deepcopy(output_schema_updates)

    def __infer_stage_dir(self, stage_definition_file: Path | None) -> Path | None:
        """Infers a possible stage directory based on the calling file.

        This is done by looking to see if the stage is defined in a file contained in a directory of the same
        name as the stage. If so, that directory is stored as the stage directory. If not, this will be None.
        """
        self.__stage_dir = None

        if stage_definition_file is None:
            return

        if not stage_definition_file.is_file():
            logger.warning(
                f"Stage definition file {stage_definition_file} is not a file. "
                "Cannot infer examples directory."
            )
            return

        # Get the directory of the calling file
        possible_stage_dir = stage_definition_file.parent

        if possible_stage_dir.name != self.stage_name:
            logger.debug(
                f"Stage definition file {stage_definition_file} is not in a directory with the same name as "
                "the stage {self.stage_name}. Cannot infer examples directory."
            )
            return

        self.__stage_dir = possible_stage_dir

    @property
    def examples_dir(self) -> Path | None:
        if self.__examples_dir is not None:
            return self.__examples_dir
        if self.__stage_dir is None:
            return None

        possible_examples_dir = self.__stage_dir / "examples"

        if possible_examples_dir.is_dir():
            return possible_examples_dir
        else:
            logger.debug(
                f"Stage definition file {self.__stage_dir} lacks an examples subdirectory. Can't infer "
                "examples directory."
            )
            return None

    @examples_dir.setter
    def examples_dir(self, examples_dir: Path | None):
        """Sets the examples directory for the stage.

        Args:
            examples_dir: The examples directory to set. This should be a directory containing nested
                test cases for the stage.
        """

        match examples_dir:
            case None | Path():
                self.__examples_dir = examples_dir
            case _:
                raise TypeError(
                    f"examples_dir must be a Path or None. Got {type(examples_dir)}: {examples_dir}"
                )

    @property
    def default_config(self) -> DictConfig:
        if self.__default_config is not None:
            return self.__default_config

        if self.__stage_dir is None:
            return DictConfig({})

        possible_default_config = self.__stage_dir / "config.yaml"
        if possible_default_config.is_file():
            return OmegaConf.load(possible_default_config)
        else:
            logger.debug(
                f"Stage definition file {self.__stage_dir} lacks a config.yaml file. Can't infer default "
                "configuration."
            )
            return DictConfig({})

    @default_config.setter
    def default_config(self, default_config: dict[str, Any] | DictConfig | Path | str | None):
        """Sets the default configuration for the stage.

        Args:
            default_config: The default configuration to set. This can be a dictionary, a DictConfig object,
                or a path to a YAML file containing the configuration.
        """
        match default_config:
            case None | DictConfig():
                self.__default_config = default_config
            case dict():
                self.__default_config = DictConfig(default_config)
            case Path() | str() as default_config_fp:
                if not Path(default_config_fp).is_file():
                    raise FileNotFoundError(f"Default configuration file {default_config_fp} does not exist.")
                self.__default_config = OmegaConf.load(default_config_fp)
            case _:
                raise TypeError(
                    "Default configuration must be a dictionary, DictConfig, or path to a YAML file. Got "
                    f"{type(default_config)}: {default_config}"
                )

    @property
    def test_cases(self) -> dict[str, StageExample]:
        if self.examples_dir is None:
            return {}

        examples_to_check = [self.examples_dir]
        test_cases = {}

        while examples_to_check:
            example_dir = examples_to_check.pop()

            if not example_dir.is_dir():
                continue

            if StageExample.is_example_dir(example_dir):
                scenario_name = example_dir.relative_to(self.examples_dir).as_posix()
                test_cases[scenario_name] = StageExample.from_dir(
                    stage_name=self.stage_name,
                    scenario_name=scenario_name,
                    example_dir=example_dir,
                    **self.output_schema_updates,
                )
            else:
                examples_to_check.extend(example_dir.iterdir())

        return StageExampleDict(**test_cases)

    def __validate_stage_entry_point_registration(self):
        """Validates that the stage is registered in the entry points."""

        registered_stages = get_all_registered_stages()

        if self.stage_name not in registered_stages:
            if self.WARN_IF_NO_ENTRY_POINT_AT_NAME:
                # If the stage is not registered, we warn.
                logger.warning(
                    f"Stage '{self.stage_name}' is not registered in the entry points.\n"
                    f"{self.ENTRY_POINT_SETUP_STRING}\n{self.DISABLE_WARNING_STRING}\n"
                    f"{self.DISABLE_ALL_STAGE_VALIDATION_STRING}",
                )
            return

        if not self.ERR_IF_ENTRY_POINT_IMPORTABLE:
            return

        with Stage.suppress_validation():
            try:
                # Attempt to reload, which should cause an error if the stage being constructed is the same
                # stage as the one being registered.
                registered_stages[self.stage_name].load()
                raise StageRegistrationError(
                    f"Stage {self.stage_name} is registered, but an attempted reload causes "
                    "no issues. If this were the stage you are constructing, a reload would cause a circular "
                    "import error, so this means you are overwriting an external, different stage, which is "
                    "a problem!\n"
                    f"{self.ENTRY_POINT_SETUP_STRING}\n{self.DISABLE_ERROR_STRING}\n"
                    f"{self.DISABLE_ALL_STAGE_VALIDATION_STRING}"
                )
            except AttributeError as e:
                if "circular import" not in str(e):
                    raise ValueError(
                        f"Failed to validate stage {self.stage_name} for an unexpected reason; it is "
                        "possible that a different stage is defined at this name."
                    ) from e

    @property
    def stage_name(self) -> str:
        if self.__stage_name is not None:
            return self.__stage_name

        return (self.main_fn or self.map_fn).__module__.split(".")[-1]

    @stage_name.setter
    def stage_name(self, name: str):
        self.__stage_name = name

    @property
    def stage_docstring(self) -> str:
        """The docstring for the stage.

        This is set automatically based on the provided functions.
        """
        if self.__stage_docstring is not None:
            return self.__stage_docstring

        match self.stage_type:
            case StageType.MAIN:
                return self.main_fn_docstring
            case StageType.MAP:
                return self.map_fn_docstring
            case StageType.MAPREDUCE:
                return f"Map Stage:\n{self.map_fn_docstring}\n\nReduce stage:\n{self.reduce_fn_docstring}"

    @stage_docstring.setter
    def stage_docstring(self, docstring: str):
        self.__stage_docstring = docstring

    @property
    def main_fn_docstring(self) -> str:
        return (inspect.getdoc(self.main_fn) or "") if self.main_fn is not None else ""

    @property
    def map_fn_docstring(self) -> str:
        return (inspect.getdoc(self.map_fn) or "") if self.map_fn is not None else ""

    @property
    def reduce_fn_docstring(self) -> str:
        return (inspect.getdoc(self.reduce_fn) or "") if self.reduce_fn is not None else ""

    @property
    def main(self) -> MAIN_FN_T:
        match self.stage_type:
            case StageType.MAIN:

                @wraps(self.main_fn)
                def main_fn(cfg: DictConfig):
                    self.main_fn(cfg)

            case StageType.MAP:

                @wraps(self.map_fn)
                def main_fn(cfg: DictConfig):
                    map_stage(cfg, self.map_fn)

            case StageType.MAPREDUCE:

                def main_fn(cfg: DictConfig):
                    mapreduce_stage(cfg, self.map_fn, self.reduce_fn)

        main_fn.__name__ = self.stage_name
        main_fn.__doc__ = self.stage_docstring
        return main_fn

    @property
    def mimic_fn(self) -> Callable | None:
        """The function that this stage object should "mimic" when treated like a plain function.

        This is useful when a stage is constructed via decorator mode of `Stage.register`. In that case, we
        want this stage to mimic the decorated function upon normal usage. This practice enables doctests and
        imports of the stage function to work as expected. This property gets the function being mimicked.

        Examples:
            >>> def compute(cfg: DictConfig):
            ...     '''base compute docstring'''
            ...     return "compute"
            >>> def baz_fn(foo: str, bar: int):
            ...     '''base baz docstring'''
            ...     return f"baz {foo} {bar}"
            >>> stage = Stage.register(map_fn=compute)
            >>> print(stage.mimic_fn)
            None
            >>> stage.mimic_fn = baz_fn
            >>> stage.__name__
            'baz_fn'
            >>> stage.__doc__
            'base baz docstring'
            >>> stage('foo', 42)
            'baz foo 42'
            >>> stage.mimic_fn = "main"
            Traceback (most recent call last):
                ...
            TypeError: Cannot set mimic_fn to non-function. Got <class 'str'>
        """
        return self.__mimic_fn

    @mimic_fn.setter
    def mimic_fn(self, fn: Callable):
        """Sets the function that this stage object should "mimic" when treated like a plain function.

        In addition to the function itself, this also overwrites variables like `__name__` and `__doc__` to
        match the decorated function. This should really only be used to mimic one of the main or map
        functions, but this is not checked currently.

        Args:
            fn: The function to mimic.

        Raises:
            TypeError: If the provided argument is not a function.
        """

        if not inspect.isfunction(fn):
            raise TypeError(f"Cannot set mimic_fn to non-function. Got {type(fn)}")

        self.__mimic_fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

        # Fix doctests
        try:
            module = sys.modules[fn.__module__]
            if hasattr(module, "__test__"):  # pragma: no cover
                module.__test__[fn.__name__] = fn
            else:
                module.__test__ = {fn.__name__: fn}
        except Exception as e:  # pragma: no cover
            warnings.warn(f"Failed to set doctest for {fn.__name__}: {e}")

    def __call__(self, *args, **kwargs):
        if self.mimic_fn is not None:
            return self.mimic_fn(*args, **kwargs)

        if len(args) == 1 and len(kwargs) == 0 and inspect.isfunction(args[0]):
            # This is likely inappropriate decorator usage, so we raise a custom error.
            raise ValueError(
                "If a Stage is constructed via keyword arguments directly (including a `main_fn` or a "
                "`map_fn`), then it cannot be used as a decorator or otherwise called directly."
            )

        raise ValueError(f"Stage {self.stage_name} has no function to mimic, so can't be called directly.")

    def __str__(self) -> str:
        lines = [
            f"Stage {self.stage_name}:",
            f"  Type: {self.stage_type}",
            f"  is_metadata: {self.is_metadata}",
            "  Docstring:",
        ]

        pretty_wrap = partial(textwrap.wrap, width=110, initial_indent="    | ", subsequent_indent="    | ")

        docstring_lines = textwrap.dedent(self.stage_docstring).splitlines()
        for line in docstring_lines:
            lines.extend(pretty_wrap(line))

        if self.default_config:
            lines.append("  Default config:")
            lines.extend(textwrap.indent(str(OmegaConf.to_yaml(self.default_config)), "    | ").splitlines())

        if self.output_schema_updates:
            lines.append("  Output schema updates:")
            lines.extend(pretty_wrap(str(self.output_schema_updates)))

        lines.extend(
            [
                f"  Map function: {self.map_fn.__name__ if self.map_fn else None}",
                f"  Reduce function: {self.reduce_fn.__name__ if self.reduce_fn else None}",
                f"  Main function: {self.main_fn.__name__ if self.main_fn else None}",
                f"  Mimic function: {self.mimic_fn.__name__ if self.mimic_fn else None}",
            ]
        )
        return "\n".join(lines)

    @classmethod
    def register(cls, *args, **kwargs) -> Callable[[Callable], Stage] | Stage:
        """This method used to define and register a MEDS-Transforms stage of any variety.

        ## Function usage modes:

        It can be used either as a decorator, a parametrized decorator, or a direct method, depending on the
        arguments provided and manner of invocation.

        ### As a decorator
        If used as a decorator (e.g., `@Stage.register`) on a single function, it will define a stage using
        that function. The stage defined will either be a `StageType.MAIN` stage if the function is named
        `main`, or a `StageType.MAP` stage otherwise. The return value will be a Stage object set to mimic the
        called function.

        ### As a parametrized decorator
        If used as a parametrized decorator (e.g., `@Stage.register(stage_name="foo")`), it will define a
        stage using the decorated function, but with the specified keyword arguments modifying the stage
        appropriately. The keyword arguments included can include any property save `main_fn` or `map_fn`. The
        stage will either be a `StageType.MAIN` stage if the function is named `main` (in which case a
        `reduce_fn` property cannot be set), a `StageType.MAP` stage if the decorated function is not named
        main and a `reduce_fn` is not set, or a `StageType.MAPREDUCE` stage if the decorated function is not
        named `main` and a `reduce_fn` is set. The return value will be a Stage object set to mimic the
        decorated function. If a `reduce_fn` is passed as a keyword argument, it will not be modified.

        ### As a direct method
        If used as a direct method, (e.g., `main = Stage.register(main_fn=main)`), it will define a stage
        using the provided arguments, and return the stage object. The returned object will not be set to
        mimic any function, and the `main` method will be the main entry point for the stage. This mode is
        only activated if sufficient keyword arguments are provided to define a stage (meaning a `main_fn` or
        `map_fn` must be set). The passed functions will not be modified.

        Args:
            *args: Positional arguments. If used as a decorator, this should be a single function, and no
                keyword arguments should be set.
            **kwargs: Keyword arguments. These can include all keyword arguments to the `Stage` constructor
                save for `_calling_file`; namely, `main_fn`, `map_fn`, `reduce_fn`, `stage_name`,
                `stage_docstring`, `examples_dir`, `output_schema_updates`, and `default_config`.
                Not all keyword arguments are required for all usages of the decorator.

        ## Inference of static data examples and the default configuration filepath.

        When this function is called, it will attempt to locate the filepath of the file in which it was
        called, and pass that as the `_calling_file` argument to the `Stage` constructor. This is used to
        infer the examples directory and default configuration option if thery are not manually set. Note that
        this means that the parameter `_calling_file` can not be used as a keyword argument to this function.
        You should never need this parameter anyways, as if you are setting something manually you can
        directly set the example directory.

        Returns:
            Either a `Stage` object or a decorator function, depending on the arguments provided.

        Raises:
            ValueError: If multiple positional arguments are passed or if the function is not callable.
            TypeError: If the first positional argument is not a function.
            ValueError: If both `main_fn` and `map_fn` or `reduce_fn` are provided.

        Examples:

            When used with only keyword arguments that fully define a function a stage set to mimic nothing is
            returned directly, with the parameters set as defined by what kind of stage is being created.

            >>> def main(cfg: DictConfig):
            ...     '''base main docstring'''
            ...     return "main"
            >>> def map_fn(cfg: DictConfig, stage_cfg: DictConfig):
            ...     '''base map docstring'''
            ...     return "map"
            >>> def reduce_fn(cfg: DictConfig, stage_cfg: DictConfig):
            ...     '''base reduce docstring'''
            ...     return "reduce"
            >>> stage = Stage.register(main_fn=main, is_metadata=True)
            >>> print(stage) # The name is inferred from the name of the file:
            Stage base:
              Type: main
              is_metadata: True
              Docstring:
                | base main docstring
              Map function: None
              Reduce function: None
              Main function: main
              Mimic function: None

            As it has no mimic function, if you try to call it directly, it will raise an error, and not act
            like the main function.

            >>> main({})
            'main'
            >>> stage({})
            Traceback (most recent call last):
                ...
            ValueError: Stage base has no function to mimic, so can't be called directly.

            >>> print(Stage.register(map_fn=map_fn, is_metadata=False))
            Stage base:
              Type: map
              is_metadata: False
              Docstring:
                | base map docstring
              Map function: map_fn
              Reduce function: None
              Main function: None
              Mimic function: None
            >>> print(Stage.register(map_fn=map_fn, reduce_fn=reduce_fn, is_metadata=True))
            Stage base:
              Type: mapreduce
              is_metadata: True
              Docstring:
                | Map Stage:
                | base map docstring
                | Reduce stage:
                | base reduce docstring
              Map function: map_fn
              Reduce function: reduce_fn
              Main function: None
              Mimic function: None

            If you call it with main and map functions or a main and a reduce function, it will raise an
            error:

            >>> Stage.register(main_fn=main, map_fn=map_fn)
            Traceback (most recent call last):
                ...
            ValueError: Only one of main_fn or map_fn/reduce_fn should be provided.
            >>> Stage.register(main_fn=main, reduce_fn=reduce_fn)
            Traceback (most recent call last):
                ...
            ValueError: Only one of main_fn or map_fn/reduce_fn should be provided.

            The `is_metadata` parameter is a bit special, in that you need to provide it if you are defining a
            "main" stage; otherwise it can be omitted and inferred.

            >>> print(Stage.register(map_fn=map_fn))
            Stage base:
              Type: map
              is_metadata: False
              Docstring:
                | base map docstring
              Map function: map_fn
              Reduce function: None
              Main function: None
              Mimic function: None
            >>> print(Stage.register(map_fn=map_fn, reduce_fn=reduce_fn))
            Stage base:
              Type: mapreduce
              is_metadata: True
              Docstring:
                | Map Stage:
                | base map docstring
                | Reduce stage:
                | base reduce docstring
              Map function: map_fn
              Reduce function: reduce_fn
              Main function: None
              Mimic function: None
            >>> Stage.register(main_fn=main)
            Traceback (most recent call last):
                ...
            ValueError: Stage type is not set to MAP or MAPREDUCE, but is_metadata is not set. Please set
            is_metadata manually.

            You can also use it as a decorator, either parametrized or not.

            >>> @Stage.register
            ... def map_fn(cfg: DictConfig, stage_cfg: DictConfig):
            ...     '''base map docstring'''
            ...     return "map"
            >>> @Stage.register(is_metadata=True)
            ... def main(cfg: DictConfig):
            ...     '''base main docstring'''
            ...     return "main"

            The output of the decorator, which is stored in the variable of the name defined by the function,
            will "mimic" the decorated function under normal usage:

            >>> map_fn.__name__
            'map_fn'
            >>> map_fn.__doc__
            'base map docstring'
            >>> map_fn({}, {})
            'map'
            >>> main.__name__
            'main'
            >>> main.__doc__
            'base main docstring'
            >>> main({})
            'main'

            ... but it is actually a stage object defined by the decorator:

            >>> print(map_fn)
            Stage base:
              Type: map
              is_metadata: False
              Docstring:
                | base map docstring
              Map function: map_fn
              Reduce function: None
              Main function: None
              Mimic function: map_fn
            >>> print(main)
            Stage base:
              Type: main
              is_metadata: True
              Docstring:
                | base main docstring
              Map function: None
              Reduce function: None
              Main function: main
              Mimic function: main

            When used as a decorator, you can also parametrize the decorator with other parameters:

            >>> @Stage.register(stage_name="foo", stage_docstring="bar", is_metadata=False)
            ... def main(cfg: DictConfig):
            ...     '''base main docstring'''
            ...     return "main"
            >>> print(main)
            Stage foo:
              Type: main
              is_metadata: False
              Docstring:
                | bar
              Map function: None
              Reduce function: None
              Main function: main
              Mimic function: main

            The acceptable keyword arguments to the decorator are the same as those to the constructor, except
            for `_calling_file` (which is automatically inferred and will be discussed more later). For
            example:

            >>> @Stage.register(
            ...     stage_name="foo",
            ...     stage_docstring="bar",
            ...     output_schema_updates={"foo": pl.Int64},
            ...     examples_dir=Path("foo"),
            ...     default_config={"arg1": {"option1": "foo"}, "arg2": [1, 2.3]},
            ... )
            ... def foobar(cfg: DictConfig):
            ...     '''base map docstring'''
            ...     return "map"
            >>> print(foobar)
            Stage foo:
              Type: map
              is_metadata: False
              Docstring:
                | bar
              Default config:
                | arg1:
                |   option1: foo
                | arg2:
                | - 1
                | - 2.3
              Output schema updates:
                | {'foo': Int64}
              Map function: foobar
              Reduce function: None
              Main function: None
              Mimic function: foobar

            The decorated function will be inferred to be a "main" function if it is named "main" and there is
            no reduce function specified in the parametrized keyword arguments to the decorator. Otherwise, it
            will be assumed to be a map function:

            >>> @Stage.register
            ... def map_fn(cfg: DictConfig, stage_cfg: DictConfig):
            ...     '''base map docstring'''
            ...     return "map"
            >>> print(map_fn)
            Stage base:
              Type: map
              is_metadata: False
              Docstring:
                | base map docstring
              Map function: map_fn
              Reduce function: None
              Main function: None
              Mimic function: map_fn
            >>> @Stage.register(reduce_fn=reduce_fn)
            ... def main(cfg: DictConfig):
            ...     '''base main docstring'''
            ...     return "main"
            >>> print(main)
            Stage base:
              Type: mapreduce
              is_metadata: True
              Docstring:
                | Map Stage:
                | base main docstring
                | Reduce stage:
                | base reduce docstring
              Map function: main
              Reduce function: reduce_fn
              Main function: None
              Mimic function: main

            You can't use it as a decorator while specifying either the main or map function in the keyword
            arguments:

            >>> @Stage.register(map_fn=map_fn)
            ... def reduce_fn(cfg: DictConfig, stage_cfg: DictConfig):
            ...     '''base reduce docstring'''
            ...     return "reduce"
            Traceback (most recent call last):
                ...
            ValueError: If a Stage is constructed via keyword arguments directly (including a `main_fn` or a
            `map_fn`), then it cannot be used as a decorator or otherwise called directly.

            You also can't specify a positional argument and keyword arguments at the same time or multiple
            positional arguments:

            >>> stage = Stage.register(main, stage_name="foo")
            Traceback (most recent call last):
                ...
            ValueError: Cannot provide keyword arguments when using as a decorator.
            >>> stage = Stage.register(main, map_fn)
            Traceback (most recent call last):
                ...
            ValueError: Stage.register can only be used with at most a single positional arg. Got 2

            Though not recommended, you could theoretically use decorator mode directly to create a function
            that would return a stage. This exposes some further error surfaces which are checked explicitly:

            >>> def main(cfg: DictConfig):
            ...     '''base main docstring'''
            ...     return "main"
            >>> def map_fn(cfg: DictConfig, stage_cfg: DictConfig):
            ...     '''base map docstring'''
            ...     return "map"
            >>> manual_decorator = Stage.register(stage_name="foo", stage_docstring="bar")
            >>> print(manual_decorator(map_fn))
            Stage foo:
              Type: map
              is_metadata: False
              Docstring:
                | bar
              Map function: map_fn
              Reduce function: None
              Main function: None
              Mimic function: map_fn
            >>> manual_decorator(main, map_fn=map_fn)
            Traceback (most recent call last):
                ...
            ValueError: Cannot provide main_fn or map_fn kwargs when using as a decorator.
            >>> manual_decorator("foo")
            Traceback (most recent call last):
                ...
            TypeError: First argument must be a function. Got <class 'str'>

            The parameter `_calling_file` is used to infer the examples directory, and can't be set manually
            in the function.

            >>> Stage.register(_calling_file="foo")
            Traceback (most recent call last):
                ...
            ValueError: Cannot provide keyword arguments that are also automatically inferred. Got
            {'_calling_file'}

            This is because this function sets the calling file to the location in the code where the
            decorator is called, and from that the stage will try to infer the "stage directory" and the
            associated default configuration file path and examples directory, if possible. We can demonstrate
            how this works by patching out the inspect module to return a file path that we'll create here:

            >>> config = DictConfig({"arg1": {"option1": "foo"}, "arg2": [1, 2.3, None], "arg3": None})
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     # Make the stage directory:
            ...     stage_dir = Path(tmpdir) / "stage_foo"
            ...     stage_dir.mkdir()
            ...     # Make the calling file:
            ...     calling_file = stage_dir / "stage_foo.py"
            ...     calling_file.touch()
            ...     # Make the examples directory:
            ...     examples_dir = stage_dir / "examples"
            ...     # We'll make two example cases:
            ...     example_1_dir = examples_dir / "example_1"
            ...     example_1_dir.mkdir(parents=True)
            ...     example_1_fp = example_1_dir / "out_data.yaml"
            ...     _ = example_1_fp.write_text("data/0.parquet: 'code,time,subject_id,numeric_value'")
            ...     example_2_dir = examples_dir / "example_2"
            ...     example_2_dir.mkdir(parents=True)
            ...     example_2_fp = example_2_dir / "out_data.yaml"
            ...     _ = example_2_fp.write_text("data/1.parquet: 'code,time,subject_id,numeric_value'")
            ...     # Save the config file:
            ...     default_config_file = stage_dir / "config.yaml"
            ...     OmegaConf.save(config, default_config_file)
            ...     # What does the directory structure look like?
            ...     print("Root directory:")
            ...     print("-----------------")
            ...     print_directory_contents(tmpdir)
            ...     # Mock out the inspect module to return the calling file we're constructing:
            ...     with patch("inspect.currentframe") as mock:
            ...         mock.return_value.f_back.f_code.co_filename = str(calling_file)
            ...         # Now we can create the stage:
            ...         @Stage.register(stage_name="stage_foo", is_metadata=True)
            ...         def main(cfg: DictConfig):
            ...             '''base main docstring'''
            ...             return "main"
            ...         # Print the stage object and see if it has set the examples directory and default
            ...         print("-----------------")
            ...         print("Stage object:")
            ...         print("-----------------")
            ...         print(main)
            ...         # Check the test cases, which aren't loaded with the stage by default as they aren't
            ...         # necessary during normal runtime:
            ...         print("-----------------")
            ...         print("Test cases:")
            ...         print("-----------------")
            ...         print(main.test_cases)
            Root directory:
            -----------------
            └── stage_foo
                ├── config.yaml
                ├── examples
                │   ├── example_1
                │   │   └── out_data.yaml
                │   └── example_2
                │       └── out_data.yaml
                └── stage_foo.py
            -----------------
            Stage object:
            -----------------
            Stage stage_foo:
              Type: main
              is_metadata: True
              Docstring:
                | base main docstring
              Default config:
                | arg1:
                |   option1: foo
                | arg2:
                | - 1
                | - 2.3
                | - null
                | arg3: null
              Map function: None
              Reduce function: None
              Main function: main
              Mimic function: main
            -----------------
            Test cases:
            -----------------
            example_2:
            │   StageExample [stage_foo/example_2]
            │     stage_cfg: {}
            │     want_data:
            │       MEDSDataset:
            │       dataset_metadata:
            │       data_shards:
            │         - 1:
            │           pyarrow.Table
            │           subject_id: int64
            │           time: timestamp[us]
            │           code: string
            │           numeric_value: float
            │           ----
            │           subject_id: [[]]
            │           time: [[]]
            │           code: []
            │           numeric_value: [[]]
            │       code_metadata:
            │         pyarrow.Table
            │         code: string
            │         description: string
            │         parent_codes: list<item: string>
            │           child 0, item: string
            │         ----
            │         code: []
            │         description: []
            │         parent_codes: []
            │       subject_splits: None
            example_1:
            │   StageExample [stage_foo/example_1]
            │     stage_cfg: {}
            │     want_data:
            │       MEDSDataset:
            │       dataset_metadata:
            │       data_shards:
            │         - 0:
            │           pyarrow.Table
            │           subject_id: int64
            │           time: timestamp[us]
            │           code: string
            │           numeric_value: float
            │           ----
            │           subject_id: [[]]
            │           time: [[]]
            │           code: []
            │           numeric_value: [[]]
            │       code_metadata:
            │         pyarrow.Table
            │         code: string
            │         description: string
            │         parent_codes: list<item: string>
            │           child 0, item: string
            │         ----
            │         code: []
            │         description: []
            │         parent_codes: []
            │       subject_splits: None

            If the example set-up is the same, but the stage name doesn't agree with the parent directory of
            the calling file, then we get a normal stage with none of the extra information as the examples
            directory and config.yaml file are assumed to be unrelated:

            >>> config = DictConfig({"arg1": {"option1": "foo"}, "arg2": [1, 2.3, None], "arg3": None})
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     # Make the stage directory:
            ...     stage_dir = Path(tmpdir) / "stage_foo"
            ...     stage_dir.mkdir()
            ...     # Make the calling file:
            ...     calling_file = stage_dir / "stage_foo.py"
            ...     calling_file.touch()
            ...     # Make the examples directory:
            ...     examples_dir = stage_dir / "examples"
            ...     # We'll make two example cases:
            ...     example_1_dir = examples_dir / "example_1"
            ...     example_1_dir.mkdir(parents=True)
            ...     example_1_fp = example_1_dir / "out_data.yaml"
            ...     _ = example_1_fp.write_text("data/0.parquet: 'code,time,subject_id,numeric_value'")
            ...     example_2_dir = examples_dir / "example_2"
            ...     example_2_dir.mkdir(parents=True)
            ...     example_2_fp = example_2_dir / "out_data.yaml"
            ...     _ = example_2_fp.write_text("data/1.parquet: 'code,time,subject_id,numeric_value'")
            ...     # Save the config file:
            ...     default_config_file = stage_dir / "config.yaml"
            ...     OmegaConf.save(config, default_config_file)
            ...     # What does the directory structure look like?
            ...     print("Root directory:")
            ...     print("-----------------")
            ...     print_directory_contents(tmpdir)
            ...     # Mock out the inspect module to return the calling file we're constructing:
            ...     with patch("inspect.currentframe") as mock:
            ...         mock.return_value.f_back.f_code.co_filename = str(calling_file)
            ...         # Now we can create the stage:
            ...         @Stage.register(stage_name="not_stage_foo", is_metadata=True)
            ...         def main(cfg: DictConfig):
            ...             '''base main docstring'''
            ...             return "main"
            ...         # Print the stage object and see if it has set the examples directory and default
            ...         print("-----------------")
            ...         print("Stage object:")
            ...         print("-----------------")
            ...         print(main)
            ...         # Check the test cases, which aren't loaded with the stage by default as they aren't
            ...         # necessary during normal runtime:
            ...         print("-----------------")
            ...         print("Test cases:")
            ...         print("-----------------")
            ...         print(main.test_cases)
            Root directory:
            -----------------
            └── stage_foo
                ├── config.yaml
                ├── examples
                │   ├── example_1
                │   │   └── out_data.yaml
                │   └── example_2
                │       └── out_data.yaml
                └── stage_foo.py
            -----------------
            Stage object:
            -----------------
            Stage not_stage_foo:
              Type: main
              is_metadata: True
              Docstring:
                | base main docstring
              Map function: None
              Reduce function: None
              Main function: main
              Mimic function: main
            -----------------
            Test cases:
            -----------------
            {}

            What about those warnings? To see the warnings in this test, we'll explicitly capture and print
            logged warnings. These will normally only be visible in the logger output, but we'll capture and
            print them here using the `print_warnings` context manager, defined in our `conftest.py` file.

            >>> with print_warnings():
            ...     @Stage.register(is_metadata=True)
            ...     def main(cfg: DictConfig):
            ...         '''base main docstring'''
            ...         return "main"
            Warning: Stage 'base' is not registered in the entry points. This may be due to a missing or
            incorrectly configured entry point in your setup.py or pyproject.toml file. If this is during
            development, you may need to run `pip install -e .` to install your package properly in editable
            mode and ensure your stage registration is detected. You can disable this warning by setting the
            class variable `WARN_IF_NO_ENTRY_POINT_AT_NAME` to `False`, or filtering out
            `StageRegistrationWarning` warnings. You can disable all validation by setting the environment
            variable `DISABLE_STAGE_VALIDATION` to `1`.
        """

        # Get the frame of the caller
        caller_frame = inspect.currentframe().f_back
        # Get the filename from the frame
        calling_file = Path(caller_frame.f_code.co_filename)
        if not calling_file.is_file():  # pragma: no cover
            # In this case, something is wrong and we can't infer what the file is, so we omit it. This mostly
            # comes up in test cases, not real usage.
            calling_file = None

        inferred_kwargs = {"_calling_file": calling_file}

        if set(inferred_kwargs.keys()).intersection(kwargs.keys()):
            raise ValueError(
                "Cannot provide keyword arguments that are also automatically inferred. "
                f"Got {set(inferred_kwargs).intersection(kwargs.keys())}"
            )

        def decorator(fn: Callable, **kwargs):
            if not inspect.isfunction(fn):
                raise TypeError(f"First argument must be a function. Got {type(fn)}")

            if "main_fn" in kwargs or "map_fn" in kwargs:
                raise ValueError("Cannot provide main_fn or map_fn kwargs when using as a decorator.")

            stage_kwargs = {**kwargs}

            if (fn.__name__ == "main") and ("reduce_fn" not in kwargs):
                stage_kwargs["main_fn"] = fn
            else:
                stage_kwargs["map_fn"] = fn

            stage = Stage(**stage_kwargs)
            stage.mimic_fn = fn
            return stage

        if len(args) == 0:
            if "main_fn" in kwargs or "map_fn" in kwargs:
                return Stage(**kwargs, **inferred_kwargs)
            return partial(decorator, **kwargs, **inferred_kwargs)
        elif len(args) == 1:
            if kwargs:
                raise ValueError("Cannot provide keyword arguments when using as a decorator.")
            return decorator(args[0], **inferred_kwargs)
        else:
            raise ValueError(
                f"Stage.register can only be used with at most a single positional arg. Got {len(args)}"
            )
