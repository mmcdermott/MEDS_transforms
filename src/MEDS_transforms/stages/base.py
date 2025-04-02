"""Functions for registering and defining MEDS-transforms stages."""

from __future__ import annotations

import inspect
import logging
import os
import textwrap
import warnings
from collections.abc import Callable
from enum import StrEnum
from functools import partial, wraps
from importlib.metadata import EntryPoint
from typing import ClassVar

from omegaconf import DictConfig

from ..mapreduce import ANY_COMPUTE_FN_T, map_stage, mapreduce_stage
from .discovery import get_all_registered_stages

logger = logging.getLogger(__name__)

MAIN_FN_T = Callable[[DictConfig], None]


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


class StageRegistrationWarning(Warning):
    pass


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
    global stage docstring, and the names of the main, map, reduce, and mimic functions, if set.

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
    """

    stage_type: StageType
    stage_name: str

    map_fn: ANY_COMPUTE_FN_T | None = None
    reduce_fn: ANY_COMPUTE_FN_T | None = None
    main_fn: MAIN_FN_T | None = None

    __mimic_fn: Callable | None = None
    __stage_docstring: str | None = None
    __stage_name: str | None = None

    WARN_IF_NO_ENTRY_POINT_AT_NAME: ClassVar[bool] = os.environ.get("DISABLE_STAGE_VALIDATION", "0") != "1"
    ERR_IF_ENTRY_POINT_IMPORTABLE: ClassVar[bool] = os.environ.get("DISABLE_STAGE_VALIDATION", "0") != "1"

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

    def __init__(
        self,
        *,
        main_fn: MAIN_FN_T | None = None,
        map_fn: ANY_COMPUTE_FN_T | None = None,
        reduce_fn: ANY_COMPUTE_FN_T | None = None,
        stage_name: str | None = None,
        stage_docstring: str | None = None,
    ) -> MAIN_FN_T:
        """Wraps or returns a function that can serve as the main function for a stage."""

        self.stage_type = StageType.from_fns(main_fn, map_fn, reduce_fn)
        self.stage_name = stage_name
        self.stage_docstring = stage_docstring

        self.main_fn = main_fn
        self.map_fn = map_fn
        self.reduce_fn = reduce_fn

        do_skip_validation = os.environ.get("DISABLE_STAGE_VALIDATION", "0") == "1"

        if do_skip_validation:
            logger.debug(
                "Skipping stage validation at constructor time due to DISABLE_STAGE_VALIDATION environment "
                "variable being set. This is normal during execution of a stage via the MEDS-Transforms CLI, "
                "as validation happens manually in the main function in that context, but is typically not "
                "normal during testing, for example."
            )
        else:
            self.__validate_stage_entry_point_registration()

    def __validate_stage_entry_point_registration(
        self,
        stage_name: str | None = None,  # For stage name inference.
        registered_stages: dict[str, EntryPoint] | None = None,  # For dependency injection.
    ):
        """Validates that the stage is registered in the entry points."""

        if not (self.WARN_IF_NO_ENTRY_POINT_AT_NAME or self.ERR_IF_ENTRY_POINT_IMPORTABLE):
            return

        if registered_stages is None:
            registered_stages = get_all_registered_stages()
        if stage_name is None:
            stage_name = self.stage_name

        if stage_name not in registered_stages:
            if self.WARN_IF_NO_ENTRY_POINT_AT_NAME:
                # If the stage is not registered, we warn.
                warnings.warn(
                    f"Stage '{stage_name}' is not registered in the entry points.\n"
                    f"{self.ENTRY_POINT_SETUP_STRING}\n{self.DISABLE_WARNING_STRING}\n"
                    f"{self.DISABLE_ALL_STAGE_VALIDATION_STRING}",
                    category=StageRegistrationWarning,
                )
            return

        if not self.ERR_IF_ENTRY_POINT_IMPORTABLE:
            return

        old_env_val = os.environ.get("DISABLE_STAGE_VALIDATION", "0")
        try:
            # Temporarily disable warnings to avoid circular imports.
            os.environ["DISABLE_STAGE_VALIDATION"] = "1"

            # Attempt to reload, which should cause an error if the stage being constructed is the same stage
            # as the one being registered.
            registered_stages[stage_name].load()
            raise StageRegistrationError(
                f"Stage {stage_name} is registered in the entry points, but an attempted reload causes "
                "no issues. If this were the stage you are constructing, a reload would cause a circular "
                "import error, so this means you are overwriting an external, different stage, which is a "
                "problem!\n"
                f"{self.ENTRY_POINT_SETUP_STRING}\n{self.DISABLE_ERROR_STRING}\n"
                f"{self.DISABLE_ALL_STAGE_VALIDATION_STRING}"
            )
        except AttributeError as e:
            if "circular import" not in str(e):
                raise ValueError(
                    f"Failed to validate stage {stage_name} for an unexpected reason; it is possible that "
                    "an upstream stage defined with the same name is invalid."
                ) from e
        finally:
            os.environ["DISABLE_STAGE_VALIDATION"] = old_env_val

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
                    return self.main_fn(cfg)

            case StageType.MAP:

                @wraps(self.map_fn)
                def main_fn(cfg: DictConfig):
                    return map_stage(cfg, self.map_fn)

            case StageType.MAPREDUCE:

                def main_fn(cfg: DictConfig):
                    return mapreduce_stage(cfg, self.map_fn, self.reduce_fn)

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
            >>> def main(cfg: DictConfig):
            ...     '''base main docstring'''
            ...     return "main"
            >>> def baz_fn(foo: str, bar: int):
            ...     '''base baz docstring'''
            ...     return f"baz {foo} {bar}"
            >>> with warnings.catch_warnings(): # We catch a warning to avoid issues with stage registration
            ...     warnings.simplefilter("ignore")
            ...     stage = Stage.register(main_fn=main)
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
            "  Docstring:",
        ]

        pretty_wrap = partial(textwrap.wrap, width=110, initial_indent="    | ", subsequent_indent="    | ")

        docstring_lines = textwrap.dedent(self.stage_docstring).splitlines()
        for line in docstring_lines:
            lines.extend(pretty_wrap(line))

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
            **kwargs: Keyword arguments. These can include `main_fn`, `map_fn`, `reduce_fn`,
                `stage_name`, and `stage_docstring`. Other keyword arguments will cause an error. Not all
                keyword arguments are required for all usages of the decorator.

        Returns:
            Either a `Stage` object or a decorator function, depending on the arguments provided.

        Raises:
            ValueError: If multiple positional arguments are passed or if the function is not callable.
            TypeError: If the first positional argument is not a function.
            ValueError: If both `main_fn` and `map_fn` or `reduce_fn` are provided.

        Examples:

            Firstly, note that in normal usage, the Stage class raises a warning if the stage is not
            registered properly in an entry point. We'll disable that and other error checking for most of the
            doctests here, then show it again at the end.

            >>> Stage.WARN_IF_NO_ENTRY_POINT_AT_NAME = False
            >>> Stage.ERR_IF_ENTRY_POINT_IMPORTABLE = False

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
            >>> stage = Stage.register(main_fn=main)
            >>> print(stage) # The name is inferred from the name of the file:
            Stage base:
              Type: main
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

            >>> print(Stage.register(map_fn=map_fn))
            Stage base:
              Type: map
              Docstring:
                | base map docstring
              Map function: map_fn
              Reduce function: None
              Main function: None
              Mimic function: None
            >>> print(Stage.register(map_fn=map_fn, reduce_fn=reduce_fn))
            Stage base:
              Type: mapreduce
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

            You can also use it as a decorator:

            >>> @Stage.register
            ... def main(cfg: DictConfig):
            ...     '''base main docstring'''
            ...     return "main"

            The output of the decorator, saved here to the variable `main`, will "mimic" the decorated
            function under normal usage:

            >>> main.__name__
            'main'
            >>> main.__doc__
            'base main docstring'
            >>> main({})
            'main'

            ... but it is actually a stage object defined by the decorator:

            >>> print(main)
            Stage base:
              Type: main
              Docstring:
                | base main docstring
              Map function: None
              Reduce function: None
              Main function: main
              Mimic function: main

            When used as a decorator, you can also parametrize the decorator:

            >>> @Stage.register(stage_name="foo", stage_docstring="bar")
            ... def main(cfg: DictConfig):
            ...     '''base main docstring'''
            ...     return "main"
            >>> print(main)
            Stage foo:
              Type: main
              Docstring:
                | bar
              Map function: None
              Reduce function: None
              Main function: main
              Mimic function: main

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

            What about those warnings?

            >>> Stage.WARN_IF_NO_ENTRY_POINT_AT_NAME = True

            To see the warnings in this test, we'll explicitly tell the warnings module to raise errors if a
            warning is thrown, then re-run something successful before:

            >>> with warnings.catch_warnings():
            ...     warnings.simplefilter("error")
            ...     @Stage.register
            ...     def main(cfg: DictConfig):
            ...         '''base main docstring'''
            ...         return "main"
            Traceback (most recent call last):
                ...
            MEDS_transforms.stages.base.StageRegistrationWarning: Stage 'base' is not registered in the entry
            points. This may be due to a missing or incorrectly configured entry point in your setup.py or
            pyproject.toml file. If this is during development, you may need to run `pip install -e .` to
            install your package properly in editable mode and ensure your stage registration is detected.
            You can disable this warning by setting the class variable `WARN_IF_NO_ENTRY_POINT_AT_NAME` to
            `False`, or filtering out `StageRegistrationWarning` warnings.
            You can disable all validation by setting the environment variable `DISABLE_STAGE_VALIDATION` to
            `1`.
        """

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
                return Stage(**kwargs)
            return partial(decorator, **kwargs)
        elif len(args) == 1:
            if kwargs:
                raise ValueError("Cannot provide keyword arguments when using as a decorator.")
            return decorator(args[0])
        else:
            raise ValueError(
                f"Stage.register can only be used with at most a single positional arg. Got {len(args)}"
            )
