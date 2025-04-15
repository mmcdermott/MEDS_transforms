"""This file defines an object-oriented backing for configuring stages in MEDS-Transforms pipelines."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from omegaconf import DictConfig

UNPARSED_STAGE_T = str | dict | DictConfig


@dataclasses.dataclass
class StageConfig:
    """A configuration class for a stage in the MEDS-transforms pipeline.

    This class is used to define the structure of a stage configuration options within a `PipelineConfig`. It
    is generally constructed from raw pipeline arguments through the `from_arg` classmethod, which parses a
    raw entry into structured form. See that method for more detailed examples of the accepted format.

    Attributes:
        name: The name of the stage.
        base_stage: The name of the underlying, registered, base stage (if any) for this stage.
        config: The configuration for the stage.

    Raises:
        TypeError: If any non-None parameter is not of the appropriate type.
        ValueError: If the base stage name is empty.

    Examples:
        >>> StageConfig(name="stage_name")
        StageConfig(name='stage_name', base_stage=None, config={})
        >>> StageConfig(name="stage_name", base_stage="base_stage", config={"param1": 1})
        StageConfig(name='stage_name', base_stage='base_stage', config={'param1': 1})

    `StageConfig`s also expose a property `resolved_name`, which returns the resolved name of the stage (i.e.,
    the base name if it exists, otherwise the name itself):

        >>> cfg = StageConfig(name="foo", base_stage="bar")
        >>> cfg.resolved_name
        'bar'
        >>> cfg = StageConfig(name="foo", config={"param1": 1})
        >>> cfg.resolved_name
        'foo'

    Errors are raised if types are invalid or the base stage name is empty:

        >>> StageConfig(name=3)
        Traceback (most recent call last):
            ...
        TypeError: Invalid stage name type <class 'int'>. Expected str.
        >>> StageConfig(name="foo", base_stage=3)
        Traceback (most recent call last):
            ...
        TypeError: Invalid base stage type <class 'int'>. Expected str.
        >>> StageConfig(name="foo", base_stage="")
        Traceback (most recent call last):
            ...
        ValueError: If specified, base stage name cannot be empty.
        >>> StageConfig(name="foo", config=3)
        Traceback (most recent call last):
            ...
        TypeError: Invalid config type <class 'int'>. Expected dict or DictConfig.
    """

    name: str
    base_stage: str | None = None
    config: dict | DictConfig = dataclasses.field(default_factory=dict)

    META_KEYS: ClassVar[set[str]] = {
        "_base_stage",
    }

    @classmethod
    def _is_meta_key(cls, key: str) -> bool:
        """Checks if a key is a meta-key for the configuration.

        Meta-keys are keys that are in the `cls.META_KEYS` set.

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
        if any(not isinstance(k, str) for k in arg):
            non_str_keys = [k for k in arg if not isinstance(k, str)]
            err_strs = [f"  - {type(key).__name__}: {key}" for key in non_str_keys]
            err_str = "\n".join(err_strs)
            return f"All keys must be strings. Got key(s):\n{err_str}"

        meta_keys, non_meta_keys = cls._split_meta_keys(arg)
        if len(non_meta_keys) != 1:
            return (
                "Expected a single non-meta key-value pair representing the stage name and configuration. "
                f"Got {len(non_meta_keys)} non-meta keys: {', '.join(non_meta_keys.keys())}."
            )

        stage_config = next(iter(non_meta_keys.values()))

        meta_key_duplicates = [k for k in meta_keys if k in stage_config]
        if meta_key_duplicates:
            return (
                "You cannot specify meta keys in both the raw and stage configuration dictionaries. "
                f"Got duplicate meta keys: {', '.join(meta_key_duplicates)}."
            )
        return None

    @classmethod
    def from_arg(cls, arg: UNPARSED_STAGE_T) -> StageConfig:
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
            StageConfig(name='stage_name', base_stage=None, config={})
            >>> StageConfig.from_arg({"stage1": {"_base_stage": "foo"}})
            StageConfig(name='stage1', base_stage='foo', config={})
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
                if error_str := cls._dict_arg_error_str(config):
                    raise ValueError(f"Invalid stage config: {error_str}")

                meta_keys, non_meta_keys = cls._split_meta_keys(config)
                stage_name = next(iter(non_meta_keys.keys()))
                stage_config = non_meta_keys[stage_name]

                meta_keys = {k[1:]: v for k, v in meta_keys.items()}
                for k in cls.META_KEYS:
                    if k in stage_config:
                        meta_keys[k[1:]] = stage_config.pop(k)

                return cls(name=stage_name, config=stage_config, **meta_keys)
            case _:
                raise TypeError(f"Invalid argument type {type(arg)}. Expected str, dict, or DictConfig.")

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError(f"Invalid stage name type {type(self.name)}. Expected str.")
        if self.base_stage is not None:
            if not isinstance(self.base_stage, str):
                raise TypeError(f"Invalid base stage type {type(self.base_stage)}. Expected str.")
            if not self.base_stage:
                raise ValueError("If specified, base stage name cannot be empty.")

        if not isinstance(self.config, dict | DictConfig):
            raise TypeError(f"Invalid config type {type(self.config)}. Expected dict or DictConfig.")

    @property
    def resolved_name(self) -> str:
        """Return the resolved (base_name, if it exists, otherwise name) of the stage configuration."""
        return self.base_stage if self.base_stage is not None else self.name
