"""This file defines two useful decorators for use in config management.

1. A decorator to define structured configs in Hydra's config store.
2. A decorator to annotate functions as OmegaConf resolvers in the global namespace.
"""

import dataclasses
from collections.abc import Callable
from functools import partial

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


def OmegaConfResolver(*args, **kwargs) -> Callable:  # noqa: N802
    """A decorator to register the decorated function as an OmegaConf resolver.

    If a name is passed via keyword arguments, it will be used as the name of the resolver. If no name is
    passed, the name of the function will be used. Other keyword arguments will be passed to the
    OmegaConf.register_new_resolver function.

    Args:
        args: The function to register as a resolver (if specified, no keyword arguments should be used)
        kwargs: The name of the resolver and other keyword arguments. If specified, this is a parametrized
            decorator and the function to register should not be explicitly passed. See the examples below.

    Returns:
        A decorator that registers the decorated function as an OmegaConf resolver.

    Raises:
        TypeError: If the arguments are not valid for the OmegaConfResolver decorator.

    Examples:
        >>> @OmegaConfResolver
        ... def my_resolver(x: int) -> str:
        ...     return str(x)

    After decorating, the function is registered as an OmegaConf resolver via its name, and can be used as a
    resolver in OmegaConf, but remains unchanged in the python scope.

        >>> OmegaConf.has_resolver("my_resolver")
        True
        >>> cfg = OmegaConf.create("foo: ${my_resolver:42}")
        >>> cfg.foo
        '42'
        >>> my_resolver(42)  # The function is unchanged
        '42'

    This can also be used as a parametrized decorator:

        >>> @OmegaConfResolver(name="my_resolver2")
        ... def my_resolver(x: int) -> str:
        ...     return str(x**2)
        >>> OmegaConf.has_resolver("my_resolver2")
        True
        >>> cfg = OmegaConf.create("foo: ${my_resolver2:8}")
        >>> cfg.foo
        '64'
        >>> my_resolver(8)  # The function is unchanged
        '64'

    Errors are raised if the arguments are not valid:

        >>> OmegaConfResolver("foo")
        Traceback (most recent call last):
            ...
        TypeError: Invalid arguments for OmegaConfResolver decorator. Expected either a positional function or
        keyword arguments; got ('foo',) and {}.
    """

    def decorator(func: Callable, name: str | None = None, **kwargs) -> Callable:
        if name is None:
            name = func.__name__
        OmegaConf.register_new_resolver(name, func, **kwargs)
        return func

    if len(args) == 0:
        return partial(decorator, **kwargs)
    elif len(args) == 1 and callable(args[0]) and not kwargs:
        return decorator(args[0])
    else:
        raise TypeError(
            "Invalid arguments for OmegaConfResolver decorator. Expected either a positional function or "
            f"keyword arguments; got {args} and {kwargs}."
        )


def hydra_registered_dataclass(*, group: str | None, name: str | None = None) -> Callable[[type], type]:
    """A simple decorator to define a class as a dataclass and register it with Hydra.

    This is a parametrized decorator that takes in the hydra config-store group (mandatory, must be specified
    by keyword) and name (optional, can be inferred from the decorated class). The flagged class will be
    registered in the hydra config store with the given name and made a dataclass.

    Args:
        group: The hydra config-store group to register the class with.
        name: The name to register the class with. If None, the class name will be used.

    Returns:
        A decorator that takes a class, makes it a dataclass, and registers it with Hydra.

    Examples:
        >>> @hydra_registered_dataclass(group="foo")
        ... class Bar:
        ...     baz: str = "baz"
        >>> dataclasses.is_dataclass(Bar)
        True
        >>> cs = ConfigStore.instance()
        >>> cs.repo["foo"]
        {'Bar.yaml': ConfigNode(name='Bar.yaml', node={'baz': 'baz'}, group='foo', ...)}
        >>> @hydra_registered_dataclass(group="foo", name="Bar2")
        ... class Bar:
        ...     qux: str = "quux"
        >>> cs.repo["foo"]
        {'Bar.yaml': ..., 'Bar2.yaml': ConfigNode(name='Bar2.yaml', node={'qux': 'quux'}, ...)}
    """

    def decorator(cls: type, name: str | None = None) -> type:
        if name is None:
            name = cls.__name__
        cls = dataclasses.dataclass(cls)
        cs = ConfigStore.instance()
        cs.store(group=group, name=name, node=cls)
        return cls

    return partial(decorator, name=name)
