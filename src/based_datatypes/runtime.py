"""Utils for reflecting the Python runtime."""

import inspect
from collections.abc import Callable, Sequence
from types import ModuleType, UnionType
from typing import Any, TypeVar

from .types import SingleTypeDef

T = TypeVar("T")


def _get_calling_frame(offset: int = 0):
    stack = inspect.stack()
    if len(stack) < offset + 3:
        raise RuntimeError("No caller!")
    return stack[offset + 2]


def get_calling_module(offset: int = 0) -> ModuleType | None:
    """Return the name of the module calling the current function."""
    return inspect.getmodule(_get_calling_frame(offset).frame)


def get_full_args_dict(
    func: Callable, args: Sequence, kwargs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return dict of all args + kwargs.

    Args:
        func: Function to inspect.
        args: Positional arguments given to the function.
        kwargs: Keyword arguments given to the function.

    Returns:
        Dictionary of all args + kwargs.
    """
    argspec = inspect.getfullargspec(func)

    posonlyargs = list(getattr(argspec, "posonlyargs", ()))
    posarg_names = [*posonlyargs, *argspec.args]

    arg_defaults = argspec.defaults or []
    defaulted_names = posarg_names[len(posarg_names) - len(arg_defaults) :]
    defaults = dict(zip(defaulted_names, arg_defaults))
    defaults |= dict(argspec.kwonlydefaults or {})

    # Handle bound methods
    if len(posarg_names) > 0 and posarg_names[0] == "self" and hasattr(func, "__self__"):
        posarg_names = posarg_names[1:]

    posargs = dict(zip(posarg_names[: len(args)], args))

    return {**defaults, **posargs, **(kwargs or {})}


def get_return_type(func: Callable) -> SingleTypeDef | UnionType | None:
    """Get the return type annotation of given function, if any."""
    sig = inspect.signature(func)
    return sig.return_annotation if sig.return_annotation != inspect.Signature.empty else None


def get_subclasses[T](cls: type[T], max_level: int | None = None, _level: int = 1) -> list[type[T]]:
    """Return all subclasses of given class.

    Args:
        cls: The class to get subclasses for.
        max_level: Maximum inheritance level to traverse.
        _level: Current inheritance level (for internal use).

    Returns:
        List of subclasses.
    """
    if max_level is not None and _level > max_level:
        return []

    subclasses = list(dict.fromkeys(cls.__subclasses__()))

    for subclass in cls.__subclasses__():
        for sub_subclass in get_subclasses(subclass, max_level=max_level, _level=_level + 1):
            if sub_subclass not in subclasses:
                subclasses.append(sub_subclass)

    return subclasses
