# Reflecting the running interpreter

`based_datatypes.runtime` answers four questions about live Python objects: which module
called me, which arguments did the caller pass, what does this function return, and which
classes derive from this one.

## When to use it

These helpers are glue for decorators, dependency-injection containers, plugin registries
and diagnostics. They inspect the running process; none of them imports anything.

## Quick start

```python exec="on" session="runtime"
from based_datatypes.runtime import get_subclasses


class Plugin:
    """Base class for registerable plugins."""


class AudioPlugin(Plugin):
    """Plays audio."""


class VideoPlugin(Plugin):
    """Plays video."""


print("plugins:", [cls.__name__ for cls in get_subclasses(Plugin)])
```

## API

| Name | Purpose |
| --- | --- |
| `get_calling_module(offset=0)` | Module that called the current function; `offset` walks further up the stack. |
| `get_full_args_dict(func, args, kwargs=None)` | Parameter-name → value mapping for a call, including defaults. |
| `get_return_type(func)` | The `return` annotation, or `None` when absent. |
| `get_subclasses(cls, max_level=None)` | All descendants, direct children first, without duplicates. |

## Which module called me

```python exec="on" session="runtime"
from based_datatypes.runtime import get_calling_module

module = get_calling_module()
print(module is not None and module.__name__ == "__main__")
```

`offset` counts frames above the direct caller, which is useful inside a wrapper:

```python exec="on" session="runtime"
from based_datatypes.runtime import get_calling_module


def wrapper() -> str | None:
    """Report the module that called the wrapper."""
    module = get_calling_module(1)
    return None if module is None else module.__name__


print(wrapper())
```

## Reconstructing the caller's arguments

`get_full_args_dict` maps the parameter names of `func` to the values supplied in `args` and
`kwargs`, filling in defaults for parameters that were not passed. It is what a decorator
needs to log or forward a call faithfully.

```python exec="on" session="runtime"
from based_datatypes.runtime import get_full_args_dict


def send(target: str, retries: int = 3, *, timeout: float = 5.0) -> None:
    """Pretend to send something."""


print(get_full_args_dict(send, ("queue",), {"timeout": 1.5}))
print(get_full_args_dict(send, ()))
```

!!! note "Only declared parameters"
    Variable positional (`*args`) and variable keyword (`**kwargs`) parameters are not
    expanded into the mapping; the caller's `kwargs` mapping is merged as given.
    Unset parameters without a default cannot appear, because there is no value to report.

## Return annotations

```python exec="on" session="runtime"
from based_datatypes.runtime import get_return_type


def load() -> list[str]:
    """Load names."""
    return []


print(get_return_type(load))
print(get_return_type(lambda value: value))
```

Annotations are returned unevaluated. A string annotation (`from __future__ import
annotations`) comes back as a string; resolve it with `typing.get_type_hints` if needed.

## Walking a class hierarchy

`get_subclasses` returns every descendant, not only direct children. Each class appears
once, even when it is reachable through several branches, and the direct children keep their
declaration order.

```python exec="on" session="runtime"
from based_datatypes.runtime import get_subclasses


class Node:
    """Root of a diamond."""


class Left(Node):
    """Left branch."""


class Right(Node):
    """Right branch."""


class Leaf(Left, Right):
    """Bottom of the diamond."""


print("all:", [cls.__name__ for cls in get_subclasses(Node)])
print("direct:", [cls.__name__ for cls in get_subclasses(Node, max_level=1)])
```

!!! warning "Subclasses are strong references"
    `type.__subclasses__()` keeps every subclass alive. Building classes dynamically in a
    long-running process keeps them registered for the lifetime of their base class.
