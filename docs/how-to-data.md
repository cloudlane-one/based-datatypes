# Dataclass and container utilities

`based_datatypes.data` turns dataclasses into plain mappings, flattens nested mappings, and
re-constructs dataclasses while overriding a few fields. It is the plumbing behind config
merging, change detection and "copy this record with one value changed" flows.

## When to use it

- `copy_and_override`: build a modified copy of an immutable or frozen dataclass.
- `container_to_dict`: feed a dataclass or pydantic model to code that expects a dict.
- `flatten_dict`: turn nested configuration into flat `tuple` keys for diffing or logging.

## Quick start

```python exec="on" session="data"
from dataclasses import dataclass, field

from based_datatypes.data import copy_and_override


@dataclass
class Server:
    """Connection settings."""

    host: str
    port: int = 8080
    tags: list[str] = field(default_factory=list)


original = Server("api.example.com", tags=["prod"])
updated = copy_and_override(Server, original, port=9090)

print(original)
print(updated)
print(original.tags is updated.tags)
```

## Re-constructing a dataclass

`copy_and_override(init, obj, *args, **kwargs)` reads every field the source defines,
applies the positional and keyword overrides, and calls the constructor. Sources may be a
single instance or a tuple of instances; later sources win.

```python exec="on" session="data"
from dataclasses import dataclass

from based_datatypes.data import copy_and_override


@dataclass
class Point:
    """A point in the plane."""

    x: int
    y: int


print(copy_and_override(Point, Point(1, 2), 9))
print(copy_and_override(Point, (Point(1, 2), Point(5, 6))))
```

Fields that are not part of the constructor (`init=False`, e.g. derived values) are copied
after construction:

```python exec="on" session="data"
from dataclasses import dataclass, field

from based_datatypes.data import copy_and_override


@dataclass
class Doubled:
    """Stores a value and its double."""

    value: int
    double: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.double = self.value * 2


print(copy_and_override(Doubled, Doubled(3)))
```

### Generic constructors and `MaskedInit`

When the constructor carries type parameters, `MaskedInit` pairs the constructor's *type*
with the runtime class, so the call site stays typed:

```python exec="on" session="data"
from dataclasses import dataclass

from based_datatypes.data import MaskedInit, copy_and_override


@dataclass
class Box[T]:
    """Holds one value."""

    item: T


mask = MaskedInit(Box, Box)
print(copy_and_override(mask, Box(1), item=2))
```

!!! warning "Known limitations"
    Keyword-only dataclasses and `InitVar` fields are not supported: the source values are
    passed positionally. Required fields missing from every source raise, rather than being
    silently defaulted.

## Converting containers to dicts

`container_to_dict` recognises dataclass instances, pydantic models, mappings, sequences
and sets. Every other object yields `None`.

```python exec="on" session="data"
from dataclasses import dataclass

import pydantic

from based_datatypes.data import container_to_dict


@dataclass
class User:
    """A user record."""

    name: str
    roles: list[str]


print(container_to_dict(User("ada", ["admin"])))
print(container_to_dict([10, 20]))
print(container_to_dict("not a container"))
```

Dataclass keys become `Attr` instances, so they cannot collide with the string keys of a
mapping that was converted inside the same structure. Recursion is on by default; pass
`recursive=False` to keep nested objects untouched.

```python exec="on" session="data"
from dataclasses import dataclass

from based_datatypes.data import container_to_dict


@dataclass
class Inner:
    """Leaf record."""

    value: int


@dataclass
class Outer:
    """Wrapper record."""

    inner: Inner


nested = Outer(Inner(1))
print(container_to_dict(nested))
print(container_to_dict(nested, recursive=False))
```

Self-referencing containers terminate: the recursive branch maps back to the result object
already built for that container.

```python exec="on" session="data"
from based_datatypes.data import container_to_dict

cyclic: dict[str, object] = {}
cyclic["self"] = cyclic

converted = container_to_dict(cyclic)
print(converted is not None and converted["self"] is converted)
```

## Flattening nested mappings

`flatten_dict` walks mappings only; sequences are values, not paths. The result maps a tuple
of keys to the leaf value.

```python exec="on" session="data"
from based_datatypes.data import flatten_dict

config = {"server": {"host": "localhost", "ports": [80, 443]}, "debug": True}
print(flatten_dict(config))
```

!!! note "Cycle handling"
    A self-reference resolves to the partially-built result for that mapping, so it
    contributes no extra leaf. The output therefore omits the recursive key entirely rather
    than emitting an empty path.
