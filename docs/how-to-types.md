# Shared types, markers and protocols

`based_datatypes.types` holds the small vocabulary the rest of the package (and its
consumers) share: nominal string types, an unresolved-state enum, structural protocols,
a named alias for ordered scalars, and a couple of typing helpers.

Nothing in this module performs I/O. Most names exist to make a type annotation precise at
a call site without defining a class at that call site.

## When to use it

Reach for this module when a value needs a name (`UUID4`), when a function accepts
"anything mapping-like" (`SupportsItems`), or when type reflection needs one of the helper
protocols (`DataclassInstance`, `StatePicklable`, `RuntimeValidated`).

## Quick start

```python exec="on" session="types"
from based_datatypes.types import Attr, Not, UUID4

identifier = UUID4("2f0c8d0e-8a5b-4f0e-9a1c-1d2e3f4a5b6c")
print(type(identifier).__name__, identifier)

status = Not.parse("resolved")
print(repr(status))

print(Attr("value"))
```

## What the module holds

The names fall into four groups; their signatures and docstrings are in the
[API reference](ref-python-api.md#based_datatypes.types).

- **Nominal strings**: `UUID4`.
- **State markers**: `Not` (a `StrEnum` of `defined`, `resolved`, `handled` and `changed`),
  `Undefined` and `ContraType`.
- **Structural protocols**: `DataclassInstance`, `StatePicklable`, `ArgsPicklable`,
  `ArgsPicklableEx`, `RuntimeValidated`, `GenericAlias`, `LiteralGenericAlias`, `SupportsItems`
  and `SupportsKeysAndGetItem`.
- **Shared aliases and vocabulary**: `Ordinal`, `Attr`, `operator_methods`, and the
  type-expression unions `TypeRoot`, `TypeNode`, `TypeMember`, `TypeDef`, `TypeScalar` and
  `TypeExpr` that the `typing` helpers accept (plus `AnnotationScanType` and `LiteralArg`).

## Protocols are structural

The protocols are `runtime_checkable`, so they can be used with `isinstance` as runtime
duck-typing checks.

```python exec="on" session="types"
import dataclasses

from based_datatypes.types import DataclassInstance, SupportsItems


@dataclasses.dataclass
class Record:
    value: int


print(isinstance(Record(1), DataclassInstance))
print(isinstance(Record, DataclassInstance))
print(isinstance({"a": 1}, SupportsItems))
print(isinstance([1], SupportsItems))
```

!!! warning "Protocol checks are shallow"
    `isinstance` on a `runtime_checkable` protocol only checks that the attributes exist,
    not their signatures. Use them for dispatch and diagnostics, not for validation.

## Nominal strings do not validate

`UUID4` is a typing marker: constructing one accepts any string. Validate at the boundary
where untrusted input enters (a pydantic model, for example), not by relying on the type.

```python exec="on" session="types"
from based_datatypes.types import UUID4

print(UUID4("not-a-uuid"))
```

## Ordered scalars

`Ordinal` names the set of types with a total order. It is a plain named alias, so it can
be used directly in annotations and introspected with `typing.get_args`.

```python exec="on" session="types"
from typing import get_args

from based_datatypes.types import Ordinal

print("aliases:", sorted(t.__name__ for t in get_args(Ordinal.__value__)))
```

## Marker types

`Undefined` is a frozen dataclass, so all instances compare equal and hash alike. Use it
as a default sentinel where `None` is a meaningful value.

`ContraType` carries a type object for contravariant APIs; the wrapped type is optional.

```python exec="on" session="types"
from based_datatypes.types import ContraType, Undefined

print(Undefined() == Undefined())
print(ContraType[int](int).type_)
```

!!! note "`@final` is static only"
    `Undefined` and `ContraType` are marked `@final`; Python does not prevent subclassing
    at runtime, the type checker does.
