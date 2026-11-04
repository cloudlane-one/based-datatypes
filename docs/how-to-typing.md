# Reflecting type hints at runtime

`based_datatypes.typing` reads, normalises and compares type hints while the program runs.
It is the layer that lets a framework answer questions such as "which concrete classes can
this annotation produce?", "is this annotation a subtype of that one?", and "what are the
type arguments of this generic class?".

## When to use it

Use it when building serialisers, dependency injection, validation layers or generic base
classes that need to know what an annotation *means* — not merely whether `isinstance`
passes.

## Quick start

```python exec="on" session="typing"
from based_datatypes.typing import TypeRef, has_type

reference = TypeRef(list[int])

print(reference.resolved_type_def)
print(reference.origin, reference.args)
print(has_type([1, 2], list[int]), has_type(["a"], list[int]))
```

## API overview

| Name | Purpose |
| --- | --- |
| `TypeRef(hint, ...)` | Resolved reference to a hint: origin, arguments, parameters, union `members`, resolved type definition, type-variable map, validation. |
| `Runtyped` | Base class whose subclasses remember the arguments they were parameterised with. |
| `TypeRef(hint).typeset` | The set of concrete classes a hint can produce (`int \| None` → `{int, NoneType}`). |
| `is_subtype(type_, supertype)` | Subtyping test over generic and special-form hints. |
| `has_type(obj, type_)` | Runtime value check driven by a hint (beartype). |
| `is_type_hint(obj)` | Whether an object describes a type rather than being a value of one. |
| `get_lowest_common_base(types)` | Most derived shared base of several classes. |
| `get_inheritance_distance(cls, base)` | Signed number of steps between two classes. |
| `is_frozen_dataclass` / `is_immutable` | Dataclass and immutability predicates. |
| `get_nondefault_methods` / `get_own_attrs` | Attribute introspection without inherited noise. |
| `TypeDef` / `TypeNode` | The unions of hint expressions the helpers accept. |

## From a hint to concrete classes

`TypeRef(...).typeset` flattens unions, resolves named aliases and reduces parameterised
generics to their origin. Literals, type variables and forward references carry no concrete
class and contribute nothing.

```python exec="on" session="typing"
from typing import Literal

from based_datatypes.typing import NoneType, TypeRef

type Identifier = int

print(TypeRef(int | None).typeset)
print(TypeRef(Identifier).typeset)
print(TypeRef(list[int]).typeset)
print(TypeRef(Literal["a", "b"]).typeset)
print(TypeRef(int | None).typeset - {NoneType} == {int})
print(NoneType is type(None))
```

## Subtyping and value checks

`is_subtype` understands generics, unions, `Annotated`, `Literal` and named aliases.
`has_type` answers the value-level question.

```python exec="on" session="typing"
from typing import Annotated

from based_datatypes.typing import has_type, is_subtype

print(is_subtype(bool, int))
print(is_subtype(list[int], list[str]))
print(is_subtype(list[int], list))
print(is_subtype(Annotated[int, "meta"], int))
print(has_type(1, int), has_type("1", int))
```

## Working with `TypeRef`

`TypeRef` is the workhorse: it resolves string and forward-reference hints, substitutes type
variables, and exposes the pieces a framework needs. All properties are cached per reference.

```python exec="on" session="typing"
from typing import TypeAliasType

from based_datatypes.typing import TypeRef

type Names = list[str]

print(TypeRef("int").type_def)
print(TypeRef(Names).origin)
print(TypeRef(Names).resolved_type_def is Names)

reference = TypeRef(dict[str, int])
print(reference.params, reference.args)
print("scalar args:", reference.scalar_args)
```

Use `subs` to replace open type variables and `overrides` when a replacement must take
precedence over existing substitutions and defaults. `pure_type_def` removes `Annotated`
metadata, while `resolved_type_def` retains it and recursively resolves type arguments:

```python exec="on" session="typing"
from typing import Annotated, TypeVar

from based_datatypes.typing import TypeRef

T = TypeVar("T")
reference = TypeRef(Annotated[list[T], "items"], subs={T: int})

print(reference.pure_type_def)
print(reference.resolved_type_def)
print(reference.annotations)
```

`validate` combines the runtime check with pydantic validation when the annotation carries
it, and returns a boolean instead of raising:

```python exec="on" session="typing"
from typing import Annotated

import pydantic

from based_datatypes.typing import TypeRef

positive = TypeRef(Annotated[int, pydantic.Field(gt=0)])

print(positive.validate(3))
print(positive.validate(-1))
print(TypeRef(int).validate("not an int"))
```

!!! note "Metadata only validates when pydantic can read it"
    Extra validation runs when the annotation is a pydantic `FieldInfo` or implements the
    `RuntimeValidated` protocol. Arbitrary `Annotated` metadata is ignored, exactly as
    `isinstance` would ignore it.

## Generic classes that know their arguments

`Runtyped` gives a class a runtime record of how it was parameterised. `Sub[int]` is a
real, cached subclass that stores `__origin__` and `__args__`, so the type arguments can be
recovered later without rebuilding the generic machinery.

```python exec="on" session="typing"
from based_datatypes.typing import Runtyped, TypeRef


class Repository[T](Runtyped):
    """Generic repository base."""


class UserRepository[T](Repository[T]):
    """Specialised repository."""


print(UserRepository[int].__origin__.__name__, UserRepository[int].__args__)
print(UserRepository[int] is UserRepository[int])

typevar_map = TypeRef(UserRepository[int]).typevar_map
print("resolved:", sorted({ref.resolved_type_def.__name__ for ref in typevar_map.values()}))
```

`TypeRef` reports the parameters a declaration is generic over, and resolves the parameters of
a specialisation to the arguments it was applied with. Unapplied parameters resolve to their
defaults, bounds, or `Any`:

```python exec="on" session="typing"
from based_datatypes.typing import Runtyped, TypeRef


class Cache[K, V](Runtyped):
    """Two-parameter generic."""


print(", ".join(param.__name__ for param in TypeRef(Cache).params))
print(TypeRef(Cache[str, int]).args)
print(sorted({ref.resolved_type_def.__name__ for ref in TypeRef(Cache[str, int]).typevar_map.values()}))
```

Validation and subtyping compare these recorded arguments structurally. Thus a `Box[int]`
instance does not validate as `Box[str]`, even though a plain `isinstance` check cannot see
the difference. Trailing arguments omitted from a supertype do not constrain the comparison.

## Replacing type variables

Pass a mapping of type variables to hints as `subs`. The same substitution is applied
through nested arguments and generic bases. `overrides` is available for the less common
case where a replacement must supersede substitutions and parameter defaults.

```python exec="on" session="typing"
from typing import TypeVar

from based_datatypes.typing import Runtyped, TypeRef

T = TypeVar("T")


class Box[T](Runtyped):
    """Single-parameter generic."""


print(TypeRef(list[T], subs={T: int}).resolved_type_def)
print(TypeRef(Box[T], subs={T: str}).resolved_type_def is Box[str])
print(TypeRef(T, subs={T: str}, overrides={T: float}).resolved_type_def)
```

## Small predicates used by the package

```python exec="on" session="typing"
from dataclasses import dataclass

from based_datatypes.typing import is_frozen_dataclass, is_immutable


@dataclass(frozen=True)
class Frozen:
    """Immutable record."""

    value: int


print(is_frozen_dataclass(Frozen), is_immutable(Frozen(1)))
print(is_immutable((1, "a")), is_immutable([1]))
```

!!! warning "`get_own_attrs(predicate=...)` filters by value"
    Attribute names are selected by applying the predicate to the class-body value, so an
    annotation without a value is only reported when no predicate is given. Pass
    `include_bases=(Base,)` to merge a base's own attributes.
