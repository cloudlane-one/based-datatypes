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

print(reference.typeform)
print(reference.origin, reference.args)
print(has_type([1, 2], list[int]), has_type(["a"], list[int]))
```

## API overview

| Name | Purpose |
| --- | --- |
| `TypeRef(hint, ...)` | Resolved reference to a hint: origin, arguments, parameters, typeform, type-variable map, validation. |
| `TypeAwareClass` | Base class whose subclasses remember the arguments they were parameterised with. |
| `typedef_to_typeset(typedef)` | The set of concrete classes a hint can produce (`int \| None` → `{int, NoneType}`). |
| `is_subtype(type_, supertype)` | Subtyping test over generic and special-form hints. |
| `has_type(obj, type_)` | Runtime value check driven by a hint (beartype). |
| `is_type_hint(obj)` | Whether an object describes a type rather than being a value of one. |
| `get_lowest_common_base(types)` | Most derived shared base of several classes. |
| `extract_nullable_type(hint)` | Non-`None` base of an optional hint. |
| `get_inheritance_distance(cls, base)` | Signed number of steps between two classes. |
| `set_typeargs(typedef, args)` | Replace type variables positionally or by name. |
| `get_typeargs(instance)` | Type arguments recorded on a generic instance. |
| `is_frozen_dataclass` / `is_immutable` | Dataclass and immutability predicates. |
| `get_nondefault_methods` / `get_own_attrs` | Attribute introspection without inherited noise. |
| `AnyTypeHint` | The union of hint expressions the helpers accept. |

## From a hint to concrete classes

`typedef_to_typeset` flattens unions, resolves named aliases and reduces parameterised
generics to their origin. Literals, type variables and forward references carry no concrete
class and yield `object`.

```python exec="on" session="typing"
from typing import Literal

from based_datatypes.typing import NoneType, typedef_to_typeset

type Identifier = int

print(typedef_to_typeset(int | None))
print(typedef_to_typeset(Identifier))
print(typedef_to_typeset(list[int]))
print(typedef_to_typeset(Literal["a", "b"]))
print(typedef_to_typeset(int | None, remove_null=True) == {int})
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

print(TypeRef("int").typedef)
print(TypeRef(Names).origin)
print(TypeRef(Names).typeform is Names)

reference = TypeRef(dict[str, int])
print(reference.params, reference.args)
print("args:", [arg.typedef for arg in reference.arg_typerefs])
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

`TypeAwareClass` gives a class a runtime record of how it was parameterised. `Sub[int]` is a
real, cached subclass that stores `__origin__` and `__args__`, so the type arguments can be
recovered later without rebuilding the generic machinery.

```python exec="on" session="typing"
from based_datatypes.typing import TypeAwareClass, TypeRef


class Repository[T](TypeAwareClass):
    """Generic repository base."""


class UserRepository[T](Repository[T]):
    """Specialised repository."""


print(UserRepository[int].__origin__.__name__, UserRepository[int].__args__)
print(UserRepository[int] is UserRepository[int])

typevar_map = TypeRef(UserRepository[int]).typevar_map
print("resolved:", sorted({ref.typeform.__name__ for ref in typevar_map.values()}))
```

The class-level `typeargs` property is the same map without constructing a `TypeRef`
yourself, and `type_params` reports the declaration's free parameters:

```python exec="on" session="typing"
from based_datatypes.typing import TypeAwareClass


class Cache[K, V](TypeAwareClass):
    """Two-parameter generic."""


print(len(Cache.type_params))
print(len(Cache[str, int].typeargs))
print(Cache[str, int].type_params)
```

## Replacing type variables

`set_typeargs` takes either a sequence (positional replacement) or a mapping from type
variable to replacement. It works for builtin generics and for `TypeAwareClass` subclasses.

```python exec="on" session="typing"
from typing import TypeVar

from based_datatypes.typing import TypeAwareClass, set_typeargs

T = TypeVar("T")


class Box[T](TypeAwareClass):
    """Single-parameter generic."""


print(set_typeargs(list[T], [int]))
print(set_typeargs(Box[T], [str]))
print(set_typeargs(Box[T], {T: float}))
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
