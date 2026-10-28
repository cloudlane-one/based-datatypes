"""Reflection utilities for types."""

from __future__ import annotations

import inspect
import operator
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from functools import cached_property, reduce
from inspect import getmodule, getmro
from itertools import chain, groupby
from types import ModuleType, NoneType, UnionType, new_class
from typing import (
    Annotated,
    Any,
    ForwardRef,
    Generic,
    Literal,
    NewType,
    Self,
    TypeAliasType,
    TypeGuard,
    TypeVar,
    TypeVarTuple,
    Union,
    cast,
    get_args,
    get_origin,
)
from typing import TypeVar as ExtTypeVar
from typing import TypeVarTuple as ExtTypeVarTuple

from beartype.door import is_bearable, is_subhint
from beartype.roar import BeartypeDoorNonpepException
from pydantic import TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from typing_extensions import TypeForm

from .descriptors import prop
from .runtime import get_subclasses
from .types import (
    AnyTypeHint,
    GenericAlias,
    RuntimeValidated,
    SingleTypeDef,
)

# Public since Python 3.14; ``ForwardRef._evaluate`` is private and deprecated.
_evaluate_forward_ref = getattr(typing, "evaluate_forward_ref", None)


def _resolve_typealias(
    type_: SingleTypeDef | UnionType | Annotated,
) -> SingleTypeDef | UnionType:
    return (
        type_.__value__
        if isinstance(type_, TypeAliasType)
        else get_args(type_)[0]
        if get_origin(type_) is Annotated
        else type_
    )


def typedef_to_typeset(  # noqa: C901
    typedef: AnyTypeHint | None,
    typevar_map: (
        dict[
            TypeVar | ExtTypeVar | TypeVarTuple | ExtTypeVarTuple,
            TypeRef | tuple[TypeRef, ...],
        ]
        | None
    ) = None,
    ctx_module: ModuleType | None = None,
    remove_null: bool = False,
) -> set[type]:
    """Convert type definition to set of types (>1 in case of union).

    Args:
        typedef: The type definition to convert.
        typevar_map: Mapping of type variables to their actual types.
        ctx_module: Context module for resolving forward-referenced types.
        remove_null: Whether to remove ``NoneType`` from the resulting typeset.

    Returns:
        Set of types.
    """
    typedef_set: set[AnyTypeHint | None] = {typedef}
    root_orig = get_origin(typedef)

    if isinstance(typedef, UnionType) or root_orig is Union:
        typedef_set = {get_origin(union_arg) or union_arg for union_arg in get_args(typedef)}

        if remove_null:
            typedef_set &= {t for t in typedef_set if t is not None and not is_subtype(t, NoneType)}

    typeset: set[type] = set()

    for t in typedef_set:
        if t is None:
            typeset.add(NoneType)
            continue

        if isinstance(t, type):
            typeset.add(t)
            continue

        if isinstance(t, TypeAliasType):
            # A type alias is a synonym, so its typeset is the value's typeset.
            typeset |= typedef_to_typeset(
                t.__value__,
                typevar_map=typevar_map,
                ctx_module=ctx_module,
                remove_null=remove_null,
            )
            continue

        if isinstance(t, NewType):
            supertype = t.__supertype__
            typeset.add(supertype if isinstance(supertype, type) else object)
            continue

        if isinstance(t, TypeVar | ExtTypeVar):
            # An unresolved type variable carries no concrete type.
            typeset.add(object)
            continue

        t_parsed = t

        orig = get_origin(t_parsed)

        if isinstance(orig, TypeAliasType):
            # A parameterized generic alias, e.g. ``MyList[int]`` where
            # ``type MyList[T] = list[T]``; resolve the alias' value.
            typeset |= typedef_to_typeset(
                TypeRef(
                    orig.__value__,
                    subs=TypeRef(cast(Any, t)).local_typevar_map,
                    ctx_module=ctx_module,
                ).typeform,
                typevar_map=typevar_map,
                ctx_module=ctx_module,
                remove_null=remove_null,
            )
            continue

        if orig is Annotated:
            typeset |= typedef_to_typeset(
                get_args(t)[0],
                typevar_map=typevar_map,
                ctx_module=ctx_module,
                remove_null=remove_null,
            )
            continue

        if orig is None or orig is Literal:
            typeset.add(object)
            continue

        if orig is UnionType or orig is Union:
            typeset |= typedef_to_typeset(
                t_parsed,
                typevar_map=typevar_map,
                ctx_module=ctx_module,
                remove_null=remove_null,
            )
            continue

        typeset.add(orig if isinstance(orig, type) else object)

    return typeset


def is_subtype[T](
    type_: AnyTypeHint, supertype: TypeForm[T] | SingleTypeDef[T] | UnionType
) -> TypeGuard[TypeForm[T]]:
    """Check if object is of given type hint.

    Args:
        type_: The type to check.
        supertype: The supertype to check against.

    Returns:
        True if type_ is a subtype of supertype.
    """
    res_type = _resolve_typealias(type_)
    res_supertype = _resolve_typealias(supertype)

    try:
        return is_subhint(
            res_type,  # pyright: ignore[reportArgumentType]
            res_supertype,  # pyright: ignore[reportArgumentType]
        )
    except (BeartypeDoorNonpepException, TypeError):
        # Fallback for hints beartype cannot compare, most notably empty
        # typevar-tuples: compare the concrete typesets instead.
        typeset = typedef_to_typeset(cast(SingleTypeDef | UnionType, type_))
        supertype_tuple = tuple(typedef_to_typeset(cast(SingleTypeDef | UnionType, supertype)))
        return len(supertype_tuple) > 0 and all(issubclass(t, supertype_tuple) for t in typeset)


def has_type[T](obj: Any, type_: TypeForm[T] | SingleTypeDef[T] | UnionType | None) -> TypeGuard[T]:
    """Check if object is of given type hint.

    Args:
        obj: The object to check.
        type_: The type to check against.

    Returns:
        True if object is of given type.
    """
    return is_bearable(obj, type_)  # pyright: ignore[reportArgumentType]


def is_type_hint[T](obj: TypeForm[T] | T) -> TypeGuard[TypeForm[T]]:
    """Check whether an object is a type hint rather than a value.

    Classes, unions, named aliases, parameterized generics and any hint with an
    origin count as type hints; instances, strings and numbers do not.

    Args:
        obj: The object to check.

    Returns:
        ``True`` if ``obj`` describes a type instead of being a value of one.
    """
    return (
        isinstance(obj, type | UnionType | TypeAliasType | GenericAlias)
        or get_origin(obj) is not None
    )


def get_lowest_common_base(types: Iterable[type]) -> type:
    """Return the lowest common base of given types.

    Args:
        types: The types to get the common base for.

    Returns:
        The lowest common base type.
    """
    type_list = list(types)
    if len(type_list) == 0:
        return object

    bases_of_all = reduce(set.intersection, (set(getmro(t)) for t in type_list))
    return max(bases_of_all, key=lambda b: sum(issubclass(b, t) for t in bases_of_all))


def extract_nullable_type[T](
    type_: TypeForm[T | None] | None,
) -> type[T] | None:
    """Extract the non-none base type of a union.

    Args:
        type_: The type to extract from.

    Returns:
        The non-none base type or ``None`` if not found.
    """
    args = get_args(type_)

    if len(args) == 0:
        typeset = typedef_to_typeset(type_) - {NoneType}
        return cast(type[T], get_lowest_common_base(typeset)) if typeset else None

    notna_args = {arg for arg in args if (get_origin(arg) or arg) is not NoneType}
    return get_lowest_common_base(notna_args) if notna_args else None


def get_inheritance_distance(cls: type, base: type) -> int | None:
    """Return the inheritance distance between two classes.

    Note: Positive direction is from subclass to base class. If arguments are
    are reversed, the sign will be negative.

    Warning: Untested function.

    Args:
        cls: The subclass.
        base: The base class.

    Returns:
        The signed inheritance distance between the two classes. If the base class is
        not a base of the subclass, None is returned.
    """
    if not isinstance(cls, type) or not isinstance(base, type):
        return None

    if cls is base:
        return 0

    if issubclass(cls, base):
        cls, base, sign = (cls, base, 1)
    elif issubclass(base, cls):
        cls, base, sign = (base, cls, -1)
    else:
        return None

    distance = 1
    bases = set(cls.__bases__)
    while base not in bases and distance < 100:
        bases = reduce(set.union, (set(b.__bases__) for b in bases))
        distance += 1

    return distance * sign


def set_typeargs[T](  # noqa: C901
    typedef: TypeForm[T],
    args: Sequence[AnyTypeHint] | dict[TypeVar, AnyTypeHint],
) -> GenericAlias[T]:
    """Set a typevar in a generic type hint.

    Args:
        typedef: The type definition to set type arguments for.
        args: The type arguments to set, either positionally or by type variable.

    Returns:
        The generic type with set type arguments.

    Raises:
        TypeError: If ``typedef`` is not a parameterized generic type.
        ValueError: If ``args`` names a type variable that ``typedef`` does not use.
    """
    is_type_aware = isinstance(typedef, type) and issubclass(typedef, TypeAwareClass)

    if is_type_aware:
        orig = getattr(typedef, "__origin__", None)
        current_args = getattr(typedef, "__args__", ())
    else:
        orig = get_origin(typedef)
        current_args = get_args(typedef)

    if orig is None:
        raise TypeError(f"`{typedef!r}` is not a generic type definition.")

    if len(current_args) == 0:
        raise TypeError(f"`{typedef!r}` has no type arguments to replace.")

    def _apply(values: tuple[Any, ...]) -> GenericAlias[T]:
        if is_type_aware:
            # A starred subscript is delivered as a single tuple key, so a
            # single argument must be passed without the star.
            return cast(GenericAlias[T], orig[values[0] if len(values) == 1 else values])

        return cast(GenericAlias[T], orig[*values])

    if not isinstance(args, dict):
        arg_seq = tuple(args)
        if len(arg_seq) == 0:
            raise ValueError("At least one type argument is required.")

        return _apply(arg_seq)

    params = getattr(orig, "__parameters__", None) or getattr(orig, "__type_params__", ())
    if not isinstance(params, tuple) or len(params) == 0:
        params = tuple(
            p
            for p in current_args
            if isinstance(p, TypeVar | ExtTypeVar | TypeVarTuple | ExtTypeVarTuple)
        )

    # Map each parameter to the argument that currently fills it.
    typearg_map = dict(zip(params, current_args))
    typevar_map = TypeRef(typedef).typevar_map

    for typevar, arg in args.items():
        if typevar in typearg_map:
            typearg_map[typevar] = arg
            continue

        targets = [p for p, supplied in typearg_map.items() if supplied is typevar]

        if len(targets) == 0:
            # Follow a substitution chain, e.g. ``Base[T]`` inside ``Sub[T]``.
            subs = typevar_map.get(typevar)
            resolved = subs.typedef if isinstance(subs, TypeRef) else None
            if isinstance(resolved, TypeVar | ExtTypeVar):
                targets = [p for p, supplied in typearg_map.items() if supplied is resolved]

        if len(targets) == 0:
            raise ValueError(f"Type variable `{typevar!r}` is not a parameter of `{typedef!r}`.")

        for target in targets:
            typearg_map[target] = arg

    return _apply(tuple(typearg_map.values()))


def get_typeargs(instance: Any) -> tuple[type, ...] | None:
    """Get the type arguments of a generic instance.

    Args:
        instance: The instance to get type arguments for.

    Returns:
        The type arguments or ``None`` if not found.
    """
    if hasattr(instance, "__orig_class__"):
        orig_class = getattr(instance, "__orig_class__")
        return get_args(orig_class)

    return None


T = TypeVar("T", covariant=True)


def _substitute_typevars(
    hint: Any,
    typevar_map: Mapping[TypeVar | ExtTypeVar, TypeRef],
) -> Any:
    """Substitute bound type variables inside a type hint, recursively.

    Args:
        hint: The type hint to substitute into.
        typevar_map: Mapping of type variables to their resolved references.

    Returns:
        The hint with every bound type variable replaced by its value.
    """
    if isinstance(hint, TypeVar | ExtTypeVar):
        ref = typevar_map.get(hint)
        return ref.typeform if ref is not None else hint

    if isinstance(hint, TypeAliasType | NewType):
        return hint

    if isinstance(hint, type) and issubclass(hint, TypeAwareClass):
        # An applied ``TypeAwareClass`` is a real class, so its arguments must
        # be read from ``__args__`` and re-applied through ``__origin__``.
        class_args = getattr(hint, "__args__", ())
        if len(class_args) == 0:
            return hint

        resolved_class_args = tuple(_substitute_typevars(arg, typevar_map) for arg in class_args)
        if resolved_class_args == class_args:
            return hint

        origin = getattr(hint, "__origin__", hint)
        return (
            origin[resolved_class_args[0]]
            if len(resolved_class_args) == 1
            else origin[resolved_class_args]
        )

    args = get_args(hint)
    origin = get_origin(hint)

    if origin is None or len(args) == 0 or origin is Literal:
        return hint

    resolved = tuple(_substitute_typevars(arg, typevar_map) for arg in args)

    if origin is Annotated:
        return Annotated[resolved[0], *args[1:]]

    if origin is Union or origin is UnionType:
        return reduce(operator.or_, resolved)

    return origin[resolved]


@dataclass
class TypeRef(Generic[T]):  # noqa: UP046
    """Reference to a typeform."""

    hint: TypeForm[T] | SingleTypeDef[T] | TypeVar | ExtTypeVar | ForwardRef | str = cast(
        type[T], object
    )
    """Type hint."""

    ctx_module: ModuleType | None = None
    """Context module for resolving forward-referenced types."""

    ctx: dict[str, Any] = field(default_factory=dict)
    """Context mapping for resolving forward-referenced types."""

    subs: Mapping[
        TypeVar | ExtTypeVar | TypeVarTuple | ExtTypeVarTuple,
        TypeRef | tuple[TypeRef, ...],
    ] = field(default_factory=dict)
    """Substitutions for type variables."""

    overrides: Mapping[TypeVar | ExtTypeVar, TypeRef] = field(default_factory=dict)
    """Overrides for type variables."""

    @cached_property
    def typedef(self) -> SingleTypeDef[T] | UnionType | Annotated:
        """Resolved type definition."""
        if isinstance(self.hint, str):
            hint = eval(
                self.hint,
                {
                    **globals(),
                    **(vars(self.ctx_module) if self.ctx_module else {}),
                    **self.ctx,
                    **{
                        k.__name__: (v.hint if isinstance(v, TypeRef) else tuple(t.hint for t in v))
                        for k, v in self.subs.items()
                    },
                    **{k.__name__: v.hint for k, v in self.overrides.items()},
                },
            )
            return TypeRef(
                hint,
                ctx_module=self.ctx_module,
                ctx=self.ctx,
                subs=self.subs,
                overrides=self.overrides,
            ).typedef

        if isinstance(self.hint, ForwardRef):
            globalns = {
                **globals(),
                **(vars(self.ctx_module) if self.ctx_module else {}),
                **self.ctx,
                **{
                    k.__name__: (v.hint if isinstance(v, TypeRef) else tuple(t.hint for t in v))
                    for k, v in self.subs.items()
                },
                **{k.__name__: v.hint for k, v in self.overrides.items()},
            }

            if _evaluate_forward_ref is not None:
                evaluated = _evaluate_forward_ref(
                    self.hint, globals=globalns, locals=None, type_params=()
                )
            else:  # Python 3.13: the public helper does not exist yet.
                evaluated = self.hint._evaluate(globalns, None, (), recursive_guard=frozenset())

            assert evaluated is not None
            return evaluated

        if isinstance(self.hint, TypeVar | ExtTypeVar):
            typeref = self.subs.get(
                self.hint,
                TypeRef(
                    (
                        self.hint.__default__
                        if isinstance(self.hint, ExtTypeVar)
                        and hasattr(self.hint, "__default__")
                        and self.hint.has_default()
                        and self.hint.__default__ is not Any
                        else (self.hint.__bound__ if self.hint.__bound__ is not None else object)
                    ),
                    ctx_module=getmodule(self.hint),
                    subs=self.subs,
                    overrides=self.overrides,
                ),
            )

            assert not isinstance(typeref, tuple)
            return typeref.typedef

        return self.hint

    @cached_property
    def origin_type(self) -> type[T] | None:
        """Origin type in case of a generic typedef (without fully evaluating the type str)."""
        if isinstance(self.hint, ForwardRef | str):
            # Avoid evaluating the full typeform if supplied as str or ForwardRef.
            hint_str = self.hint.__forward_arg__ if isinstance(self.hint, ForwardRef) else self.hint

            if hint_str is None:
                return None

            type_str = (
                hint_str
                if "|" in hint_str or hint_str.startswith("Union[")
                else hint_str.split("[", 1)[0]
            )
            try:
                return eval(
                    type_str,
                    {
                        **globals(),
                        **(vars(self.ctx_module) if self.ctx_module else {}),
                        **self.ctx,
                    },
                )
            except Exception:
                return None

        return get_origin(self.typedef)

    @cached_property
    def typeset(self) -> set[type[T]]:
        """Set of concrete types (>1 in case of union typeform)."""
        return typedef_to_typeset(self.typedef)

    @cached_property
    def common_type(self) -> type[T]:
        """Common base type of the typeset."""
        return get_lowest_common_base(
            typedef_to_typeset(
                self.typedef,
                remove_null=True,
            )
        )

    @cached_property
    def origin(self) -> type | TypeAliasType:
        """Origin of the type reference."""
        if isinstance(self.typedef, type) and issubclass(self.typedef, TypeAwareClass):
            return getattr(self.typedef, "__origin__", None) or self.typedef

        return (
            get_origin(self.typedef)
            or getattr(self.typedef, "__origin__", None)
            or self.common_type
        )

    @cached_property
    def params(self) -> tuple[TypeVar | ExtTypeVar, ...]:
        """Type parameters of the type reference."""
        origin = self.origin

        if isinstance(origin, TypeVar | ExtTypeVar):
            return (origin,)

        for attr in ("__parameters__", "__type_params__"):
            params = getattr(origin, attr, None)
            if isinstance(params, tuple):
                return params

        return ()

    @cached_property
    def args(self) -> tuple[Any, ...]:
        """Type arguments of the type reference."""
        return (
            self.typedef.__args__
            if isinstance(self.typedef, type) and issubclass(self.typedef, TypeAwareClass)
            else (get_args(self.typedef) or getattr(self.typedef, "__args__", ()))
        )

    @cached_property
    def annotation(self) -> Any | None:
        """The dynamic annotation supplied via ``typing.Annotated``."""
        if self.origin is Annotated:
            return self.args[1] if len(self.args) > 1 else None
        return None

    @cached_property
    def literal_values(self) -> tuple[str | int | float | bool | Enum, ...]:
        """Distinct literal values, recursively resolving aliases and unions.

        Values retain declaration order and are deduplicated by both type and
        value. Non-literal union members contribute no values.
        """
        pending = [self.typedef]
        seen: set[int] = set()
        values: dict[tuple[type, Any], str | int | float | bool | Enum] = {}

        while pending:
            hint = pending.pop()
            if id(hint) in seen:
                continue
            seen.add(id(hint))

            if isinstance(hint, TypeAliasType):
                pending.append(hint.__value__)
                continue

            origin = get_origin(hint)
            if origin is Union or origin is UnionType:
                pending.extend(reversed(get_args(hint)))
            elif origin is Literal:
                for value in get_args(hint):
                    values.setdefault((type(value), value), value)

        return tuple(values.values())

    @property
    def single_typedef(self) -> SingleTypeDef[T]:
        """Resolved type definition without union."""
        return self.typedef if not isinstance(self.typedef, UnionType) else self.common_type

    @cached_property
    def bases(self) -> tuple[type, *tuple[type, ...]]:
        """Bases of the type reference."""
        bases = reduce(
            set.intersection,
            (set(t.__dict__.get("__orig_bases__", t.__bases__)) for t in self.typeset),
        )

        return tuple(bases) or (object,)

    @cached_property
    def arg_typerefs(self) -> tuple[TypeRef, ...]:
        """TypeRefs of the type arguments."""
        ctx_module = self.ctx_module or getmodule(self.typedef)

        return tuple(
            TypeRef(
                v,
                ctx_module=ctx_module,
            )
            for v in self.args
        )

    @cached_property
    def local_typevar_map(
        self,
    ) -> Mapping[
        TypeVar | ExtTypeVar | TypeVarTuple | ExtTypeVarTuple,
        TypeRef | tuple[TypeRef, ...],
    ]:
        """Mapping of argument type variables to their values."""
        ctx_module = self.ctx_module or getmodule(self.typedef)

        # First get the typevar params of the typehint.
        raw_arg_map: dict[Any, Any] = {}

        if len(self.params) > 0 and len(self.args) > 0:
            # Map typevar params to typeargs, which may be typevars themselves.
            raw_arg_map = {}

            # Initialize a list of remaining type arguments,
            # which are yet to be matched to params.
            # Order is reversed for efficient popping.
            remaining_args = list(reversed(self.args))

            for i, p in enumerate(self.params):
                # If no more args are left, fill the rest of the map
                # with the type params themselves (identity map).
                if len(remaining_args) == 0:
                    raw_arg_map[p] = p
                    continue

                if isinstance(p, TypeVar | ExtTypeVar):
                    # In case of a normal typevar, map the next arg.
                    raw_arg_map[p] = remaining_args.pop()
                elif isinstance(p, TypeVarTuple | ExtTypeVarTuple):
                    # In case of a typevar-tuple, map all remaining args
                    # minus those associated to downstream typevars.
                    remaining_params = len(self.params) - i - 1
                    n_mapped = len(remaining_args) - remaining_params

                    if n_mapped < 0:
                        raise TypeError("Incompatible type arguments for typevar-tuple")

                    raw_arg_map[p] = tuple(remaining_args.pop() for _ in range(n_mapped))
                else:
                    raise TypeError("Unsupported typevar type")

        # Merge with defaults, substitutions, and overrides.
        local_typevar_map = (
            {
                p: TypeRef(
                    p,
                    ctx_module=ctx_module,
                )
                for p in self.params
            }
            | dict(self.subs)
            | {
                k: (
                    tuple(
                        TypeRef(
                            item,
                            ctx_module=ctx_module,
                        )
                        for item in v
                    )
                    if isinstance(v, tuple)
                    else TypeRef(
                        v,
                        ctx_module=ctx_module,
                    )
                )
                for k, v in raw_arg_map.items()
            }
            | dict(self.overrides)
        )

        # Make sure every arg-typeref can access all other arg-typerefs.
        for s in local_typevar_map.values():
            if isinstance(s, TypeRef):
                s = [s]

            for t in s:
                t.subs = local_typevar_map

        return local_typevar_map

    @cached_property
    def typevar_map(
        self,
    ) -> dict[
        TypeVar | ExtTypeVar | TypeVarTuple | ExtTypeVarTuple,
        TypeRef | tuple[TypeRef, ...],
    ]:
        """Return a mapping of type variables to their values, including those of bases."""
        ctx_module = self.ctx_module or getmodule(self.typedef)
        typevar_map: dict[
            TypeVar | ExtTypeVar | TypeVarTuple | ExtTypeVarTuple,
            TypeRef | tuple[TypeRef, ...],
        ] = {}

        if isinstance(self.typedef, TypeAliasType):
            # A named alias only carries its own parameters; its value is a
            # separate hint whose variables are already bound by the alias.
            return dict(self.local_typevar_map)

        if isinstance(self.typedef, UnionType) or self.origin is Union:
            # Resolve typevar map of each union arg individually and
            # union the resulting types per typevar.
            base_typevar_items = chain(
                *(
                    TypeRef(arg, subs=self.subs, overrides=self.overrides).typevar_map.items()
                    for arg in self.args
                )
            )
            groups = groupby(
                sorted(base_typevar_items, key=lambda x: x[0].__name__),
                key=lambda x: x[0].__name__,
            )
            group_values = [list(g) for _, g in groups]
            typevar_map = {
                (tv := g[0][0]): (
                    TypeRef(
                        cast(
                            Any,
                            reduce(operator.or_, (cast(TypeRef, v).typedef for _, v in g)),
                        ),
                        subs=self.subs,
                        overrides=self.overrides,
                        ctx_module=ctx_module,
                    )
                    if isinstance(tv, TypeVar | ExtTypeVar)
                    else tuple(
                        TypeRef(
                            reduce(operator.or_, (v.typedef for v in t)),
                            subs=self.subs,
                            overrides=self.overrides,
                            ctx_module=ctx_module,
                        )
                        for t in zip(*(cast(tuple[TypeRef, ...], v) for _, v in g))
                    )
                )
                for g in group_values
            }

        else:
            # Anything else means we have a pure type or a generic typehint.
            # Ascend to generic base classes, stopping at the terminal bases
            # (``object`` and ``Generic``) so the ascent cannot cycle.
            base_arg_map: dict[
                TypeVar | ExtTypeVar | TypeVarTuple | ExtTypeVarTuple,
                TypeRef | tuple[TypeRef, ...],
            ] = {}

            for base in self.bases:
                if base in (object, Generic) or get_origin(base) is Generic:
                    continue

                base_arg_map |= TypeRef(
                    base,
                    subs=self.local_typevar_map,
                ).typevar_map

            # Merge with base typevar map.
            typevar_map = base_arg_map | dict(self.local_typevar_map)

        return typevar_map

    @property
    def scalar_typevar_map(
        self,
    ) -> dict[
        TypeVar | ExtTypeVar,
        TypeRef,
    ]:
        """Return a mapping of type variables to their values, excluding typevar tuples."""
        return {
            k: v
            for k, v in self.typevar_map.items()
            if isinstance(k, TypeVar | ExtTypeVar) and isinstance(v, TypeRef)
        }

    @cached_property
    def typeform(self) -> TypeForm[T]:
        """Resolved type definition with arguments recursively resolved."""
        return cast(Any, _substitute_typevars(self.typedef, self.scalar_typevar_map))

    def validate(self, obj: Any) -> TypeGuard[T]:
        """Check if object is of this type.

        ``Annotated`` metadata that carries pydantic validation (a
        ``FieldInfo`` or an object implementing the core-schema hook) is
        enforced as well; a failed validation yields ``False`` rather than
        raising.
        """
        if not has_type(obj, cast(Any, self.typeform)):
            return False

        if isinstance(self.typeform, RuntimeValidated) or isinstance(
            self.annotation, RuntimeValidated | FieldInfo
        ):
            try:
                TypeAdapter(self.typeform).validate_python(obj)
            except ValidationError:
                return False

        return True


class TypeAwareClass:
    """Class, which is aware of its type arguments."""

    __origin__: type[TypeAwareClass]
    __args__: tuple[AnyTypeHint, ...]

    @prop(cached=True, mode="class")
    @classmethod
    def typeargs(
        cls,
    ) -> dict[
        TypeVar | ExtTypeVar | TypeVarTuple | ExtTypeVarTuple,
        TypeRef | tuple[TypeRef, ...],
    ]:
        """Type arguments of this class."""
        return TypeRef(cls).typevar_map

    @prop(mode="class")
    @classmethod
    def type_params(
        cls,
    ) -> tuple[TypeVar | ExtTypeVar | TypeVarTuple | ExtTypeVarTuple, ...]:
        """Type arguments of this class."""
        return cls.__dict__.get("__parameters__", ()) or cls.__dict__.get("__type_params__", ())

    @prop(mode="class", cached=True)
    @classmethod
    def _generic_subclasses(cls) -> dict[Any, type[Self]]:
        return {}

    def __init_subclass__(cls):  # noqa: D105
        if "__origin__" not in cls.__dict__:
            cls.__origin__ = cls
        if "__args__" not in cls.__dict__:
            cls.__args__ = ()
        if "__parameters__" not in cls.__dict__ and Generic not in cls.__bases__:
            cls.__parameters__ = ()
        if "__type_params__" not in cls.__dict__ and Generic not in cls.__bases__:
            cls.__type_params__ = ()

        super().__init_subclass__()

    def __class_getitem__(cls, item: Any) -> type[Self]:  # noqa: D105
        if item in cls._generic_subclasses:
            return cls._generic_subclasses[item]

        subclass = new_class(
            f"{cls.__name__}[{item.__name__ if hasattr(item, '__name__') else item}]",
            bases=(cls,),
            exec_body=lambda ns: ns.update(
                {
                    "__origin__": cls,
                    "__args__": item if isinstance(item, tuple) else (item,),
                    "__module__": cls.__module__,
                }
            ),
        )
        cls._generic_subclasses[item] = subclass
        return subclass


def is_frozen_dataclass(obj: Any) -> bool:
    """Check if an object is a frozen dataclass.

    Args:
        obj: The object to check.

    Returns:
        ``True`` if the object is a frozen dataclass.
    """
    cls = obj if isinstance(obj, type) else type(obj)
    return is_dataclass(obj) and getattr(cls, "__dataclass_params__").frozen


def is_immutable(obj: Any) -> bool:
    """Check if an object is immutable.

    Args:
        obj: The object to check.

    Returns:
        ``True`` if the object is immutable.
    """
    # Built-in immutables
    if isinstance(obj, int | float | bool | str | bytes | frozenset):
        return True

    # Tuples: check recursively
    if isinstance(obj, tuple):
        return all(is_immutable(el) for el in obj)

    # Frozen dataclass
    if is_frozen_dataclass(obj):
        return True

    return False


def get_nondefault_methods(cls: type) -> dict[str, Any]:
    """Get the all attributes on a class without those inherited from ``object``.

    Args:
        cls: The class to get methods for.

    Returns:
        Mapping of method names to method objects.
    """
    return {
        name: member
        for name, member in inspect.getmembers(
            cls,
            predicate=lambda m: (
                inspect.ismethoddescriptor(m) or inspect.isfunction(m) or inspect.isbuiltin(m)
            ),
        )
        if not (
            (
                inspect.getmodule(member) is None
                and str(member.__qualname__).startswith("object" + ".")
            )
            or name == "__subclasshook__"
        )
    }


def get_own_attrs(
    cls: type,
    predicate: Callable[[object], bool] | None = None,
    include_bases: tuple[type, ...] = (),
) -> set[str]:
    """Get all non-inherited attributes on a class.

    Args:
        cls: The class to get attributes for.
        predicate: Optional predicate to filter attributes.
        include_bases: Additional base classes to include attributes from.

    Returns:
        Set of attribute names.
    """
    if predicate is None:
        attrs = set(cls.__dict__.keys())
        attrs.update(getattr(cls, "__annotations__", {}).keys())
    else:
        attrs = {name for name, value in cls.__dict__.items() if predicate(value)}

    included_cls_mods = [
        supcls
        for base in include_bases
        for supcls in (base, *get_subclasses(base))
        if issubclass(cls, supcls) and supcls is not cls
    ]
    sup_attrs = reduce(
        set.__or__,
        (
            get_own_attrs(
                supcls,
                predicate=predicate,
                include_bases=(),
            )
            for supcls in included_cls_mods
        ),
        set(),
    )

    return sup_attrs | attrs
