"""Reflection utilities for types."""

from __future__ import annotations

import inspect
import operator
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field, is_dataclass
from enum import Enum
from functools import cached_property, reduce
from inspect import getmodule, getmro, ismethod, signature
from types import ModuleType, UnionType, new_class
from typing import (
    Annotated,
    Any,
    ClassVar,
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

from annotationlib import get_annotations
from beartype.door import is_bearable, is_subhint
from beartype.roar import BeartypeDoorNonpepException
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from typing_extensions import Sentinel, TypeForm

from .descriptors import prop
from .runtime import get_subclasses
from .types import (
    GenericAlias,
    RuntimeValidated,
    TypeDef,
    TypeExpr,
    TypeNode,
    TypeOrigin,
)

NoneType = type(None)
"""The type of ``None``, re-exported for convenience."""

# Public since Python 3.14; ``ForwardRef._evaluate`` is private and deprecated.
_evaluate_forward_ref = getattr(typing, "evaluate_forward_ref", None)


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
        isinstance(obj, type | UnionType | TypeAliasType | TypeVar | TypeVarTuple | ForwardRef)
        or isinstance(obj, GenericAlias)
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


T = TypeVar("T", covariant=True)


def _merge_typevar_values(
    left: TypeRef | tuple[TypeRef, ...],
    right: TypeRef | tuple[TypeRef, ...],
    ctx_module: ModuleType | None,
    subs: Mapping[TypeVar | TypeVarTuple, TypeRef | tuple[TypeRef, ...]],
    overrides: Mapping[TypeVar, TypeRef],
) -> TypeRef | tuple[TypeRef, ...]:
    """Union two values bound to the same type variable.

    Args:
        left: The first bound value.
        right: The second bound value.
        ctx_module: Context module for the resulting reference.
        subs: Substitutions to carry over to the resulting reference.
        overrides: Overrides to carry over to the resulting reference.

    Returns:
        The union of both values, element-wise for typevar tuples.
    """
    if isinstance(left, tuple) or isinstance(right, tuple):
        left_items = left if isinstance(left, tuple) else (left,)
        right_items = right if isinstance(right, tuple) else (right,)
        return tuple(
            cast(
                TypeRef,
                _merge_typevar_values(a, b, ctx_module, subs, overrides),
            )
            for a, b in zip(left_items, right_items)
        )

    return TypeRef(
        reduce(operator.or_, (left.resolved_type_def, right.resolved_type_def)),
        ctx_module=ctx_module,
        subs=subs,
        overrides=overrides,
    )


def _validate_attrs(obj: Any, anno: dict[str, TypeDef | TypeForm]) -> bool:
    """Validate the attributes of an object against their type annotations.

    Args:
        obj: The object to validate.
        anno: The type annotations to validate against.

    Returns:
        True if all attributes are valid, False otherwise.
    """
    for attr_name, attr_type in anno.items():
        if not hasattr(obj, attr_name):
            return False
        attr_value = getattr(obj, attr_name)
        try:
            if not TypeRef(attr_type).validate(attr_value):
                return False
        except Exception:  # noqa: BLE001 - unresolvable third-party hints
            continue
    return True


@dataclass
class TypeRef(Generic[T]):  # noqa: UP046
    """Reference to a typeform."""

    hint: TypeDef[T] | TypeForm[T] = cast(type[T], object)
    """The type hint."""

    ctx_module: ModuleType | None = None
    """Context module for resolving forward-referenced and string types."""

    ctx: dict[str, Any] = field(default_factory=dict)
    """Context mapping for resolving forward-referenced and string types."""

    subs: Mapping[
        TypeVar | TypeVarTuple,
        TypeRef | tuple[TypeRef, ...],
    ] = field(default_factory=dict)
    """Substitutions for type variables.

    These only act on open type variables in the inheritance / parameter tree,
    meaning variables that have not yet been substituted.
    """

    overrides: Mapping[TypeVar, TypeRef] = field(default_factory=dict)
    """Overrides for type variables.

    These override any matching type variables in the inheritance / parameter tree,
    meaning that they take precedence over any existing substitutions or defaults.
    Use with care.
    """

    @staticmethod
    def from_instance[I](instance: I) -> TypeRef[I]:
        """Get the type associated with this instance, may be generic.

        Args:
            instance: The instance to get type for.

        Returns:
            The type associated with the instance.
        """
        if hasattr(instance, "__orig_class__"):
            orig_class = getattr(instance, "__orig_class__")
            typ = TypeRef(orig_class).exact_type_node
            if typ is not None:
                return TypeRef(typ, ctx_module=getmodule(typ))

        return TypeRef(type(instance))

    # Top-level hint evaluation:

    @cached_property
    def origin_from_str(self) -> type[T] | None:
        """Origin type in case of a generic typedef (without fully evaluating the string hints)."""
        if not isinstance(self.hint, ForwardRef | str):
            return get_origin(self.type_def)

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

    @cached_property
    def type_expr(self) -> TypeExpr[T]:
        """Resolved type expression.

        This evaluates string hints and forward references to rich objects.
        """
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
            ).type_def

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

        return self.hint

    def _resolve_typevar(self, var: TypeVar) -> TypeRef | tuple[TypeRef, ...]:
        return self.overrides.get(
            var,
            self.subs.get(
                var,
                TypeRef(
                    (
                        var.__default__
                        if isinstance(var, TypeVar)
                        and hasattr(var, "__default__")
                        and var.has_default()
                        and var.__default__ is not Any
                        else (var.__bound__ if var.__bound__ is not None else object)
                    ),
                    ctx_module=getmodule(var),
                    subs=self.subs,
                    overrides=self.overrides,
                ),
            ),
        )

    @cached_property
    def type_def(self) -> TypeDef[T]:
        """Resolved type definition.

        This evaluates string hints, forward references, and top-level type variables
        to rich objects.
        """
        if self.type_expr is None:
            return NoneType

        if isinstance(self.type_expr, TypeVar):
            typeref = self._resolve_typevar(self.type_expr)
            assert not isinstance(typeref, tuple)
            return typeref.type_def

        return self.type_expr

    @cached_property
    def origin(self) -> type | TypeAliasType:
        """Origin of the type hint."""
        if isinstance(self.type_def, Sentinel):
            return Sentinel

        if isinstance(self.type_def, NewType):
            return cast(type, self.type_def.__supertype__)

        if isinstance(self.type_def, TypeAliasType):
            # A named alias reports the origin of its value. Substitutions are not
            # needed here, since the origin does not depend on the arguments.
            return TypeRef(
                self.type_def.__value__,
                ctx_module=self.ctx_module,
            ).origin

        orig = get_origin(self.type_def) or getattr(self.type_def, "__origin__", None)

        if orig is not None:
            return orig

        assert isinstance(self.type_def, type | TypeAliasType)
        return self.type_def

    # Next-level typearg evaluation:

    @cached_property
    def params(self) -> tuple[TypeVar, ...]:
        """Type parameters of the type hint."""
        for source in (self.type_def, self.origin):
            for attr in ("__parameters__", "__type_params__"):
                params = getattr(source, attr, None)
                if isinstance(params, tuple) and len(params) > 0:
                    return params

        return ()

    @cached_property
    def raw_args(self) -> tuple[Any, ...]:
        """Type arguments of the type hint, unevaluated."""
        return (
            self.type_def.__args__
            if isinstance(self.type_def, type) and issubclass(self.type_def, TypeAwareClass)
            else (get_args(self.type_def) or getattr(self.type_def, "__args__", ()))
        )

    @cached_property
    def arg_typerefs(self) -> tuple[TypeRef, ...]:
        """TypeRefs of the type arguments."""
        ctx_module = self.ctx_module or getmodule(self.type_def)

        return tuple(
            TypeRef(
                v,
                ctx_module=ctx_module,
            )
            for v in self.raw_args
        )

    @cached_property
    def annotations(self) -> tuple[Any, ...]:
        """The dynamic annotations supplied via ``typing.Annotated``."""
        if get_origin(self.type_def) is Annotated:
            return self.raw_args[1:] if len(self.raw_args) > 1 else ()

        return ()

    @property
    def annotation(self) -> Any | None:
        """The first dynamic annotation supplied via ``typing.Annotated``, if any."""
        return self.annotations[0] if self.annotations else None

    @cached_property
    def local_typevar_map(
        self,
    ) -> Mapping[
        TypeVar | TypeVarTuple,
        TypeRef | tuple[TypeRef, ...],
    ]:
        """Mapping of argument type variables to their values."""
        ctx_module = self.ctx_module or getmodule(self.type_def)

        # First get the typevar params of the typehint.
        raw_arg_map: dict[Any, Any] = {}

        if len(self.params) > 0 and len(self.raw_args) > 0:
            # Map typevar params to typeargs, which may be typevars themselves.
            raw_arg_map = {}

            # Initialize a list of remaining type arguments,
            # which are yet to be matched to params.
            # Order is reversed for efficient popping.
            remaining_args = list(reversed(self.raw_args))

            for i, p in enumerate(self.params):
                # If no more args are left, fill the rest of the map
                # with the type params themselves (identity map).
                if len(remaining_args) == 0:
                    raw_arg_map[p] = p
                    continue

                if isinstance(p, TypeVar):
                    # In case of a normal typevar, map the next arg.
                    raw_arg_map[p] = remaining_args.pop()
                elif isinstance(p, TypeVarTuple):
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

    # Recursive typevar evaluation and substitution:

    @cached_property
    def typevar_map(
        self,
    ) -> dict[
        TypeVar | TypeVarTuple,
        TypeRef | tuple[TypeRef, ...],
    ]:
        """Return a mapping of type variables to their values, including those of bases and args."""
        ctx_module = self.ctx_module or getmodule(self.type_def)
        typevar_map: dict[
            TypeVar | TypeVarTuple,
            TypeRef | tuple[TypeRef, ...],
        ] = {}

        # A named alias only carries its own parameters; its value is a separate
        # hint whose variables are already bound by the alias.
        # Sentinels and literals bind no type variables.
        if (
            isinstance(self.type_def, TypeAliasType | Sentinel)
            or get_origin(self.type_def) is Literal
        ):
            return dict(self.local_typevar_map)

        if get_origin(self.type_def) is Annotated:
            return TypeRef(
                self.raw_args[0],
                ctx_module=ctx_module,
                subs=self.subs,
                overrides=self.overrides,
            ).typevar_map

        if isinstance(self.origin, TypeAliasType):
            return TypeRef(
                self.origin.__value__,
                ctx_module=ctx_module,
                subs=self.local_typevar_map,
                overrides=self.overrides,
            ).typevar_map

        if self.origin is Union or self.origin is UnionType:
            # Resolve each union member individually and union the values bound
            # to the same type variable.
            for arg in self.raw_args:
                for var, value in TypeRef(
                    arg,
                    ctx_module=ctx_module,
                    subs=self.subs,
                    overrides=self.overrides,
                ).typevar_map.items():
                    typevar_map[var] = (
                        _merge_typevar_values(
                            typevar_map[var],
                            value,
                            ctx_module,
                            self.subs,
                            self.overrides,
                        )
                        if var in typevar_map
                        else value
                    )

            return typevar_map

        # Ascend to generic base classes, stopping at the terminal bases
        # (``object`` and ``Generic``) so the ascent cannot cycle.
        origin = self.origin
        if isinstance(origin, type):
            for base in origin.__dict__.get("__orig_bases__", origin.__bases__):
                if base is object or base is Generic or get_origin(base) is Generic:
                    continue

                typevar_map |= TypeRef(
                    base,
                    ctx_module=ctx_module,
                    subs=self.local_typevar_map,
                    overrides=self.overrides,
                ).typevar_map

        # The local bindings take precedence over inherited ones.
        return typevar_map | dict(self.local_typevar_map)

    @property
    def scalar_typevar_map(
        self,
    ) -> dict[
        TypeVar,
        TypeRef,
    ]:
        """Return a mapping of type variables to their values, excluding typevar tuples."""
        return {
            k: v
            for k, v in self.typevar_map.items()
            if isinstance(k, TypeVar) and isinstance(v, TypeRef)
        }

    @cached_property
    def resolved_typevar_map(
        self,
    ) -> dict[
        TypeVar | TypeVarTuple,
        TypeRef | tuple[TypeRef, ...],
    ]:
        """Return a mapping of type variables to their resolved values."""
        return {
            k: v
            for k, v in self.typevar_map.items()
            if all(
                not isinstance(t.hint, TypeVar | TypeVarTuple)
                for t in (v if isinstance(v, tuple) else (v,))
            )
        }

    @cached_property
    def resolved_type_def(self) -> TypeDef[T]:
        """Resolved type definition with arguments recursively resolved."""
        if isinstance(self.type_def, type) and issubclass(self.type_def, TypeAwareClass):
            # An applied ``TypeAwareClass`` is a real class, so its arguments must be
            # read from ``__args__`` and re-applied through ``__origin__``.
            resolved = tuple(
                TypeRef(
                    arg,
                    ctx_module=self.ctx_module,
                    subs=self.subs,
                    overrides=self.overrides,
                ).resolved_type_def
                for arg in self.raw_args
            )

            if resolved == self.raw_args:
                return self.type_def

            origin = self.type_def.__origin__
            return origin[resolved[0]] if len(resolved) == 1 else origin[resolved]

        if is_bearable(self.type_def, TypeOrigin):
            return self.type_def

        if get_origin(self.type_def) is Annotated:
            return Annotated[
                TypeRef(
                    self.raw_args[0],
                    ctx_module=self.ctx_module,
                    subs=self.subs,
                    overrides=self.overrides,
                ).resolved_type_def,
                *self.raw_args[1:],
            ]

        if self.origin is None or len(self.raw_args) == 0 or self.origin is Literal:
            return self.type_def

        resolved = tuple(
            (
                TypeRef(
                    arg,
                    ctx_module=self.ctx_module,
                    subs=self.subs,
                    overrides=self.overrides,
                ).resolved_type_def
                if is_type_hint(arg)
                else arg
            )
            for arg in self.raw_args
        )

        if self.origin is Union or self.origin is UnionType:
            return reduce(operator.or_, resolved)

        # Single-argument generics must be re-applied with the argument itself,
        # not a one-element tuple: ``ClassVar[(set[str],)]`` is invalid while
        # ``ClassVar[set[str]]`` is not.
        return (
            self.origin[resolved[0]]  # pyright: ignore[reportIndexIssue]
            if len(resolved) == 1
            else self.origin[resolved]  # pyright: ignore[reportIndexIssue]
        )

    # Recursive type algebra evaluation:

    @cached_property
    def members(self) -> tuple[TypeNode, ...]:
        """Union members of the type hint, or the type itself if it is not a union.

        Resolves all type aliases, which consist of unions, recursively. Members keep
        their declaration order and any type arguments.
        """
        ctx_module = self.ctx_module or getmodule(self.type_def)

        if isinstance(self.type_def, TypeAliasType):
            return TypeRef(
                self.type_def.__value__,
                ctx_module=ctx_module,
                subs=self.local_typevar_map,
                overrides=self.overrides,
            ).members

        if isinstance(self.origin, TypeAliasType):
            return TypeRef(
                self.origin.__value__,
                ctx_module=ctx_module,
                subs=self.local_typevar_map,
                overrides=self.overrides,
            ).members

        if self.origin is Union or self.origin is UnionType:
            return tuple(
                member
                for arg in self.raw_args
                for member in TypeRef(
                    arg,
                    ctx_module=ctx_module,
                    subs=self.subs,
                    overrides=self.overrides,
                ).members
            )

        return (cast(TypeNode, self.type_def),)

    @cached_property
    def literal_values(self) -> tuple[str | int | float | bool | Enum, ...]:
        """Distinct literal values, recursively resolving aliases and unions.

        Values retain declaration order and are deduplicated by both type and
        value. Non-literal union members contribute no values.
        """
        ctx_module = self.ctx_module or getmodule(self.type_def)
        values: dict[tuple[type, Any], str | int | float | bool | Enum] = {}
        pending: list[TypeRef] = [self]
        seen: set[int] = set()

        while pending:
            ref = pending.pop()
            hint = ref.type_def

            if id(hint) in seen:
                continue
            seen.add(id(hint))

            if isinstance(hint, TypeAliasType):
                pending.append(
                    TypeRef(
                        hint.__value__,
                        ctx_module=ctx_module,
                        subs=ref.local_typevar_map,
                        overrides=ref.overrides,
                    )
                )
                continue

            origin = get_origin(hint)
            if origin is Union or origin is UnionType:
                pending.extend(
                    reversed(
                        [
                            TypeRef(
                                arg,
                                ctx_module=ctx_module,
                                subs=ref.subs,
                                overrides=ref.overrides,
                            )
                            for arg in get_args(hint)
                        ]
                    )
                )
            elif origin is Literal:
                for value in get_args(hint):
                    values.setdefault((type(value), value), value)

        return tuple(values.values())

    @property
    def exact_type_node(self) -> TypeNode[T] | None:
        """Resolved type definition, or None if it is a union."""
        if (
            self.origin is Union
            or self.origin is UnionType
            or self.origin is Literal
            or self.origin is Sentinel
        ):
            return None

        if get_origin(self.type_def) is Annotated:
            return TypeRef(
                self.raw_args[0],
                subs=self.subs,
                overrides=self.overrides,
                ctx_module=self.ctx_module,
            ).exact_type_node

        return cast(TypeNode[T], self.resolved_type_def)

    @cached_property
    def concrete_type(self) -> type[T] | None:
        """Concrete type of the type hint, if any.

        Unions, literals, sentinels and unresolved type variables carry no single
        concrete type and yield ``None``; parameterised generics reduce to their origin.
        """
        if self.origin is Union or self.origin is UnionType:
            return None

        if self.origin is Literal or self.origin is Sentinel:
            return None

        if self.origin is Annotated:
            return TypeRef(
                self.raw_args[0],
                ctx_module=self.ctx_module,
                subs=self.subs,
                overrides=self.overrides,
            ).concrete_type

        if isinstance(self.type_def, TypeVar):
            return None

        if isinstance(self.type_def, type) and issubclass(self.type_def, TypeAwareClass):
            # An applied ``TypeAwareClass`` is itself the concrete class.
            return cast(type[T], self.type_def)

        origin = self.origin
        return cast(type[T], origin) if isinstance(origin, type) else None

    @cached_property
    def typeset(self) -> set[type[T]]:
        """Set of concrete types (>1 in case of union typeform)."""
        return {
            ref.concrete_type
            for member in self.members
            if (ref := TypeRef(member, subs=self.subs, overrides=self.overrides)).concrete_type
            is not None
        }

    @cached_property
    def bases(self) -> tuple[type, *tuple[type, ...]]:
        """Bases of the type reference."""
        bases = reduce(
            set.intersection,
            (set(t.__dict__.get("__orig_bases__", t.__bases__)) for t in self.typeset),
        )

        return tuple(bases) or (object,)

    @cached_property
    def common_type(self) -> type[T]:
        """Common base type of the typeset, ignoring ``NoneType``."""
        return get_lowest_common_base(self.typeset - {NoneType})

    @property
    def approx_type_node(self) -> TypeNode[T]:
        """Resolved type definition without union, falling back to the common type."""
        return self.exact_type_node or self.common_type

    @property
    def approx_single_typeform(self) -> TypeNode[T]:
        """Resolved type definition without union, falling back to the common type."""
        return self.approx_type_node

    @cached_property
    def attr_annotations(self) -> dict[str, TypeDef | TypeForm] | None:
        """Type annotations of the attributes of the type reference.

        Only classes whose annotations describe instance attributes are
        inspected: dataclasses and pydantic models. For arbitrary classes the
        annotations are class-level declarations (often ``ClassVar`` or types
        whose instances are stored), so validating them against an instance
        would produce false negatives. Class variables and init vars are skipped.
        Descriptors are validated against the return type of their ``__get__`` method.

        Returns:
            A dictionary mapping attribute names to their type annotations, or
            ``None`` when the type does not declare instance attributes.
        """
        common_type = self.common_type

        if not (is_dataclass(common_type) or issubclass(common_type, BaseModel)):
            return None

        annos: dict[str, TypeDef | TypeForm] = {}

        for name, anno in get_annotations(common_type).items():
            resolved = TypeRef(
                anno, ctx_module=getmodule(common_type), subs=self.resolved_typevar_map
            )

            if resolved.origin in (ClassVar, InitVar):
                continue

            if resolved.concrete_type is not None and hasattr(resolved.concrete_type, "__get__"):
                # If the attribute is a descriptor, use the return type of its __get__ method.
                get_method = getattr(resolved.concrete_type, "__get__")

                if ismethod(get_method):
                    get_return_type = signature(get_method).return_annotation
                    annos[name] = TypeRef(
                        get_return_type,
                        ctx_module=getmodule(resolved.concrete_type),
                        subs=self.resolved_typevar_map,
                    ).resolved_type_def

            annos[name] = resolved.resolved_type_def

        return annos

    # Validation:

    def validate(self, obj: Any, with_attrs: bool = True) -> TypeGuard[T]:
        """Check if object is of this type.

        The runtime check is performed with beartype. ``Annotated`` metadata that
        carries pydantic validation (a ``FieldInfo`` or an object implementing the
        core-schema hook) is enforced as well; a failed validation yields ``False``
        rather than raising.

        Args:
            obj: The object to check.
            with_attrs: Whether to validate attributes by their annotations.

        Returns:
            ``True`` if the object is of this type, ``False`` otherwise.
        """
        if not is_bearable(obj, cast(Any, self.resolved_type_def)):
            return False

        if isinstance(self.resolved_type_def, RuntimeValidated) or any(
            isinstance(a, RuntimeValidated | FieldInfo) for a in self.annotations
        ):
            try:
                TypeAdapter(self.resolved_type_def).validate_python(obj)
            except ValidationError:
                return False

        if with_attrs and self.attr_annotations is not None:
            if not _validate_attrs(obj, self.attr_annotations):
                return False

        return True

    def is_subtype(self, supertype: TypeForm[T] | TypeDef[T]) -> TypeGuard[TypeForm[T]]:
        """Check if this type reference is a subtype of the given type hint.

        Args:
            supertype: The supertype to check against.

        Returns:
            ``True`` if this reference is a subtype of ``supertype``.
        """
        super_typeref = TypeRef(supertype)
        try:
            return is_subhint(
                cast(Any, self.resolved_type_def),
                cast(Any, super_typeref.resolved_type_def),
            )
        except (BeartypeDoorNonpepException, TypeError):
            # Fallback for hints beartype cannot compare: compare the concrete typesets.
            return len(super_typeref.typeset) > 0 and all(
                issubclass(t, tuple(super_typeref.typeset)) for t in self.typeset
            )


def set_typeargs[T](  # noqa: C901
    typedef: TypeForm[T],
    args: Sequence[TypeForm] | dict[TypeVar, TypeForm],
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
    ref = TypeRef(typedef)

    # Prefer the type variables actually applied in the hint; fall back to the
    # declaration's parameters for unapplied generics.
    params = (
        tuple(arg for arg in ref.raw_args if isinstance(arg, TypeVar | TypeVarTuple)) or ref.params
    )

    if len(params) == 0:
        raise TypeError(f"{typedef!r} is not a generic type definition")

    if isinstance(args, Mapping):
        if len(args) == 0:
            raise ValueError("At least one type argument is required")

        unknown = [var for var in args if var not in params]
        if unknown:
            raise ValueError(f"{unknown[0]!r} is not a parameter of {typedef!r}")

        replacements: dict[Any, Any] = dict(args)
    else:
        if len(args) == 0:
            raise ValueError("At least one type argument is required")

        replacements = dict(zip(params, args))

    resolved = tuple(replacements.get(param, param) for param in params)

    origin = cast(Any, ref.origin)
    return cast(
        GenericAlias[T],
        origin[resolved[0]] if len(resolved) == 1 else origin[resolved],
    )


class TypeAwareClass:
    """Class, which is aware of its type arguments."""

    __origin__: type[TypeAwareClass]
    __args__: tuple[TypeForm, ...]

    @prop(mode="class")
    @classmethod
    def type_params(
        cls,
    ) -> tuple[TypeVar | TypeVarTuple, ...]:
        """Type arguments of this class."""
        return cls.__dict__.get("__parameters__", ()) or cls.__dict__.get("__type_params__", ())

    @prop(cached=True, mode="class")
    @classmethod
    def typevar_map(
        cls,
    ) -> dict[
        TypeVar | TypeVarTuple,
        TypeRef | tuple[TypeRef, ...],
    ]:
        """Type variable map of this class."""
        return TypeRef(cls).typevar_map

    @prop(cached=True, mode="class")
    @classmethod
    def typeargs(
        cls,
    ) -> tuple[Any, ...]:
        """Type arguments of this class."""
        return TypeRef(cls).raw_args

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
