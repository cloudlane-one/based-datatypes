"""Reflection utilities for types."""

from __future__ import annotations

import inspect
import operator
import typing
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from dataclasses import InitVar, dataclass, field, is_dataclass
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
    Self,
    TypeAliasType,
    TypeGuard,
    TypeIs,
    TypeVar,
    TypeVarTuple,
    Union,
    Unpack,
    cast,
    final,
    get_args,
    get_origin,
)

from annotationlib import get_annotations
from beartype.door import is_bearable, is_subhint
from beartype.roar import BeartypeDoorNonpepException, BeartypeException
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from typing_extensions import Sentinel, TypeForm

from .runtime import get_calling_module, get_subclasses
from .types import (
    GenericAlias,
    LiteralArg,
    RuntimeValidated,
    TypeDef,
    TypeExpr,
    TypeMember,
    TypeNode,
    TypeRoot,
    TypeScalar,
)

NoneType = type(None)
"""The type of ``None``, re-exported for convenience."""

# Public since Python 3.14; ``ForwardRef._evaluate`` is private and deprecated.
_evaluate_forward_ref = getattr(typing, "evaluate_forward_ref", None)


class OrderedSet[T](set[T]):
    """A set that iterates in insertion order.

    Membership, equality and hashing behave exactly as for :class:`set`, so an
    ordered set compares equal to a plain set holding the same items. Only
    iteration follows the order in which items were added, which type reflection
    relies on: a union keeps its members in declaration order, and a serializer
    declares its preferred format first.
    """

    __slots__ = ("_order",)

    _order: dict[T, None]

    def __init__(self, items: Iterable[T] = ()) -> None:
        """Create an ordered set from the given items."""
        super().__init__()
        self._order = {}
        self.update(items)

    def add(self, item: T) -> None:
        """Add an item, keeping its first insertion position."""
        if item not in self:
            super().add(item)
            self._order[item] = None

    def update(self, *others: Iterable[T]) -> None:
        """Add the items of every given iterable."""
        for other in others:
            for item in other:
                self.add(item)

    def discard(self, item: T) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Remove an item if it is present."""
        if item in self:
            super().discard(item)
            self._order.pop(item, None)

    def remove(self, item: T) -> None:
        """Remove an item, raising if it is absent."""
        super().remove(item)
        self._order.pop(item, None)

    def pop(self) -> T:
        """Remove and return the first item in insertion order."""
        item = next(iter(self._order))
        self.remove(item)
        return item

    def clear(self) -> None:
        """Remove every item."""
        super().clear()
        self._order.clear()

    def copy(self) -> OrderedSet[T]:
        """Return a shallow copy that keeps the insertion order."""
        return OrderedSet(self)

    def __iter__(self) -> Iterator[T]:
        """Iterate in insertion order."""
        return iter(self._order)

    def __repr__(self) -> str:
        """Show the items in insertion order."""
        return f"{type(self).__name__}({list(self)!r})"


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


def _validate_attrs(obj: Any, anno: Mapping[str, TypeDef]) -> bool:
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


T = TypeVar("T", covariant=True)


@final
class Undefined:
    """Sentinel for undefined values."""


@dataclass(frozen=True)
class RepType[T]:
    """A type hint that repeats a type definition multiple times."""

    type: TypeScalar[T]


type TypeHint[T] = TypeExpr[T] | TypeForm[T] | str | ForwardRef


@dataclass(frozen=True)
class TypeRef(Generic[T]):  # noqa: UP046
    """Reference to a typeform."""

    hint: TypeHint[T] = cast(type[T], object)
    """The type hint."""

    ctx_module: ModuleType | None = None
    """Context module for resolving forward-referenced and string types."""

    ctx: dict[str, Any] = field(default_factory=dict)
    """Context mapping for resolving forward-referenced and string types."""

    subs: Mapping[
        TypeVar | TypeVarTuple,
        TypeHint,
    ] = field(default_factory=dict)
    """Substitutions for type variables.

    These only act on open type variables in the inheritance / parameter tree,
    meaning variables that have not yet been substituted.
    """

    overrides: Mapping[TypeVar | TypeVarTuple, TypeHint] = field(default_factory=dict)
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
                    **{k.__name__: v for k, v in self.subs.items()},
                    **{k.__name__: v for k, v in self.overrides.items()},
                },
            )
            return TypeRef(
                hint,
                ctx_module=self.ctx_module,
                ctx=self.ctx,
                subs=self.subs,
                overrides=self.overrides,
            ).pure_type_def

        if isinstance(self.hint, ForwardRef):
            globalns = {
                **globals(),
                **(vars(self.ctx_module) if self.ctx_module else {}),
                **self.ctx,
                **{k.__name__: v for k, v in self.subs.items()},
                **{k.__name__: v for k, v in self.overrides.items()},
            }

            if _evaluate_forward_ref is not None:
                evaluated = _evaluate_forward_ref(
                    self.hint, globals=globalns, locals=None, type_params=()
                )
            else:  # Python 3.13: the public helper does not exist yet.
                evaluated = self.hint._evaluate(globalns, None, (), recursive_guard=frozenset())

            assert evaluated is not None
            return evaluated

        return cast(TypeExpr[T], self.hint)

    def _resolve_typevar(self, var: TypeVar | TypeVarTuple) -> TypeHint:
        res = self.overrides.get(
            var,
            self.subs.get(
                var,
                Undefined(),
            ),
        )

        if isinstance(res, Undefined) or res is var:
            if var.has_default():
                res = var.__default__
                typeref = TypeRef(
                    res, ctx_module=self.ctx_module, subs=self.subs, overrides=self.overrides
                )
                if typeref.origin is Unpack:
                    res = typeref.args[0]
            elif isinstance(var, TypeVarTuple):
                res = tuple
            elif var.__bound__ is not None:
                res = var.__bound__
            else:
                res = Any

        return res

    @cached_property
    def type_def(self) -> TypeDef[T]:
        """Resolved type definition.

        This evaluates string hints, forward references, and top-level type variables
        to rich objects.
        """
        if self.type_expr is None:
            return cast(type[T], NoneType)

        if isinstance(self.type_expr, TypeVar | TypeVarTuple):
            return TypeRef(
                self._resolve_typevar(self.type_expr),
                ctx_module=self.ctx_module,
                subs=self.subs,
                overrides=self.overrides,
            ).pure_type_def

        return self.type_expr

    @cached_property
    def origin(self) -> Any:
        """Origin of the type hint."""
        if isinstance(self.hint, str | ForwardRef):
            # Avoid evaluating the full typeform if supplied as str or ForwardRef.
            hint_str = self.hint.__forward_arg__ if isinstance(self.hint, ForwardRef) else self.hint

            if hint_str is not None:
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
                except:  # noqa: E722
                    pass

        orig = get_origin(self.type_def) or getattr(self.type_def, "__origin__", None)

        if orig is not None:
            return orig

        return self.type_def

    @property
    def args(self) -> tuple[TypeHint[T], ...]:
        """Type arguments of the type hint, unevaluated."""
        return get_args(self.type_def) or getattr(self.type_def, "__args__", ())

    @cached_property
    def annotations(self) -> tuple[Any, ...]:
        """The dynamic annotations supplied via ``typing.Annotated``."""
        annos = ()

        if self.origin is Annotated and len(self.args) > 1:
            annos = self.args[1:]
            if (
                TypeRef(
                    self.args[0],
                    ctx_module=self.module,
                    subs=self.subs,
                    overrides=self.overrides,
                ).origin
                is Annotated
            ):
                annos = (
                    TypeRef(
                        self.args[0],
                        ctx_module=self.module,
                        subs=self.subs,
                        overrides=self.overrides,
                    ).annotations
                    + annos
                )

        return annos

    @cached_property
    def pure_type_def(self) -> TypeDef[T]:
        """Resolved type definition.

        This evaluates string hints, forward references, and top-level type variables
        to rich objects, and resolves annotated types .
        """
        if self.origin is Annotated:
            return TypeRef(
                self.args[0],
                ctx_module=self.ctx_module,
                subs=self.subs,
                overrides=self.overrides,
            ).pure_type_def

        return self.type_def

    @cached_property
    def pure_origin(self) -> Any:
        """Origin of the type hint, with annotations removed."""
        if self.origin is Annotated:
            return TypeRef(
                self.args[0],
                ctx_module=self.ctx_module,
                subs=self.subs,
                overrides=self.overrides,
            ).pure_origin

        return self.origin

    @cached_property
    def module(self) -> ModuleType | None:
        """Module where the type hint is defined."""
        return getmodule(self.pure_type_def) or self.ctx_module

    @property
    def pure_args(self) -> tuple[TypeHint | LiteralArg, ...]:
        """Type arguments of the type hint, with dynamic annotations removed."""
        return get_args(self.pure_type_def) or getattr(self.pure_type_def, "__args__", ())

    @property
    def type_args(self) -> tuple[TypeHint, ...]:
        """Type arguments of the type hint, with dynamic annotations removed."""
        if self.pure_origin is Literal:
            return ()

        return cast(tuple[TypeHint, ...], self.pure_args)

    @cached_property
    def scalar_args(self) -> tuple[TypeScalar | RepType, ...]:
        """Type arguments of the type hint, recursively resolved to type scalars."""
        args = []

        for arg in self.type_args:
            if isinstance(arg, TypeVar):
                args.append(arg)
                continue

            arg_typeref = TypeRef(
                arg, ctx_module=self.module, subs=self.subs, overrides=self.overrides
            )

            if arg_typeref.pure_origin is Unpack:
                tt_typeref = TypeRef(
                    arg_typeref.type_args[0],
                    ctx_module=self.module,
                    subs=self.subs,
                    overrides=self.overrides,
                )
                args.extend(tt_typeref.scalar_args)
                continue

            if is_subtype(arg_typeref.pure_origin, tuple) and getattr(
                arg_typeref.pure_type_def, "__unpacked__", False
            ):
                if len(arg_typeref.scalar_args) == 2 and arg_typeref.scalar_args[-1] is Ellipsis:
                    rep_type = arg_typeref.scalar_args[0]
                    assert not isinstance(rep_type, RepType)
                    args.append(RepType(rep_type))
                else:
                    args.extend(arg_typeref.scalar_args)

                continue

            args.append(arg_typeref.pure_type_def)

        return tuple(args)

    @cached_property
    def members(self) -> OrderedSet[TypeMember]:
        """Union members of the type hint, or the type itself if it is not a union.

        Resolves all type aliases, which consist of unions, recursively. Members
        keep their declaration order, which callers rely on to pick a preferred
        format among several.
        """
        if isinstance(self.pure_type_def, TypeAliasType) or isinstance(
            self.pure_origin, TypeAliasType
        ):
            return TypeRef(
                self.pure_origin.__value__,
                ctx_module=self.module,
                subs=self.param_map,
                overrides=self.overrides,
            ).members

        if self.pure_origin is Union or self.pure_origin is UnionType:
            members: OrderedSet[TypeMember] = OrderedSet()

            for arg in self.type_args:
                members.update(
                    TypeRef(
                        arg,
                        ctx_module=self.module,
                        subs=self.subs,
                        overrides=self.overrides,
                    ).members
                )

            return members

        return OrderedSet([cast(TypeMember, self.pure_type_def)])

    @cached_property
    def literal_values(self) -> OrderedSet[LiteralArg]:
        """Distinct literal values, recursively resolving aliases and unions.

        Values keep their declaration order, deduplicated by both type and value.
        """
        values: OrderedSet[LiteralArg] = OrderedSet()

        for member in self.members:
            typeref = TypeRef(member, ctx_module=self.module, overrides=self.overrides)
            if typeref.pure_origin is Literal:
                values.update(member.__args__)  # pyright: ignore[reportArgumentType]

        return values

    @cached_property
    def resolved_args(self) -> tuple[TypeDef[T], ...]:
        """Fully resolved type arguments."""
        if (
            (
                is_bearable(self.pure_type_def, TypeRoot[T])
                and not (
                    isinstance(self.pure_type_def, type)
                    and issubclass(self.pure_type_def, Runtyped)
                )
            )
            or self.pure_origin is None
            or len(self.scalar_args) == 0
        ):
            return ()

        resolved = [
            (
                TypeRef(
                    arg.type if isinstance(arg, RepType) else arg,
                    ctx_module=self.module,
                    subs=self.subs,
                    overrides=self.overrides,
                ).resolved_type_def,
                isinstance(arg, RepType),
            )
            for arg in self.scalar_args
        ]

        return cast(
            tuple[TypeDef[T], ...],
            tuple((Unpack[tuple[r[0], ...]] if r[1] else r[0]) for r in resolved),
        )

    @cached_property
    def resolved_type_def(self) -> TypeDef[T]:
        """Resolved type definition with arguments recursively resolved."""
        if len(self.resolved_args) == 0:
            return self.pure_type_def

        if self.pure_origin is Union or self.pure_origin is UnionType:
            return reduce(operator.or_, (r for r in self.resolved_args))

        getitem_func = (
            self.pure_origin.__class_getitem__
            if isinstance(self.pure_origin, type)
            else self.pure_origin.__getitem__
        )

        return cast(
            TypeDef[T],
            (
                getitem_func(
                    self.resolved_args[0] if len(self.resolved_args) == 1 else self.resolved_args
                )
            ),
        )

    @cached_property
    def runtime_type_def(self) -> TypeDef[T]:
        """Resolved type definition with arguments recursively resolved and annotations applied."""
        typedef = self.resolved_type_def

        if self.origin is Annotated:
            typedef = cast(TypeDef[T], Annotated[typedef, *self.annotations])

        return typedef

    @property
    def exact_type_node(self) -> TypeNode[T] | None:
        """Resolved type definition, or None if it is a union."""
        if (
            self.pure_origin is Union
            or self.pure_origin is UnionType
            or self.pure_origin is Literal
        ):
            return None

        return cast(TypeNode[T], self.resolved_type_def)

    @cached_property
    def concrete_type(self) -> type[T] | None:
        """Concrete type of the type hint, if any.

        Unions, literals, sentinels and unresolved type variables carry no single
        concrete type and yield ``None``; parameterised generics reduce to their origin.
        """
        if self.pure_origin is Union or self.pure_origin is UnionType:
            return None

        if self.pure_origin is Literal or self.pure_origin is Sentinel:
            return None

        if isinstance(self.pure_type_def, TypeVar):
            return None
        return cast(type[T], self.pure_origin) if isinstance(self.pure_origin, type) else None

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
    def bases(self) -> set[type]:
        """Bases of the type reference."""
        return reduce(
            set.intersection,
            (set((t, *t.__bases__)) for t in self.typeset),
        )

    @cached_property
    def common_type(self) -> type[T]:
        """Common base type of the typeset, ignoring ``NoneType``."""
        return get_lowest_common_base(
            (self.typeset - {NoneType}) if len(self.typeset) > 1 else self.typeset
        )

    @property
    def approx_type_node(self) -> TypeNode[T]:
        """Resolved type definition without union, falling back to the common type."""
        return self.exact_type_node or self.common_type

    @property
    def approx_single_typeform(self) -> TypeNode[T]:
        """Resolved type definition without union, falling back to the common type."""
        return self.approx_type_node

    @property
    def params(self) -> tuple[TypeVar | TypeVarTuple, ...]:
        """Type parameters of the type hint."""
        for attr in ("__parameters__", "__type_params__"):
            params = getattr(self.pure_type_def, attr, None)
            if isinstance(params, tuple) and len(params) > 0:
                return params

        return ()

    @cached_property
    def param_map(
        self,
    ) -> Mapping[
        TypeVar | TypeVarTuple,
        TypeScalar,
    ]:
        """Mapping of argument type variables to their values.

        Each matched argument is resolved through the current substitutions and
        overrides, so the map is already fully substituted. Callers that ascend
        into a base class pass this map on as their substitutions, which only
        works when the values are resolved rather than the raw arguments.
        """
        # First get the typevar params of the typehint.
        param_args: list[tuple[TypeVar | TypeVarTuple, TypeScalar]] = []
        params = TypeRef(self.pure_origin).params

        if len(params) > 0:
            # Initialize a list of remaining type params and args,
            # which are yet to be matched.
            remaining_params = list(params)
            remaining_args = list(self.scalar_args)

            param_arg_pointer: int = 0
            next_item: Literal[-1, 0] = 0

            assert (
                sum(isinstance(a, RepType) for a in remaining_args) <= 1
            ), "Only one repeating type is allowed in type arguments"

            while len(remaining_params) > 0:
                p = remaining_params.pop(next_item)

                # If no more args are left, fill the rest of the map
                # with the type param defaults.
                if len(remaining_args) == 0:
                    param_args.insert(
                        param_arg_pointer,
                        (
                            p,
                            TypeRef(
                                p, ctx_module=self.module, subs=self.subs, overrides=self.overrides
                            ).pure_type_def,
                        ),
                    )
                    continue

                next_arg = remaining_args[next_item]
                if isinstance(next_arg, RepType):
                    if len(remaining_args) > 1:
                        # Switch direction to evaluate from the end of the remaining args
                        # backwards, so that the repeating type is evaluated last.
                        next_item = -1
                        param_arg_pointer = len(param_args)
                        continue

                    next_arg = next_arg.type
                else:
                    remaining_args.pop(next_item)

                if isinstance(p, TypeVar | TypeVarTuple):
                    # In case of a normal typevar, map the next arg.
                    param_args.insert(
                        param_arg_pointer,
                        (p, next_arg),
                    )

                if isinstance(p, TypeVarTuple):
                    # In case of a typevar-tuple, map all remaining args
                    # minus those associated to downstream typevars.
                    # There should not be any other TypeVarTuples downstream,
                    # as those are not supported in the typing module.
                    while len(remaining_args) > len(remaining_params) and not isinstance(
                        next_arg, RepType
                    ):
                        next_arg = remaining_args[next_item]
                        if isinstance(next_arg, RepType):
                            next_arg = next_arg.type
                        else:
                            remaining_args.pop(next_item)
                        param_args.insert(
                            param_arg_pointer,
                            (p, next_arg),
                        )

        return {k: v for k, v in param_args} | dict(self.subs) | dict(self.overrides)

    @cached_property
    def typevar_map(
        self,
    ) -> dict[
        TypeVar | TypeVarTuple,
        TypeRef[Any],
    ]:
        """Return a mapping of type variables to their values, including those of bases."""
        typevar_map: dict[
            TypeVar | TypeVarTuple,
            TypeRef,
        ] = {
            k: TypeRef(v, ctx_module=self.module, subs=self.param_map, overrides=self.overrides)
            for k, v in self.param_map.items()
        }

        if self.pure_origin is Union or self.pure_origin is UnionType:
            # Resolve each union member individually and union the values bound
            # to the same type variable.
            for arg in self.type_args:
                for var, value in TypeRef(
                    arg,
                    ctx_module=self.module,
                    subs=self.subs,
                    overrides=self.overrides,
                ).typevar_map.items():
                    typevar_map[var] = TypeRef(
                        (  # pyright: ignore[reportArgumentType]
                            typevar_map[var].pure_type_def
                            | value.pure_type_def  # pyright: ignore[reportOperatorIssue]
                            if var in typevar_map
                            else value.pure_type_def
                        ),
                        ctx_module=self.module,
                        subs=self.subs,
                        overrides=self.overrides,
                    )

        elif isinstance(self.pure_origin, type):
            # Resolve the type variables of the base classes, which may be parameterized.
            for base in getattr(self.pure_origin, "__orig_bases__", self.pure_origin.__bases__):
                if base is object or base is Generic or get_origin(base) is Generic:
                    continue

                typevar_map |= TypeRef(
                    base,
                    ctx_module=self.module,
                    subs=self.param_map,
                    overrides=self.overrides,
                ).typevar_map

        return typevar_map

    @cached_property
    def attr_typehints(self) -> dict[str, TypeDef] | None:
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

        annos: dict[str, TypeDef] = {}

        for name, anno in get_annotations(common_type).items():
            subs = {k: v.pure_type_def for k, v in self.typevar_map.items()}
            resolved = TypeRef(anno, ctx_module=getmodule(common_type), subs=subs)

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
                        subs=subs,
                    ).resolved_type_def

            annos[name] = resolved.resolved_type_def

        return annos

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
        if isinstance(obj, Runtyped):
            return is_subtype(type(obj), self.runtime_type_def)

        if not is_bearable(obj, cast(Any, self.runtime_type_def)):
            return False

        if isinstance(self.runtime_type_def, RuntimeValidated) or any(
            isinstance(a, RuntimeValidated | FieldInfo) for a in self.annotations
        ):
            try:
                TypeAdapter(self.runtime_type_def).validate_python(obj)
            except ValidationError:
                return False

        if with_attrs and self.attr_typehints is not None:
            if not _validate_attrs(obj, self.attr_typehints):
                return False

        return True

    def is_subtype_of(self, supertype: TypeDef[T] | TypeForm[T]) -> TypeGuard[TypeForm[T]]:
        """Check if this type reference is a subtype of the given type hint.

        Args:
            supertype: The supertype to check against.

        Returns:
            ``True`` if this reference is a subtype of ``supertype``.
        """
        super_typeref = TypeRef(supertype)

        try:
            return is_subhint(
                cast(Any, self.runtime_type_def),
                cast(Any, super_typeref.runtime_type_def),
            )
        except (BeartypeDoorNonpepException, BeartypeException, TypeError):
            # Fallback for hints beartype cannot compare: compare the concrete typesets.
            return len(super_typeref.typeset) > 0 and all(
                issubclass(t, tuple(super_typeref.typeset)) for t in self.typeset
            )

    def is_supertype_of(self, subtype: TypeDef[T] | TypeForm[T]) -> TypeGuard[TypeForm[T]]:
        """Check if this type reference is a supertype of the given type hint.

        Args:
            subtype: The subtype to check against.

        Returns:
            ``True`` if this reference is a supertype of ``subtype``.
        """
        return TypeRef(subtype).is_subtype_of(self.runtime_type_def)


def has_type[T](instance: Any, typedef: TypeForm[T] | TypeDef[T]) -> TypeIs[T]:
    """Check if an instance is of a given type hint.

    Args:
        instance: The instance to check.
        typedef: The type hint to check against.

    Returns:
        ``True`` if the instance is of the given type hint, ``False`` otherwise.
    """
    return TypeRef(typedef).validate(instance)


def is_subtype[T](
    subtype: TypeForm | TypeDef, supertype: TypeForm[T] | TypeDef[T]
) -> TypeIs[TypeForm[T]]:
    """Check if a type hint is a subtype of another type hint.

    Args:
        subtype: The subtype to check.
        supertype: The supertype to check against.

    Returns:
        ``True`` if ``subtype`` is a subtype of ``supertype``, ``False`` otherwise.
    """
    return TypeRef(subtype).is_subtype_of(supertype)


def validated[T](typedef: TypeForm[T] | TypeDef[T], obj: Any) -> T:
    """Validate an object against a type hint and return it if valid.

    Args:
        obj: The object to validate.
        typedef: The type hint to validate against.

    Returns:
        The object if it is valid.

    Raises:
        TypeError: If the object is not valid for the given type hint.
    """
    typeref = TypeRef(typedef)
    if not typeref.validate(obj):
        raise TypeError(
            f"Object {obj!r} of type {type(obj).__name__} is not valid "
            f"for type {typeref.runtime_type_def}"
        )
    return cast(T, obj)


@final
class NotProvided:
    """Sentinel for missing type arguments."""


class Runtyped:
    """Class, which is aware of its type arguments."""

    __origin__: type[Runtyped]
    __args__: tuple[TypeForm, ...]

    _typeref: TypeRef[Self]
    _generic_subclasses: dict[tuple[Hashable, ...], type[Self]]
    _generic_alias: GenericAlias | None

    def __init_subclass__(cls):  # noqa: D105
        if "__origin__" not in cls.__dict__:
            cls.__origin__ = cls
        if "__args__" not in cls.__dict__:
            cls.__args__ = ()
        if "__parameters__" not in cls.__dict__ and Generic not in cls.__mro__:
            cls.__parameters__ = ()
        if "__type_params__" not in cls.__dict__ and Generic not in cls.__mro__:
            cls.__type_params__ = ()

        orig_bases = []
        for base in cls.__bases__:
            if (
                issubclass(base, Runtyped)
                and base is not Runtyped
                and base._generic_alias is not None
            ):
                orig_bases.append(base._generic_alias)

        if len(orig_bases) > 0:
            cls.__orig_bases__ = tuple(orig_bases)

        super().__init_subclass__()

        if "_typeref" not in cls.__dict__:
            cls._typeref = TypeRef(cls)
        if "_generic_subclasses" not in cls.__dict__:
            cls._generic_subclasses = {}
        if "_generic_alias" not in cls.__dict__:
            cls._generic_alias = None

    def __class_getitem__(cls, item: TypeForm | tuple[TypeForm, ...]) -> type[Self]:  # noqa: D105
        origin = cls.__origin__

        module = get_calling_module(1)
        generic: (
            GenericAlias
        ) = super().__class_getitem__(  # pyright: ignore[reportAttributeAccessIssue]
            item
        )
        typeref = TypeRef(generic, ctx_module=module)

        full_args = tuple(
            (Unpack[tuple[r.type, ...]] if isinstance(r, RepType) else r)
            for r in typeref.scalar_args
        )

        if full_args in origin._generic_subclasses:
            return cast(type[Self], origin._generic_subclasses[full_args])

        subclass = new_class(
            f"{origin.__name__}[{', '.join(map(str, full_args))}]",
            bases=(origin,),
            exec_body=lambda ns: ns.update(
                {
                    "__origin__": origin,
                    "__orig_bases__": (generic,),
                    "__args__": tuple(full_args),
                    "__module__": module.__name__ if module is not None else None,
                    "_generic_subclasses": origin._generic_subclasses,
                    "_generic_alias": generic,
                }
            ),
        )
        origin._generic_subclasses[full_args] = subclass
        return subclass

    def narrow_instance_type(self, typevar_map: dict[TypeVar | TypeVarTuple, TypeDef]) -> None:
        """Set the type of this instance to a subclass of its current type.

        Args:
            typevar_map: A mapping of type variables to their concrete types.

        Raises:
            TypeError: If ``cls`` is not a subclass of the current type.
        """
        new_cls = TypeRef(self.__origin__, subs=typevar_map).runtime_type_def
        assert isinstance(new_cls, type)
        if not issubclass(new_cls, self.__class__):
            raise TypeError(
                f"Cannot narrow instance type to {new_cls}, "
                f"which is not a subclass of {self.__class__}"
            )
        self.__class__ = new_cls
