"""Tests for the type-reflection functions in :mod:`based_datatypes.typing`."""

import typing
from collections.abc import Callable, Sequence
from decimal import Decimal
from types import ModuleType
from typing import Annotated, Any, ForwardRef, Literal, TypeVar, get_args

import pydantic
import pytest
from beartype.roar import BeartypeDoorNonpepException

from based_datatypes import typing as bt
from based_datatypes.typing import TypeAwareClass, TypeRef

T = TypeVar("T")
T_bound = TypeVar("T_bound", bound=int)
T_default = TypeVar("T_default", default=str)

type AliasInt = int
type AliasList = list[int]
type AliasUnion = int | str
type AliasGeneric[T] = list[T]
type MaybeInt = int | None

type LiteralAlias = Literal["a", 1]
type NestedLiteralAlias = LiteralAlias
type LiteralUnionAlias = NestedLiteralAlias | Literal["b", 2]
type RecursiveLiteralAlias = Literal["a"] | list[RecursiveLiteralAlias]

K = TypeVar("K")
V = TypeVar("V")


def only_ref(value: TypeRef | tuple[TypeRef, ...]) -> TypeRef:
    """Narrow a typevar-map value to a scalar reference."""
    assert isinstance(value, TypeRef)
    return value


class Box[T](TypeAwareClass):
    """Generic helper class for the type-argument tests."""


class ChildBox[T](Box[T]):
    """Generic subclass used to exercise base ascent."""


class Pair[K, V](TypeAwareClass):
    """Two-parameter generic class."""


def test_typedef_to_typeset_of_a_class() -> None:
    """A plain class maps to itself."""
    assert bt.typedef_to_typeset(int) == {int}


def test_typedef_to_typeset_of_none() -> None:
    """``None`` maps to ``NoneType``."""
    assert bt.typedef_to_typeset(None) == {bt.NoneType}


def test_typedef_to_typeset_of_a_union() -> None:
    """Every union member is represented."""
    assert bt.typedef_to_typeset(int | str) == {int, str}


def test_typedef_to_typeset_of_a_nullable_union() -> None:
    """An optional type includes ``NoneType`` by default."""
    assert bt.typedef_to_typeset(int | None) == {int, bt.NoneType}


def test_typedef_to_typeset_can_remove_null() -> None:
    """``remove_null`` drops ``NoneType`` from the result."""
    assert bt.typedef_to_typeset(int | None, remove_null=True) == {int}


def test_typedef_to_typeset_of_a_generic_alias() -> None:
    """A parameterised generic maps to its origin."""
    assert bt.typedef_to_typeset(list[int]) == {list}
    assert bt.typedef_to_typeset(dict[str, int]) == {dict}


def test_typedef_to_typeset_of_a_literal() -> None:
    """Literals carry no concrete class and map to ``object``."""
    assert bt.typedef_to_typeset(Literal["a", 1]) == {object}


def test_typedef_to_typeset_of_a_typevar() -> None:
    """An unresolved type variable maps to ``object``."""
    assert bt.typedef_to_typeset(T) == {object}


def test_typedef_to_typeset_of_a_newtype() -> None:
    """A ``NewType`` maps to its supertype."""
    UserId = typing.NewType("UserId", int)

    assert bt.typedef_to_typeset(UserId) == {int}


def test_typedef_to_typeset_of_a_named_alias() -> None:
    """A named alias maps to the typeset of its value."""
    assert bt.typedef_to_typeset(AliasInt) == {int}
    assert bt.typedef_to_typeset(AliasList) == {list}
    assert bt.typedef_to_typeset(AliasUnion) == {int, str}


def test_typedef_to_typeset_of_a_parameterised_alias() -> None:
    """A parameterised named alias resolves its value's origin."""
    assert bt.typedef_to_typeset(AliasGeneric[int]) == {list}


def test_typedef_to_typeset_of_a_union_containing_an_alias() -> None:
    """Aliases nested inside a union are flattened."""
    assert bt.typedef_to_typeset(AliasInt | None) == {  # pyright: ignore[reportArgumentType]
        int,
        bt.NoneType,
    }


def test_typedef_to_typeset_of_a_forward_ref() -> None:
    """An unevaluated forward reference maps to ``object``."""
    assert bt.typedef_to_typeset(ForwardRef("int")) == {object}


def test_typedef_to_typeset_of_an_annotated_hint() -> None:
    """``Annotated`` maps to its underlying type."""
    assert bt.typedef_to_typeset(Annotated[list[int], "meta"]) == {list}


def test_is_subtype_of_a_builtin() -> None:
    """``bool`` is a subtype of ``int``, but not the other way round."""
    assert bt.is_subtype(bool, int)
    assert not bt.is_subtype(int, bool)


def test_is_subtype_against_object_and_any() -> None:
    """Every type is a subtype of ``object`` and of ``Any``."""
    assert bt.is_subtype(int, object)
    assert bt.is_subtype(int, Any)


def test_is_subtype_of_a_union() -> None:
    """Each union member is a subtype of the union."""
    assert bt.is_subtype(int, int | str)
    assert bt.is_subtype(int | str, object)


def test_is_subtype_of_a_generic() -> None:
    """Generic arguments are compared, not just the origin."""
    assert bt.is_subtype(list[int], list)
    assert not bt.is_subtype(list[int], list[str])
    assert bt.is_subtype(dict[str, int], typing.Mapping[str, int])


def test_is_subtype_of_an_annotated_hint() -> None:
    """Annotations do not change subtyping."""
    assert bt.is_subtype(Annotated[int, "meta"], int)


def test_is_subtype_of_a_literal() -> None:
    """A literal value type is a subtype of its value type."""
    assert bt.is_subtype(Literal["a"], str)
    assert not bt.is_subtype(str, Literal["a"])


def test_is_subtype_of_a_named_alias() -> None:
    """A named alias is resolved before comparison."""
    assert bt.is_subtype(AliasInt, object)
    assert not bt.is_subtype(AliasUnion, int)


def test_is_subtype_falls_back_to_typesets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hints beartype refuses are compared through their typesets."""

    def _raise(*args: object, **kwargs: object) -> bool:
        raise BeartypeDoorNonpepException("unsupported")

    monkeypatch.setattr(bt, "is_subhint", _raise)

    assert bt.is_subtype(int, object)
    assert not bt.is_subtype(list[int], dict)


def test_has_type_matches_values() -> None:
    """``has_type`` validates a runtime value against a hint."""
    assert bt.has_type(1, int)
    assert not bt.has_type("1", int)
    assert bt.has_type([1], list[int])
    assert not bt.has_type(["x"], list[int])


def test_get_lowest_common_base_of_an_empty_iterable() -> None:
    """An empty input yields ``object``."""
    assert bt.get_lowest_common_base([]) is object


def test_get_lowest_common_base_of_related_types() -> None:
    """The most derived shared base is chosen."""
    assert bt.get_lowest_common_base([bool, int]) is int


def test_get_lowest_common_base_of_unrelated_types() -> None:
    """Unrelated classes share only ``object``."""
    assert bt.get_lowest_common_base([list, dict]) is object


def test_get_lowest_common_base_accepts_a_generator() -> None:
    """A one-shot iterable is consumed only once."""
    assert bt.get_lowest_common_base(t for t in (bool, int)) is int


def test_extract_nullable_type_of_a_plain_type() -> None:
    """A non-union hint is returned unchanged."""
    assert bt.extract_nullable_type(int) is int


def test_extract_nullable_type_of_an_optional() -> None:
    """The non-``None`` member is extracted."""
    assert bt.extract_nullable_type(int | None) is int


def test_extract_nullable_type_of_none() -> None:
    """``None`` has no non-null base."""
    assert bt.extract_nullable_type(None) is None


def test_extract_nullable_type_uses_the_common_base() -> None:
    """Multiple members collapse to their common base."""
    assert bt.extract_nullable_type(int | str | None) is object


def test_extract_nullable_type_of_an_alias() -> None:
    """A nullable named alias resolves through its typeset."""
    assert bt.extract_nullable_type(MaybeInt) is int


def test_get_inheritance_distance_for_the_same_class() -> None:
    """A class is at distance zero from itself."""
    assert bt.get_inheritance_distance(int, int) == 0


def test_get_inheritance_distance_down_the_hierarchy() -> None:
    """Descending two levels reports a positive distance."""
    assert bt.get_inheritance_distance(bool, object) == 2


def test_get_inheritance_distance_is_signed() -> None:
    """Reversing the arguments reverses the sign."""
    assert bt.get_inheritance_distance(bool, int) == 1
    assert bt.get_inheritance_distance(int, bool) == -1


def test_get_inheritance_distance_of_unrelated_classes() -> None:
    """Unrelated classes have no distance."""
    assert bt.get_inheritance_distance(int, str) is None


def test_get_inheritance_distance_requires_classes() -> None:
    """Non-class arguments yield ``None``."""
    assert bt.get_inheritance_distance(1, int) is None  # type: ignore[arg-type]


def test_set_typeargs_positionally() -> None:
    """Positional type arguments replace the parameters."""
    assert bt.set_typeargs(list[T], [int]) == list[int]  # pyright: ignore[reportGeneralTypeIssues]
    assert bt.set_typeargs(Box[T], [int]) == Box[int]  # pyright: ignore[reportGeneralTypeIssues]


def test_set_typeargs_by_typevar() -> None:
    """A mapping replaces the selected type variable."""
    assert (
        bt.set_typeargs(list[T], {T: str}) == list[str]  # pyright: ignore[reportGeneralTypeIssues]
    )
    assert (
        bt.set_typeargs(Pair[K, V], {V: int})  # pyright: ignore[reportGeneralTypeIssues]
        == Pair[K, int]  # pyright: ignore[reportGeneralTypeIssues]
    )


def test_set_typeargs_rejects_non_generic_hints() -> None:
    """A bare class has no type arguments to set."""
    with pytest.raises(TypeError, match="not a generic type definition"):
        bt.set_typeargs(int, [str])  # type: ignore[arg-type]


def test_set_typeargs_rejects_an_empty_argument_list() -> None:
    """At least one replacement is required."""
    with pytest.raises(ValueError, match="At least one type argument"):
        bt.set_typeargs(list[T], [])  # pyright: ignore[reportGeneralTypeIssues]


def test_set_typeargs_rejects_an_unknown_typevar() -> None:
    """A type variable the hint does not use is refused."""
    with pytest.raises(ValueError, match="not a parameter"):
        bt.set_typeargs(
            list[T],  # pyright: ignore[reportGeneralTypeIssues]
            {TypeVar("Other"): int},
        )


def test_get_typeargs_of_a_non_generic_instance() -> None:
    """Instances without ``__orig_class__`` report no arguments."""
    assert bt.get_typeargs([1]) is None
    assert bt.get_typeargs(object()) is None


def test_get_typeargs_of_a_generic_instance() -> None:
    """A parameterised instance exposes its arguments."""

    class Generic[T]:
        pass

    instance = Generic[int]()

    assert bt.get_typeargs(instance) == (int,)


def test_typeref_resolves_a_string_hint() -> None:
    """A string hint is evaluated in the given context."""
    assert TypeRef("int").typedef is int


def test_typeref_resolves_a_string_hint_with_context() -> None:
    """The ``ctx`` mapping supplies names for evaluation."""
    assert TypeRef("Missing", ctx={"Missing": int}).typedef is int


def test_typeref_reports_an_unknown_string_hint() -> None:
    """An unresolvable name raises ``NameError``."""
    with pytest.raises(NameError):
        TypeRef("definitely_not_defined").typedef  # noqa: B018


def test_typeref_resolves_a_forward_ref() -> None:
    """A ``ForwardRef`` is evaluated to a real hint."""
    assert TypeRef(ForwardRef("int")).typedef is int
    assert TypeRef(ForwardRef("list[int]")).typedef == list[int]


def test_typeref_resolves_an_unbound_typevar_to_object() -> None:
    """A free type variable carries no constraint."""
    assert TypeRef(T).typedef is object


def test_typeref_resolves_a_bounded_typevar() -> None:
    """A bound type variable resolves to its bound."""
    assert TypeRef(T_bound).typedef is int


def test_typeref_resolves_a_defaulted_typevar() -> None:
    """A PEP 696 default is used when present."""
    assert TypeRef(T_default).typedef is str


def test_typeref_substitutes_a_typevar() -> None:
    """Substitutions replace a free type variable."""
    ref = TypeRef(T, subs={T: TypeRef(int)})

    assert ref.typedef is int


def test_typeref_origin_of_a_generic_alias() -> None:
    """The origin is the unsubscripted class."""
    assert TypeRef(list[int]).origin is list


def test_typeref_origin_of_a_typeref_aware_class() -> None:
    """A ``TypeAwareClass`` instance reports its declaration class."""
    assert TypeRef(Box[int]).origin is Box


def test_typeref_origin_of_a_named_alias() -> None:
    """A named alias reports its value's origin."""
    assert TypeRef(AliasList).origin is list


def test_typeref_params_of_a_builtin_generic_are_empty() -> None:
    """Builtin generics declare no parameters."""
    assert TypeRef(list[int]).params == ()


def test_typeref_params_of_a_typeref_aware_class() -> None:
    """A ``TypeAwareClass`` exposes the declaration's parameters."""
    assert len(TypeRef(Box[int]).params) == 1


def test_typeref_params_of_literal_and_annotated_are_empty() -> None:
    """Special forms have no type parameters."""
    assert TypeRef(Literal["a"]).params == ()
    assert TypeRef(Annotated[int, "meta"]).params == ()


def test_typeref_args_of_a_generic_alias() -> None:
    """Arguments are reported positionally."""
    assert TypeRef(dict[str, int]).args == (str, int)


def test_typeref_args_of_a_typeref_aware_class() -> None:
    """A ``TypeAwareClass`` instance reports its applied arguments."""
    assert TypeRef(Box[int]).args == (int,)


def test_typeref_typeform_round_trips_builtin_generics() -> None:
    """Resolved typeforms equal the original hints."""
    assert TypeRef(list[int]).typeform == list[int]
    assert TypeRef(dict[str, int]).typeform == dict[str, int]
    assert TypeRef(tuple[int, ...]).typeform == tuple[int, ...]
    assert TypeRef(int | None).typeform == int | None


def test_typeref_typeform_round_trips_special_forms() -> None:
    """Literals and annotations keep their metadata."""
    assert TypeRef(Literal["a", 1]).typeform == Literal["a", 1]
    assert TypeRef(Annotated[int, "meta"]).typeform == Annotated[int, "meta"]


def test_typeref_typeform_round_trips_a_typeref_aware_class() -> None:
    """An applied ``TypeAwareClass`` round-trips unchanged."""
    assert TypeRef(Box[int]).typeform is Box[int]


def test_typeref_typeform_substitutes_typevars() -> None:
    """A free type variable is replaced in the resolved typeform."""
    assert (
        TypeRef(
            list[T],  # pyright: ignore[reportGeneralTypeIssues]
            subs={T: TypeRef(int)},
        ).typeform
        == list[int]
    )
    assert (
        TypeRef(Box[T], subs={T: TypeRef(int)}).typeform  # pyright: ignore[reportGeneralTypeIssues]
        == Box[int]
    )


def test_typeref_typeform_of_a_named_alias_is_the_alias() -> None:
    """A named alias resolves to itself."""
    assert TypeRef(AliasList).typeform is AliasList


def test_typeref_typevar_map_of_a_builtin_generic_is_empty() -> None:
    """A builtin generic binds no type variables."""
    assert TypeRef(list[int]).typevar_map == {}


def test_typeref_typevar_map_of_a_typeref_aware_class() -> None:
    """Both the class' own and inherited type variables resolve to ``int``."""
    mapping = TypeRef(ChildBox[int]).typevar_map

    assert len(mapping) > 0
    assert all(only_ref(ref).typeform is int for ref in mapping.values())


def test_typeref_typevar_map_of_a_generic_alias() -> None:
    """A generic alias binds its parameter to the argument."""
    mapping = TypeRef(Box[int]).typevar_map

    assert len(mapping) == 1
    assert only_ref(next(iter(mapping.values()))).typedef is int


def test_typeref_typevar_map_of_a_union_is_empty_for_concrete_members() -> None:
    """A union of concrete types binds nothing."""
    assert TypeRef(int | str).typevar_map == {}


def test_typeref_scalar_typevar_map_excludes_typevar_tuples() -> None:
    """Only scalar type-variable entries survive."""
    mapping = TypeRef(Box[int]).scalar_typevar_map

    assert set(mapping) == set(TypeRef(Box[int]).typevar_map)


def test_typeref_local_typevar_map_maps_parameters_to_arguments() -> None:
    """The local map binds the declaration's parameters."""
    mapping = TypeRef(Box[int]).local_typevar_map

    assert len(mapping) == 1
    assert only_ref(next(iter(mapping.values()))).typedef is int


def test_typeref_local_typevar_map_of_a_typevar_argument() -> None:
    """An applied type variable is bound to the argument that was supplied."""
    mapping = TypeRef(Box[T]).local_typevar_map  # pyright: ignore[reportGeneralTypeIssues]

    assert len(mapping) == 1
    assert only_ref(next(iter(mapping.values()))).hint is T


def test_typeref_typeset_and_common_type() -> None:
    """The typeset and its common base describe a union."""
    ref = TypeRef(bool | int | None)

    assert ref.typeset == {bool, int, bt.NoneType}
    assert ref.common_type is int


def test_typeref_single_typedef_of_a_union() -> None:
    """A union collapses to its common type."""
    assert TypeRef(bool | int).single_typedef is int
    assert TypeRef(int).single_typedef is int


def test_typeref_bases_of_a_plain_class() -> None:
    """A plain class reports its direct bases."""
    assert TypeRef(list[int]).bases == (object,)


def test_typeref_bases_of_a_typeref_aware_class() -> None:
    """An applied class reports its declaration class as base."""
    assert TypeRef(Box[int]).bases == (Box,)


def test_typeref_annotation_and_literal_values() -> None:
    """Dynamic metadata and literal values are exposed."""
    assert TypeRef(Annotated[int, "meta"]).annotation == "meta"
    assert TypeRef(int).annotation is None
    assert TypeRef(Literal["a", 1]).literal_values == ("a", 1)
    assert TypeRef(int).literal_values == ()


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        (LiteralAlias, ("a", 1)),
        (NestedLiteralAlias, ("a", 1)),
        (LiteralUnionAlias, ("a", 1, "b", 2)),
        (Literal["a", 1] | Literal["b", 2], ("a", 1, "b", 2)),
        (typing.Union[Literal["a"], Literal["b"]], ("a", "b")),  # noqa: UP007
        (LiteralUnionAlias | Literal["c"], ("a", 1, "b", 2, "c")),
        (LiteralAlias | NestedLiteralAlias, ("a", 1)),
        (LiteralAlias | Literal["a", "b"], ("a", 1, "b")),
        (Literal[1] | Literal[True], (1, True)),
        (LiteralAlias | int | None, ("a", 1)),
        (AliasInt, ()),
        (AliasUnion, ()),
        (list[Literal["a"]], ()),
        (RecursiveLiteralAlias, ("a",)),
    ],
)
def test_typeref_literal_values_resolves_aliases_and_unions(
    hint: Any, expected: tuple[Any, ...]
) -> None:
    """Aliases and unions expose distinct literal values in declaration order."""
    values = TypeRef(hint).literal_values

    assert values == expected
    assert tuple(type(value) for value in values) == tuple(type(value) for value in expected)


def test_typeref_arg_typerefs() -> None:
    """Each type argument is wrapped in its own reference."""
    refs = TypeRef(dict[str, int]).arg_typerefs

    assert len(refs) == 2
    assert refs[0].typedef is str
    assert refs[1].typedef is int


def test_typeref_origin_type_of_a_forward_ref() -> None:
    """A forward reference's head is resolved without full evaluation."""
    assert TypeRef(ForwardRef("list[int]")).origin_type is list
    assert TypeRef("dict[str, int]").origin_type is dict


def test_typeref_origin_type_of_a_union_string() -> None:
    """A union string evaluates to the union itself."""
    assert TypeRef("int | str").origin_type == int | str


def test_typeref_origin_type_of_an_invalid_string() -> None:
    """An unresolvable string reports no origin type."""
    assert TypeRef("not a type!").origin_type is None


def test_typeref_origin_type_of_a_plain_hint() -> None:
    """A non-string hint delegates to ``get_origin``."""
    assert TypeRef(list[int]).origin_type is list
    assert TypeRef(int).origin_type is None


def test_typeref_validate_accepts_matching_values() -> None:
    """Values of the right type validate."""
    assert TypeRef(int).validate(1)
    assert TypeRef(list[int]).validate([1])


def test_typeref_validate_rejects_other_values() -> None:
    """Values of the wrong type are rejected without raising."""
    assert not TypeRef(int).validate("x")
    assert not TypeRef(list[int]).validate(["x"])


def test_typeref_validate_enforces_annotated_field_constraints() -> None:
    """Pydantic ``Field`` metadata in ``Annotated`` is enforced."""
    ref = TypeRef(Annotated[int, pydantic.Field(gt=0)])

    assert ref.validate(3)
    assert not ref.validate(-1)


def test_typeref_validate_accepts_pydantic_instances() -> None:
    """A pydantic model hint validates instances, not raw mappings."""

    class Model(pydantic.BaseModel):
        value: int

    assert TypeRef(Model).validate(Model(value=1))
    assert not TypeRef(Model).validate({"value": 1})


def test_typeref_validate_ignores_plain_metadata() -> None:
    """Non-validating metadata does not change the outcome."""
    assert TypeRef(Annotated[int, "meta"]).validate(1)
    assert not TypeRef(Annotated[int, "meta"]).validate("x")


def test_typeref_equality_is_field_based() -> None:
    """References compare by their configuration."""
    assert TypeRef(int) == TypeRef(int)
    assert TypeRef(list[int]) != TypeRef(dict[str, int])


def test_typeref_accepts_a_context_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """The context module is used to resolve a string hint."""
    module = ModuleType("fake_ctx_module")
    module.__dict__["Alias"] = list[int]
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)

    assert TypeRef("Alias", ctx_module=module).typedef == list[int]


def test_none_type_is_exported() -> None:
    """The module re-exports ``NoneType`` for convenience."""
    assert bt.NoneType is type(None)


def test_public_names_are_defined() -> None:
    """The documented public helpers exist."""
    for name in (
        "typedef_to_typeset",
        "is_subtype",
        "has_type",
        "get_lowest_common_base",
        "extract_nullable_type",
        "get_inheritance_distance",
        "set_typeargs",
        "get_typeargs",
        "TypeRef",
        "TypeAwareClass",
        "is_frozen_dataclass",
        "is_immutable",
        "get_nondefault_methods",
        "get_own_attrs",
    ):
        assert hasattr(bt, name)


def test_decimal_is_a_supported_ordinal() -> None:
    """``Decimal`` is part of the exported ordinal alias."""
    from based_datatypes.types import Ordinal

    assert Decimal in get_args(Ordinal.__value__)


def test_sequence_matches_list() -> None:
    """``list`` satisfies the ``Sequence`` ABC contract used by beartype."""
    assert bt.is_subtype(list[int], Sequence[int])


def test_callable_subtyping() -> None:
    """Callables are compared structurally."""
    assert bt.is_subtype(Callable[[int], str], Callable)
