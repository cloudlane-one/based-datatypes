"""Tests for the type-reflection functions in :mod:`based_datatypes.typing`."""

import typing
from decimal import Decimal
from types import ModuleType
from typing import Annotated, Any, ForwardRef, Literal, TypeVar, get_args

import pydantic
import pytest

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


def test_is_type_hint_accepts_hints() -> None:
    """Classes, unions, aliases and parameterized generics are type hints."""
    assert bt.is_type_hint(int)
    assert bt.is_type_hint(int | str)
    assert bt.is_type_hint(list[int])
    assert bt.is_type_hint(AliasInt)
    assert bt.is_type_hint(AliasGeneric[int])
    assert bt.is_type_hint(Annotated[int, "meta"])


def test_members_of_a_union_keep_declaration_order() -> None:
    """A union reference yields its members in declaration order."""
    assert TypeRef(int | str).members == (int, str)
    assert TypeRef(typing.Union[int, None]).members == (int, type(None))  # noqa: UP007


def test_members_resolve_a_named_alias() -> None:
    """A named alias is resolved before splitting."""
    assert TypeRef(AliasUnion).members == (int, str)
    assert TypeRef(AliasInt).members == (int,)


def test_members_keep_a_parameterized_hint_whole() -> None:
    """A non-union hint is returned as-is, arguments intact."""
    assert TypeRef(list[int]).members == (list[int],)
    assert TypeRef(int).members == (int,)


def test_is_type_hint_rejects_values() -> None:
    """Instances and plain values are not type hints."""
    assert not bt.is_type_hint(1)
    assert not bt.is_type_hint("int")
    assert not bt.is_type_hint([1])
    assert not bt.is_type_hint({"a": 1})
    assert not bt.is_type_hint(None)


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


def test_typeref_resolves_a_string_hint() -> None:
    """A string hint is evaluated in the given context."""
    assert TypeRef("int").type_def is int


def test_typeref_resolves_a_string_hint_with_context() -> None:
    """The ``ctx`` mapping supplies names for evaluation."""
    assert TypeRef("Missing", ctx={"Missing": int}).type_def is int


def test_typeref_reports_an_unknown_string_hint() -> None:
    """An unresolvable name raises ``NameError``."""
    with pytest.raises(NameError):
        TypeRef("definitely_not_defined").type_def  # noqa: B018


def test_typeref_resolves_a_forward_ref() -> None:
    """A ``ForwardRef`` is evaluated to a real hint."""
    assert TypeRef(ForwardRef("int")).type_def is int
    assert TypeRef(ForwardRef("list[int]")).type_def == list[int]


def test_typeref_resolves_an_unbound_typevar_to_object() -> None:
    """A free type variable carries no constraint."""
    assert TypeRef(T).type_def is object


def test_typeref_resolves_a_bounded_typevar() -> None:
    """A bound type variable resolves to its bound."""
    assert TypeRef(T_bound).type_def is int


def test_typeref_resolves_a_defaulted_typevar() -> None:
    """A PEP 696 default is used when present."""
    assert TypeRef(T_default).type_def is str


def test_typeref_substitutes_a_typevar() -> None:
    """Substitutions replace a free type variable."""
    ref = TypeRef(T, subs={T: TypeRef(int)})

    assert ref.type_def is int


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
    assert TypeRef(dict[str, int]).raw_args == (str, int)


def test_typeref_args_of_a_typeref_aware_class() -> None:
    """A ``TypeAwareClass`` instance reports its applied arguments."""
    assert TypeRef(Box[int]).raw_args == (int,)


def test_typeref_typeform_round_trips_builtin_generics() -> None:
    """Resolved typeforms equal the original hints."""
    assert TypeRef(list[int]).resolved_type_def == list[int]
    assert TypeRef(dict[str, int]).resolved_type_def == dict[str, int]
    assert TypeRef(tuple[int, ...]).resolved_type_def == tuple[int, ...]
    assert TypeRef(int | None).resolved_type_def == int | None


def test_typeref_typeform_round_trips_special_forms() -> None:
    """Literals and annotations keep their metadata."""
    assert TypeRef(Literal["a", 1]).resolved_type_def == Literal["a", 1]
    assert TypeRef(Annotated[int, "meta"]).resolved_type_def == Annotated[int, "meta"]


def test_typeref_typeform_round_trips_a_typeref_aware_class() -> None:
    """An applied ``TypeAwareClass`` round-trips unchanged."""
    assert TypeRef(Box[int]).resolved_type_def is Box[int]


def test_typeref_typeform_substitutes_typevars() -> None:
    """A free type variable is replaced in the resolved typeform."""
    assert (
        TypeRef(
            list[T],  # pyright: ignore[reportGeneralTypeIssues]
            subs={T: TypeRef(int)},
        ).resolved_type_def
        == list[int]
    )
    assert (
        TypeRef(
            Box[T], subs={T: TypeRef(int)}  # pyright: ignore[reportGeneralTypeIssues]
        ).resolved_type_def
        == Box[int]
    )


def test_typeref_typeform_of_a_named_alias_is_the_alias() -> None:
    """A named alias resolves to itself."""
    assert TypeRef(AliasList).resolved_type_def is AliasList


def test_typeref_typevar_map_of_a_builtin_generic_is_empty() -> None:
    """A builtin generic binds no type variables."""
    assert TypeRef(list[int]).typevar_map == {}


def test_typeref_typevar_map_of_a_typeref_aware_class() -> None:
    """Both the class' own and inherited type variables resolve to ``int``."""
    mapping = TypeRef(ChildBox[int]).typevar_map

    assert len(mapping) > 0
    assert all(only_ref(ref).resolved_type_def is int for ref in mapping.values())


def test_typeref_typevar_map_of_a_generic_alias() -> None:
    """A generic alias binds its parameter to the argument."""
    mapping = TypeRef(Box[int]).typevar_map

    assert len(mapping) == 1
    assert only_ref(next(iter(mapping.values()))).type_def is int


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
    assert only_ref(next(iter(mapping.values()))).type_def is int


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
    assert TypeRef(bool | int).approx_single_typeform is int
    assert TypeRef(int).approx_single_typeform is int


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
    assert refs[0].type_def is str
    assert refs[1].type_def is int


def test_typeref_origin_type_of_a_forward_ref() -> None:
    """A forward reference's head is resolved without full evaluation."""
    assert TypeRef(ForwardRef("list[int]")).origin_from_str is list
    assert TypeRef("dict[str, int]").origin_from_str is dict


def test_typeref_origin_type_of_a_union_string() -> None:
    """A union string evaluates to the union itself."""
    assert TypeRef("int | str").origin_from_str == int | str


def test_typeref_origin_type_of_an_invalid_string() -> None:
    """An unresolvable string reports no origin type."""
    assert TypeRef("not a type!").origin_from_str is None


def test_typeref_origin_type_of_a_plain_hint() -> None:
    """A non-string hint delegates to ``get_origin``."""
    assert TypeRef(list[int]).origin_from_str is list
    assert TypeRef(int).origin_from_str is None


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

    assert TypeRef("Alias", ctx_module=module).type_def == list[int]


def test_none_type_is_exported() -> None:
    """The module re-exports ``NoneType`` for convenience."""
    assert bt.NoneType is type(None)


def test_decimal_is_a_supported_ordinal() -> None:
    """``Decimal`` is part of the exported ordinal alias."""
    from based_datatypes.types import Ordinal

    assert Decimal in get_args(Ordinal.__value__)
