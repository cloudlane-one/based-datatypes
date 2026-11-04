"""Tests for the type-reflection functions in :mod:`based_datatypes.typing`."""

import dataclasses
import typing
from decimal import Decimal
from types import ModuleType
from typing import Annotated, Any, ForwardRef, Generic, Literal, TypeVar, get_args

import pydantic
import pytest
from beartype.door import is_bearable

from based_datatypes import typing as bt
from based_datatypes.files import File, LocalFile
from based_datatypes.typing import Runtyped, TypeRef, is_subtype

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


class Box[T](Runtyped):
    """Generic class aware of its type arguments."""

    def __init__(self, item: T | None = None) -> None:
        self.item = item


class ChildBox[T](Box[T]):
    """Generic subclass used to exercise base ascent."""


class Pair[K, V](Runtyped):
    """Two-parameter generic class."""

    def __init__(self, key: K | None = None, value: V | None = None) -> None:
        self.key = key
        self.value = value


def test_is_type_hint_accepts_hints() -> None:
    """Classes, unions, aliases and parameterized generics are type hints."""
    assert bt.is_type_hint(int)
    assert bt.is_type_hint(int | str)
    assert bt.is_type_hint(list[int])
    assert bt.is_type_hint(AliasInt)
    assert bt.is_type_hint(AliasGeneric[int])
    assert bt.is_type_hint(Annotated[int, "meta"])


def test_members_resolve_a_named_alias() -> None:
    """A named alias is resolved before splitting."""
    assert TypeRef(AliasUnion).members == {int, str}
    assert TypeRef(AliasInt).members == {int}


def test_members_keep_a_parameterized_hint_whole() -> None:
    """A non-union hint is returned as-is, arguments intact."""
    assert TypeRef(list[int]).members == {list[int]}
    assert TypeRef(int).members == {int}


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


def test_typeref_resolves_a_string_hint() -> None:
    """A string hint is evaluated in the given context."""
    assert TypeRef("int").pure_type_def is int


def test_typeref_resolves_a_string_hint_with_context() -> None:
    """The ``ctx`` mapping supplies names for evaluation."""
    assert TypeRef("Missing", ctx={"Missing": int}).pure_type_def is int


def test_typeref_reports_an_unknown_string_hint() -> None:
    """An unresolvable name raises ``NameError``."""
    with pytest.raises(NameError):
        TypeRef("definitely_not_defined").pure_type_def  # noqa: B018


def test_typeref_resolves_a_forward_ref() -> None:
    """A ``ForwardRef`` is evaluated to a real hint."""
    assert TypeRef(ForwardRef("int")).pure_type_def is int
    assert TypeRef(ForwardRef("list[int]")).pure_type_def == list[int]


def test_typeref_resolves_an_unbound_typevar_to_object() -> None:
    """A free type variable carries no constraint."""
    assert TypeRef(T).pure_type_def is Any


def test_typeref_resolves_a_bounded_typevar() -> None:
    """A bound type variable resolves to its bound."""
    assert TypeRef(T_bound).pure_type_def is int


def test_typeref_resolves_a_defaulted_typevar() -> None:
    """A PEP 696 default is used when present."""
    assert TypeRef(T_default).pure_type_def is str


def test_typeref_substitutes_a_typevar() -> None:
    """Substitutions replace a free type variable."""
    ref = TypeRef(T, subs={T: int})

    assert ref.pure_type_def is int


def test_typeref_origin_of_a_generic_alias() -> None:
    """The origin is the unsubscripted class."""
    assert TypeRef(list[int]).origin is list


def test_typeref_origin_of_a_runtyped_class() -> None:
    """A ``Runtyped`` instance reports its declaration class."""
    assert TypeRef(Box[int]).origin is Box


def test_typeref_params_of_a_builtin_generic_are_empty() -> None:
    """Builtin generics declare no parameters."""
    assert TypeRef(list[int]).params == ()


def test_typeref_params_of_literal_and_annotated_are_empty() -> None:
    """Special forms have no type parameters."""
    assert TypeRef(Literal["a"]).params == ()
    assert TypeRef(Annotated[int, "meta"]).params == ()


def test_typeref_args_of_a_generic_alias() -> None:
    """Arguments are reported positionally."""
    assert TypeRef(dict[str, int]).args == (str, int)


def test_typeref_args_of_a_runtyped_class() -> None:
    """A ``Runtyped`` instance reports its applied arguments."""
    assert TypeRef(Box[int]).args == (int,)


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


def test_typeref_typeform_round_trips_a_runtyped_class() -> None:
    """An applied ``Runtyped`` round-trips unchanged."""
    assert TypeRef(Box[int]).resolved_type_def is Box[int]


def test_typeref_typeform_substitutes_typevars() -> None:
    """A free type variable is replaced in the resolved typeform."""
    assert (
        TypeRef(
            list[T],  # pyright: ignore[reportGeneralTypeIssues]
            subs={T: int},
        ).resolved_type_def
        == list[int]
    )
    assert (
        TypeRef(
            Box[T],  # pyright: ignore[reportGeneralTypeIssues]
            subs={T: int},
        ).resolved_type_def
        == Box[int]
    )


def test_typeref_typeform_of_a_named_alias_is_the_alias() -> None:
    """A named alias resolves to itself."""
    assert TypeRef(AliasList).resolved_type_def is AliasList


def test_typeref_typevar_map_of_a_builtin_generic_is_empty() -> None:
    """A builtin generic binds no type variables."""
    assert TypeRef(list[int]).typevar_map == {}


def test_typeref_typevar_map_of_a_runtyped_class() -> None:
    """Both the class' own and inherited type variables resolve to ``int``."""
    mapping = TypeRef(ChildBox[int]).typevar_map

    assert len(mapping) > 0
    assert all(only_ref(ref).resolved_type_def is int for ref in mapping.values())


def test_typeref_typevar_map_of_a_generic_alias() -> None:
    """A generic alias binds its parameter to the argument."""
    mapping = TypeRef(Box[int]).typevar_map

    assert len(mapping) == 1
    assert only_ref(next(iter(mapping.values()))).pure_type_def is int


def test_typeref_typevar_map_of_a_union_is_empty_for_concrete_members() -> None:
    """A union of concrete types binds nothing."""
    assert TypeRef(int | str).typevar_map == {}


def test_typeref_scalar_typevar_map_excludes_typevar_tuples() -> None:
    """Only scalar type-variable entries survive."""
    mapping = TypeRef(Box[int]).typevar_map

    assert set(mapping) == set(TypeRef(Box[int]).typevar_map)


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
    assert TypeRef(list[int]).bases == {list, object}


def test_typeref_bases_of_a_runtyped_class() -> None:
    """An applied class reports its declaration class as base."""
    assert TypeRef(Box[int]).bases == {Generic, Runtyped, Box}


def test_typeref_annotation_and_literal_values() -> None:
    """Dynamic metadata and literal values are exposed."""
    assert TypeRef(Annotated[int, "meta"]).annotations[0] == "meta"
    assert len(TypeRef(int).annotations) == 0
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


def test_typeref_origin_type_of_a_forward_ref() -> None:
    """A forward reference's head is resolved without full evaluation."""
    assert TypeRef(ForwardRef("list[int]")).origin is list
    assert TypeRef("dict[str, int]").origin is dict


def test_typeref_origin_type_of_a_union_string() -> None:
    """A union string evaluates to the union itself."""
    assert TypeRef("int | str").origin is int | str


def test_typeref_origin_type_of_an_invalid_string() -> None:
    """An unresolvable string reports no origin type."""
    assert TypeRef("not a type!").origin is None


def test_typeref_origin_type_of_a_plain_hint() -> None:
    """A non-string hint delegates to ``get_origin``."""
    assert TypeRef(list[int]).origin is list
    assert TypeRef(int).origin is None


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

    assert TypeRef("Alias", ctx_module=module).pure_type_def == list[int]


def test_none_type_is_exported() -> None:
    """The module re-exports ``NoneType`` for convenience."""
    assert bt.NoneType is type(None)


def test_decimal_is_a_supported_ordinal() -> None:
    """``Decimal`` is part of the exported ordinal alias."""
    from based_datatypes.types import Ordinal

    assert Decimal in get_args(Ordinal.__value__)


class TestRuntypedComparison:
    """Tests for comparing applied ``Runtyped`` hints."""

    def test_arguments_decide_the_subtype_relation(self) -> None:
        """A file hint fits only a hint whose MIME type and IO type it satisfies."""
        assert is_subtype(
            File[Literal["application/json"], Any],
            File[Literal["application/json", "application/yaml"], Any],
        )
        assert not is_subtype(
            File[Literal["application/pickle"], Any],
            File[Literal["application/json"], Any],
        )

    def test_trailing_parameters_may_be_omitted(self) -> None:
        """A hint that supplies fewer arguments still compares the ones it has."""
        assert is_subtype(
            LocalFile[Literal["application/json"], Any, Any],
            File[Literal["application/json"], Any],
        )

    def test_beartype_does_not_inspect_arguments_of_a_value(self) -> None:
        """beartype ignores the type arguments of an instance, which is why we compare them.

        This pins the reason :meth:`TypeRef.validate` compares ``Runtyped``
        hints structurally: if beartype ever starts honouring generic arguments of
        values, that comparison could delegate to it.
        """
        assert is_bearable(Box[int](), Box[str])


class Nested[T](Box[T]):
    """Generic subclass of another aware class."""


@dataclasses.dataclass(frozen=True)
class FrozenRecord:
    """Frozen dataclass used by the immutability checks."""

    value: int


@dataclasses.dataclass
class MutableRecord:
    """Mutable dataclass used by the immutability checks."""

    value: int


class BaseWithMethod:
    """Base class contributing an inherited method."""

    base_attr: int = 0

    def base_method(self) -> str:
        """Return a marker from the base class."""
        return "base"


class WithMethod(BaseWithMethod):
    """Class defining its own attributes on top of a base."""

    own_attr: int = 1

    def own_method(self) -> str:
        """Return a marker from the subclass."""
        return "own"

    def __str__(self) -> str:
        return "with-method"


def test_class_getitem_creates_a_subclass() -> None:
    """Subscripting a ``Runtyped`` yields a real subclass."""
    assert issubclass(Box[int], Box)
    assert isinstance(Box[int], type)


def test_class_getitem_records_origin_and_arguments() -> None:
    """The specialised class records its declaration and arguments."""
    assert Box[int].__origin__ is Box
    assert Box[int].__args__ == (int,)


def test_class_getitem_caches_specialisations() -> None:
    """Repeated subscription returns the same class object."""
    assert Box[int] is Box[int]
    assert Box[int] is not Box[str]


def test_class_getitem_supports_multiple_parameters() -> None:
    """Several arguments are stored as a tuple."""
    assert Pair[str, int].__args__ == (str, int)
    assert Pair[str, int].__origin__ is Pair


def test_specialised_class_is_instantiable() -> None:
    """A specialised class is a normal, instantiable class."""
    instance = Box[int](1)

    assert isinstance(instance, Box)
    assert instance.item == 1


def test_type_params_of_the_declaration() -> None:
    """The declaration exposes its type parameters."""
    assert len(TypeRef(Box).params) == 1
    assert len(TypeRef(Pair).params) == 2


def test_typeargs_of_a_specialisation_resolve_to_the_argument() -> None:
    """The type-argument map resolves the parameter to the applied type."""
    mapping = TypeRef(Box[int]).typevar_map

    assert len(mapping) == 1
    value = next(iter(mapping.values()))
    assert not isinstance(value, tuple)
    assert value.pure_type_def is int


def test_typeargs_of_a_two_parameter_specialisation() -> None:
    """Each parameter is bound to its own argument."""
    mapping = TypeRef(Pair[str, int]).typevar_map

    assert len(mapping) == 2
    resolved = {
        key.__name__: value.pure_type_def
        for key, value in mapping.items()
        if not isinstance(value, tuple)
    }
    assert set(resolved.values()) == {str, int}


def test_typeargs_are_stable_per_class() -> None:
    """Resolving the same specialisation twice yields the same bindings."""
    first = TypeRef(Box[int]).typevar_map
    second = TypeRef(Box[int]).typevar_map

    assert first == second
    ((_, value),) = first.items()
    assert not isinstance(value, tuple)
    assert value.pure_type_def is int


def test_typeargs_of_an_unapplied_class_are_identity_bound() -> None:
    """An unapplied class binds its parameters to themselves."""
    mapping = TypeRef(Box).typevar_map

    assert len(mapping) == 1
    ((key, value),) = mapping.items()
    assert not isinstance(value, tuple)
    assert value.hint is key


def test_generic_subclass_specialisation() -> None:
    """A generic subclass specialises through its own origin."""
    assert Nested[int].__origin__ is Nested
    assert Nested[int].__args__ == (int,)
    assert issubclass(Nested[int], Nested)


def test_concrete_subclass_of_a_specialisation() -> None:
    """Subclassing a specialised class yields a concrete class."""

    class Concrete(Box[int]):
        pass

    assert Concrete.__origin__ is Concrete
    assert Concrete.__args__ == ()
    assert TypeRef(Concrete).params == ()
    assert Concrete().item is None


def test_typeref_round_trips_nested_specialisations() -> None:
    """A nested generic alias resolves to the applied class."""
    assert TypeRef(Nested[int]).resolved_type_def is Nested[int]
    assert TypeRef(Box[list[int]]).resolved_type_def is Box[list[int]]


def test_non_generic_subclass_gets_default_origin_and_args() -> None:
    """A plain subclass still satisfies the ``Runtyped`` contract."""

    class Plain(Runtyped):
        pass

    assert Plain.__origin__ is Plain
    assert Plain.__args__ == ()
    assert TypeRef(Plain).params == ()


def test_is_frozen_dataclass_for_class_and_instance() -> None:
    """Frozen dataclasses are recognised from either the class or an instance."""
    assert bt.is_frozen_dataclass(FrozenRecord)
    assert bt.is_frozen_dataclass(FrozenRecord(1))
    assert not bt.is_frozen_dataclass(MutableRecord)
    assert not bt.is_frozen_dataclass(MutableRecord(1))
    assert not bt.is_frozen_dataclass(int)


def test_is_immutable_for_builtin_scalars() -> None:
    """Scalar builtins are immutable."""
    for value in (1, 1.5, True, "text", b"bytes", frozenset({1})):
        assert bt.is_immutable(value)


def test_is_immutable_for_tuples_only_when_contents_are() -> None:
    """Tuples are immutable when every element is."""
    assert bt.is_immutable((1, "a", (2,)))
    assert not bt.is_immutable((1, [2]))


def test_is_immutable_for_frozen_dataclasses() -> None:
    """Frozen dataclass instances are immutable."""
    assert bt.is_immutable(FrozenRecord(1))
    assert not bt.is_immutable(MutableRecord(1))


def test_is_immutable_for_mutable_containers_and_objects() -> None:
    """Lists, dicts and plain objects are mutable."""
    assert not bt.is_immutable([1])
    assert not bt.is_immutable({"a": 1})
    assert not bt.is_immutable(object())


def test_get_nondefault_methods_excludes_object_methods() -> None:
    """Inherited ``object`` methods are filtered out."""
    methods = bt.get_nondefault_methods(WithMethod)

    assert "own_method" in methods
    assert "base_method" in methods
    assert "__str__" in methods
    assert "__eq__" not in methods
    assert "partialmethod" not in methods


def test_get_own_attrs_reports_annotated_and_defined_names() -> None:
    """Own attributes include class-body definitions and annotations."""
    attrs = bt.get_own_attrs(WithMethod)

    assert {"own_attr", "own_method"} <= attrs
    assert "base_attr" not in attrs


def test_get_own_attrs_includes_annotations() -> None:
    """Class-level annotations without a value are reported."""

    class Annotated:
        declared: int

    assert "declared" in bt.get_own_attrs(Annotated)


def test_get_own_attrs_can_filter_by_value() -> None:
    """The predicate filters attributes by their bound value."""
    callables = bt.get_own_attrs(WithMethod, predicate=callable)

    assert "own_method" in callables
    assert "own_attr" not in callables


def test_get_own_attrs_includes_requested_bases() -> None:
    """``include_bases`` merges the bases' own attributes."""
    attrs = bt.get_own_attrs(WithMethod, include_bases=(BaseWithMethod,))

    assert "own_method" in attrs
    assert "base_method" in attrs
    assert "base_attr" in attrs


def test_get_own_attrs_ignores_bases_not_in_the_hierarchy() -> None:
    """A base the class does not inherit is not merged."""

    class Unrelated:
        unrelated_attr = 2

    attrs = bt.get_own_attrs(WithMethod, include_bases=(Unrelated,))

    assert "unrelated_attr" not in attrs


def test_get_own_attrs_of_a_builtin_class() -> None:
    """The helper works for builtin classes too."""
    attrs = bt.get_own_attrs(bool)

    assert "__and__" in attrs
    assert "__class__" not in attrs


def test_runtyped_class_does_not_require_arguments() -> None:
    """A ``Runtyped`` subclass may itself accept no arguments."""
    instance: Any = Box()
    assert instance.item is None
