"""Tests for ``TypeAwareClass`` and the attribute-reflection helpers."""

import dataclasses
from typing import Any

from based_datatypes import typing as bt
from based_datatypes.typing import TypeAwareClass, TypeRef


class Box[T](TypeAwareClass):
    """Generic class aware of its type arguments."""

    def __init__(self, item: T | None = None) -> None:
        self.item = item


class Pair[K, V](TypeAwareClass):
    """Two-parameter generic class."""

    def __init__(self, key: K | None = None, value: V | None = None) -> None:
        self.key = key
        self.value = value


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
    """Subscripting a ``TypeAwareClass`` yields a real subclass."""
    assert issubclass(Box[int], Box)
    assert isinstance(Box[int], type)


def test_class_getitem_records_origin_and_arguments() -> None:
    """The specialised class records its declaration and arguments."""
    assert Box[int].__origin__ is Box
    assert Box[int].__args__ == (int,)
    assert Box[int].__module__ == Box.__module__


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
    assert len(Box.type_params) == 1
    assert len(Pair.type_params) == 2


def test_type_params_of_a_specialisation_are_empty() -> None:
    """An applied class has no free parameters left."""
    assert Box[int].type_params == ()
    assert Pair[str, int].type_params == ()


def test_typeargs_of_a_specialisation_resolve_to_the_argument() -> None:
    """The type-argument map resolves the parameter to the applied type."""
    mapping = Box[int].typeargs

    assert len(mapping) == 1
    value = next(iter(mapping.values()))
    assert not isinstance(value, tuple)
    assert value.typedef is int


def test_typeargs_of_a_two_parameter_specialisation() -> None:
    """Each parameter is bound to its own argument."""
    mapping = Pair[str, int].typeargs

    assert len(mapping) == 2
    resolved = {
        key.__name__: value.typedef
        for key, value in mapping.items()
        if not isinstance(value, tuple)
    }
    assert set(resolved.values()) == {str, int}


def test_typeargs_are_cached_per_class() -> None:
    """The class-level property caches its result."""
    first = Box[int].typeargs
    second = Box[int].typeargs

    assert first is second


def test_typeargs_of_an_unapplied_class_are_identity_bound() -> None:
    """An unapplied class binds its parameters to themselves."""
    mapping = Box.typeargs

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
    assert Concrete.type_params == ()
    assert Concrete().item is None


def test_typeref_round_trips_nested_specialisations() -> None:
    """A nested generic alias resolves to the applied class."""
    assert TypeRef(Nested[int]).typeform is Nested[int]
    assert TypeRef(Box[list[int]]).typeform is Box[list[int]]


def test_non_generic_subclass_gets_default_origin_and_args() -> None:
    """A plain subclass still satisfies the ``TypeAwareClass`` contract."""

    class Plain(TypeAwareClass):
        pass

    assert Plain.__origin__ is Plain
    assert Plain.__args__ == ()
    assert Plain.type_params == ()


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


def test_type_aware_class_does_not_require_arguments() -> None:
    """A ``TypeAwareClass`` subclass may itself accept no arguments."""
    instance: Any = Box()
    assert instance.item is None
