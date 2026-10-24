"""Tests for :mod:`based_datatypes.runtime`."""

from types import ModuleType

import pytest

from based_datatypes.runtime import (
    _get_calling_frame,
    get_calling_module,
    get_full_args_dict,
    get_return_type,
    get_subclasses,
)


class Base:
    """Root of the test hierarchy."""


class Left(Base):
    """First branch."""


class Right(Base):
    """Second branch."""


class Leaf(Left, Right):
    """Diamond bottom."""


class GrandChild(Left):
    """Second level below ``Base``."""


def test_get_calling_module_returns_the_caller() -> None:
    """The caller's module is reported."""
    module = get_calling_module()

    assert module is not None
    assert module.__name__ == __name__


def helper_that_calls_back() -> ModuleType | None:
    """Call ``get_calling_module`` with a non-zero offset."""
    return get_calling_module(1)


def test_get_calling_module_honours_the_offset() -> None:
    """A larger offset walks further up the stack."""
    module = helper_that_calls_back()

    assert module is not None
    assert module.__name__ == __name__


def test_get_calling_frame_rejects_an_impossible_offset() -> None:
    """An offset beyond the stack raises ``RuntimeError``."""
    with pytest.raises(RuntimeError, match="No caller"):
        _get_calling_frame(1000)


def test_get_full_args_dict_maps_positional_arguments() -> None:
    """Positional arguments are keyed by parameter name."""

    def func(a: int, b: int = 2) -> None:
        pass

    assert get_full_args_dict(func, (1, 5)) == {"a": 1, "b": 5}


def test_get_full_args_dict_includes_defaults() -> None:
    """Unset keyword defaults are still reported."""

    def func(a: int, b: int = 2, *, c: int = 3) -> None:
        pass

    assert get_full_args_dict(func, (1,)) == {"b": 2, "a": 1, "c": 3}


def test_get_full_args_dict_merges_keywords() -> None:
    """Explicit keyword arguments override positional values."""

    def func(a: int, b: int = 2) -> None:
        pass

    result = get_full_args_dict(func, (1,), {"b": 9})

    assert result == {"a": 1, "b": 9}


def test_get_full_args_dict_accepts_no_keywords() -> None:
    """``kwargs=None`` is treated as an empty mapping."""

    def func(a: int) -> None:
        pass

    assert get_full_args_dict(func, (1,), None) == {"a": 1}


def test_get_full_args_dict_keeps_self_for_unbound_functions() -> None:
    """An unbound function keeps ``self`` in the mapping."""

    class Example:
        def method(self, value: int = 1) -> None:
            pass

    instance = Example()

    assert get_full_args_dict(Example.method, (instance, 2)) == {"self": instance, "value": 2}


def test_get_full_args_dict_drops_self_for_bound_methods() -> None:
    """A bound method is described without ``self``."""

    class Example:
        def method(self, value: int = 1) -> None:
            pass

    assert get_full_args_dict(Example().method, (5,)) == {"value": 5}


def test_get_full_args_dict_of_a_zero_argument_function() -> None:
    """A parameterless function yields an empty mapping."""

    def func() -> None:
        pass

    assert get_full_args_dict(func, ()) == {}


def test_get_return_type_returns_the_annotation() -> None:
    """An annotated function reports its return type."""

    def func() -> int:
        return 1

    assert get_return_type(func) is int


def test_get_return_type_returns_none_without_annotation() -> None:
    """An unannotated function reports ``None``."""

    def func():
        return 1

    assert get_return_type(func) is None


def test_get_return_type_supports_union_annotations() -> None:
    """Union return annotations are returned unchanged."""

    def func() -> int | str:
        return 1

    assert get_return_type(func) == int | str


def test_get_subclasses_lists_direct_and_indirect_children() -> None:
    """All descendants are returned, direct children first."""
    subclasses = get_subclasses(Base)

    assert Left in subclasses
    assert Right in subclasses
    assert Leaf in subclasses
    assert GrandChild in subclasses


def test_get_subclasses_respects_max_level() -> None:
    """``max_level`` bounds the traversal depth."""
    assert set(get_subclasses(Base, max_level=1)) >= {Left, Right}
    assert GrandChild not in get_subclasses(Base, max_level=1)


def test_get_subclasses_deduplicates_diamonds() -> None:
    """A class reachable through two branches appears once."""
    subclasses = get_subclasses(Base)

    assert subclasses.count(Leaf) == 1


def test_get_subclasses_of_a_leaf_is_empty() -> None:
    """A class without children yields an empty list."""
    assert get_subclasses(Leaf) == []


def test_get_subclasses_preserves_declaration_order() -> None:
    """Direct subclasses keep their declaration order."""
    assert get_subclasses(Base)[:2] == [Left, Right]


def test_get_subclasses_works_for_builtin_types() -> None:
    """Builtin hierarchies are traversed as well."""
    assert bool in get_subclasses(int)
