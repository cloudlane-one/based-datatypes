"""Tests for :mod:`based_datatypes.data`."""

import dataclasses
from dataclasses import dataclass, field
import pydantic
import pytest

from based_datatypes.data import (
    MaskedInit,
    container_to_dict,
    copy_and_override,
    flatten_dict,
)
from based_datatypes.types import Attr


@dataclass
class Point:
    """Simple two-dimensional point."""

    x: int
    y: int = 2


@dataclass
class WithFactory:
    """Dataclass with a default factory and a post-init field."""

    x: int
    tags: list[str] = field(default_factory=list)


@dataclass
class WithDerived:
    """Dataclass with a non-init field."""

    x: int
    double: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.double = self.x * 2


@dataclass
class Box[T]:
    """Generic dataclass used to test generic-alias constructors."""

    item: T


@dataclass
class Nested:
    """Dataclass holding another dataclass."""

    point: Point
    label: str = "nested"


class Model(pydantic.BaseModel):
    """Pydantic model used to exercise the ``BaseModel`` branch."""

    a: int
    b: list[int] = []


class Other:
    """Dataclass sharing a field name with the source type."""

    def __init__(self) -> None:
        self.x = 1


@dataclass
class Overlapping:
    """Dataclass with a superset of ``Point``'s fields."""

    x: int
    z: int = 4


def test_copy_and_override_without_overrides() -> None:
    """The source values are reused verbatim."""
    assert copy_and_override(Point, Point(1, 2)) == Point(1, 2)


def test_copy_and_override_with_positional_override() -> None:
    """Positional arguments replace the leading fields."""
    assert copy_and_override(Point, Point(1, 2), 9) == Point(9, 2)


def test_copy_and_override_with_keyword_override() -> None:
    """Keyword arguments replace the named fields."""
    assert copy_and_override(Point, Point(1, 2), y=7) == Point(1, 7)


def test_copy_and_override_with_mixed_override() -> None:
    """Positional and keyword overrides combine."""
    assert copy_and_override(Point, Point(1, 2), 3, y=4) == Point(3, 4)


def test_copy_and_override_fills_defaults_from_the_source() -> None:
    """Fields the source does not define fall back to the target defaults."""
    source = Overlapping(1, 9)

    assert copy_and_override(Point, source) == Point(1, 2)


def test_copy_and_override_uses_default_factories() -> None:
    """Missing list fields get a fresh factory value."""
    result = copy_and_override(WithFactory, WithFactory(1, ["a"]))

    assert result == WithFactory(1, ["a"])
    assert copy_and_override(WithFactory, WithFactory(1)).tags == []


def test_copy_and_override_copies_non_init_fields() -> None:
    """Fields excluded from ``__init__`` are copied after construction."""
    source = WithDerived(3)

    result = copy_and_override(WithDerived, source)

    assert result.x == 3
    assert result.double == 6


def test_copy_and_override_merges_multiple_sources() -> None:
    """Later sources win for fields they define."""
    result = copy_and_override(Point, (Point(1, 2), Point(5)))

    assert result == Point(5, 2)


def test_copy_and_override_unwraps_masked_init() -> None:
    """``MaskedInit`` supplies only the constructor's type."""
    mask = MaskedInit(Point, Point)

    assert mask.mask is Point
    assert mask.constructor is Point
    assert copy_and_override(mask, Point(1, 2), x=10) == Point(10, 2)


def test_masked_init_is_callable() -> None:
    """Calling the mask constructs the underlying type."""
    assert MaskedInit(Point, Point)(1, 2) == Point(1, 2)


def test_copy_and_override_accepts_a_generic_alias_constructor() -> None:
    """A parameterised generic alias is resolved to its origin."""
    result = copy_and_override(Box[int], Box[int](1), item=5)

    assert result == Box(5)


def test_copy_and_override_rejects_missing_required_fields() -> None:
    """A required field absent from every source fails loudly."""

    @dataclass
    class Target:
        x: int

    source = Other()

    with pytest.raises(AttributeError):
        copy_and_override(Target, source)  # type: ignore[arg-type]


def test_copy_and_override_passes_extra_positionals_to_the_constructor() -> None:
    """Too many positional overrides raise the constructor's error."""
    with pytest.raises(TypeError):
        copy_and_override(Point, Point(1, 2), 1, 2, 3)


def test_copy_and_override_does_not_support_kw_only_dataclasses() -> None:
    """Keyword-only sources are a documented limitation."""
    kw_only = dataclasses.make_dataclass("KwOnly", [("x", int)], kw_only=True)
    source = kw_only(x=1)

    with pytest.raises(TypeError):
        copy_and_override(kw_only, source, 5)


def test_container_to_dict_of_a_dataclass() -> None:
    """Dataclass fields become ``Attr``-keyed entries."""
    assert container_to_dict(Point(1, 2)) == {
        Attr("x"): 1,
        Attr("y"): 2,
    }


def test_container_to_dict_of_a_pydantic_model() -> None:
    """Pydantic models are dumped through ``model_dump``."""
    assert container_to_dict(Model(a=1, b=[2])) == {
        Attr("a"): 1,
        Attr("b"): {0: 2},
    }


def test_container_to_dict_recurses_into_nested_dataclasses() -> None:
    """Nested dataclass values are converted as well."""
    assert container_to_dict(Nested(Point(1, 2))) == {
        Attr("point"): {Attr("x"): 1, Attr("y"): 2},
        Attr("label"): "nested",
    }


def test_container_to_dict_can_skip_recursion() -> None:
    """``recursive=False`` keeps nested objects untouched."""
    point = Point(1, 2)

    assert container_to_dict(Nested(point), recursive=False) == {
        Attr("point"): point,
        Attr("label"): "nested",
    }


def test_container_to_dict_of_a_mapping() -> None:
    """Mapping keys are preserved and values recursed."""
    assert container_to_dict({"a": [1, 2]}) == {"a": {0: 1, 1: 2}}


def test_container_to_dict_of_a_sequence_and_set() -> None:
    """Sequences and sets are indexed by position."""
    assert container_to_dict([10, 20]) == {0: 10, 1: 20}
    assert container_to_dict((10,)) == {0: 10}
    set_result = container_to_dict({1, 2})
    assert set_result is not None
    assert set(set_result.values()) == {1, 2}


def test_container_to_dict_of_scalars_is_none() -> None:
    """Scalars are not containers."""
    assert container_to_dict(1) is None  # type: ignore[arg-type]
    assert container_to_dict(None) is None  # type: ignore[arg-type]
    assert container_to_dict(object()) is None  # type: ignore[arg-type]


def test_container_to_dict_of_strings_is_none() -> None:
    """Strings are iterable but are not treated as containers."""
    assert container_to_dict("abc") is None  # type: ignore[arg-type]
    assert container_to_dict(b"abc") is None  # type: ignore[arg-type]
    assert container_to_dict(bytearray(b"abc")) is None  # type: ignore[arg-type]


def test_container_to_dict_preserves_cycles() -> None:
    """Self-referencing containers keep a reference to the converted result."""
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic

    result = container_to_dict(cyclic)

    assert result is not None
    assert result["self"] is result


def test_container_to_dict_shared_references_are_converted_once() -> None:
    """The same object encountered twice reuses the first conversion."""
    point = Point(1, 2)

    result = container_to_dict([point, point])

    assert result is not None
    assert result[0] is result[1]


def test_container_to_dict_accepts_an_explicit_registry() -> None:
    """A caller-supplied registry seeds the cycle detection."""
    point = Point(1, 2)
    registry: dict[int, dict[object, object]] = {id(point): {"cached": True}}

    assert container_to_dict(point, _registry=registry) == {"cached": True}


def test_flatten_dict_of_a_nested_mapping() -> None:
    """Nested keys become tuples."""
    assert flatten_dict({"a": {"b": 1}, "c": 2}) == {("a", "b"): 1, ("c",): 2}


def test_flatten_dict_with_a_prefix_path() -> None:
    """An initial path is prepended."""
    assert flatten_dict({"a": 1}, path=("root",)) == {("root", "a"): 1}


def test_flatten_dict_of_an_empty_mapping() -> None:
    """An empty mapping flattens to an empty dict."""
    assert flatten_dict({}) == {}


def test_flatten_dict_ignores_recursive_references() -> None:
    """A self-reference terminates instead of recursing forever."""
    cyclic: dict[str, object] = {"a": 1}
    cyclic["self"] = cyclic

    # The recursive branch resolves to the already-collected result, so the
    # self-reference contributes no further leaf.
    assert flatten_dict(cyclic) == {("a",): 1}


def test_flatten_dict_requires_a_mapping() -> None:
    """A non-mapping root raises ``AttributeError``."""
    with pytest.raises(AttributeError):
        flatten_dict([1, 2])  # type: ignore[arg-type]


def test_flatten_dict_keeps_sequences_as_values() -> None:
    """Only mappings are traversed."""
    assert flatten_dict({"a": [1, {"b": 2}]}) == {("a",): [1, {"b": 2}]}
