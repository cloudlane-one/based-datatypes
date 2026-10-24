"""Tests for :mod:`based_datatypes.types`."""

import dataclasses
import typing
from typing import get_args

import pydantic
import pytest

from based_datatypes import types
from based_datatypes.enums import StrEnum


def test_uuid4_is_a_string() -> None:
    """``UUID4`` is a nominal string type."""
    value = types.UUID4("2f0c8d0e-8a5b-4f0e-9a1c-1d2e3f4a5b6c")

    assert isinstance(value, str)
    assert value.upper() == "2F0C8D0E-8A5B-4F0E-9A1C-1D2E3F4A5B6C"


def test_uuid4_does_not_validate_its_content() -> None:
    """``UUID4`` performs no format validation; it is a typing marker only."""
    assert types.UUID4("not-a-uuid") == "not-a-uuid"


def test_not_is_a_string_enum() -> None:
    """``Not`` derives from the documentation-friendly ``StrEnum``."""
    assert issubclass(types.Not, StrEnum)
    assert types.Not.defined.value == "defined"


def test_not_members_cover_the_unresolved_states() -> None:
    """The declared members are the four unresolved/handled states."""
    assert [member.name for member in types.Not] == [
        "defined",
        "resolved",
        "handled",
        "changed",
    ]


def test_not_parses_its_members() -> None:
    """``parse`` is inherited from ``StrEnum``."""
    assert types.Not.parse("handled") is types.Not.handled


def test_contratype_defaults_to_none() -> None:
    """``ContraType`` wraps an optional type argument."""
    wrapped = types.ContraType[int]()

    assert wrapped.type_ is None


def test_contratype_stores_the_type() -> None:
    """The wrapped type is stored as given."""
    wrapped = types.ContraType[int](int)

    assert wrapped.type_ is int


def test_contratype_is_a_dataclass_and_final() -> None:
    """``ContraType`` is a final dataclass (``@final`` is static only)."""
    assert dataclasses.is_dataclass(types.ContraType)
    assert getattr(types.ContraType, "__dataclass_params__").frozen is False
    assert getattr(types.ContraType, "__final__", False) is True


def test_undefined_is_frozen_and_comparable() -> None:
    """``Undefined`` is an immutable marker with value equality."""
    first = types.Undefined()
    second = types.Undefined()

    assert first == second
    assert hash(first) == hash(second)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.attribute = 1  # type: ignore[attr-defined]


def test_undefined_is_final() -> None:
    """``Undefined`` is marked final (``@final`` is static only)."""
    assert getattr(types.Undefined, "__final__", False) is True


def test_dataclass_instance_protocol_accepts_instances_and_classes() -> None:
    """The protocol checks for the ``__dataclass_fields__`` attribute."""

    @dataclasses.dataclass
    class Record:
        value: int

    assert isinstance(Record(1), types.DataclassInstance)
    assert isinstance(Record, types.DataclassInstance)
    assert not isinstance(1, types.DataclassInstance)


def test_state_picklable_protocol_matches_getstate() -> None:
    """Objects exposing ``__getstate__`` satisfy the protocol."""

    class WithState:
        def __getstate__(self) -> dict[str, int]:
            return {"a": 1}

    assert isinstance(WithState(), types.StatePicklable)
    # ``object`` has exposed ``__getstate__`` since Python 3.11, so every
    # instance satisfies the protocol; it is a marker, not a capability test.
    assert isinstance(object(), types.StatePicklable)


def test_args_picklable_protocol_matches_getnewargs() -> None:
    """Objects exposing ``__getnewargs__`` satisfy the protocol."""

    class WithArgs:
        def __getnewargs__(self) -> tuple[int]:
            return (1,)

    assert isinstance(WithArgs(), types.ArgsPicklable)
    assert not isinstance(object(), types.ArgsPicklable)


def test_args_picklable_ex_protocol_matches_getnewargs_ex() -> None:
    """Objects exposing ``__getnewargs_ex__`` satisfy the extended protocol."""

    class WithArgsEx:
        def __getnewargs_ex__(self) -> tuple[tuple[int], dict[str, int]]:
            return (1,), {"b": 2}

    assert isinstance(WithArgsEx(), types.ArgsPicklableEx)
    assert not isinstance(object(), types.ArgsPicklableEx)


def test_ordinal_alias_covers_scalar_ordered_types() -> None:
    """``Ordinal`` is a named alias over every orderable scalar type."""
    assert isinstance(types.Ordinal, typing.TypeAliasType)
    members = set(get_args(types.Ordinal.__value__))

    assert {
        bool,
        int,
        float,
        types.Decimal,
        types.datetime,
        types.date,
        types.time,
        types.timedelta,
        str,
    } <= members


def test_annotation_scan_type_and_single_typedef_exist() -> None:
    """The public aliases are usable as type hints."""
    assert types.AnnotationScanType is not None
    assert types.SingleTypeDef[int] is not None


def test_generic_alias_protocol_recognises_parameterised_generics() -> None:
    """``GenericAlias`` matches ``list[int]`` but not the bare class."""
    assert isinstance(list[int], types.GenericAlias)
    assert not isinstance(list, types.GenericAlias)


def test_supports_items_protocol_matches_mappings() -> None:
    """Objects with ``keys``/``values``/``items`` satisfy the protocol."""
    assert isinstance({}, types.SupportsItems)
    assert not isinstance([], types.SupportsItems)


def test_supports_keys_and_get_item_protocol() -> None:
    """Objects with ``keys`` and ``__getitem__`` satisfy the protocol."""
    assert isinstance({}, types.SupportsKeysAndGetItem)
    assert not isinstance([], types.SupportsKeysAndGetItem)


def test_runtime_validated_protocol_matches_pydantic_models() -> None:
    """Pydantic models expose the core-schema hook the protocol requires."""

    class Model(pydantic.BaseModel):
        value: int

    assert isinstance(Model, types.RuntimeValidated)
    assert not isinstance(int, types.RuntimeValidated)


def test_attr_is_a_frozen_reference() -> None:
    """``Attr`` is an immutable, hashable attribute name."""
    attr = types.Attr("value")

    assert attr.name == "value"
    assert attr == types.Attr("value")
    assert hash(attr) == hash(types.Attr("value"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        attr.name = "other"  # type: ignore[misc]


def test_operator_methods_are_dunder_names() -> None:
    """Every listed operator method is a dunder string."""
    assert len(types.operator_methods) > 0
    assert all(name.startswith("__") and name.endswith("__") for name in types.operator_methods)
    assert "__add__" in types.operator_methods
    assert "__class_getitem__" not in types.operator_methods


def test_operator_methods_are_unique() -> None:
    """The operator list carries no duplicates."""
    assert len(types.operator_methods) == len(set(types.operator_methods))


def test_public_helpers_have_no_unexpected_annotations() -> None:
    """The module exposes concrete annotations for its markers."""
    assert types.UUID4.__mro__[1] is str
    assert "type_" in typing.get_type_hints(types.ContraType)
