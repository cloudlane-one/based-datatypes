"""Tests for :mod:`based_datatypes.enums`."""

import json

import pytest

from based_datatypes.enums import StrEnum


class Color(StrEnum):
    """Example enum used by the tests."""

    red = "RED"
    green = "GREEN"


class Auto(StrEnum):
    """Enum whose values are derived automatically."""

    first = "first"
    second = "second"


def test_repr_renders_only_the_value() -> None:
    """``repr`` quotes the value only, without the class name."""
    assert repr(Color.red) == "'RED'"


def test_str_renders_the_value() -> None:
    """``str`` renders the raw value."""
    assert str(Color.red) == "RED"
    assert f"{Color.green}" == "GREEN"


def test_repr_agrees_with_str_value() -> None:
    """``repr`` is the quoted ``str`` form."""
    assert repr(Color.red) == f"'{Color.red}'"


def test_members_compare_equal_to_their_value() -> None:
    """A ``StrEnum`` member is a string equal to its value."""
    assert Color.red == "RED"
    assert isinstance(Color.red, str)


def test_parse_returns_the_matching_member() -> None:
    """``parse`` maps a value back to its member."""
    assert Color.parse("RED") is Color.red


def test_parse_returns_the_subclass_type_not_the_base() -> None:
    """``parse`` is a classmethod and stays on the subclass."""
    assert type(Color.parse("GREEN")) is Color


def test_parse_rejects_unknown_values() -> None:
    """An unknown value raises ``ValueError`` with the enum's name."""
    with pytest.raises(ValueError, match="'BLUE' is not a valid Color"):
        Color.parse("BLUE")


def test_parse_rejects_non_string_values() -> None:
    """Non-string input is rejected by the underlying enum lookup."""
    with pytest.raises((ValueError, TypeError)):
        Color.parse(1)  # type: ignore[arg-type]


def test_json_serialises_to_the_value() -> None:
    """JSON treats the member as its string value."""
    assert json.dumps(Color.red) == '"RED"'


def test_iteration_order_follows_declaration() -> None:
    """Iteration keeps the declaration order."""
    assert [member.name for member in Color] == ["red", "green"]


def test_lookup_by_name_and_value() -> None:
    """Both the member map and the value index the enum."""
    assert Color.__members__["red"] is Color.red
    assert Color("RED") is Color.red


def test_auto_values_are_member_names() -> None:
    """Declared values are preserved for auto-style members."""
    assert Auto.first.value == "first"
    assert Auto.parse("second") is Auto.second


def test_member_is_hashable_and_usable_as_dict_key() -> None:
    """Members behave like their string value in mappings."""
    mapping = {Color.red: 1}
    assert mapping[Color("RED")] == 1
    assert mapping[Color.red] == 1


def test_parse_preserves_identity_for_repeated_calls() -> None:
    """Parsing the same value repeatedly yields the same singleton member."""
    assert Color.parse("RED") is Color.parse("RED")
