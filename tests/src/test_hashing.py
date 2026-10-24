"""Tests for :mod:`based_datatypes.hashing`."""

import dataclasses
import typing
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from based_datatypes import hashing
from based_datatypes.hashing import gen_int_hash, gen_str_hash


@dataclasses.dataclass
class UnhashableRecord:
    """Unhashable dataclass (``eq=True`` without ``frozen``)."""

    name: str
    value: int


class PicklableState:
    """Unhashable object whose state is a mapping."""

    __hash__ = None  # type: ignore[assignment]

    def __init__(self, state: dict[str, int]) -> None:
        self.state = state

    def __getstate__(self) -> dict[str, int]:
        return self.state


class EmptyState:
    """Unhashable object whose state has no mapping."""

    __hash__ = None  # type: ignore[assignment]

    def __getstate__(self) -> tuple[None]:
        return (None,)


class ArgPicklable:
    """Unhashable object carrying positional reconstruction args."""

    __hash__ = None  # type: ignore[assignment]

    def __getnewargs__(self) -> tuple[int, str]:
        return 1, "a"


class ArgPicklableEx:
    """Unhashable object carrying positional and keyword args."""

    __hash__ = None  # type: ignore[assignment]

    def __getnewargs_ex__(self) -> tuple[tuple[int], dict[str, str]]:
        return (1,), {"b": "c"}


class Unsupported:
    """Unhashable object without any recognised hashing hook."""

    __hash__ = None  # type: ignore[assignment]


def fake_reference(url: str) -> typing.Callable[[object], SimpleNamespace]:
    """Build a ``PyObjectRef.reference`` replacement returning a fixed URL."""

    def _reference(obj: object) -> SimpleNamespace:
        return SimpleNamespace(url=url)

    return _reference


def test_none_hashes_to_zero() -> None:
    """``None`` is the zero element of the hash."""
    assert gen_int_hash(None) == 0


def test_module_hashes_like_its_name() -> None:
    """A module is identified by its name."""
    assert gen_int_hash(hashing) == gen_int_hash(hashing.__name__)


def test_type_hashes_like_module_and_name() -> None:
    """A class is identified by its module and qualified name."""
    assert gen_int_hash(UnhashableRecord) == gen_int_hash(
        (UnhashableRecord.__module__, UnhashableRecord.__name__)
    )


def test_type_alias_hashes_like_module_and_name() -> None:
    """A named type alias hashes like a class."""
    from based_datatypes.types import Ordinal

    assert gen_int_hash(Ordinal) == gen_int_hash((Ordinal.__module__, Ordinal.__name__))


def test_function_hashes_like_its_reference_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A named function is identified by its persistent reference URL."""
    monkeypatch.setattr(
        hashing.PyObjectRef,
        "reference",
        staticmethod(fake_reference("py+https://example.com/repo#mod:func")),
    )

    assert gen_int_hash(gen_str_hash) == gen_int_hash("py+https://example.com/repo#mod:func")


def test_generic_alias_hashes_like_origin_and_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parameterised generic hashes its origin and its arguments."""
    monkeypatch.setattr(
        hashing.PyObjectRef,
        "reference",
        staticmethod(fake_reference("py+https://example.com/repo#builtins:list")),
    )

    expected = gen_int_hash(
        (
            "py+https://example.com/repo#builtins:list",
            gen_int_hash(int),
        )
    )

    assert gen_int_hash(list[int]) == expected


def test_union_hashes_like_its_members(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PEP 604 union hashes its members under a stable label."""
    monkeypatch.setattr(
        hashing.PyObjectRef,
        "reference",
        staticmethod(fake_reference("py+https://example.com/repo#builtins:str")),
    )

    result = gen_int_hash(int | str)

    expected = gen_int_hash(("types.UnionType", gen_int_hash(int), gen_int_hash(str)))

    assert result == expected


def test_scalars_are_type_sensitive() -> None:
    """Equal values of different types hash differently."""
    assert gen_int_hash(1) != gen_int_hash(True)
    assert gen_int_hash(1) != gen_int_hash(1.0)


def test_scalars_are_stable_within_and_across_calls() -> None:
    """Repeated hashing of a scalar is deterministic."""
    assert gen_int_hash("value") == gen_int_hash("value")
    assert gen_int_hash(b"value") == gen_int_hash(b"value")
    assert gen_int_hash(datetime(2024, 1, 1)) == gen_int_hash(datetime(2024, 1, 1))
    assert gen_int_hash(date(2024, 1, 1)) == gen_int_hash(date(2024, 1, 1))
    assert gen_int_hash(time(12, 30)) == gen_int_hash(time(12, 30))
    assert gen_int_hash(timedelta(days=2)) == gen_int_hash(timedelta(days=2))


def test_tuples_are_order_sensitive() -> None:
    """Sequence order contributes to the hash."""
    assert gen_int_hash((1, 2)) != gen_int_hash((2, 1))


def test_lists_and_tuples_with_equal_content_hash_alike() -> None:
    """A list and a tuple of equal content hash alike (documented)."""
    assert gen_int_hash([1, 2]) == gen_int_hash((1, 2))


def test_list_hash_is_deterministic() -> None:
    """A list's hash does not depend on its identity."""
    assert gen_int_hash([1, {"a": 2}]) == gen_int_hash([1, {"a": 2}])


def test_cyclic_list_terminates() -> None:
    """A list containing itself hashes without recursing forever."""
    cyclic: list[object] = [1]
    cyclic.append(cyclic)

    result = gen_int_hash(cyclic)

    assert isinstance(result, int)


def test_cyclic_dict_terminates() -> None:
    """A dict containing itself hashes without recursing forever."""
    cyclic: dict[str, object] = {"a": 1}
    cyclic["self"] = cyclic

    assert isinstance(gen_int_hash(cyclic), int)


def test_dict_hash_is_key_order_independent() -> None:
    """Inserting the same items in a different order hashes alike."""
    assert gen_int_hash({"a": 1, "b": 2}) == gen_int_hash({"b": 2, "a": 1})


def test_set_hash_is_insertion_order_independent() -> None:
    """A set hashes independently of its iteration order."""
    values = {3, 1, 2}

    assert gen_int_hash(values) == gen_int_hash(set(values))


def test_pandas_dataframe_and_series_are_deterministic() -> None:
    """Pandas containers hash through ``hash_pandas_object``."""
    frame = pd.DataFrame({"a": [1, 2]})
    series = pd.Series([1, 2])

    assert gen_int_hash(frame) == gen_int_hash(pd.DataFrame({"a": [1, 2]}))
    assert gen_int_hash(series) == gen_int_hash(pd.Series([1, 2]))
    assert gen_int_hash(frame) == gen_int_hash(series)
    assert gen_int_hash(pd.Index([1, 2])) == gen_int_hash(pd.Index([1, 2]))


def test_pandas_hash_depends_on_data() -> None:
    """Different pandas data hashes differently."""
    assert gen_int_hash(pd.DataFrame({"a": [1, 2]})) != gen_int_hash(pd.DataFrame({"a": [2, 1]}))


def test_hashable_objects_fall_back_to_identity_hash() -> None:
    """Any other hashable object uses its own ``__hash__``."""
    value = object()

    assert gen_int_hash(value) == hash(value)


def test_custom_hashable_is_used() -> None:
    """A custom ``__hash__`` is honoured."""

    class Custom:
        def __hash__(self) -> int:
            return 12345

    assert gen_int_hash(Custom()) == 12345


def test_unhashable_dataclass_hashes_structurally() -> None:
    """An unhashable dataclass hashes through its fields."""
    record = UnhashableRecord("a", 1)

    assert gen_int_hash(record) == gen_int_hash({"name": "a", "value": 1})


def test_dataclass_field_order_does_not_matter() -> None:
    """Field order within the mapping is not significant."""

    @dataclasses.dataclass
    class Reordered:
        value: int
        name: str

    assert gen_int_hash(UnhashableRecord("a", 1)) == gen_int_hash(Reordered(1, "a"))


def test_unhashable_stateful_object_hashes_its_state() -> None:
    """An unhashable object with ``__getstate__`` hashes its state."""
    obj = PicklableState({"a": 1})

    expected = sum(gen_int_hash((key, value)) for key, value in {"a": 1}.items())

    assert gen_int_hash(obj) == expected


def test_state_without_a_mapping_is_rejected() -> None:
    """A state tuple without a mapping cannot be hashed."""
    with pytest.raises(ValueError):
        gen_int_hash(EmptyState())


def test_unhashable_arg_picklable_object_hashes_its_args() -> None:
    """``__getnewargs__`` provides the hash input."""
    assert gen_int_hash(ArgPicklable()) == gen_int_hash(1) + gen_int_hash("a")


def test_unhashable_arg_picklable_ex_object_hashes_args_and_kwargs() -> None:
    """``__getnewargs_ex__`` provides positional and keyword input."""
    expected = gen_int_hash(1) + gen_int_hash(("b", "c"))

    assert gen_int_hash(ArgPicklableEx()) == expected


def test_unsupported_unhashable_object_is_rejected() -> None:
    """An object without a hashing hook raises ``ValueError``."""
    with pytest.raises(ValueError):
        gen_int_hash(Unsupported())


def test_hashing_context_is_reset_after_a_call() -> None:
    """The hashing context does not leak between top-level calls."""
    assert hashing.current_ctx.get() is None
    gen_int_hash([1, 2])
    assert hashing.current_ctx.get() is None


def test_hashing_context_is_reset_after_an_error() -> None:
    """The hashing context is cleared even when hashing fails."""
    with pytest.raises(ValueError):
        gen_int_hash(Unsupported())

    assert hashing.current_ctx.get() is None


def test_gen_str_hash_returns_a_fixed_width_hex_string() -> None:
    """The string hash is hex, zero-padded to the requested width."""
    value = gen_str_hash("abc")

    assert len(value) == 10
    assert int(value, 16) >= 0


def test_gen_str_hash_respects_the_length() -> None:
    """The requested width truncates the hex digest."""
    assert gen_str_hash("abc", length=4) == gen_str_hash("abc")[:4]


def test_gen_str_hash_pads_zero() -> None:
    """A zero hash is padded to the requested width."""
    assert gen_str_hash(None) == "0" * 10


def test_gen_str_hash_supports_a_zero_length() -> None:
    """A zero width yields an empty string."""
    assert gen_str_hash("abc", length=0) == ""


def test_gen_str_hash_is_deterministic() -> None:
    """Equal inputs yield equal string hashes."""
    assert gen_str_hash({"a": 1}) == gen_str_hash({"a": 1})


def test_gen_str_hash_of_different_values_differs() -> None:
    """Different inputs yield different string hashes."""
    assert gen_str_hash("a") != gen_str_hash("b")
