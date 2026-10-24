"""Utilities for producing hashes of common objects."""

import hashlib
from collections.abc import Callable, Hashable, Sequence
from contextvars import ContextVar
from dataclasses import fields
from datetime import date, datetime, time, timedelta
from functools import reduce
from numbers import Number
from types import FunctionType, GenericAlias, ModuleType, UnionType
from typing import (
    Any,
    Concatenate,
    TypeAliasType,
    get_args,
    get_origin,
)

import pandas as pd
from pandas.util import hash_pandas_object

from .ref import PyObjectRef
from .types import (
    ArgsPicklable,
    ArgsPicklableEx,
    DataclassInstance,
    StatePicklable,
)

try:
    import polars as pl  # type: ignore
except ImportError:
    pl = None  # type: ignore


def _stable_hash(
    data: Number | str | bytes | date | time | datetime | timedelta,
) -> int:
    data = f"{type(data)}:{data}"
    m = hashlib.md5(usedforsecurity=False)
    m.update(bytes(data, "utf-8"))
    return int.from_bytes(m.digest())


def _hash_sequence(s: Sequence, _ctx: set[int]) -> int:
    return reduce(lambda x, y: gen_int_hash(str(x) + str(gen_int_hash(y))), s, 0)


current_ctx: ContextVar[set[int] | None] = ContextVar("current_ctx", default=None)


def _hashing_context[**P, T](
    func: Callable[Concatenate[set[int], P], T],
) -> Callable[P, T]:
    def inner(*args: P.args, **kwargs: P.kwargs) -> T:
        ctx = current_ctx.get()
        if ctx is None:
            new_ctx: set[int] = set()
            current_ctx.set(new_ctx)
            try:
                return func(new_ctx, *args, **kwargs)
            finally:
                current_ctx.set(None)
        else:
            return func(ctx, *args, **kwargs)

    return inner


@_hashing_context
def gen_int_hash(_ctx: set[int], obj: Any) -> int:  # noqa: C901
    """Generate stable hash for obj (must be known, hashable or composed of such).

    Args:
        obj: Object to hash.

    Returns:
        Hash of the object as integer.
    """
    match obj:
        case None:
            return 0
        case ModuleType():
            return gen_int_hash(obj.__name__)
        case type() | TypeAliasType():
            return gen_int_hash((obj.__module__, obj.__name__))
        case FunctionType() if hasattr(obj, "__name__"):
            return gen_int_hash(str(PyObjectRef.reference(obj).url))
        case FunctionType():
            # This may be unstable as closures are not considered.
            return gen_int_hash(list(obj.__code__.co_lines()))
        case GenericAlias() | UnionType():
            base = get_origin(obj)
            args = get_args(obj)
            return gen_int_hash(
                (
                    (
                        "types.UnionType"
                        if base is UnionType
                        else str(PyObjectRef.reference(base).url)
                    ),
                    *(gen_int_hash(arg) for arg in args),
                )
            )
        case Number() | str() | bytes() | date() | time() | datetime() | timedelta():
            return _stable_hash(obj)
        case tuple():
            return _hash_sequence(obj, _ctx)
        case _ if id(obj) in _ctx:
            return 0
        case pd.DataFrame() | pd.Series() | pd.Index():
            return sum(hash_pandas_object(obj))
        case list():
            _ctx.add(id(obj))
            return _hash_sequence(obj, _ctx)
        case dict():
            _ctx.add(id(obj))
            return sum(_hash_sequence(item, _ctx) for item in obj.items())
        case set():
            _ctx.add(id(obj))
            return sum(gen_int_hash(item) for item in obj)
        case Hashable():
            _ctx.add(id(obj))
            return hash(obj)
        case DataclassInstance():
            _ctx.add(id(obj))
            return gen_int_hash({f.name: getattr(obj, f.name) for f in fields(obj)})
        case StatePicklable():
            state = obj.__getstate__()
            state = state if isinstance(state, tuple) else (state,)
            state_dicts = [s for s in state if isinstance(s, dict)]

            if len(state_dicts) == 0:
                raise ValueError()

            return sum(sum(_hash_sequence((k, v), _ctx) for k, v in s.items()) for s in state_dicts)
        case ArgsPicklable():
            args = obj.__getnewargs__()
            return sum(gen_int_hash(item) for item in args)
        case ArgsPicklableEx():
            args, kwargs = obj.__getnewargs_ex__()
            return sum(gen_int_hash(item) for item in args) + sum(
                gen_int_hash(item) for item in kwargs.items()
            )
        case _:
            if pl is not None:
                match obj:
                    case pl.DataFrame():
                        return sum(obj.hash_rows()) if not obj.is_empty() else 0
                    case pl.Series():
                        return sum(obj.hash())
                    case _:
                        pass

            raise ValueError()


def gen_str_hash(x: Any, length: int = 10) -> str:
    """Generate stable hash for obj (must be known, hashable or composed of such).

    Args:
        x: Object to hash.
        length: Length of the hash.
        raw_str:
            Whether to use the raw string representation of the object,
            if it is a string.

    Returns:
        Hash of the object as string.
    """
    s = hex(abs(gen_int_hash(x)))[2:]
    return s[:length].rjust(length, "0")
