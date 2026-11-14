"""Arbitrary function caching with sane handling of arguments."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from fractions import Fraction
from typing import Any, Literal, final, overload

from cachebox import (
    BaseCacheImpl,
    FIFOCache,
    LFUCache,
    LRUCache,
    RRCache,
    TTLCache,
    VTTLCache,
)
from cachebox import (
    Cache as NaiveCache,
)
from slugify import slugify

from based_datatypes.hashing import gen_str_hash
from based_datatypes.runtime import get_full_args_dict

type EvictionPolicy[**P] = (
    Literal["age", "last-usage", "usage-frequency", "random-choice"]
    | timedelta
    | float
    | Callable[P, timedelta | float]
    | None
)


def gen_id(x: Any, max_str_len: int = 24) -> str:
    """Generate a unique id for the given object.

    Args:
        x: Object to generate id for.
        max_str_len: Maximum length of the string representation.

    Returns:
        Unique id as a string.
    """
    s = None
    match x:
        case bool() | int() | float() | complex() | Decimal():
            s = str(x)
        case Fraction():
            s = str(x).replace("/", "_over_")
        case date():
            s = x.isoformat()
        case time():
            s = f"{x.hour:02}{x.minute:02}{x.second:02}{x.microsecond:06}"
        case datetime():
            s = str(x.timestamp())
        case timedelta():
            s = str(x.total_seconds())
        case str() if len(slugified := slugify(x)) <= max_str_len:
            s = slugified
        case bytes() if len(x) / 2 <= max_str_len:
            s = x.hex()
        case _:
            s = gen_str_hash(x, length=max_str_len)

    return s


@final
class MISSING:
    """Singleton to represent a missing value in the cache."""


type ArgsCallback[**P] = Callable[P, Mapping[str, Any]] | Mapping[str, Callable[[Any], Any]]


@dataclass
class CachedFunction[**P, R]:
    """In-memory cache."""

    func: Callable[P, R]
    """Function to cache."""

    max_size: int = 1000
    """Maximum number of items to store in the cache."""

    args_callback: ArgsCallback[P] | None = None
    """Callback to transform the function arguments into a map of arguments to use for caching."""

    evict_by: EvictionPolicy[P] = "age"
    """Eviction strategy for the cache.
    Following values are supported:
    - ``"age"``: Evict the oldest item in the cache.
    - ``"last-usage"``: Evict the least recently used item in the cache.
    - ``"usage-frequency"``: Evict the least frequently used item in the cache.
    - ``"random-choice"``: Evict a random item in the cache.
    - ``timedelta``: Evict items that have been in the cache longer than the given timedelta.
    - ``float``: Evict items that have been in the cache longer than the given number of seconds.
    - ``Callable``: A callable that takes the same arguments as the cached function and returns ttl.
    - ``None``: No eviction, the cache will grow until ``max_size`` is reached, then raise an error.
    """

    method_caching: Literal["per-instance", "per-class"] = "per-instance"
    """Whether to keep a separate cache for each instance of the class (per-instance) or
    a single cache for all instances of the class (per-class)."""

    _memory_cache: BaseCacheImpl | None = field(default=None, init=False, repr=False)

    def get_memory_cache(self, instance: object | None = None) -> BaseCacheImpl:  # noqa: C901
        """Return the cache implementation."""
        if self._memory_cache is not None and instance is None:
            return self._memory_cache

        elif self.method_caching == "per-instance":
            assert instance is not None, "Instance must be provided for per-instance caching."

            inst_cache = getattr(instance, "_cache", None)
            if inst_cache is not None:
                return inst_cache

        match self.evict_by:
            case "age":
                cache = FIFOCache(maxsize=self.max_size)
            case "last-usage":
                cache = LRUCache(maxsize=self.max_size)
            case "usage-frequency":
                cache = LFUCache(maxsize=self.max_size)
            case "random-choice":
                cache = RRCache(maxsize=self.max_size)
            case timedelta() as ttl:
                cache = TTLCache(maxsize=self.max_size, global_ttl=ttl.total_seconds())
            case float() as ttl:
                cache = TTLCache(maxsize=self.max_size, global_ttl=ttl)
            case _ if isinstance(self.evict_by, Callable):
                cache = VTTLCache(maxsize=self.max_size)
            case None:
                cache = NaiveCache(maxsize=self.max_size)
            case _:
                raise ValueError(f"Invalid evict_by value: {self.evict_by}")

        if self.method_caching == "per-instance" and instance is not None:
            setattr(instance, "_cache", cache)
        else:
            self._memory_cache = cache

        return cache

    def _get_from_cache(self, key: Any, mem_cache: BaseCacheImpl) -> R | MISSING:
        """Get the value from the cache, or MISSING if not found."""
        return mem_cache.get(key, MISSING())

    def _save_to_cache(self, key: Any, value: R, mem_cache: BaseCacheImpl) -> None:
        """Save the value to the cache."""
        mem_cache[key] = value

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the cached function."""
        args_dict = (
            get_full_args_dict(self.func, args, kwargs)
            if self.args_callback is None
            else self.args_callback(*args, **kwargs)
            if callable(self.args_callback)
            else {
                k: (self.args_callback[k](v) if k in self.args_callback else v)
                for k, v in get_full_args_dict(self.func, args, kwargs).items()
            }
        )

        key = gen_id(next(iter(args_dict.values()))) if len(args_dict) == 1 else gen_id(args_dict)

        instance = None
        if "self" == next(iter(args_dict.keys())):
            instance = args_dict["self"]

        if self.method_caching == "per-class" and instance is not None:
            key = (id(instance), key)

        cache = self.get_memory_cache(instance=instance)

        cached = self._get_from_cache(key, cache)
        if not isinstance(cached, MISSING):
            return cached

        result = self.func(*args, **kwargs)
        self._save_to_cache(key, result, cache)
        return result

    @property
    def __wrapped__(self) -> Callable[P, R]:
        """Return the undecorated function, as expected by introspection helpers."""
        return self.func


@overload
def cached[**P, R](
    func: Callable[P, R],
) -> CachedFunction[P, R]: ...


@overload
def cached[**P, R](
    *,
    max_size: int = 1000,
    args_callback: ArgsCallback[P] | None = None,
    evict_by: EvictionPolicy[P] = "age",
    method_caching: Literal["per-instance", "per-class"] = "per-instance",
) -> Callable[[Callable[P, R]], CachedFunction[P, R]]: ...


def cached[**P, R](
    func: Callable[P, R] | None = None,
    max_size: int = 1000,
    args_callback: ArgsCallback[P] | None = None,
    evict_by: EvictionPolicy[P] = "age",
    method_caching: Literal["per-instance", "per-class"] = "per-instance",
) -> Callable[[Callable[P, R]], CachedFunction[P, R]] | CachedFunction[P, R]:
    """Decorator to cache a function.

    Args:
        func: Function to cache.
        max_size: Maximum number of items to store in the cache.
        args_callback:
            Callback to transform the function arguments into a map of arguments to use for caching.
        evict_by: Eviction strategy for the cache.
        method_caching:
            Whether to keep a separate cache for each instance of the class (per-instance)
            or a single cache for all instances of the class (per-class).

    Returns:
        Decorated function with caching.
    """

    def decorator(func: Callable[P, R]) -> CachedFunction[P, R]:
        return CachedFunction(
            func=func,
            max_size=max_size,
            args_callback=args_callback,
            evict_by=evict_by,
            method_caching=method_caching,
        )

    return decorator if func is None else decorator(func)
