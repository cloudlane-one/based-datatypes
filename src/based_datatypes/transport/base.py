"""Base classes for transport of arbitrary types and data formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence, Set
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import InitVar, dataclass
from types import UnionType
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import polars as pl

from based_datatypes.descriptors import cached_method
from based_datatypes.typing import SingleTypeDef, TypeAwareClass, is_subtype

T = TypeVar("T", contravariant=True)
T2 = TypeVar("T2")


@dataclass(frozen=True)
class TransportScope:
    """Base class for the scope of a transport operation."""


@dataclass(frozen=True)
class IPC(TransportScope):
    """Transport across processes within the same OS."""


@dataclass(frozen=True)
class LAN(IPC):
    """Transport across a local area network."""

    private_subnet: str
    """The subnet for LAN communication in CIDR notation (e.g., '192.168.1.0/24')."""


@dataclass(frozen=True)
class WAN(LAN):
    """Indicates serialization for communication purposes across a wide area network."""

    source_region: str | None = None
    """Name of the region for WAN communication (e.g., 'us-east-1')."""


@dataclass(frozen=True)
class Open(TransportScope):
    """No restriction to transport scope, meaning full offline serialization."""


current_ser_purpose: ContextVar[TransportScope] = ContextVar("current_ser_purpose", default=Open())
"""Context variable to hold the current serialization purpose."""


@contextmanager
def serialization_purpose(purpose: TransportScope) -> Generator[None]:
    """Context manager to set the current serialization purpose."""
    token = current_ser_purpose.set(purpose)
    try:
        yield
    finally:
        current_ser_purpose.reset(token)


class Realm:
    """A class representing the realm into / from which storage / loading occurs."""


@dataclass
class TransportTypes(Generic[T]):  # noqa: UP046
    """A set of interfaces that can be used as transport formats."""

    types: InitVar[Sequence[SingleTypeDef[T]]]

    def __post_init__(self, types: Sequence[SingleTypeDef[T]]) -> None:
        """Initialize the TypeSet with a set of types."""
        self.__types = tuple(types)

    def match_targets(
        self: TransportTypes[T2], targets: Set[SingleTypeDef[T2]]
    ) -> list[SingleTypeDef[T2]]:
        """Filter the types in this set to only those that are in the provided set."""
        return [t2 for t in self.__types for t2 in targets if is_subtype(t, t2)]


U = TypeVar("U", bound="Transportable")

B = TypeVar("B", contravariant=True, default=None)
B2 = TypeVar("B2")


@runtime_checkable
class Transportable(Protocol[T]):  # pyright: ignore[reportInvalidTypeVarUse]
    """Protocol for classes that can be imported / exported."""

    @classmethod
    def __transport_types__(cls) -> TransportTypes[T]:
        """Return the set of transport format types that this class supports."""
        ...

    @classmethod
    def __import__(
        cls: type[U],
        source: T,
        realm: Realm,
        annotation: SingleTypeDef[U] | None = None,
    ) -> U:
        """Import from the source type or format."""
        ...

    def __export__(
        self: Transportable[T2],
        target: T2,
        realm: Realm,
        annotation: SingleTypeDef[Transportable[T2]] | None = None,
    ) -> None:
        """Export to the target type or format."""
        ...


V = TypeVar("V")
V2 = TypeVar("V2")


@dataclass
class TransportDriver(ABC, TypeAwareClass, Generic[V, T, B]):
    """A class representing a transport driver for a specific type."""

    @classmethod
    @cached_method
    def obj_type(cls) -> SingleTypeDef[V]:
        """Return the type of object this transport driver handles."""
        t = cls.typeargs[V]
        assert not isinstance(t, UnionType), "U must not be a Union type"
        assert not isinstance(t, tuple)
        return t.single_typedef

    @classmethod
    @cached_method
    def transport_types(cls) -> TransportTypes[T]:
        """Return the set of types that this transport driver supports."""
        t = cls.typeargs[T]
        assert not isinstance(t, tuple)

        return TransportTypes(tuple(t.typeset))

    @classmethod
    @abstractmethod
    def import_object(
        cls: type[TransportDriver[V2, Any]],
        source: T,
        realm: Realm,
        annotation: SingleTypeDef[V2] | None = None,
    ) -> V2:
        """Import an object from the source type or format."""
        ...

    @classmethod
    @abstractmethod
    def export_object(
        cls,
        instance: V,
        target: T,
        realm: Realm,
        annotation: SingleTypeDef[V] | None = None,
    ) -> None:
        """Export an object to the target type or format."""
        ...

    @classmethod
    def batch_import(
        cls,
        source: B,
        realm: Realm,
        annotation: SingleTypeDef[V] | None = None,
    ) -> pl.Series:
        """Import a batch of data from the source type or format."""
        raise NotImplementedError()

    @classmethod
    def batch_export(
        cls: type[TransportDriver[Any, Any, B2]],
        batch: pl.Series,
        target: B2,
        realm: Realm,
        annotation: SingleTypeDef[V] | None = None,
    ) -> None:
        """Export a batch of data to the target type or format."""
        raise NotImplementedError()
