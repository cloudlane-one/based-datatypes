"""Base classes for transport of arbitrary types and data formats."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Collection, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import UnionType
from typing import (
    Any,
    Generic,
    Protocol,
    TypeAliasType,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    runtime_checkable,
)

import polars as pl
from typing_extensions import TypeForm

from based_datatypes.descriptors import cached_method
from based_datatypes.files import File
from based_datatypes.typing import SingleTypeDef, TypeAwareClass, is_subtype

T = TypeVar("T", contravariant=True)
T2 = TypeVar("T2")


@dataclass(frozen=True)
class TransportScope:
    """Base class for the scope of a transport operation."""

    def matches(self, scopes: Collection[type[TransportScope]]) -> bool:
        """Check whether any declared scope can serve this request.

        A declared scope fulfils the request when it is the same scope or a
        narrower one: moving data further only ever restricts the formats that
        may carry it, so a driver that supports WAN payloads also serves an IPC
        request. An :class:`Open` request imposes no restriction at all, so
        every declaration satisfies it.

        Args:
            scopes: The scope classes a driver or transportable declares.

        Returns:
            ``True`` if one of the declared scopes is this scope or narrower.
        """
        if isinstance(self, Open):
            return True

        return any(issubclass(scope_type, type(self)) for scope_type in scopes)


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


ALL_TRANSPORT_SCOPES: frozenset[type[TransportScope]] = frozenset({Open, IPC, LAN, WAN})
"""Every built-in transport scope, for drivers that can travel anywhere."""


current_ser_purpose: ContextVar[TransportScope] = ContextVar("current_ser_purpose", default=Open())
"""Context variable to hold the current serialization purpose."""


@contextmanager
def transport_scope(purpose: TransportScope) -> Generator[None]:
    """Context manager to set the current transport scope."""
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

    types: Sequence[SingleTypeDef[T]]
    """Declared transport type hints, e.g. ``File[Literal["application/json"], TextIO]``."""

    def __post_init__(self) -> None:
        """Normalize the declared hints to a tuple."""
        self.types = tuple(self.types)

    def match_targets(self, targets: Collection[TypeForm | File]) -> list[SingleTypeDef[T]]:
        """Return the supported types that any of the given targets satisfies.

        Args:
            targets: Transport mediums to match, either type hints or instances. A
                :class:`~based_datatypes.files.File` instance is matched on its
                ``mime`` and ``io_type`` so that, say, a JSON driver rejects a
                pickle target.

        Returns:
            The declared transport type hints able to read from or write to at
            least one of the targets.
        """
        return [
            t
            for t in self.types
            if any(tg.match(t) if isinstance(tg, File) else is_subtype(tg, t) for tg in targets)
        ]


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
    def __transport_scopes__(cls) -> frozenset[type[TransportScope]]:
        """Return the transport scopes this class supports.

        The declaration is static; implementations are expected to consult
        :data:`current_ser_purpose` inside ``__import__`` / ``__export__`` to
        adapt the payload to the scope actually in effect.
        """
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

    def __estimate_bytes__(self, target: T, realm: Realm) -> int:
        """Estimate the byte size of exporting this object to ``target``."""
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
        assert not isinstance(t, tuple)
        return t.single_typedef

    @classmethod
    def __transport_scopes__(cls) -> frozenset[type[TransportScope]]:
        """Return the transport scopes this driver supports.

        The declaration is static; :meth:`import_object` and
        :meth:`export_object` are expected to consult
        :data:`current_ser_purpose` when the payload must depend on the scope
        actually in effect. The default accepts every built-in scope.
        """
        return ALL_TRANSPORT_SCOPES

    @classmethod
    def __lossless_formats__(cls) -> frozenset[str]:
        """Return the MIME types this driver round-trips without loss.

        A lossless format reproduces the original object exactly; a lossy one
        only preserves a reduced representation (CSV drops dtypes, JPEG drops
        pixel precision). Empty by default, i.e. every format is considered
        lossy.
        """
        return frozenset()

    @classmethod
    @cached_method
    def transport_types(cls) -> TransportTypes[T]:
        """Return the set of types that this transport driver supports."""
        t = cls.typeargs[T]
        assert not isinstance(t, tuple)

        typedef = t.typedef
        if isinstance(typedef, TypeAliasType):
            typedef = typedef.__value__

        hints = (
            get_args(typedef)
            if isinstance(typedef, UnionType) or get_origin(typedef) is Union
            else (typedef,)
        )

        return TransportTypes(cast(tuple[SingleTypeDef[T], ...], tuple(hints)))

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
    def estimate_bytes(
        cls,
        instance: V,
        target: T,
        realm: Realm,
        annotation: SingleTypeDef[V] | None = None,
    ) -> int:
        """Estimate the byte size of exporting ``instance`` to ``target``.

        Drivers override this when they can measure their format cheaply. The
        default falls back to :func:`sys.getsizeof`, which measures the object
        in memory rather than the encoded representation.

        Args:
            instance: The object to be exported.
            target: The medium the object would be written to.
            realm: The realm the export would happen in.
            annotation: The declared type of the object, if any.

        Returns:
            The estimated number of bytes.
        """
        return sys.getsizeof(instance)

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
