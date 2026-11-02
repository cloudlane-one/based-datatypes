"""Base classes for transport of arbitrary types and data formats."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Generator, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar, cast, runtime_checkable

from based_datatypes.descriptors import prop
from based_datatypes.files import File
from based_datatypes.types import TypeDef
from based_datatypes.typing import TypeAwareClass, TypeRef


@dataclass(frozen=True)
class TransportScope(ABC):
    """Base class for the scope of a transport operation."""

    def match(self, scopes: Collection[TransportScope]) -> bool:
        """Check whether this scope matches any of the requested scopes.

        A scope matches the request when it is the same scope or a
        narrower one.

        Args:
            scopes: The scope classes a driver or transportable declares.

        Returns:
            ``True`` if one of the declared scopes is this scope or narrower.
        """
        return any(isinstance(scope_type, type(self)) for scope_type in scopes)


# @dataclass(frozen=True)
# class IPC(TransportScope):
#     """Transport across processes within the same OS."""

#     source_process_id: int
#     """The process ID of the source process."""


# @dataclass(frozen=True)
# class LAN(IPC):
#     """Transport across a local area network."""

#     private_subnet: str
#     """The subnet for LAN communication in CIDR notation (e.g., '192.168.1.0/24')."""


# @dataclass(frozen=True)
# class WAN(LAN):
#     """Indicates serialization for communication purposes across a wide area network."""

#     source_region: str | None = None
#     """Name of the region for WAN communication (e.g., 'us-east-1')."""


@dataclass(frozen=True)
class Open(TransportScope):
    """No restriction to transport scope, meaning full offline serialization."""


class Realm:
    """A class representing the realm into / from which import / export occurs."""


type JSONScalar = str | int | float | bool | None
type JSONDict = dict[str, JSONValue]
type JSONValue = JSONScalar | JSONDict | list[JSONValue]
type Serialized = JSONValue | File


T = TypeVar("T", bound=Serialized, contravariant=True)
T2 = TypeVar("T2", bound=Serialized)
S = TypeVar("S", bound=TransportScope, default=Open, contravariant=True)
L = TypeVar("L", bound=bool, default=Literal[False])


@dataclass(frozen=True)
class SerializationTarget(Generic[T, S, L]):
    """A set of supported serialization formats."""

    type: TypeDef[T]
    """The type of the target format."""

    scope: S
    """The transport scope for this target format."""

    lossless: L
    """Whether this target format supports lossless serialization."""

    @staticmethod
    def match(
        targets: Iterable[SerializationTarget[Any, Open, Any]],
        types: Collection[TypeDef[JSONValue | File]] | None = None,
        scope: TransportScope | None = None,
        lossless: bool = True,
    ) -> list[SerializationTarget[T, S, L]]:
        """Return the subset of targets that match any of the given types.

        Args:
            targets: The list of serialization targets to filter.
            types: The collection of types or files to match against.
            scope: The transport scope to match.
            lossless: Whether to only include lossless targets.

        Returns:
            The list of matching serialization targets.
        """
        return cast(
            list[SerializationTarget[T, S, L]],
            [
                t
                for t in targets
                if (
                    any((TypeRef(tg).is_subtype(t.type)) for tg in types)
                    if types is not None
                    else True
                )
                and (t.scope.match([scope]) if scope is not None else True)
                and (not lossless or t.lossless)
            ],
        )


current_ser_ctx: ContextVar[tuple[SerializationTarget[JSONDict, Open, bool], Realm | None]] = (
    ContextVar(
        "current_ser_ctx",
        default=(SerializationTarget(type=JSONDict, scope=Open(), lossless=True), None),
    )
)
"""The current serialization context: (serialization target, realm)."""


@contextmanager
def serialization_context(
    target: SerializationTarget[Any, Any, Any], realm: Realm | None = None
) -> Generator[None]:
    """Context manager to set the current transport scope."""
    token = current_ser_ctx.set((target, realm))
    try:
        yield
    finally:
        current_ser_ctx.reset(token)


@runtime_checkable
class Serializable(Protocol[T, S]):  # pyright: ignore[reportInvalidTypeVarUse]
    """Protocol for classes that can be imported / exported."""

    @classmethod
    def __serialization_targets__(cls) -> list[SerializationTarget[T, S, Any]]:
        """Serialization targets supported by this class."""
        ...

    @classmethod
    def __deserialize__(
        cls: type[U],
        source: Serialized,
        scope: S = Open(),
        realm: Realm | None = None,
        annotation: TypeDef[U] | None = None,
    ) -> U:
        """Import from the source type or format.

        Args:
            source: The source object to import from.
            scope: The transport scope to use for the import.
            realm: The realm to import from.
            annotation: The declared type of the object, if any.

        Returns:
            An instance of the class, imported from the source.
        """
        ...

    def __serialize__(
        self: Serializable[T2],
        target_type: TypeDef[T2],
        lossless: bool = True,
        scope: S = Open(),
        annotation: TypeDef[Serializable[T2]] | None = None,
        realm: Realm | None = None,
        file_factory: Callable[[], File] | None = None,
    ) -> T2:
        """Export to the target type or format.

        Args:
            target_type: The target type or format to export to.
            lossless: Whether to require a lossless export.
            scope: The transport scope to use for the export.
            annotation: The declared type of the object, if any.
            realm: The realm to export to.
            file_factory: An optional factory function to create new files.

        Returns:
            The exported target object.
        """
        ...

    def __estimate_serialized_size__(
        self: Serializable[T2],
        target_type: TypeDef[T2],
        lossless: bool = True,
        scope: S = Open(),
        annotation: TypeDef[Serializable[T2]] | None = None,
    ) -> int:
        """Estimate the byte size of exporting this object to ``target_type``.

        Args:
            target_type: The target type or format to estimate the byte size for.
            lossless: Whether to require a lossless export.
            scope: The transport scope to use for the estimation.
            annotation: The declared type of the object, if any.

        Returns:
            The estimated byte size of exporting this object to ``target_type``.
        """
        ...


V = TypeVar("V")
V2 = TypeVar("V2")
U = TypeVar("U", bound="Serializable")


@dataclass
class Serializer(ABC, TypeAwareClass, Generic[V, T, S]):  # noqa: UP046
    """A class representing a serialization driver for a specific type."""

    @prop(cached=True, mode="class")
    def obj_type(self) -> TypeDef[V]:
        """Return the type of object this transport driver handles."""
        return self.typeref.scalar_typevar_map[V].resolved_type_def

    @prop(cached=True, mode="class")
    def serialization_targets(self) -> list[SerializationTarget[T, S, Any]]:
        """Serialization targets supported by this transport driver."""
        typeref = self.typeref.scalar_typevar_map[T]
        return cast(
            list[SerializationTarget[T, S, Any]],
            [SerializationTarget(type=t, scope=Open(), lossless=False) for t in typeref.members],
        )

    @abstractmethod
    def deserialize(
        self: Serializer[V2, T2],
        source: T2,
        scope: S = Open(),
        realm: Realm | None = None,
        annotation: TypeDef[V2] | None = None,
    ) -> V2:
        """Import an object from the source type or format.

        Args:
            source: The source object to import from.
            scope: The transport scope to use for the import.
            realm: The realm to import from.
            annotation: The declared type of the object, if any.

        Returns:
            An instance of the object, imported from the source.
        """
        ...

    @abstractmethod
    def serialize(
        self: Serializer[V2, T2],
        instance: V2,
        target_type: TypeDef[T2],
        lossless: bool = True,
        scope: S = Open(),
        annotation: TypeDef[V2] | None = None,
        realm: Realm | None = None,
        file_factory: Callable[[], File] | None = None,
    ) -> T2:
        """Export an object to the target type or format.

        Args:
            instance: The object to export.
            target_type: The target type or format to export to.
            lossless: Whether to require a lossless export.
            scope: The transport scope to use for the export.
            annotation: The declared type of the object, if any.
            realm: The realm to export to.
            file_factory: An optional factory function to create new files.

        Returns:
            The exported target object.
        """
        ...

    def estimate_serialized_size(
        self: Serializer[V2, T2],
        instance: V2,
        target_type: TypeDef[T2],
        lossless: bool = True,
        scope: S = Open(),
        annotation: TypeDef[V2] | None = None,
    ) -> int:
        """Estimate the byte size of exporting ``instance`` to ``target_type``.

        Drivers override this when they can measure their format cheaply. The
        default falls back to :func:`sys.getsizeof`, which measures the object
        in memory rather than the encoded representation.

        Args:
            instance: The object to be exported.
            target_type: The target type or format to export to.
            lossless: Whether to require a lossless export.
            scope: The transport scope to use for the export.
            annotation: The declared type of the object, if any.

        Returns:
            The estimated number of bytes.
        """
        return sys.getsizeof(instance)
