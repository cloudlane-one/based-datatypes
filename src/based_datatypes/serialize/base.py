"""Base classes for the serialization of arbitrary types and data formats."""

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
from based_datatypes.typing import Runtyped, TypeRef


@dataclass(frozen=True)
class TransportScope(ABC):
    """Base class for the scope of a serialization operation."""

    def match(self, scopes: Collection[TransportScope]) -> bool:
        """Check whether this scope matches any of the requested scopes.

        A scope matches the request when it is the same scope or a
        narrower one.

        Args:
            scopes: The scope classes a serializer or serializable declares.

        Returns:
            ``True`` if one of the declared scopes is this scope or narrower.
        """
        return any(isinstance(scope_type, type(self)) for scope_type in scopes)


# @dataclass(frozen=True)
# class IPC(TransportScope):
#     """Serialization across processes within the same OS."""

#     source_process_id: int
#     """The process ID of the source process."""


# @dataclass(frozen=True)
# class LAN(IPC):
#     """Serialization across a local area network."""

#     private_subnet: str
#     """The subnet for LAN communication in CIDR notation (e.g., '192.168.1.0/24')."""


# @dataclass(frozen=True)
# class WAN(LAN):
#     """Indicates serialization for communication purposes across a wide area network."""

#     source_region: str | None = None
#     """Name of the region for WAN communication (e.g., 'us-east-1')."""


@dataclass(frozen=True)
class Open(TransportScope):
    """No restriction to the serialization scope, meaning full offline serialization."""


class Realm:
    """A class representing the realm into / from which serialization happens."""


type JSONScalar = str | int | float | bool | None
"""A value JSON represents directly."""

type JSONDict = dict[str, JSONValue]
"""A JSON object."""

type JSONValue = JSONScalar | JSONDict | list[JSONValue]
"""Any value a JSON document can hold."""

type SerializedValue = JSONScalar | SerializedDict | list[SerializedValue] | File
"""A serialized value: JSON-shaped data or a file, nested at any depth."""

type SerializedDict = dict[str, SerializedValue]
"""A serialized object: JSON-shaped data or files, keyed by name."""

type Serialized = JSONScalar | dict[str, Any] | list[Any] | File
"""Upper bound of everything a serializer may produce or consume.

The containers hold ``Any`` rather than :data:`SerializedValue`, so a specific
serialized artifact (``list[float]``, ``dict[str, str]``, ...) fits the bound
despite the invariance of the container type parameters.
"""


T = TypeVar("T", bound=Serialized)
T2 = TypeVar("T2", bound=Serialized)
S = TypeVar("S", bound=TransportScope, default=Open, contravariant=True)
L = TypeVar("L", bound=bool, default=Literal[False])


@dataclass(frozen=True)
class SerializationTarget(Generic[T, S, L]):
    """A set of supported serialization formats."""

    type: TypeDef[T]
    """The type of the target format."""

    scope: S
    """The scope for this target format."""

    lossless: L
    """Whether this target format supports lossless serialization."""

    @staticmethod
    def match(
        targets: Iterable[SerializationTarget[Any, Open, Any]],
        types: Collection[TypeDef[Serialized]] | None = None,
        scope: TransportScope | None = None,
        lossless: bool = True,
    ) -> list[SerializationTarget[T, S, L]]:
        """Return the subset of targets that match any of the given types.

        Args:
            targets: The list of serialization targets to filter.
            types: The collection of types to match against.
            scope: The scope to match.
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
                    any((TypeRef(tt).is_subtype_of(t.type)) for tt in types)
                    if types is not None
                    else True
                )
                and (t.scope.match([scope]) if scope is not None else True)
                and (not lossless or t.lossless)
            ],
        )


current_ser_ctx: ContextVar[
    tuple[SerializationTarget[SerializedDict, Open, bool], Realm | None]
] = ContextVar(
    "current_ser_ctx",
    default=(SerializationTarget(type=SerializedDict, scope=Open(), lossless=True), None),
)
"""The current serialization context: (serialization target, realm)."""


@contextmanager
def serialization_context(
    target: SerializationTarget[Any, Any, Any], realm: Realm | None = None
) -> Generator[None]:
    """Context manager to set the current serialization context."""
    token = current_ser_ctx.set((target, realm))
    try:
        yield
    finally:
        current_ser_ctx.reset(token)


@runtime_checkable
class Serializable(Protocol[T, S]):  # pyright: ignore[reportInvalidTypeVarUse]
    """Protocol for classes that can serialize and deserialize themselves."""

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
        """Deserialize from the source type or format.

        Args:
            source: The source object to deserialize from.
            scope: The serialization scope to use for the deserialization.
            realm: The realm to deserialize from.
            annotation: The declared type of the object, if any.

        Returns:
            An instance of the class, deserialized from the source.
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
        """Serialize to the target type or format.

        Args:
            target_type: The target type or format to serialize to.
            lossless: Whether to require a lossless serialization.
            scope: The serialization scope to use for the serialization.
            annotation: The declared type of the object, if any.
            realm: The realm to serialize to.
            file_factory: An optional factory function to create new files.

        Returns:
            The serialized target object.
        """
        ...

    def __estimate_serialized_size__(
        self: Serializable[T2],
        target_type: TypeDef[T2],
        lossless: bool = True,
        scope: S = Open(),
        annotation: TypeDef[Serializable[T2]] | None = None,
    ) -> int:
        """Estimate the byte size of serializing this object to ``target_type``.

        Args:
            target_type: The target type or format to estimate the byte size for.
            lossless: Whether to require a lossless serialization.
            scope: The serialization scope to use for the estimation.
            annotation: The declared type of the object, if any.

        Returns:
            The estimated byte size of serializing this object to ``target_type``.
        """
        ...


V = TypeVar("V")
V2 = TypeVar("V2")
U = TypeVar("U", bound="Serializable")


def target_mimes(hint: Any) -> tuple[str, ...]:
    """Return the MIME types a file target declares.

    Args:
        hint: A target type, a :class:`TypeRef`, or ``None``.

    Returns:
        The declared MIME types, in declaration order; empty when the hint is not
        a file target.
    """
    if hint is None:
        return ()

    ref = hint if isinstance(hint, TypeRef) else TypeRef(hint)

    for var, value in ref.typevar_map.items():
        if var.__name__ == "M":
            return cast(tuple[str, ...], value.literal_values)

    return ()


@dataclass
class Serializer(ABC, Runtyped, Generic[V, T, S]):  # noqa: UP046
    """A class representing a serialization driver for a specific type."""

    @prop(cached=True, mode="class")
    def obj_type(self) -> TypeDef[V]:
        """Return the type of object this serializer handles."""
        return self._typeref.typevar_map[V].resolved_type_def

    __lossless_mimes__: frozenset[str] = frozenset()
    """MIME types this serializer round-trips exactly; empty marks every file format lossy."""

    @classmethod
    def lossless_target(cls, hint: TypeDef[Any]) -> bool:
        """Whether a target format round-trips the object exactly.

        Primitive targets always round-trip; a file target does exactly when all
        of its MIME types are declared lossless by the serializer.

        Args:
            hint: The target type to check.

        Returns:
            ``True`` if the target is lossless for this serializer.
        """
        mimes = target_mimes(hint)

        return all(mime in cls.__lossless_mimes__ for mime in mimes) if mimes else True

    @prop(cached=True, mode="class")
    def serialization_targets(self) -> list[SerializationTarget[T, S, Any]]:
        """Serialization targets supported by this serializer, as declared in its type arguments."""
        typeref = self._typeref.typevar_map[T]
        return cast(
            list[SerializationTarget[T, S, Any]],
            [
                SerializationTarget(
                    type=member, scope=Open(), lossless=self.lossless_target(member)
                )
                for member in typeref.members
            ],
        )

    @abstractmethod
    def deserialize(
        self,
        source: T,
        scope: S = Open(),
        realm: Realm | None = None,
        annotation: TypeDef[V] | None = None,
    ) -> V:
        """Deserialize an object from the source type or format.

        Args:
            source: The source object to deserialize from.
            scope: The serialization scope to use for the deserialization.
            realm: The realm to deserialize from.
            annotation: The declared type of the object, if any.

        Returns:
            An instance of the object, deserialized from the source.
        """
        ...

    @abstractmethod
    def serialize(
        self,
        instance: V,
        target_type: TypeDef[T] | None = None,
        lossless: bool = True,
        scope: S = Open(),
        annotation: TypeDef[V] | None = None,
        realm: Realm | None = None,
        file_factory: Callable[[], File] | None = None,
    ) -> T:
        """Serialize an object to the target type or format.

        Args:
            instance: The object to serialize.
            target_type: The target type or format to serialize to.
            lossless: Whether to require a lossless serialization.
            scope: The serialization scope to use for the serialization.
            annotation: The declared type of the object, if any.
            realm: The realm to serialize to.
            file_factory: An optional factory function to create new files.

        Returns:
            The serialized target object.
        """
        ...

    def estimate_serialized_size(
        self,
        instance: V,
        target_type: TypeDef[T] | None = None,
        lossless: bool = True,
        scope: S = Open(),
        annotation: TypeDef[V] | None = None,
    ) -> int:
        """Estimate the byte size of serializing ``instance`` to ``target_type``.

        Drivers override this when they can measure their format cheaply. The
        default falls back to :func:`sys.getsizeof`, which measures the object
        in memory rather than the encoded representation.

        Args:
            instance: The object to be serialized.
            target_type: The target type or format to serialize to.
            lossless: Whether to require a lossless serialization.
            scope: The serialization scope to use for the serialization.
            annotation: The declared type of the object, if any.

        Returns:
            The estimated number of bytes.
        """
        return sys.getsizeof(instance)
