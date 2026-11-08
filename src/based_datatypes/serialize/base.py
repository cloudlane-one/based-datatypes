"""Base classes for the serialization of arbitrary types and data formats."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Collection, Generator, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
    Any,
    BinaryIO,
    Generic,
    Literal,
    Protocol,
    TextIO,
    TypeVar,
    cast,
    override,
    runtime_checkable,
)

from based_datatypes.descriptors import prop
from based_datatypes.files import File, I, MemoryFile, Write
from based_datatypes.types import TypeDef
from based_datatypes.typing import Runtyped, TypeRef, is_subtype


@dataclass(frozen=True)
class TransportScope(ABC):
    """Base class for the scope of a serialization operation."""

    def match(self, scopes: Collection[TransportScope | type[TransportScope]]) -> bool:
        """Check whether this declared scope can serve any of the requested scopes.

        The declaration is the receiver and the requests are the arguments: a
        declaration serves a request when the request is an instance of the
        declared scope, so a serializer declared for :class:`Open` serves every
        request, while one declared for a narrower scope cannot serve an
        unrestricted one. Scopes form a hierarchy rooted at :class:`Open`, and
        the comparison is the class relation, never an exact-class match.

        Args:
            scopes: The requested scopes, as classes or values.

        Returns:
            ``True`` if this declaration can serve at least one request.
        """
        return any(_scope_serves(self, scope) for scope in scopes)


def _scope_serves(
    declared: TransportScope | type[TransportScope],
    requested: TransportScope | type[TransportScope],
) -> bool:
    """Check whether a declared scope can serve a requested scope.

    Args:
        declared: The scope a serializer or serializable declares.
        requested: The scope the operation requests.

    Returns:
        ``True`` if the request is the declared scope or a narrower one.
    """
    declared_cls = declared if isinstance(declared, type) else type(declared)
    requested_cls = requested if isinstance(requested, type) else type(requested)

    return issubclass(requested_cls, declared_cls)


@dataclass(frozen=True)
class Open(TransportScope):
    """No restriction to the serialization scope, meaning full offline serialization."""


class Backend(ABC):
    """A class representing the data backend into / from which serialization happens."""

    @abstractmethod
    def get_writable_file[K: TextIO | BinaryIO](
        self, typedef: TypeDef[File[Any, K, Write]]
    ) -> File[Any, K, Write]:
        """Return a file to write to.

        Returns:
            A file object.
        """
        ...


class Memory(Backend):
    """A backend that writes to in-memory files."""

    @override
    def get_writable_file[K: TextIO | BinaryIO](
        self, typedef: TypeDef[File[Any, K, Write]]
    ) -> MemoryFile[Any, K, Write]:

        return MemoryFile(
            mime=File.get_all_mimes(typedef)[0],
            io=TypeRef(typedef).typevar_map[I].resolved_type_def,
            rw="w",
        )


class UnsupportedTargetError(ValueError):
    """A serializer cannot produce the requested target.

    A driver raises this when the requested format cannot carry what it was
    asked to write — a JSON or YAML file cannot hold a file produced for a
    nested item, for example. :func:`serialize` and
    :func:`estimate_serialized_size` treat it as a selection miss: they try the
    next matching serializer and only raise when none can fulfil the target.
    """


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

    obj_type: TypeDef
    """The type of the object being serialized."""

    tgt_type: type[T]
    """The type of the target format."""

    scope: type[S] | S
    """The scope this target format is declared for.

    A serializer's type argument names the scope class, while an explicit target
    declaration may carry a scope value; :meth:`TransportScope.match` accepts
    either. The requested scope of an operation is applied to the target when it
    is dispatched, so this field only ever describes the declaration.
    """

    lossy: L
    """Whether this target format is lossy."""

    def __post_init__(self) -> None:  # noqa: D105
        if not isinstance(self.tgt_type, type):
            raise TypeError(f"SerializationTarget.type must be a type, got {self.tgt_type!r}")

        if not isinstance(self.scope, (type, TransportScope)):
            raise TypeError(
                f"SerializationTarget.scope must be a type or TransportScope, got {self.scope!r}"
            )

    @staticmethod
    def match(
        targets: Iterable[SerializationTarget[Any, Any, Any]],
        types: Collection[TypeDef[Serialized]] | None = None,
        scope: TransportScope | None = None,
        allow_lossy: bool = False,
    ) -> list[SerializationTarget[T, S, L]]:
        """Return the subset of targets that match any of the given types.

        Args:
            targets: The list of serialization targets to filter.
            types: The collection of types to match against.
            scope: The scope to match.
            allow_lossy: Whether to include lossy targets.

        Returns:
            The list of matching serialization targets.
        """
        return cast(
            list[SerializationTarget[T, S, L]],
            [
                t
                for t in targets
                if (
                    any((TypeRef(tt).is_subtype_of(t.tgt_type)) for tt in types)
                    if types is not None
                    else True
                )
                and (_scope_serves(t.scope, scope) if scope is not None else True)
                and (allow_lossy or not t.lossy)
            ],
        )


current_ser_ctx: ContextVar[
    tuple[SerializationTarget[SerializedDict, Open, bool], Backend] | None
] = ContextVar(
    "current_ser_ctx",
    default=None,
)
"""The current serialization context: (serialization target, realm)."""


@contextmanager
def serialization_context(
    target: SerializationTarget[Any, Any, Any], realm: Backend = Memory()
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
        annotation: TypeDef[U] | None = None,
    ) -> U:
        """Deserialize from the source type or format.

        Args:
            source: The source object to deserialize from.
            scope: The serialization scope to use for the deserialization.
            annotation: The declared type of the object, if any.

        Returns:
            An instance of the class, deserialized from the source.
        """
        ...

    def __serialize__(
        self: Serializable[T2],
        target_type: TypeDef[T2],
        allow_lossy: bool = False,
        scope: S = Open(),
        annotation: TypeDef[Serializable[T2]] | None = None,
        backend: Backend = Memory(),
    ) -> T2:
        """Serialize to the target type or format.

        Args:
            target_type: The target type or format to serialize to.
            allow_lossy: Whether to allow lossy serialization.
            scope: The serialization scope to use for the serialization.
            annotation: The declared type of the object, if any.
            backend: The backend to serialize to.

        Returns:
            The serialized target object.
        """
        ...

    def __estimate_serialized_size__(
        self: Serializable[T2],
        target_type: TypeDef[T2],
        allow_lossy: bool = False,
        scope: S = Open(),
        annotation: TypeDef[Serializable[T2]] | None = None,
    ) -> int:
        """Estimate the byte size of serializing this object to ``target_type``.

        Args:
            target_type: The target type or format to estimate the byte size for.
            allow_lossy: Whether to allow lossy serialization.
            scope: The serialization scope to use for the estimation.
            annotation: The declared type of the object, if any.

        Returns:
            The estimated byte size of serializing this object to ``target_type``.
        """
        ...


V = TypeVar("V")
V2 = TypeVar("V2")
U = TypeVar("U", bound="Serializable")


@dataclass
class Serializer(ABC, Runtyped, Generic[V, T, S]):  # noqa: UP046
    """A class representing a serialization driver for a specific type."""

    @prop(cached=True, mode="class")
    def obj_type(self) -> TypeDef[V]:
        """Return the type of object this serializer handles."""
        return self._typeref.typevar_map[V].resolved_type_def

    __lossless_mimes__: frozenset[str] = frozenset()
    """MIME types this serializer round-trips exactly; empty marks every file format lossy."""

    __lossless_values__: bool = True
    """Whether the serializer's non-file targets round-trip the object exactly.

    A value target is the same encoding as the file target of the same format,
    so a serializer whose text output is lossy — SQL and HTML re-render the
    object rather than reproducing it — declares ``False`` here as well.
    """

    @classmethod
    def lossless_target(cls, filetype: type[File]) -> bool:
        """Whether a target format round-trips the object exactly.

        A file target does exactly when all of its MIME types are declared
        lossless by the serializer; a target that is not a file follows
        :attr:`__lossless_values__`.

        Args:
            filetype: The target file type to check.

        Returns:
            ``True`` if the target is lossless for this serializer.
        """
        return (
            all(mime in cls.__lossless_mimes__ for mime in filetype.mimes)
            if filetype.mimes
            else cls.__lossless_values__
        )

    @prop(cached=True, mode="class")
    def scopes(self) -> tuple[type[S], ...]:
        """The scope classes this serializer declares through its ``S`` type argument.

        A scope value carries data (a process id, a subnet), so the class-level
        declaration is the type argument itself; selection compares it with the
        requested scope instead of materializing a value that would lose that data.
        """
        return (cast(type[S], self._typeref.typevar_map[S].resolved_type_def),)

    @prop(cached=True, mode="class")
    def serialization_targets(self) -> list[SerializationTarget[T, S, Any]]:
        """Serialization targets supported by this serializer, as declared in its type arguments."""
        typeref = self._typeref.typevar_map[T]
        # The declarations are the driver's: the requested scope of an operation
        # is applied to the target when it is dispatched, so this metadata only
        # states what the serializer itself can serve.
        declared = self.scopes
        return cast(
            list[SerializationTarget[T, S, Any]],
            [
                SerializationTarget(
                    obj_type=self.obj_type,
                    tgt_type=member,
                    scope=scope,
                    lossy=not self.lossless_target(member),
                )
                for member in typeref.members
                if isinstance(member, type) and issubclass(member, File)
                for scope in declared
            ],
        )

    @abstractmethod
    def deserialize(
        self,
        source: T,
        annotation: TypeDef[V],
        scope: S = Open(),
    ) -> V:
        """Deserialize an object from the source type or format.

        Args:
            source: The source object to deserialize from.
            annotation: The declared type of the object.
            scope: The serialization scope to use for the deserialization.

        Returns:
            An instance of the object, deserialized from the source.
        """
        ...

    @abstractmethod
    def serialize(
        self,
        instance: V,
        target_type: TypeDef[T] | None = None,
        allow_lossy: bool = False,
        scope: S = Open(),
        annotation: TypeDef[V] | None = None,
        backend: Backend = Memory(),
    ) -> T:
        """Serialize an object to the target type or format.

        Args:
            instance: The object to serialize.
            target_type: The target type or format to serialize to.
            allow_lossy: Whether to allow lossy serialization.
            scope: The serialization scope to use for the serialization.
            annotation: The declared type of the object, if any.
            backend: The backend to serialize to.

        Returns:
            The serialized target object.
        """
        ...

    def estimate_serialized_size(
        self,
        instance: V,
        target_type: TypeDef[T] | None = None,
        allow_lossy: bool = False,
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
            allow_lossy: Whether to allow lossy serialization.
            scope: The serialization scope to use for the serialization.
            annotation: The declared type of the object, if any.

        Returns:
            The estimated number of bytes.
        """
        return sys.getsizeof(instance)


_serializers: list[Serializer[Any, Any, Any]] = []


def _handles(
    serializer: Serializer[Any, Any, Any], typ: TypeDef[Any], value: object | None
) -> bool:
    """Check whether a serializer handles a type, and the given value when there is one."""
    if not is_subtype(typ, serializer.obj_type):
        return False

    return value is None or TypeRef(serializer.obj_type).validate(value)


def get_serializers[U, T: Serialized, S: TransportScope, L: bool](
    typ: TypeDef[U],
    targets: Collection[TypeDef[T]] | None = None,
    scope: S = Open(),
    allow_lossy: L = False,
    value: object | None = None,
) -> list[tuple[int, SerializationTarget[T, S, L], Serializer[U, T, S] | type[Serializable[T, S]]]]:
    """Get the serializers that can handle a given type or instance.

    Args:
        typ: The type or instance to get the serializer for.
        targets: The serialization targets to write to or read from. Instances are
            matched on their MIME type and IO type, so the serializer for the actual
            format wins; types only narrow by class. ``None`` skips the target
            check and returns the serializers for the object type alone.
        scope: The scope to select for. A serializer declared for a supertype
            scope (e.g. Open) also serves a request for a narrower one.
        allow_lossy: When ``True``, only serializers that round-trip the target
            format without loss are eligible; ``False`` accepts either.
        value: The instance to select for. A type alone cannot express element
            constraints — an SQL statement list and ``[1, 2]`` are both ``list`` —
            so a value is validated against the driver's declared object type as
            well. ``None`` matches on the type only, which is what introspection
            wants.

    Returns:
        A list of tuples of the form ``(index, target, driver)``, where
        ``index`` is the index of the matching target in the list of targets,
        ``target`` is the matching serialization target and ``driver`` is the
        serializer or serializable class to handle the object.
    """
    candidates = []

    cls = TypeRef(typ).common_type

    if issubclass(cls, Serializable):
        candidates = sorted(
            [
                (i, t, cls)
                for i, t in enumerate(
                    SerializationTarget.match(
                        cls.__serialization_targets__(), targets, scope, allow_lossy
                    )
                )
            ],
            key=lambda x: x[0],
        )  # Sort by index to prioritize the first match

    if len(candidates) > 0:
        return candidates  # pyright: ignore[reportReturnType]

    candidates = sorted(
        [
            (i, t, driver)
            for driver in _serializers
            if _handles(driver, typ, value)
            for i, t in enumerate(
                SerializationTarget.match(driver.serialization_targets, targets, scope, allow_lossy)
            )
        ],
        key=lambda x: x[0],
    )  # Sort by index to prioritize the first match

    return candidates  # pyright: ignore[reportReturnType]


def deserialize[V2](
    source: Serialized,
    annotation: TypeDef[V2],
    scope: TransportScope = Open(),
) -> V2:
    """Deserialize an object from the source type or format.

    Args:
        source: The source object to deserialize from.
        annotation: The declared type of the object.
        scope: The serialization scope to use for the deserialization.
        realm: The realm to deserialize from.

    Returns:
        An instance of the object, deserialized from the source.
    """
    # Reading is not restricted to lossless formats: the lossiness of a format
    # describes what an export can drop, not what an import can restore.
    candidates = get_serializers(
        annotation,
        targets=[cast(type[V2], type(source))],
        scope=scope,
        allow_lossy=True,
    )

    if len(candidates) == 0:
        raise ValueError(
            f"No suitable serializer found for {type(source).__name__} as "
            f"{annotation!r} under scope {scope!r}."
        )

    _, target, serializer = candidates[0]

    with serialization_context(target):
        return cast(
            V2,
            (
                serializer.__deserialize__(
                    source,
                    scope=scope,
                    annotation=annotation,
                )
                if isinstance(serializer, type)
                else serializer.deserialize(
                    source,
                    scope=scope,
                    annotation=annotation,
                )
            ),
        )


def serialize[T2: Serialized](
    instance: Any,
    target_type: TypeDef[T2] | None = None,
    allow_lossy: bool = False,
    scope: TransportScope = Open(),
    annotation: TypeDef[Serializable[T2]] | None = None,
    backend: Backend = Memory(),
) -> T2:
    """Serialize an object to the target type or format.

    When no target type is given, the serializer picks its preferred format: the
    first one it declares, which is the lossless one wherever it has a choice.
    ``allow_lossy`` therefore only filters the selection once a target is requested.

    Args:
        instance: The object to serialize.
        target_type: The target type or format to serialize to.
        allow_lossy: Whether to allow a lossy serialization.
        scope: The serialization scope to use for the serialization.
        annotation: The declared type of the object, if any.
        backend: The backend to serialize to.
        file_factory: An optional factory function to create new files.

    Returns:
        The serialized target object.
    """
    candidates = get_serializers(
        typ=annotation if annotation is not None else type(instance),
        targets=[target_type] if target_type is not None else None,
        scope=scope,
        # Without a requested target the driver picks its preferred format, so
        # there is no format to judge: the lossless filter applies only once a
        # target is asked for.
        allow_lossy=allow_lossy if target_type is not None else False,
        # With no annotation the instance itself narrows the selection: a bare
        # ``list`` cannot say whether it holds SQL statements or integers.
        value=instance if annotation is None else None,
    )

    if len(candidates) == 0:
        raise ValueError(
            f"No suitable serializer found for {type(instance).__name__} to "
            f"{target_type!r} under scope {scope!r} "
            f"with allow_lossy={allow_lossy}."
        )

    _, target, serializer = candidates[0]

    with serialization_context(target, backend):
        return (
            serializer.__serialize__(
                instance,
                target_type=target_type if target_type is not None else target.tgt_type,
                allow_lossy=target.lossy,
                scope=scope,
                backend=backend,
                annotation=annotation,
            )
            if isinstance(serializer, type)
            else serializer.serialize(
                instance,
                target_type=target_type if target_type is not None else target.tgt_type,
                allow_lossy=target.lossy,
                scope=scope,
                backend=backend,
                annotation=annotation,
            )
        )


def estimate_serialized_size[T2: Serialized](
    instance: Any,
    target_type: TypeDef[T2],
    allow_lossy: bool = False,
    scope: TransportScope = Open(),
    annotation: TypeDef[Serializable[T2]] | None = None,
    backend: Backend = Memory(),
) -> int:
    """Estimate the byte size of serializing ``instance`` to ``target_type``.

    Serializers override this when they can measure their format cheaply. The
    default falls back to :func:`sys.getsizeof`, which measures the object
    in memory rather than the encoded representation.

    Args:
        instance: The object to be serialized.
        target_type: The target type or format to serialize to.
        allow_lossy: Whether to allow a lossy serialization.
        scope: The serialization scope to use for the serialization.
        annotation: The declared type of the object, if any.
        backend: The backend to serialize to.
        file_factory: An optional factory function to create new files.

    Returns:
        The estimated number of bytes.
    """
    candidates = get_serializers(
        typ=annotation if annotation is not None else type(instance),
        targets=[target_type] if target_type is not None else None,
        scope=scope,
        allow_lossy=allow_lossy,
        value=instance if annotation is None else None,
    )

    if len(candidates) == 0:
        raise ValueError(
            f"No suitable serializer found for {type(instance).__name__} to "
            f"{target_type!r} under scope {scope!r} "
            f"with allow_lossy={allow_lossy}."
        )

    _, target, serializer = candidates[0]

    with serialization_context(target, backend):
        return (
            serializer.__estimate_serialized_size__(
                instance,
                target_type=target_type,
                allow_lossy=target.lossy,
                scope=scope,
                annotation=annotation,
            )
            if isinstance(serializer, type)
            else serializer.estimate_serialized_size(
                instance,
                target_type=target_type,
                allow_lossy=target.lossy,
                scope=scope,
                annotation=annotation,
            )
        )


def register_serializer(ser: Serializer[Any, Any, Any]) -> None:
    """Register a custom serializer."""
    _serializers.append(ser)
