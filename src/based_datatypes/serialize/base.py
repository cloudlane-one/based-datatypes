"""Base classes for the serialization of arbitrary types and data formats."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Collection, Generator, Iterable, Mapping, Set
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Generic,
    Literal,
    LiteralString,
    Protocol,
    Self,
    TypeVar,
    cast,
    override,
    runtime_checkable,
)
from uuid import UUID

from based_datatypes.caching.base import cached
from based_datatypes.descriptors import prop
from based_datatypes.files import E, File, MemoryFile, Write
from based_datatypes.ref import PyObjectRef
from based_datatypes.types import TypeDef
from based_datatypes.typing import Runtyped, TypeRef, is_subtype

type Scalar = (
    None
    | bool
    | int
    | float
    | Decimal
    | Fraction
    | complex
    | datetime
    | date
    | time
    | timedelta
    | UUID
    | Path
    | Enum
)

type PrimitiveLeaf[S: Scalar = Scalar] = S | str | bytes | bytearray | memoryview
type PrimitiveContainer[S: Scalar = Scalar] = Mapping[str, Primitive[S]] | Collection[Primitive[S]]
type UntaggedPrimitive[S: Scalar = Scalar] = PrimitiveLeaf[S] | PrimitiveContainer[S]
type Primitive[S: Scalar = Scalar] = UntaggedPrimitive[S] | Tagged[UntaggedPrimitive[S], Any]

type SerializedLeaf[S: Scalar = Scalar, M: LiteralString = Any] = (
    PrimitiveLeaf[S] | File[M, Any, Any]
)
type SerializedContainer[S: Scalar = Scalar, M: LiteralString = Any] = (
    Mapping[str, Serialized[S, M]] | Collection[Serialized[S, M]]
)
type UntaggedSerialized[S: Scalar = Scalar, M: LiteralString = Any] = (
    SerializedLeaf[S, M] | SerializedContainer[S, M]
)
type Serialized[S: Scalar = Scalar, M: LiteralString = Any] = (
    UntaggedSerialized[S, M] | Tagged[UntaggedSerialized[S, M], Any]
)


@dataclass(frozen=True)
class Tagged[T: Serialized[Any, Any] = Serialized, V = Any]:
    """A wrapper for a serialized value that carries a type tag."""

    value: T
    type_tag: type[V]

    @cached_property
    def tag_str(self) -> str:
        """Return the string representation of the type tag."""
        ref = PyObjectRef.reference(self.type_tag)
        return f"{ref.module_name}/{ref.qualname}"


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


class Backend[S: Scalar = Scalar](ABC):
    """A class representing the data backend into / from which serialization happens."""

    supported_scalars: Set[type[S]] = TypeRef(Scalar).typeset  # pyright: ignore[reportAssignmentType]

    @abstractmethod
    def get_writable_file[K: LiteralString | None](
        self, typedef: TypeDef[File[Any, K, Write]]
    ) -> File[Any, K, Write]:
        """Return a file to write to.

        Returns:
            A file object.
        """
        ...

    def store(self, data: Serialized[S]) -> None:
        """Store the serialized data in the backend.

        Args:
            data: The serialized data to store.
        """
        ...

    def load(self) -> Serialized[S]:
        """Load the serialized data from the backend.

        Returns:
            The serialized data.
        """
        ...


@dataclass
class Memory(Backend[Scalar]):
    """A backend that writes to in-memory files."""

    _data: Serialized | None = None

    @override
    def get_writable_file[K: LiteralString | None](
        self, typedef: TypeDef[File[Any, K, Write]]
    ) -> MemoryFile[Any, K, Write]:
        enc_type = TypeRef(typedef).typevar_map[E]

        return MemoryFile(
            mime=File.get_all_mimes(typedef)[0],
            encoding=cast(
                K, enc_type.literal_values.pop() if len(enc_type.literal_values) > 0 else None
            ),
            rw="w",
        )

    def store(self, data: Serialized) -> None:
        """Store the serialized data in the backend.

        Args:
            data: The serialized data to store.
        """
        self._data = data

    def load(self) -> Serialized:
        """Load the serialized data from the backend.

        Returns:
            The serialized data.
        """
        return self._data


V = TypeVar("V")
T = TypeVar("T", bound=Serialized)
TS = TypeVar("TS", bound=TransportScope, default=Open, contravariant=True)
L = TypeVar("L", bound=bool, default=Literal[False])


@dataclass(frozen=True)
class SerializationTarget(Generic[V, T, TS, L]):
    """A set of supported serialization formats."""

    obj_type: TypeDef[V]
    """The type of the object being serialized."""

    tgt_type: type[T]
    """The type of the target format."""

    scope: type[TS]
    """The scope this target format is declared for."""

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
        targets: Iterable[SerializationTarget[Any, Any, Any, Any]],
        types: Collection[TypeDef[Serialized]] | None = None,
        scope: TransportScope | None = None,
        allow_lossy: bool = False,
    ) -> list[SerializationTarget[V, T, TS, L]]:
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
            list[SerializationTarget[V, T, TS, L]],
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


current_ser_ctx: ContextVar[tuple[SerializationTarget, TransportScope, Backend] | None] = (
    ContextVar(
        "current_ser_ctx",
        default=None,
    )
)
"""The current serialization context: (serialization target, realm)."""


@contextmanager
def serialization_context(
    target: SerializationTarget[Any, Any, Any, Any],
    scope: TransportScope = Open(),
    backend: Backend = Memory(),
) -> Generator[None]:
    """Context manager to set the current serialization context."""
    token = current_ser_ctx.set((target, scope, backend))
    try:
        yield
    finally:
        current_ser_ctx.reset(token)


@runtime_checkable
class Serializable(Protocol[T, TS, L]):  # pyright: ignore[reportInvalidTypeVarUse]
    """Protocol for classes that can serialize and deserialize themselves."""

    @classmethod
    def __serialization_targets__(cls) -> list[SerializationTarget[Self, T, TS, L]]:
        """Serialization targets supported by this class."""
        ...

    @classmethod
    def __deserialize__(
        cls,
        source: Serialized,
        scope: TS = Open(),
        annotation: TypeDef[Self] | None = None,
    ) -> Self:
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
        self,
        target_type: TypeDef[T],
        allow_lossy: bool = False,
        scope: TS = Open(),
        annotation: TypeDef[Serializable[T, Any, Any]] | None = None,
        backend: Backend = Memory(),
    ) -> T:
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
        self,
        target_type: TypeDef[T],
        allow_lossy: bool = False,
        scope: TS = Open(),
        annotation: TypeDef[Serializable[T, Any, Any]] | None = None,
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


@dataclass
class Serializer(Runtyped, Generic[V, T, TS, L]):  # noqa: UP046
    """A class representing a serialization driver for a specific type."""

    @prop(cached=True, mode="class")
    @classmethod
    def obj_type(cls) -> TypeRef[V]:
        """Return the type of object this serializer handles."""
        return cls._typeref.typevar_map[V]

    @prop(cached=True, mode="class")
    @classmethod
    def target_type(cls) -> TypeRef[T]:
        """Return the target type this serializer handles."""
        return cls._typeref.typevar_map[T]

    @prop(cached=True, mode="class")
    @classmethod
    def scopes(cls) -> tuple[type[TS], ...]:
        """The scope classes this serializer declares through its ``S`` type argument.

        A scope value carries data (a process id, a subnet), so the class-level
        declaration is the type argument itself; selection compares it with the
        requested scope instead of materializing a value that would lose that data.
        """
        return (cast(type[TS], cls._typeref.typevar_map[TS].resolved_type_def),)

    @prop(cached=True, mode="class")
    @classmethod
    def lossless(cls) -> bool:
        """The losslessness, which this serializer declares through its ``L`` type argument."""
        literal_values = cls._typeref.typevar_map[L].literal_values
        return len(literal_values) == 1 and next(iter(literal_values)) is True

    @prop(cached=True, mode="class")
    @classmethod
    def serialization_targets(cls) -> list[SerializationTarget[V, T, TS, L]]:
        """Serialization targets supported by this serializer, as declared in its type arguments."""
        typeref = cls._typeref.typevar_map[T]
        # The declarations are the driver's: the requested scope of an operation
        # is applied to the target when it is dispatched, so this metadata only
        # states what the serializer itself can serve.
        declared = cls.scopes
        return cast(
            list[SerializationTarget[V, T, TS, L]],
            [
                SerializationTarget(
                    obj_type=cls.obj_type,
                    tgt_type=member,
                    scope=scope,
                    lossy=not cls.lossless,
                )
                for member in typeref.members
                if isinstance(member, type) and issubclass(member, File)
                for scope in declared
            ],
        )

    def __init_subclass__(cls, **kwargs) -> None:  # noqa: D105, N805
        if is_subtype(cls.obj_type.resolved_type_def, Serializable):
            raise TypeError(
                "Cannot register a serializer for a Serializable type: "
                "the class itself handles its serialization."
            )

    @abstractmethod
    def deserialize(
        self,
        source: T,
        annotation: TypeDef[V],
        scope: TS = Open(),
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
        scope: TS = Open(),
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
        scope: TS = Open(),
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


@cached(args_callback={"value": lambda v: id(v) if v is not None else None})
def get_serializers[V, T: Serialized, S: TransportScope, L: bool](
    typ: TypeDef[V],
    targets: Collection[TypeDef[T]] | None = None,
    scope: S = Open(),
    allow_lossy: L = False,
    value: object | None = None,
) -> list[
    tuple[
        int, SerializationTarget[V, T, S, L], Serializer[V, T, S, L] | type[Serializable[T, S, L]]
    ]
]:
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


def _get_defaults_from_ctx(
    scope: TransportScope | None = None,
    backend: Backend | None = None,
) -> tuple[TransportScope, Backend]:
    context = current_ser_ctx.get()
    if scope is None:
        scope = context[1] if context is not None else Open()
    if backend is None:
        backend = context[2] if context is not None else Memory()
    return scope, backend


def deserialize[V2](
    source: Serialized,
    annotation: TypeDef[V2],
    scope: TransportScope | None = None,
    backend: Backend | None = None,
) -> V2:
    """Deserialize an object from the source type or format.

    Args:
        source: The source object to deserialize from.
        annotation: The declared type of the object.
        scope: The serialization scope to use for the deserialization.
        backend:
            The backend to deserialize from. If ``None``, the current context's backend is used.

    Returns:
        An instance of the object, deserialized from the source.
    """
    scope, backend = _get_defaults_from_ctx(scope, backend)

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

    with serialization_context(target, scope, backend):
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


def serialize[T2: Serialized = Serialized](
    instance: Any,
    target_type: TypeDef[T2] | None = None,
    allow_lossy: bool = False,
    scope: TransportScope | None = None,
    annotation: TypeDef[Serializable[T2]] | None = None,
    backend: Backend | None = None,
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

    Returns:
        The serialized target object.
    """
    scope, backend = _get_defaults_from_ctx(scope, backend)

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

    with serialization_context(target, scope, backend):
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
    scope, backend = _get_defaults_from_ctx(scope, backend)

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

    with serialization_context(target, scope, backend):
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


C = TypeVar("C")


class Convertible(Runtyped, Serializable[T, TS, L], Generic[C, T, TS, L]):
    """Serialize via converting to another object type ``C``."""

    @prop(cached=True, mode="class")
    @classmethod
    def conversion_type(cls) -> TypeDef[C]:
        """Return the type of the intermediate object this converter uses."""
        return cls._typeref.typevar_map[C].resolved_type_def
    
    @prop(cached=True, mode="class")
    def conversion_chain(cls) -> tuple[*tuple[type[Convertible], ...], type]:
        

    @classmethod
    @abstractmethod
    def __revert__(
        cls,
        source: C,
        annotation: TypeDef[V],
    ) -> Self:
        """Revert a converted object back to the original type.

        Args:
            source: The converted object to revert.
            annotation: The declared type of the original object.

        Returns:
            The reverted original object.
        """
        ...

    @abstractmethod
    def __convert__(
        self,
        target_type: TypeDef[T] | None = None,
    ) -> C:
        """Convert an object to the target type or format.

        Args:
            target_type: The target type or format to convert to.

        Returns:
            The converted target object.
        """
        ...

    @classmethod
    def __serialization_targets__(cls) -> list[SerializationTarget[Self, T, TS, L]]:
        """Serialization targets supported by this class."""
        ...

    @classmethod
    def __deserialize__(
        cls,
        source: Serialized,
        scope: TS = Open(),
        annotation: TypeDef[Self] | None = None,
    ) -> Self:
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
        self,
        target_type: TypeDef[T],
        allow_lossy: bool = False,
        scope: TS = Open(),
        annotation: TypeDef[Serializable[T, Any, Any]] | None = None,
        backend: Backend = Memory(),
    ) -> T:
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
        self,
        target_type: TypeDef[T],
        allow_lossy: bool = False,
        scope: TS = Open(),
        annotation: TypeDef[Serializable[T, Any, Any]] | None = None,
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
