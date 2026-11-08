"""Arbitrary object serialization to files and JSON-supported values."""

from collections.abc import Callable, Collection
from dataclasses import replace
from typing import Any, cast

from based_datatypes.files import File
from based_datatypes.types import TypeDef
from based_datatypes.typing import (
    TypeRef,
    is_subtype,
)

from .base import (
    Open,
    Realm,
    Serializable,
    SerializationTarget,
    Serialized,
    SerializedDict,
    SerializedValue,
    Serializer,
    TransportScope,
    UnsupportedTargetError,
    current_ser_ctx,
    serialization_context,
)
from .common import common_serializers
from .local_fs import INDEX_FILENAME, read_structure, write_structure

__all__ = [
    "INDEX_FILENAME",
    "Open",
    "Realm",
    "SerializationTarget",
    "Serialized",
    "SerializedDict",
    "SerializedValue",
    "TransportScope",
    "Serializable",
    "Serializer",
    "UnsupportedTargetError",
    "current_ser_ctx",
    "custom_serializers",
    "estimate_serialized_size",
    "serialize",
    "get_serializers",
    "deserialize",
    "register_serializer",
    "read_structure",
    "serialization_context",
    "write_structure",
]

custom_serializers: list[Serializer[Any, Any, Any]] = []


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
            for driver in custom_serializers + common_serializers
            if _handles(driver, typ, value)
            for i, t in enumerate(
                SerializationTarget.match(driver.serialization_targets, targets, scope, allow_lossy)
            )
        ],
        key=lambda x: x[0],
    )  # Sort by index to prioritize the first match

    return candidates  # pyright: ignore[reportReturnType]


def _handles(driver: Serializer[Any, Any, Any], typ: TypeDef[Any], value: object | None) -> bool:
    """Check whether a driver handles a type, and the given value when there is one.

    Args:
        driver: The serializer to check.
        typ: The type of the object to serialize.
        value: The object itself, or ``None`` to match on the type alone.

    Returns:
        ``True`` if the driver declares the type and, when a value is given,
        accepts that value.
    """
    if not is_subtype(typ, driver.obj_type):
        return False

    return value is None or TypeRef(driver.obj_type).validate(value)


def _operation_target(
    target: SerializationTarget[Any, Any, Any], scope: TransportScope
) -> SerializationTarget[Any, Any, Any]:
    """Return the target of one dispatch, carrying the requested scope.

    The targets a serializer declares describe what it can serve; the scope of
    the operation is the caller's, so it replaces the declaration before the
    context is set and the driver is called.

    Args:
        target: The matching target from selection.
        scope: The scope the operation requests.

    Returns:
        The same target with the requested scope.
    """
    return replace(target, scope=scope)


def _dispatch[T](
    candidates: Collection[tuple[int, SerializationTarget[Any, Any, Any], Any]],
    call: Callable[[Any, SerializationTarget[Any, Any, Any]], T],
    scope: TransportScope,
    realm: Realm | None,
) -> T:
    """Run the first candidate that can fulfil the requested target.

    A driver raises :class:`UnsupportedTargetError` when the requested format cannot
    carry what it produced — a JSON file cannot hold a file written for a nested
    item, for instance. That is a selection miss rather than a failure of the
    request, so the next matching serializer is tried; the last refusal is
    raised when none of them can fulfil the target.

    Args:
        candidates: The matching ``(index, target, driver)`` triples.
        call: Invokes one driver with its operation target.
        scope: The scope the operation requests.
        realm: The realm of the operation.

    Returns:
        Whatever the first successful candidate returns.

    Raises:
        UnsupportedTargetError: If every candidate refuses the requested target.
    """
    failure: UnsupportedTargetError | None = None

    for _, target, driver in candidates:
        operation = _operation_target(target, scope)

        with serialization_context(target=operation, realm=realm):
            try:
                return call(driver, operation)
            except UnsupportedTargetError as error:
                failure = error

    assert failure is not None
    raise failure


def _source_type(source: Serialized) -> TypeDef[Any]:
    """Return the type a source value is matched against.

    A file is matched by its file type. A bare container carries no element
    types, so it is widened to its content: any list can serve a target that
    declares a list of a concrete item type, because the target decides what the
    items are validated as.

    Args:
        source: The source value to describe.

    Returns:
        The type of the source value.
    """
    if isinstance(source, File):
        return source.to_file_type()

    if isinstance(source, dict):
        return dict[str, Any]

    if isinstance(source, list):
        return list[Any]

    if isinstance(source, tuple):
        return tuple[Any, ...]

    if isinstance(source, set):
        return set[Any]

    if isinstance(source, frozenset):
        return frozenset[Any]

    return type(source)


def deserialize[V2](
    source: Serialized,
    annotation: TypeDef[V2],
    scope: TransportScope = Open(),
    realm: Realm | None = None,
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
        targets=[_source_type(source)],
        scope=scope,
        allow_lossy=True,
    )

    if len(candidates) == 0:
        raise ValueError(
            f"No suitable serializer found for {type(source).__name__} as "
            f"{annotation!r} under scope {scope!r}."
        )

    _, _, driver = candidates[0]

    if isinstance(driver, type) and issubclass(driver, Serializable):
        return cast(
            V2,
            driver.__deserialize__(
                source,
                scope=scope,
                realm=realm,
                annotation=cast(TypeDef[Serializable[Any, Any]], annotation),
            ),
        )
    else:
        return cast(V2, driver.deserialize(source, scope=scope, realm=realm, annotation=annotation))


def serialize[T2: Serialized](
    instance: Any,
    target_type: TypeDef[T2] | None = None,
    allow_lossy: bool = False,
    scope: TransportScope = Open(),
    annotation: TypeDef[Serializable[T2]] | None = None,
    realm: Realm | None = None,
    file_factory: Callable[[], File] | None = None,
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
        realm: The realm to serialize to.
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

    return cast(
        T2,
        _dispatch(
            candidates,
            lambda serializer, target: (
                serializer.__serialize__(
                    instance,
                    target_type=target_type if target_type is not None else target.type,
                    allow_lossy=target.lossy,
                    scope=scope,
                    realm=realm,
                    file_factory=file_factory,
                    annotation=annotation,
                )
                if isinstance(serializer, type)
                else serializer.serialize(
                    instance,
                    target_type=target_type if target_type is not None else target.type,
                    allow_lossy=target.lossy,
                    scope=scope,
                    realm=realm,
                    file_factory=file_factory,
                    annotation=annotation,
                )
            ),
            scope,
            realm,
        ),
    )


def estimate_serialized_size[T2: Serialized](
    instance: Any,
    target_type: TypeDef[T2],
    allow_lossy: bool = False,
    scope: TransportScope = Open(),
    annotation: TypeDef[Serializable[T2]] | None = None,
    realm: Realm | None = None,
    file_factory: Callable[[], File] | None = None,
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
        realm: The realm to serialize to.
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

    return _dispatch(
        candidates,
        lambda driver, target: (
            driver.__estimate_serialized_size__(
                instance,
                target_type=target_type,
                allow_lossy=target.lossy,
                scope=scope,
                annotation=annotation,
            )
            if isinstance(driver, type)
            else driver.estimate_serialized_size(
                instance,
                target_type=target_type,
                allow_lossy=target.lossy,
                scope=scope,
                annotation=annotation,
            )
        ),
        scope,
        realm,
    )


def register_serializer(ser: Serializer[Any, Any, Any]) -> None:
    """Register a custom serializer."""
    custom_serializers.append(ser)
