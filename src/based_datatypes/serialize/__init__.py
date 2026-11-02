"""Arbitrary object (de-)serialization for transport across various mediums."""

from collections.abc import Callable, Collection
from typing import Any, cast

from based_datatypes.files import File
from based_datatypes.types import TypeDef
from based_datatypes.typing import (
    TypeRef,
    is_subtype,
)

from .base import (
    JSONValue,
    Open,
    Realm,
    Serializable,
    SerializationTarget,
    Serialized,
    Serializer,
    TransportScope,
    current_ser_ctx,
    serialization_context,
)
from .common import common_serializers

__all__ = [
    "Open",
    "Realm",
    "TransportScope",
    "Serializable",
    "Serializer",
    "current_ser_ctx",
    "custom_serializers",
    "estimate_serialized_size",
    "serialize",
    "get_serializers",
    "deserialize",
    "register_serializer",
    "serialization_context",
]

custom_serializers: list[Serializer] = []


def get_serializers[U, T: JSONValue | File, S: TransportScope](
    typ: TypeDef[U],
    targets: Collection[TypeDef[T]] | None = None,
    scope: S = Open(),
    lossless: bool = False,
) -> list[tuple[int, type[T], Serializer[U, T, S] | type[Serializable[T, S]]]]:
    """Get the serializer for a given type or instance.

    Args:
        typ: The type or instance to get the serializer for.
        targets: The transport mediums to write to or read from. Instances are
            matched on their MIME type and IO type, so the driver for the actual
            format wins; types only narrow by class. ``None`` skips the medium
            check and returns the driver for the object type alone.
        scope: The transport scope to select for. ``None`` uses the ambient
            :data:`current_ser_purpose`. A driver declared for a subclass scope
            (e.g. WAN) also serves a request for its base scope (e.g. IPC).
        lossless: When ``True``, only drivers that round-trip the target format
            without loss are eligible; when ``False``, only lossy ones; ``None``
            accepts either.

    Returns:
        A list of tuples of the form ``(index, target, driver)``, where
        ``index`` is the index of the matching target in the list of targets
        ``target`` is the matching transport medium
        ``driver`` is the serializer or serializable class to handle the object.
    """
    candidates = []

    cls = TypeRef(typ).common_type

    if issubclass(cls, Serializable):
        candidates = sorted(
            [
                (i, t, cls)
                for i, t in enumerate(
                    SerializationTarget.match(
                        cls.__serialization_targets__(), targets, scope, lossless
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
            if is_subtype(typ, driver.obj_type)
            for i, t in enumerate(
                SerializationTarget.match(driver.serialization_targets, targets, scope, lossless)
            )
        ],
        key=lambda x: x[0],
    )  # Sort by index to prioritize the first match

    return candidates  # pyright: ignore[reportReturnType]


def deserialize[V2](
    source: Serialized,
    annotation: TypeDef[V2],
    scope: TransportScope = Open(),
    realm: Realm | None = None,
) -> V2:
    """Import an object from the source type or format.

    Args:
        source: The source object to import from.
        annotation: The declared type of the object.
        scope: The transport scope to use for the import.
        realm: The realm to import from.

    Returns:
        An instance of the object, imported from the source.
    """
    source_type = source.to_file_type() if isinstance(source, File) else type(source)

    candidates = get_serializers(
        annotation,
        targets=[source_type],
        scope=scope,
        lossless=True,
    )

    if len(candidates) == 0:
        raise ValueError(
            f"No suitable serializer found for {type(source)}"
            f"with scope {scope} and lossless={True}"
        )

    _, _, driver = candidates[0]

    if isinstance(driver, type) and issubclass(driver, Serializable):
        return cast(
            V2,
            driver.__deserialize__(
                source, scope=scope, realm=realm, annotation=cast(TypeDef[Serializable], annotation)
            ),
        )
    else:
        return cast(V2, driver.deserialize(source, scope=scope, realm=realm, annotation=annotation))


def serialize[T2: JSONValue | File](
    instance: Any,
    target_type: TypeDef[T2] | None = None,
    lossless: bool = True,
    scope: TransportScope = Open(),
    annotation: TypeDef[Serializable[T2]] | None = None,
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
    candidates = get_serializers(
        typ=annotation if annotation is not None else type(instance),
        targets=[target_type] if target_type is not None else None,
        scope=scope,
        lossless=lossless,
    )

    if len(candidates) == 0:
        raise ValueError(
            f"No suitable serializer found for {type(instance)}"
            f"with scope {scope} and lossless={lossless}"
        )

    _, _, driver = candidates[0]

    if isinstance(driver, type) and issubclass(driver, Serializable):
        return cast(
            T2,
            driver.__serialize__(
                instance,
                target_type=target_type,
                lossless=lossless,
                scope=scope,
                realm=realm,
                file_factory=file_factory,
                annotation=annotation,
            ),
        )
    else:
        return cast(
            T2,
            driver.serialize(
                instance,
                target_type=target_type,
                lossless=lossless,
                scope=scope,
                realm=realm,
                file_factory=file_factory,
                annotation=annotation,
            ),
        )


def estimate_serialized_size[T2: JSONValue | File](
    instance: Any,
    target_type: TypeDef[T2],
    lossless: bool = True,
    scope: TransportScope = Open(),
    realm: Realm | None = None,
    target: T2 | None = None,
    annotation: TypeDef[Serializable[T2]] | None = None,
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
        realm: The realm to export to.
        target: An optional pre-allocated target object to export into.
        annotation: The declared type of the object, if any.

    Returns:
        The estimated number of bytes.
    """
    candidates = get_serializers(
        typ=annotation if annotation is not None else type(instance),
        targets=[target_type],
        scope=scope,
        lossless=lossless,
    )

    if len(candidates) == 0:
        raise ValueError(
            f"No suitable serializer found for {type(instance)}"
            f"with scope {scope} and lossless={lossless}"
        )

    _, _, driver = candidates[0]

    if isinstance(driver, type) and issubclass(driver, Serializable):
        return driver.__estimate_serialized_size__(
            instance,
            target_type=target_type,
            lossless=lossless,
            scope=scope,
            annotation=annotation,
        )
    else:
        return driver.estimate_serialized_size(
            instance,
            target_type=target_type,
            lossless=lossless,
            scope=scope,
            annotation=annotation,
        )


def register_serializer(ser: Serializer) -> None:
    """Register a custom transport driver."""
    custom_serializers.append(ser)
