"""Arbitrary object (de-)serialization for transport across various mediums."""

from collections.abc import Collection
from types import GenericAlias, UnionType
from typing import Any, TypeAliasType, cast, get_origin

from typing_extensions import TypeForm

from based_datatypes.files import File
from based_datatypes.typing import (
    SingleTypeDef,
    TypeRef,
    is_subtype,
    typedef_to_typeset,
)

from .base import Realm, Transportable, TransportDriver, TransportTypes
from .common import common_drivers

custom_drivers: dict[SingleTypeDef | UnionType, type[TransportDriver]] = {}


def _covers_type(registered: SingleTypeDef | UnionType, obj_type: type) -> bool:
    """Check whether a registered driver key covers the given object type."""
    return any(issubclass(obj_type, t) for t in typedef_to_typeset(registered))


def _is_type_hint(obj: object) -> bool:
    """Check whether an object is a type hint rather than a value."""
    return (
        isinstance(obj, type | UnionType | TypeAliasType | GenericAlias)
        or get_origin(obj) is not None
    )


def get_transport_driver[U](
    instance: type[U] | U, targets: Collection[TypeForm | File] | None = None
) -> type[TransportDriver[U, Any]] | None:
    """Get the transporter for a given type or instance.

    Args:
        instance: The object, its type, or its type hint, to be imported or
            exported.
        targets: The transport mediums to write to or read from. Instances are
            matched on their MIME type and IO type, so the driver for the actual
            format wins; types only narrow by class. ``None`` skips the medium
            check and returns the driver for the object type alone.

    Returns:
        The first matching driver, or ``None`` if no driver applies.
    """
    obj_types = (
        typedef_to_typeset(cast(SingleTypeDef, instance))
        if _is_type_hint(instance)
        else {type(instance)}
    )

    candidates = [
        (t, c)
        for t, c in (common_drivers | custom_drivers).items()
        if any(_covers_type(t, obj_type) for obj_type in obj_types)
        and (targets is None or len(c.transport_types().match_targets(targets)) > 0)
    ]

    if len(candidates) == 0:
        return None

    # The ``object`` registration is the catch-all; anything more specific wins.
    candidates.sort(key=lambda item: item[0] is object)

    return cast(type[TransportDriver[U, Any]], candidates[0][1])


def get_transport_types(src_type: SingleTypeDef | UnionType) -> TransportTypes:
    """Get the transport types for a given type."""
    if is_subtype(src_type, Transportable):
        return TypeRef(
            src_type
        ).common_type.__transport_types__()  # pyright: ignore[reportAbstractUsage]

    driver = get_transport_driver(src_type)
    if driver is not None:
        return driver.transport_types()

    raise TypeError(f"No transport types found for {src_type}")


def export_object(obj: Any, target: Any, realm: Realm) -> None:
    """Export an object to a target."""
    if isinstance(obj, Transportable):
        return obj.__export__(target, realm)

    driver = get_transport_driver(obj, [target])
    if driver is None:
        raise TypeError(f"No driver found for exporting {type(obj)} to {target}")

    return driver.export_object(obj, target, realm)


def import_object[T](source: Any, realm: Realm, annotation: SingleTypeDef[T]) -> T:
    """Import an object from a source."""
    if is_subtype(annotation, Transportable):
        conv = [
            t
            for t in typedef_to_typeset(annotation)
            if issubclass(t, Transportable) and t.__transport_types__().match_targets([source])
        ]

        if len(conv) > 0:
            return cast(
                T,
                conv[0].__import__(source, realm, cast(SingleTypeDef, annotation)),
            )

    annotation = cast(SingleTypeDef[T], annotation)  # avoid pyright issue due to type narrowing.
    converter = get_transport_driver(annotation, [source])
    if converter is None:
        raise TypeError(f"No parser found for {type(source)} with annotation {annotation}")

    return cast(T, converter.import_object(source, realm, cast(Any, annotation)))


def register_transport_driver(
    type_: SingleTypeDef | UnionType, driver: type[TransportDriver]
) -> None:
    """Register a custom transport driver."""
    custom_drivers[type_] = driver
