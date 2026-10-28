"""Arbitrary object (de-)serialization for transport across various mediums."""

from collections.abc import Set
from types import UnionType
from typing import Any, cast

from based_datatypes.typing import (
    SingleTypeDef,
    TypeRef,
    is_subtype,
    typedef_to_typeset,
)

from .base import Realm, Transportable, TransportDriver, TransportTypes
from .common import common_drivers

custom_drivers: dict[SingleTypeDef | UnionType, type[TransportDriver]] = {}


def get_transport_driver[U, T](
    instance: type[U] | U, targets: Set[SingleTypeDef[T]]
) -> type[TransportDriver[U, T]] | None:
    """Get the transporter for a given type or instance."""
    obj_type = instance if isinstance(instance, type) else type(instance)

    matching_sto = [
        c
        for t, c in (common_drivers | custom_drivers).items()
        if is_subtype(t, obj_type) and len(c.transport_types().match_targets(targets)) > 0
    ]

    if len(matching_sto) == 0:
        return None

    return matching_sto[0]


def get_transport_types(src_type: SingleTypeDef | UnionType) -> TransportTypes:
    """Get the transport types for a given type."""
    if is_subtype(src_type, Transportable):
        return TypeRef(
            src_type
        ).common_type.__transport_types__()  # pyright: ignore[reportAbstractUsage]

    driver = get_transport_driver(src_type, {object})
    if driver is not None:
        return driver.transport_types()

    raise TypeError(f"No transport types found for {src_type}")


def export_object(obj: Any, target: Any, realm: Realm) -> None:
    """Export an object to a target."""
    if isinstance(obj, Transportable):
        return obj.__export__(target, realm)

    driver = get_transport_driver(obj, {type(target)})
    if driver is None:
        raise TypeError(f"No driver found for exporting {type(obj)} to {target}")

    return driver.export_object(obj, target, realm)


def import_object[T](source: Any, realm: Realm, annotation: SingleTypeDef[T]) -> T:
    """Import an object from a source."""
    if is_subtype(annotation, Transportable):
        conv = [
            t
            for t in typedef_to_typeset(annotation)
            if issubclass(t, Transportable) and t.__transport_types__().match_targets({source})
        ]

        if len(conv) > 0:
            return cast(
                T,
                conv[0].__import__(source, realm, cast(SingleTypeDef, annotation)),
            )

    annotation = cast(SingleTypeDef[T], annotation)  # avoid pyright issue due to type narrowing.
    converter = get_transport_driver(source, {annotation})
    if converter is None:
        raise TypeError(f"No parser found for {type(source)} with annotation {annotation}")

    return converter.import_object(source, realm, annotation)


def register_transport_driver(
    type_: SingleTypeDef | UnionType, driver: type[TransportDriver]
) -> None:
    """Register a custom transport driver."""
    custom_drivers[type_] = driver
