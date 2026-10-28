"""Arbitrary object (de-)serialization for transport across various mediums."""

from collections.abc import Collection
from types import GenericAlias, UnionType
from typing import Any, TypeAliasType, Union, cast, get_args, get_origin

from typing_extensions import TypeForm

from based_datatypes.files import File
from based_datatypes.types import AnyTypeHint
from based_datatypes.typing import (
    SingleTypeDef,
    TypeRef,
    has_type,
    is_subtype,
    typedef_to_typeset,
)

from .base import (
    ALL_TRANSPORT_SCOPES,
    IPC,
    LAN,
    WAN,
    Open,
    Realm,
    Transportable,
    TransportDriver,
    TransportScope,
    TransportTypes,
    current_ser_purpose,
    transport_scope,
)
from .common import common_drivers

__all__ = [
    "ALL_TRANSPORT_SCOPES",
    "IPC",
    "LAN",
    "Open",
    "WAN",
    "Realm",
    "TransportScope",
    "TransportTypes",
    "Transportable",
    "TransportDriver",
    "current_ser_purpose",
    "custom_drivers",
    "estimate_bytes",
    "export_object",
    "get_transport_driver",
    "get_transport_types",
    "import_object",
    "register_transport_driver",
    "transport_scope",
]

custom_drivers: dict[SingleTypeDef | UnionType, type[TransportDriver]] = {}


def _is_type_hint(obj: object) -> bool:
    """Check whether an object is a type hint rather than a value."""
    return (
        isinstance(obj, type | UnionType | TypeAliasType | GenericAlias)
        or get_origin(obj) is not None
    )


def _alternatives(registered: SingleTypeDef | UnionType) -> tuple[object, ...]:
    """Return the union members of a driver key, or the key itself."""
    if isinstance(registered, UnionType) or get_origin(registered) is Union:
        return get_args(registered)

    return (registered,)


def _hint_covers(subject: object, registered: SingleTypeDef | UnionType) -> bool:
    """Check whether a driver key covers a type hint, keeping its arguments."""
    hint = cast(AnyTypeHint, subject)
    supertype = cast("SingleTypeDef[Any] | TypeForm[Any]", registered)

    if is_subtype(hint, supertype):
        return True

    # A bare ``dict`` is not a subtype of ``dict[str, ...]`` even though the
    # alias describes it, so resolve named aliases to their value's typeset.
    return isinstance(registered, TypeAliasType) and any(
        is_subtype(hint, cast("SingleTypeDef[Any] | TypeForm[Any]", member))
        for member in typedef_to_typeset(registered)
    )


def _value_matches(value: object, hint: object) -> bool:
    """Check one concrete value against one driver-key alternative.

    Container hints keep their element constraint: a list only matches when
    every item satisfies the declared element type.
    """
    origin = get_origin(hint)
    args = get_args(hint)

    if origin is None:
        if isinstance(hint, TypeAliasType):
            return any(_value_matches(value, member) for member in typedef_to_typeset(hint))
        if isinstance(hint, type):
            return isinstance(value, hint)
        return is_subtype(type(value), cast("SingleTypeDef[Any] | TypeForm[Any]", hint))

    if not isinstance(origin, type) or not isinstance(value, origin):
        return False

    if origin in (list, tuple) and args and args[0] is not Ellipsis:
        return all(has_type(item, args[0]) for item in cast(Collection[object], value))

    return True


def _covers(registered: SingleTypeDef | UnionType, subject: object) -> bool:
    """Check whether a registered driver key covers a value, type or type hint."""
    if _is_type_hint(subject):
        return _hint_covers(subject, registered)

    return any(_value_matches(subject, alternative) for alternative in _alternatives(registered))


def _lossless_for(driver: type[TransportDriver], target: object) -> bool:
    """Check whether a driver round-trips a single target without loss."""
    declared = driver.__lossless_formats__()

    if isinstance(target, File):
        return target.mime in declared

    origin = get_origin(target) or target
    if isinstance(origin, type) and issubclass(origin, File):
        args = get_args(target)
        if args:
            mimes = get_args(args[0]) or (args[0],)
            return all(mime in declared for mime in mimes)

    return len(declared) > 0


def _matches_lossless(
    driver: type[TransportDriver],
    targets: Collection[TypeForm | File] | None,
    wanted: bool,
) -> bool:
    """Check whether the driver is lossless for any of the requested targets.

    Losslessness is judged per target: a driver that is lossless for parquet may
    still be lossy for CSV, and a mixed target list matches either way.
    """
    if targets is None:
        return (len(driver.__lossless_formats__()) > 0) is wanted

    return any(_lossless_for(driver, target) is wanted for target in targets)


def _check_scope(
    supported: frozenset[type[TransportScope]], name: str, scope: TransportScope
) -> None:
    """Raise when ``scope`` is not one of the declared scopes."""
    if scope.matches(supported):
        return

    names = ", ".join(sorted(scope_type.__name__ for scope_type in supported))
    raise ValueError(f"{name} does not support transport scope {scope!r}; supported: {names}")


def get_transport_driver[U](
    instance: type[U] | U,
    targets: Collection[TypeForm | File] | None = None,
    *,
    scope: TransportScope | None = None,
    lossless: bool | None = None,
) -> type[TransportDriver[U, Any]] | None:
    """Get the transporter for a given type or instance.

    Args:
        instance: The object, its type, or its type hint, to be imported or
            exported.
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
        The first matching driver, or ``None`` if no driver applies.
    """
    effective_scope = scope if scope is not None else current_ser_purpose.get()

    candidates = [
        (t, c)
        for t, c in (common_drivers | custom_drivers).items()
        if _covers(t, instance)
        and effective_scope.matches(c.__transport_scopes__())
        and (targets is None or len(c.transport_types().match_targets(targets)) > 0)
        and (lossless is None or _matches_lossless(c, targets, lossless))
    ]

    if len(candidates) == 0:
        return None

    # The ``object`` registration is the catch-all; anything more specific wins.
    candidates.sort(key=lambda item: item[0] is object)

    return cast(type[TransportDriver[U, Any]], candidates[0][1])


def get_transport_types(src_type: SingleTypeDef | UnionType) -> TransportTypes:
    """Get the transport types for a given type."""
    if is_subtype(src_type, Transportable):
        common_type = TypeRef(src_type).common_type
        _check_scope(
            common_type.__transport_scopes__(),  # pyright: ignore[reportAbstractUsage]
            common_type.__name__,
            current_ser_purpose.get(),
        )
        return common_type.__transport_types__()  # pyright: ignore[reportAbstractUsage]

    driver = get_transport_driver(src_type)
    if driver is not None:
        return driver.transport_types()

    raise TypeError(
        f"No transport types found for {src_type} under transport scope "
        f"{current_ser_purpose.get()!r}"
    )


def export_object(obj: Any, target: Any, realm: Realm) -> None:
    """Export an object to a target."""
    scope = current_ser_purpose.get()

    if isinstance(obj, Transportable):
        _check_scope(obj.__transport_scopes__(), type(obj).__name__, scope)
        return obj.__export__(target, realm)

    driver = get_transport_driver(obj, [target], scope=scope)
    if driver is None:
        raise TypeError(
            f"No driver found for exporting {type(obj)} to {target} under transport scope {scope!r}"
        )

    return driver.export_object(obj, target, realm)


def import_object[T](source: Any, realm: Realm, annotation: SingleTypeDef[T]) -> T:
    """Import an object from a source."""
    scope = current_ser_purpose.get()

    if is_subtype(annotation, Transportable):
        transportables = [t for t in typedef_to_typeset(annotation) if issubclass(t, Transportable)]
        convertible = [t for t in transportables if t.__transport_types__().match_targets([source])]
        allowed = [
            candidate
            for candidate in convertible
            if scope.matches(candidate.__transport_scopes__())
        ]

        if allowed:
            return cast(
                T,
                allowed[0].__import__(source, realm, cast(SingleTypeDef, annotation)),
            )

        if convertible:
            # The source format is supported, only the scope is not.
            _check_scope(convertible[0].__transport_scopes__(), convertible[0].__name__, scope)

    annotation = cast(SingleTypeDef[T], annotation)  # avoid pyright issue due to type narrowing.
    converter = get_transport_driver(annotation, [source], scope=scope)
    if converter is None:
        raise TypeError(
            f"No parser found for {type(source)} with annotation {annotation} under "
            f"transport scope {scope!r}"
        )

    return cast(T, converter.import_object(source, realm, cast(Any, annotation)))


def estimate_bytes(obj: Any, target: Any, realm: Realm) -> int:
    """Estimate how many bytes exporting ``obj`` to ``target`` would produce.

    The estimate is format-aware but never writes to ``target``: drivers measure
    their own encoding, sampling where a full pass would be too expensive, and
    fall back to :func:`sys.getsizeof` when the format cannot be measured
    cheaply.

    Args:
        obj: The object that would be exported.
        target: The medium the object would be written to.
        realm: The realm the export would happen in.

    Returns:
        The estimated number of bytes.
    """
    scope = current_ser_purpose.get()

    if isinstance(obj, Transportable):
        _check_scope(obj.__transport_scopes__(), type(obj).__name__, scope)
        return obj.__estimate_bytes__(target, realm)

    driver = get_transport_driver(obj, [target], scope=scope)
    if driver is None:
        raise TypeError(
            f"No driver found for estimating {type(obj)} to {target} under transport "
            f"scope {scope!r}"
        )

    return driver.estimate_bytes(obj, target, realm)


def register_transport_driver(
    type_: SingleTypeDef | UnionType, driver: type[TransportDriver]
) -> None:
    """Register a custom transport driver."""
    custom_drivers[type_] = driver
