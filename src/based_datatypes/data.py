"""Tools for working with dataclasses."""

from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import MISSING, Field, dataclass, fields
from typing import (
    Any,
    ClassVar,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
    get_origin,
    runtime_checkable,
)

from pydantic import BaseModel

from .types import Attr


@runtime_checkable
class DataclassInstance(Protocol):
    """Protocol for dataclass instances."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


Params = ParamSpec("Params")
DC = TypeVar("DC", bound=DataclassInstance)


@dataclass
class MaskedInit[**P, T]:
    """Mask a type constructor with a compatible type."""

    mask: Callable[P, T]
    constructor: type

    def __call__(self, *args: P.args, **kwds: P.kwargs) -> T:  # noqa: D102
        return self.constructor(*args, **kwds)


def copy_and_override[**Params, DC: DataclassInstance](
    _init: Callable[Params, DC],
    _obj: DataclassInstance | tuple[DataclassInstance, ...],
    *args: Params.args,
    **kwargs: Params.kwargs,
) -> DC:
    """Re-construct a dataclass instance with all its init params + override.

    Args:
        _init: The dataclass constructor or a `MaskedInit` wrapping it.
        _obj: The dataclass instance(s) to copy values from.
        *args: Positional arguments to override.
        **kwargs: Keyword arguments to override.

    Returns:
        A new dataclass instance with the specified overrides.

    Note:
        Required attributes of the target dataclass type must always be provided as args / kwargs.

    Warning:
        Does not work for kw_only dataclasses and InitVars (yet).
    """
    if isinstance(_init, MaskedInit):
        _init = _init.constructor
    if not isinstance(_init, type):
        orig_init = get_origin(_init)
        assert orig_init is not None
        _init = orig_init

    constr_func = _init or type(_obj)
    target_fields = list(fields(cast(type, _init)))

    objs = _obj if isinstance(_obj, tuple) else (_obj,)

    obj_fields: dict[Field, Any] = {}
    for obj in objs:
        src_fields = set(f.name for f in obj.__dataclass_fields__.values())
        all_obj_fields = {f: getattr(obj, f.name) for f in target_fields if f.name in src_fields}
        obj_fields = {
            f: (
                v
                if v is not MISSING
                else obj_fields.get(
                    f,
                    (
                        f.default
                        if f.default is not MISSING
                        else (f.default_factory() if f.default_factory is not MISSING else MISSING)
                    ),
                )
            )
            for f, v in all_obj_fields.items()
        }

    assert not any(v is MISSING for v in obj_fields.values())

    obj_kwargs = list({f.name: v for f, v in obj_fields.items() if f.init}.items())
    new_kwargs = {
        **dict(obj_kwargs[len(args) :] if len(args) < len(obj_kwargs) else {}),
        **kwargs,
    }

    new_obj = constr_func(*args, **new_kwargs)  # pyright: ignore[reportCallIssue]

    for f, v in obj_fields.items():
        if not f.init:
            setattr(new_obj, f.name, v)

    return new_obj


def container_to_dict(
    container: Sequence | Set | Mapping | DataclassInstance | BaseModel,
    recursive: bool = True,
    _registry: dict[int, dict[Any, Any]] | None = None,
) -> dict[Any, Any] | None:
    """Turn an arbitrary container object into a dict.

    Args:
        container: The container to convert.
        recursive: Whether to recursively convert nested containers.
        _registry: Internal registry to track visited objects for cycle detection.

    Returns:
        A dict representation of the container, or None if not convertible.
    """
    _registry = _registry or {}

    obj_id = id(container)
    if obj_id in _registry:
        return _registry[obj_id]

    match container:
        case DataclassInstance():
            res = {
                Attr(f_name): getattr(container, f_name)
                for f_name in container.__dataclass_fields__.keys()
            }
        case _ if isinstance(container, BaseModel):
            res = {Attr(k): v for k, v in container.model_dump().items()}
        case Mapping():
            res = dict(container)
        case Iterable():
            res = {i: v for i, v in enumerate(container)}
        case _:
            return None

    _registry[obj_id] = res

    if recursive:
        for k, v in list(res.items()):
            m = container_to_dict(v, True, _registry)
            res[k] = m if m is not None else v

    return res


def flatten_dict(
    d: Mapping[Any, Any],
    path: tuple = (),
    _registry: dict[int, dict[tuple, Any]] | None = None,
) -> dict[tuple, Any]:
    """Turn a nested mapping into a flat dict (ignoring recursive references).

    Args:
        d: The mapping to flatten.
        path: The current path of keys (used for recursion).
        _registry: Internal registry to track visited objects for cycle detection.

    Returns:
        A flat dict mapping tuple paths to values.
    """
    items: dict[tuple, Any] = {}

    _registry = _registry or {}

    obj_id = id(d)
    if obj_id in _registry:
        return _registry[obj_id]

    _registry[obj_id] = items

    for k, v in d.items():
        new_path = path + (k,)
        if isinstance(v, Mapping):
            items.update(flatten_dict(v, new_path, _registry))
        else:
            items[new_path] = v

    return items
