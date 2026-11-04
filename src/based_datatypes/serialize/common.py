"""Common serializers for common types."""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
import pickle
import uuid
import xml.etree.ElementTree as xml  # noqa: N813
from abc import abstractmethod
from collections.abc import Callable, Sequence
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Literal, TextIO, TypeVar, cast, get_origin, override

import polars as pl
import sqlparse
import yaml

from based_datatypes.files import File, MemoryFile
from based_datatypes.types import TypeDef
from based_datatypes.typing import TypeRef, is_subtype

from .base import (
    JSONDict,
    JSONScalar,
    Open,
    Realm,
    Serialized,
    SerializedDict,
    SerializedValue,
    Serializer,
    TransportScope,
    target_mimes,
)

common_serializers: list[Serializer] = []
"""The serializers registered for the types this library supports out of the box."""

V = TypeVar("V")
T = TypeVar("T", bound=Serialized)


def check_type_and_cast[T](
    obj: Any, annotation: TypeDef[Any] | None = None
) -> T:  # pyright: ignore[reportInvalidTypeVarUse]
    """Check if the object matches the annotation type and cast it."""
    if annotation is not None and not TypeRef(annotation).validate(obj):
        raise TypeError(f"Object does not match expected type: {annotation}")
    return cast(T, obj)


def _encoded_size(text: str) -> int:
    """Return the UTF-8 encoded size of ``text``."""
    return len(text.encode("utf-8"))


def _serialize_json(instance: Any, mime: str) -> str:
    """Serialize JSON-compatible data for the given MIME type.

    The payload is built before the target is opened, so a value the encoder
    rejects cannot truncate an existing file.

    Args:
        instance: The data to serialize.
        mime: The target MIME type, ``application/yaml`` or JSON otherwise.

    Returns:
        The serialized text.
    """
    if mime == "application/yaml":
        return yaml.safe_dump(instance, allow_unicode=True)

    return json.dumps(instance, indent=2)


def _sample_step(total: int, limit: int = 100) -> int:
    """Return a stride that yields at most ``limit`` evenly spread rows.

    Sampling by stride rather than by head avoids a systematic bias on sorted
    data, where the first rows are unrepresentatively narrow.

    Args:
        total: The number of rows in the frame.
        limit: The maximum number of rows to sample.

    Returns:
        The stride, at least ``1``.
    """
    return max(total // limit, 1)


def _json_estimate(instance: Any, mime: str) -> int:
    """Measure a JSON or YAML payload cheaply.

    ``json.dumps`` is implemented in C and beats any Python size walk of the same
    data, so the JSON size is measured directly instead of estimated. YAML
    dumping is pure Python and about two orders of magnitude slower, while its
    output stays close to the compact JSON encoding, so a YAML target is measured
    as that compact JSON payload — a rough upper bound at C speed.

    Args:
        instance: The data to measure.
        mime: The target MIME type, ``application/yaml`` or JSON otherwise.

    Returns:
        The estimated encoded size in bytes.
    """
    if mime == "application/yaml":
        return _encoded_size(json.dumps(instance))

    return _encoded_size(json.dumps(instance, indent=2))


# Inspecting targets


def _target_io(hint: Any) -> TypeDef[Any]:
    """Return the IO type a file target declares, defaulting to text."""
    if hint is not None:
        ref = hint if isinstance(hint, TypeRef) else TypeRef(hint)

        for var, value in ref.scalar_typevar_map.items():
            if var.__name__ == "I":
                return value.resolved_type_def

    return TextIO


def _is_file_hint(hint: Any) -> bool:
    """Whether a target type asks for a file."""
    if hint is None:
        return False

    ref = hint if isinstance(hint, TypeRef) else TypeRef(hint)
    origin = ref.origin

    return isinstance(origin, type) and issubclass(origin, File)


def _memory_file(mime: str = "text/plain", io: TypeDef[Any] = TextIO) -> File:
    """Create an in-memory file with the given MIME type and IO type."""
    binary = not is_subtype(io, TextIO)
    file_type = cast(Any, MemoryFile)[Literal[cast(Any, mime)], BinaryIO if binary else TextIO]

    return cast(File, file_type())


# Reading and writing payloads


def _read_text(source: SerializedValue) -> str:
    """Read text from a file, or pass text through unchanged."""
    if isinstance(source, File):
        with source.open("r") as stream:
            return cast(str, stream.read())

    if isinstance(source, str):
        return source

    raise TypeError(f"Expected a text file or a string, got {type(source).__name__}.")


def _write_text(target: File, text: str) -> None:
    """Write text into a file target."""
    with target.open("w") as stream:
        stream.write(text)


def _read_value(source: SerializedValue) -> SerializedValue:
    """Read JSON-shaped data from a file, or pass a value through unchanged."""
    if not isinstance(source, File):
        return source

    with source.open("r") as stream:
        if source.default_mime == "application/yaml":
            # ``safe_load`` refuses to construct arbitrary Python objects.
            return cast(SerializedValue, yaml.safe_load(stream))

        return cast(SerializedValue, json.load(stream))


def _require_file(source: SerializedValue, serializer: str) -> File:
    """Return the source as a file, or refuse a source that is not one."""
    if isinstance(source, File):
        return source

    raise TypeError(f"{serializer} deserializes from a file, got {type(source).__name__}.")


# Base classes for the common serializers


class CommonSerializer(Serializer[V, T]):
    """Shared target handling for the common serializers.

    A serializer declares its formats through the type parameters of its class: a
    :class:`File` hint is written to, and a JSON-supported value (``str``,
    ``list[float]``, a ``dict``, ...) is returned as it is. Without a requested
    target type the first declared format is used, and a file format is delivered
    as an in-memory :class:`MemoryFile`.
    """

    def _default_target(self) -> File | None:
        """Return an in-memory file of the preferred file format, if there is one."""
        for target in self.serialization_targets:
            mimes = target_mimes(target.type)
            if len(mimes) > 0:
                return _memory_file(mimes[0], _target_io(target.type))

        return None

    def _resolve_target(
        self, target_type: TypeDef[Any] | None, file_factory: Callable[[], File] | None
    ) -> File | None:
        """Resolve the file a serialization writes to, if the target is a file.

        Args:
            target_type: The requested target type, or ``None`` for the default.
            file_factory: An optional factory that creates the file to write to.

        Returns:
            The file to write to, or ``None`` when the target is not a file and the
            serialized value is returned directly.

        Raises:
            ValueError: If the target is a file that cannot be created here.
        """
        if file_factory is not None:
            return file_factory()

        if target_type is None:
            return self._default_target()

        if not _is_file_hint(target_type):
            return None

        origin = TypeRef(target_type).origin

        if isinstance(origin, type) and origin is not File and not issubclass(origin, MemoryFile):
            raise ValueError(
                f"{type(self).__name__} cannot create a {origin.__name__} target; pass a "
                "file_factory that returns the file to write to."
            )

        mimes = target_mimes(target_type)

        if len(mimes) == 0:
            default = self._default_target()
            if default is None:
                raise ValueError(f"{type(self).__name__} has no file format to serialize to.")
            return default

        return _memory_file(mimes[0], _target_io(target_type))

    def _file_target(
        self, target_type: TypeDef[Any] | None, file_factory: Callable[[], File] | None
    ) -> File:
        """Resolve the file a serialization writes to, or refuse a non-file target."""
        target = self._resolve_target(target_type, file_factory)

        if target is None:
            raise ValueError(
                f"{type(self).__name__} can only serialize to a file; "
                f"{target_type!r} is not a file target."
            )

        return target


class PrimitiveSerializer(CommonSerializer[V, T]):
    """Shared handling for serializers that map onto a JSON-supported value."""

    @abstractmethod
    def _encode(self, instance: V) -> T:
        """Convert the object into its primitive representation."""

    @abstractmethod
    def _decode(self, source: T, annotation: TypeDef[Any] | None = None) -> V:
        """Restore the object from its primitive representation."""

    @override
    def serialize(
        self,
        instance: V,
        target_type: TypeDef[Any] | None = None,
        lossless: bool = True,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
        realm: Realm | None = None,
        file_factory: Callable[[], File] | None = None,
    ) -> T:
        """Serialize an object to a JSON-supported value.

        Args:
            instance: The object to serialize.
            target_type: The target type or format to serialize to.
            lossless: Whether to require a lossless serialization.
            scope: The serialization scope to use for the serialization.
            annotation: The declared type of the object, if any.
            realm: The realm to serialize to.
            file_factory: Unused; these serializers never write files.

        Returns:
            The serialized value.

        Raises:
            ValueError: If a file target is requested.
        """
        if file_factory is not None or _is_file_hint(target_type):
            raise ValueError(
                f"{type(self).__name__} serializes to a JSON value, not to a file; embed "
                "the value in a dict or list to write it to a file."
            )

        return self._encode(instance)

    @override
    def deserialize(
        self,
        source: T,
        scope: TransportScope = Open(),
        realm: Realm | None = None,
        annotation: TypeDef[Any] | None = None,
    ) -> V:
        """Deserialize an object from a JSON-supported value.

        Args:
            source: The value to deserialize from.
            scope: The serialization scope to use for the deserialization.
            realm: The realm to deserialize from.
            annotation: The declared type of the object, if any.

        Returns:
            An instance of the object, deserialized from the value.

        Raises:
            TypeError: If the source is a file.
        """
        if isinstance(source, File):
            raise TypeError(
                f"{type(self).__name__} deserializes from a JSON value, "
                f"got a {type(source).__name__}."
            )

        return check_type_and_cast(self._decode(source, annotation), annotation)

    @override
    def estimate_serialized_size(
        self,
        instance: V,
        target_type: TypeDef[Any] | None = None,
        lossless: bool = True,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
    ) -> int:
        """Measure the encoded value."""
        return _encoded_size(json.dumps(self._encode(instance)))


# JSON / YAML

type JSONFile = (
    File[Literal["application/json"], TextIO] | File[Literal["application/yaml"], TextIO]
)


class JSONSerializer(CommonSerializer[JSONDict, SerializedDict | JSONFile]):
    """Serializer for JSON-shaped dictionaries."""

    __lossless_mimes__ = frozenset({"application/json", "application/yaml"})

    @override
    def serialize(
        self,
        instance: JSONDict,
        target_type: TypeDef[Any] | None = None,
        lossless: bool = True,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
        realm: Realm | None = None,
        file_factory: Callable[[], File] | None = None,
    ) -> SerializedDict | JSONFile:
        """Serialize a dictionary to a JSON or YAML file, or return it as it is.

        Args:
            instance: The dictionary to serialize.
            target_type: The target type or format to serialize to.
            lossless: Whether to require a lossless serialization.
            scope: The serialization scope to use for the serialization.
            annotation: The declared type of the object, if any.
            realm: The realm to serialize to.
            file_factory: An optional factory function to create new files.

        Returns:
            The written file, or the dictionary when a dict target was requested.
        """
        target = self._resolve_target(target_type, file_factory)

        if target is None:
            return cast(SerializedDict, instance)

        _write_text(target, _serialize_json(instance, target.default_mime))

        return target

    @override
    def deserialize(
        self,
        source: SerializedDict | JSONFile,
        scope: TransportScope = Open(),
        realm: Realm | None = None,
        annotation: TypeDef[Any] | None = None,
    ) -> JSONDict:
        """Deserialize a dictionary from a JSON or YAML file, or from a dictionary.

        Args:
            source: The file or dictionary to deserialize from.
            scope: The serialization scope to use for the deserialization.
            realm: The realm to deserialize from.
            annotation: The declared type of the object, if any.

        Returns:
            The deserialized dictionary.
        """
        return check_type_and_cast(_read_value(source), annotation)

    @override
    def estimate_serialized_size(
        self,
        instance: JSONDict,
        target_type: TypeDef[Any] | None = None,
        lossless: bool = True,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
    ) -> int:
        """Measure the encoded dictionary.

        The JSON encoder is C, so measuring costs less than any Python size walk
        of the same data; a YAML target is measured from the compact JSON payload
        instead, because the YAML dumper is pure Python and far slower.
        """
        mimes = target_mimes(target_type)

        return _json_estimate(instance, mimes[0] if len(mimes) > 0 else "application/json")


# XML

type XMLFile = File[Literal["application/xml"], TextIO] | File[Literal["text/xml"], TextIO]


def _serialize_xml(instance: xml.ElementTree) -> str:
    """Serialize a tree, keeping comments and processing instructions intact."""
    root = instance.getroot()
    if root is None:
        return ""

    return xml.tostring(root, encoding="unicode")


class XMLSerializer(CommonSerializer[xml.ElementTree, str | XMLFile]):
    """Serializer for XML element trees."""

    __lossless_mimes__ = frozenset({"application/xml", "text/xml"})

    @override
    def serialize(
        self,
        instance: xml.ElementTree,
        target_type: TypeDef[Any] | None = None,
        lossless: bool = True,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
        realm: Realm | None = None,
        file_factory: Callable[[], File] | None = None,
    ) -> str | XMLFile:
        """Serialize a tree to XML text or to an XML file.

        Args:
            instance: The tree to serialize.
            target_type: The target type or format to serialize to.
            lossless: Whether to require a lossless serialization.
            scope: The serialization scope to use for the serialization.
            annotation: The declared type of the object, if any.
            realm: The realm to serialize to.
            file_factory: An optional factory function to create new files.

        Returns:
            The written file, or the XML text when a text target was requested.
        """
        text = _serialize_xml(instance)
        target = self._resolve_target(target_type, file_factory)

        if target is None:
            return text

        _write_text(target, text)

        return target

    @override
    def deserialize(
        self,
        source: str | XMLFile,
        scope: TransportScope = Open(),
        realm: Realm | None = None,
        annotation: TypeDef[Any] | None = None,
    ) -> xml.ElementTree:
        """Deserialize a tree from XML text or from an XML file.

        Args:
            source: The file or text to deserialize from.
            scope: The serialization scope to use for the deserialization.
            realm: The realm to deserialize from.
            annotation: The declared type of the object, if any.

        Returns:
            The deserialized tree.
        """
        # The default builder drops comments and processing instructions, which
        # would break the lossless round trip this serializer declares.
        parser = xml.XMLParser(target=xml.TreeBuilder(insert_comments=True, insert_pis=True))
        tree = xml.parse(io.StringIO(_read_text(source)), parser=parser)

        return check_type_and_cast(tree, annotation)

    @override
    def estimate_serialized_size(
        self,
        instance: xml.ElementTree,
        target_type: TypeDef[Any] | None = None,
        lossless: bool = True,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
    ) -> int:
        """Measure the serialized tree without writing it.

        ``xml.tostring`` is a C-level pass over the tree, so it is cheaper than
        walking the elements in Python — which would also under-count the markup
        the serializer emits.
        """
        return _encoded_size(_serialize_xml(instance))


# Pydantic models
try:
    import pydantic as pyd

    class PydSerializer(CommonSerializer[pyd.BaseModel, SerializedDict | JSONFile]):
        """Serializer for Pydantic models."""

        @override
        def serialize(
            self,
            instance: pyd.BaseModel,
            target_type: TypeDef[Any] | None = None,
            lossless: bool = True,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
            realm: Realm | None = None,
            file_factory: Callable[[], File] | None = None,
        ) -> SerializedDict | JSONFile:
            """Serialize a model to a JSON or YAML file, or return its dump.

            ``mode="json"`` converts dates, UUIDs and decimals to JSON-native
            values, so a value the encoder rejects cannot truncate an existing
            file and YAML's safe dumper accepts it. Aliases and excluded fields
            do not survive the dump, which is why the format is not lossless.

            Args:
                instance: The model to serialize.
                target_type: The target type or format to serialize to.
                lossless: Whether to require a lossless serialization.
                scope: The serialization scope to use for the serialization.
                annotation: The declared type of the object, if any.
                realm: The realm to serialize to.
                file_factory: An optional factory function to create new files.

            Returns:
                The written file, or the model dump when a dict target was requested.
            """
            dump = cast(SerializedDict, instance.model_dump(mode="json"))
            target = self._resolve_target(target_type, file_factory)

            if target is None:
                return dump

            _write_text(target, _serialize_json(dump, target.default_mime))

            return target

        @override
        def deserialize(
            self,
            source: SerializedDict | JSONFile,
            scope: TransportScope = Open(),
            realm: Realm | None = None,
            annotation: TypeDef[Any] | None = None,
        ) -> pyd.BaseModel:
            """Deserialize a model from a JSON or YAML file, or from its dump.

            Args:
                source: The file or dictionary to deserialize from.
                scope: The serialization scope to use for the deserialization.
                realm: The realm to deserialize from.
                annotation: The model type to validate against.

            Returns:
                The deserialized model.

            Raises:
                ValueError: If no annotation is given.
            """
            if annotation is None:
                raise ValueError("Deserializing a Pydantic model requires the model as annotation.")

            return pyd.TypeAdapter(annotation).validate_python(_read_value(source))

        @override
        def estimate_serialized_size(
            self,
            instance: pyd.BaseModel,
            target_type: TypeDef[Any] | None = None,
            lossless: bool = True,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
        ) -> int:
            """Measure the serialized model without writing it.

            The JSON-mode dump is unavoidable — it is what the model's fields mean
            once serialized — but it is a C-level pass, and the payload is then
            measured the same way as a plain dictionary.
            """
            mimes = target_mimes(target_type)

            return _json_estimate(
                instance.model_dump(mode="json"),
                mimes[0] if len(mimes) > 0 else "application/json",
            )

    _pyd_serializer: type[Serializer[Any, Any, Any]] | None = PydSerializer
except ImportError:
    _pyd_serializer = None


# DataFrames

type DfFile = (
    File[Literal["application/vnd.apache.parquet"], BinaryIO]
    | File[Literal["application/vnd.ms-excel"], BinaryIO]
    | File[Literal["text/csv"], TextIO]
)


# Pandas
try:
    import pandas as pd

    class PdSerializer(CommonSerializer[pd.DataFrame, DfFile]):
        """Serializer for Pandas DataFrames."""

        __lossless_mimes__ = frozenset({"application/vnd.apache.parquet"})

        @override
        def serialize(
            self,
            instance: pd.DataFrame,
            target_type: TypeDef[Any] | None = None,
            lossless: bool = True,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
            realm: Realm | None = None,
            file_factory: Callable[[], File] | None = None,
        ) -> DfFile:
            """Serialize a DataFrame to Parquet, Excel or CSV.

            Args:
                instance: The frame to serialize.
                target_type: The target type or format to serialize to.
                lossless: Whether to require a lossless serialization.
                scope: The serialization scope to use for the serialization.
                annotation: The declared type of the object, if any.
                realm: The realm to serialize to.
                file_factory: An optional factory function to create new files.

            Returns:
                The written file.

            Raises:
                ValueError: If the target MIME type is not supported.
            """
            target = self._file_target(target_type, file_factory)
            mime = target.default_mime

            # The format is checked before the target is opened, so a rejected
            # format cannot truncate an existing file.
            if mime not in {
                "text/csv",
                "application/vnd.apache.parquet",
                "application/vnd.ms-excel",
            }:
                raise ValueError(f"Unsupported MIME type: {mime}")

            with target.open("w") as stream:
                if mime == "text/csv":
                    instance.to_csv(stream, index=False)
                elif mime == "application/vnd.apache.parquet":
                    instance.to_parquet(cast(BinaryIO, stream))
                else:
                    instance.to_excel(cast(BinaryIO, stream), index=False)

            return target

        @override
        def deserialize(
            self,
            source: DfFile,
            scope: TransportScope = Open(),
            realm: Realm | None = None,
            annotation: TypeDef[Any] | None = None,
        ) -> pd.DataFrame:
            """Deserialize a DataFrame from Parquet, Excel or CSV.

            Args:
                source: The file to deserialize from.
                scope: The serialization scope to use for the deserialization.
                realm: The realm to deserialize from.
                annotation: The declared type of the object, if any.

            Returns:
                The deserialized frame.

            Raises:
                ValueError: If the source MIME type is not supported.
            """
            file = _require_file(source, type(self).__name__)

            with file.open("r") as stream:
                if file.default_mime == "text/csv":
                    frame = pd.read_csv(stream)
                elif file.default_mime == "application/vnd.apache.parquet":
                    frame = pd.read_parquet(cast(BinaryIO, stream))
                elif file.default_mime == "application/vnd.ms-excel":
                    frame = pd.read_excel(cast(BinaryIO, stream))
                else:
                    raise ValueError(f"Unsupported MIME type: {file.default_mime}")

            return check_type_and_cast(frame, annotation)

        @override
        def estimate_serialized_size(
            self,
            instance: pd.DataFrame,
            target_type: TypeDef[Any] | None = None,
            lossless: bool = True,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
        ) -> int:
            """Estimate the encoded size, sampling CSV output by row."""
            if target_mimes(target_type)[:1] == ("text/csv",):
                total = len(instance)
                if total == 0:
                    return 0

                sample = instance.iloc[:: _sample_step(total)]
                return int(_encoded_size(sample.to_csv(index=False)) * total / len(sample))

            return int(instance.memory_usage(deep=True).sum())

    _pd_serializer: type[Serializer[Any, Any, Any]] | None = PdSerializer
except ImportError:
    _pd_serializer = None


# Polars


class PlSerializer(CommonSerializer[pl.DataFrame, DfFile]):
    """Serializer for Polars DataFrames."""

    __lossless_mimes__ = frozenset({"application/vnd.apache.parquet"})

    @override
    def serialize(
        self,
        instance: pl.DataFrame,
        target_type: TypeDef[Any] | None = None,
        lossless: bool = True,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
        realm: Realm | None = None,
        file_factory: Callable[[], File] | None = None,
    ) -> DfFile:
        """Serialize a DataFrame to Parquet, Excel or CSV.

        Args:
            instance: The frame to serialize.
            target_type: The target type or format to serialize to.
            lossless: Whether to require a lossless serialization.
            scope: The serialization scope to use for the serialization.
            annotation: The declared type of the object, if any.
            realm: The realm to serialize to.
            file_factory: An optional factory function to create new files.

        Returns:
            The written file.

        Raises:
            ValueError: If the target MIME type is not supported.
        """
        target = self._file_target(target_type, file_factory)
        mime = target.default_mime

        # The format is checked before the target is opened, so a rejected
        # format cannot truncate an existing file.
        if mime not in {"text/csv", "application/vnd.apache.parquet", "application/vnd.ms-excel"}:
            raise ValueError(f"Unsupported MIME type: {mime}")

        with target.open("w") as stream:
            if mime == "text/csv":
                instance.write_csv(stream)
            elif mime == "application/vnd.apache.parquet":
                instance.write_parquet(cast(BinaryIO, stream))
            else:
                instance.write_excel(cast(BinaryIO, stream))

        return target

    @override
    def deserialize(
        self,
        source: DfFile,
        scope: TransportScope = Open(),
        realm: Realm | None = None,
        annotation: TypeDef[Any] | None = None,
    ) -> pl.DataFrame:
        """Deserialize a DataFrame from Parquet, Excel or CSV.

        Args:
            source: The file to deserialize from.
            scope: The serialization scope to use for the deserialization.
            realm: The realm to deserialize from.
            annotation: The declared type of the object, if any.

        Returns:
            The deserialized frame.

        Raises:
            ValueError: If the source MIME type is not supported.
        """
        file = _require_file(source, type(self).__name__)

        with file.open("r") as stream:
            if file.default_mime == "text/csv":
                frame = pl.read_csv(stream)
            elif file.default_mime == "application/vnd.apache.parquet":
                frame = pl.read_parquet(cast(BinaryIO, stream))
            elif file.default_mime == "application/vnd.ms-excel":
                frame = pl.read_excel(cast(BinaryIO, stream))
            else:
                raise ValueError(f"Unsupported MIME type: {file.default_mime}")

        return check_type_and_cast(frame, annotation)

    @override
    def estimate_serialized_size(
        self,
        instance: pl.DataFrame,
        target_type: TypeDef[Any] | None = None,
        lossless: bool = True,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
    ) -> int:
        """Estimate the encoded size, sampling CSV output by row."""
        if target_mimes(target_type)[:1] == ("text/csv",):
            total = instance.height
            if total == 0:
                return 0

            sample = instance.gather_every(_sample_step(total))
            return int(_encoded_size(sample.write_csv()) * total / sample.height)

        return int(instance.estimated_size())


# NumPy arrays
try:
    import numpy as np

    type NpFile = (
        File[Literal["application/x-npy"], BinaryIO]
        | File[Literal["application/x-npz"], BinaryIO]
        | File[Literal["text/csv"], TextIO]
    )

    def _require_native_dtype(instance: np.ndarray) -> None:
        """Refuse arrays whose native-format payload would need unpickling.

        ``np.save`` pickles object arrays, but loading them back requires
        ``allow_pickle=True``, which the deserialization side never enables:
        that would bypass the pickle serializer's restriction to trusted data.

        Args:
            instance: The array to check.

        Raises:
            TypeError: If the array holds Python objects.
        """
        if instance.dtype.hasobject:
            raise TypeError(
                "NumPy arrays with object dtype cannot be exported to npy/npz: the "
                "native formats would require pickling. Convert the array to a "
                "concrete dtype first."
            )

    class NpSerializer(CommonSerializer[np.ndarray, NpFile]):
        """Serializer for NumPy arrays."""

        __lossless_mimes__ = frozenset({"application/x-npy", "application/x-npz"})

        @override
        def serialize(
            self,
            instance: np.ndarray,
            target_type: TypeDef[Any] | None = None,
            lossless: bool = True,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
            realm: Realm | None = None,
            file_factory: Callable[[], File] | None = None,
        ) -> NpFile:
            """Serialize an array to ``.npy``, ``.npz`` or CSV.

            Args:
                instance: The array to serialize.
                target_type: The target type or format to serialize to.
                lossless: Whether to require a lossless serialization.
                scope: The serialization scope to use for the serialization.
                annotation: The declared type of the object, if any.
                realm: The realm to serialize to.
                file_factory: An optional factory function to create new files.

            Returns:
                The written file.

            Raises:
                ValueError: If the target MIME type is not supported.
                TypeError: If the array holds Python objects.
            """
            target = self._file_target(target_type, file_factory)
            mime = target.default_mime

            # Both checks run before the target is opened, so a rejected array or
            # format cannot truncate an existing file.
            if mime not in {"application/x-npy", "application/x-npz", "text/csv"}:
                raise ValueError(f"Unsupported MIME type: {mime}")

            if mime in {"application/x-npy", "application/x-npz"}:
                _require_native_dtype(instance)

            with target.open("w") as stream:
                if mime == "application/x-npy":
                    np.save(cast(BinaryIO, stream), instance)
                elif mime == "application/x-npz":
                    np.savez(cast(BinaryIO, stream), arr_0=instance)
                else:
                    np.savetxt(cast(TextIO, stream), instance, delimiter=",")

            return target

        @override
        def deserialize(
            self,
            source: NpFile,
            scope: TransportScope = Open(),
            realm: Realm | None = None,
            annotation: TypeDef[Any] | None = None,
        ) -> np.ndarray:
            """Deserialize an array from ``.npy``, ``.npz`` or CSV.

            Args:
                source: The file to deserialize from.
                scope: The serialization scope to use for the deserialization.
                realm: The realm to deserialize from.
                annotation: The declared type of the object, if any.

            Returns:
                The deserialized array.

            Raises:
                ValueError: If the source MIME type is not supported.
            """
            file = _require_file(source, type(self).__name__)

            with file.open("r") as stream:
                if file.default_mime == "application/x-npy":
                    array = np.load(cast(BinaryIO, stream), allow_pickle=False)
                elif file.default_mime == "application/x-npz":
                    array = np.load(cast(BinaryIO, stream), allow_pickle=False)["arr_0"]
                elif file.default_mime == "text/csv":
                    array = np.loadtxt(cast(TextIO, stream), delimiter=",")
                else:
                    raise ValueError(f"Unsupported MIME type: {file.default_mime}")

            return check_type_and_cast(array, annotation)

        @override
        def estimate_serialized_size(
            self,
            instance: np.ndarray,
            target_type: TypeDef[Any] | None = None,
            lossless: bool = True,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
        ) -> int:
            """Estimate the encoded size from the array's memory footprint."""
            if target_mimes(target_type)[:1] in (
                ("application/x-npy",),
                ("application/x-npz",),
            ):
                _require_native_dtype(instance)

            if target_mimes(target_type)[:1] == ("application/x-npy",):
                # The ``.npy`` format prepends a fixed 128-byte header.
                return instance.nbytes + 128

            return instance.nbytes

    _np_serializer: type[Serializer[Any, Any, Any]] | None = NpSerializer
except ImportError:
    _np_serializer = None


# SQL statements
try:
    import sqlalchemy as sqla  # pyright: ignore[reportMissingImports]

    type SQLFile = File[Literal["application/sql"], TextIO]

    type SQLStatement = sqla.Executable | list[sqla.Executable] | tuple[sqla.Executable, ...]

    _SQL_ESTIMATE_SAMPLE = 10
    """Statements formatted for a size estimate; formatting costs more than writing."""

    def _format_statements(instance: SQLStatement) -> list[str]:
        """Format statements the way they are written to disk.

        Args:
            instance: One statement or an iterable of statements.

        Returns:
            The formatted statement texts.
        """
        statements = [instance] if isinstance(instance, sqla.Executable) else instance
        return [
            sqlparse.format(str(stmt), reindent=True, keyword_case="upper") for stmt in statements
        ]

    def _join_statements(instance: SQLStatement) -> str:
        """Serialize statements the way they are written to disk.

        Args:
            instance: One statement or an iterable of statements.

        Returns:
            The serialized statement text.
        """
        return ";\n\n".join(_format_statements(instance))

    class SQLSerializer(CommonSerializer[SQLStatement, str | SQLFile]):
        """Serializer for SQL statements."""

        @override
        def serialize(
            self,
            instance: SQLStatement,
            target_type: TypeDef[Any] | None = None,
            lossless: bool = True,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
            realm: Realm | None = None,
            file_factory: Callable[[], File] | None = None,
        ) -> str | SQLFile:
            """Serialize statements to SQL text or to an SQL file.

            Args:
                instance: One statement or an iterable of statements.
                target_type: The target type or format to serialize to.
                lossless: Whether to require a lossless serialization.
                scope: The serialization scope to use for the serialization.
                annotation: The declared type of the object, if any.
                realm: The realm to serialize to.
                file_factory: An optional factory function to create new files.

            Returns:
                The written file, or the SQL text when a text target was requested.
            """
            text = _join_statements(instance)
            target = self._resolve_target(target_type, file_factory)

            if target is None:
                return text

            _write_text(target, text)

            return target

        @override
        def deserialize(
            self,
            source: str | SQLFile,
            scope: TransportScope = Open(),
            realm: Realm | None = None,
            annotation: TypeDef[Any] | None = None,
        ) -> SQLStatement:
            """Deserialize statements from SQL text or from an SQL file.

            Args:
                source: The file or text to deserialize from.
                scope: The serialization scope to use for the deserialization.
                realm: The realm to deserialize from.
                annotation: The declared type of the object, if any.

            Returns:
                The deserialized statements.
            """
            statements = [sqla.text(str(stmt)) for stmt in sqlparse.parse(_read_text(source))]

            # Rebuild the container the annotation asks for; the statement count
            # alone cannot tell a scalar from a one-element list.
            origin = get_origin(annotation)
            shaped: object
            if origin is list or origin is Sequence:
                shaped = statements
            elif origin is tuple:
                shaped = tuple(statements)
            elif len(statements) == 1:
                shaped = statements[0]
            else:
                shaped = statements

            return check_type_and_cast(shaped, annotation)

        @override
        def estimate_serialized_size(
            self,
            instance: SQLStatement,
            target_type: TypeDef[Any] | None = None,
            lossless: bool = True,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
        ) -> int:
            """Measure the formatted statements from a sample.

            ``sqlparse.format`` parses and re-renders every statement, which costs more
            than writing them out; a strided sample is formatted and extrapolated
            instead, the same way CSV output is sampled by row.
            """
            statements = [instance] if isinstance(instance, sqla.Executable) else list(instance)

            total = len(statements)
            if total == 0:
                return 0

            sample = statements[:: _sample_step(total, _SQL_ESTIMATE_SAMPLE)]
            formatted = _encoded_size(
                ";\n\n".join(
                    sqlparse.format(str(statement), reindent=True, keyword_case="upper")
                    for statement in sample
                )
            )

            return int(formatted * total / len(sample))

    _sql_serializer: type[Serializer[Any, Any, Any]] | None = SQLSerializer
except ImportError:
    _sql_serializer = None


# PIL images, if PIL is installed
try:
    from PIL import Image  # pyright: ignore[reportMissingImports]

    type ImageFile = (
        File[Literal["image/png"], BinaryIO]
        | File[Literal["image/tiff"], BinaryIO]
        | File[Literal["image/bmp"], BinaryIO]
        | File[Literal["image/jpeg"], BinaryIO]
        | File[Literal["image/gif"], BinaryIO]
        | File[Literal["image/webp"], BinaryIO]
    )

    _BYTES_PER_BAND: dict[str, float] = {
        "1": 0.125,
        "I": 4,
        "F": 4,
        "I;16": 2,
        "I;16B": 2,
        "I;16L": 2,
        "I;16N": 2,
    }
    """Bytes per pixel and band for modes that are not one byte per band."""

    def _image_format_for_mime(mime: str) -> str | None:
        """Return the PIL encoder format registered for a MIME type.

        Pillow keys ``Image.MIME`` by format in newer releases and by MIME type
        in older ones, so both directions are accepted.

        Args:
            mime: The MIME type to look up.

        Returns:
            The PIL format name, or ``None`` when Pillow cannot write the format.
        """
        Image.init()
        for key, value in Image.MIME.items():
            if key == mime:
                return value
            if value == mime:
                return key

        return None

    class ImageSerializer(CommonSerializer[Image.Image, ImageFile]):
        """Serializer for PIL images."""

        __lossless_mimes__ = frozenset({"image/png", "image/tiff", "image/bmp"})

        @override
        def serialize(
            self,
            instance: Image.Image,
            target_type: TypeDef[Any] | None = None,
            lossless: bool = True,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
            realm: Realm | None = None,
            file_factory: Callable[[], File] | None = None,
        ) -> ImageFile:
            """Serialize an image to an image file.

            Args:
                instance: The image to serialize.
                target_type: The target type or format to serialize to.
                lossless: Whether to require a lossless serialization.
                scope: The serialization scope to use for the serialization.
                annotation: The declared type of the object, if any.
                realm: The realm to serialize to.
                file_factory: An optional factory function to create new files.

            Returns:
                The written file.

            Raises:
                ValueError: If the target MIME type is not supported.
            """
            target = self._file_target(target_type, file_factory)

            # The encoder comes from the target format, never from the source
            # image: a JPEG loaded from disk must still be written as PNG when
            # the target is a PNG, or the payload contradicts the declared loss.
            image_format = _image_format_for_mime(target.default_mime)
            if image_format is None:
                raise ValueError(f"Unsupported image MIME type: {target.default_mime}")

            with target.open("w") as stream:
                instance.save(stream, format=image_format)

            return target

        @override
        def deserialize(
            self,
            source: ImageFile,
            scope: TransportScope = Open(),
            realm: Realm | None = None,
            annotation: TypeDef[Any] | None = None,
        ) -> Image.Image:
            """Deserialize an image from an image file.

            Args:
                source: The file to deserialize from.
                scope: The serialization scope to use for the deserialization.
                realm: The realm to deserialize from.
                annotation: The declared type of the object, if any.

            Returns:
                The deserialized image.
            """
            file = _require_file(source, type(self).__name__)

            with file.open("r") as stream:
                image = Image.open(stream)
                image.load()

            return check_type_and_cast(image, annotation)

        @override
        def estimate_serialized_size(
            self,
            instance: Image.Image,
            target_type: TypeDef[Any] | None = None,
            lossless: bool = True,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
        ) -> int:
            """Estimate the uncompressed pixel payload.

            Container overhead and compression are not modelled, so the result
            can be smaller than a tiny compressed file.
            """
            bytes_per_band = _BYTES_PER_BAND.get(instance.mode, 1)

            return int(instance.width * instance.height * len(instance.getbands()) * bytes_per_band)

    _image_serializer: type[Serializer[Any, Any, Any]] | None = ImageSerializer
except ImportError:
    # PIL is not installed, so no serializer is registered for images.
    _image_serializer = None


# BeautifulSoup HTML, if BeautifulSoup is installed
try:
    from bs4 import BeautifulSoup
    from bs4.element import Tag

    type HTMLFile = File[Literal["text/html"], TextIO]

    _HTML_TAG_OVERHEAD = 5
    """Bytes of markup a tag costs beyond its name, its attributes and its content."""

    def _html_estimate(instance: BeautifulSoup) -> int:
        """Estimate the serialized soup without serializing it.

        ``str(soup)`` renders every node in Python; walking the parsed tree and
        charging a fixed markup overhead per tag is about ten times cheaper, and
        lands slightly above the real size.

        Args:
            instance: The soup to measure.

        Returns:
            The estimated encoded size in bytes.
        """
        size = 0

        for node in instance.descendants:
            if isinstance(node, Tag):
                attributes = sum(
                    len(name) + 4 + len(str(value)) for name, value in node.attrs.items()
                )
                size += 2 * len(node.name or "") + _HTML_TAG_OVERHEAD + attributes
            else:
                size += _encoded_size(str(node))

        return size

    class HTMLSerializer(CommonSerializer[BeautifulSoup, str | HTMLFile]):
        """Serializer for BeautifulSoup HTML."""

        @override
        def serialize(
            self,
            instance: BeautifulSoup,
            target_type: TypeDef[Any] | None = None,
            lossless: bool = True,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
            realm: Realm | None = None,
            file_factory: Callable[[], File] | None = None,
        ) -> str | HTMLFile:
            """Serialize a soup to HTML text or to an HTML file.

            Args:
                instance: The soup to serialize.
                target_type: The target type or format to serialize to.
                lossless: Whether to require a lossless serialization.
                scope: The serialization scope to use for the serialization.
                annotation: The declared type of the object, if any.
                realm: The realm to serialize to.
                file_factory: An optional factory function to create new files.

            Returns:
                The written file, or the HTML text when a text target was requested.
            """
            text = str(instance)
            target = self._resolve_target(target_type, file_factory)

            if target is None:
                return text

            _write_text(target, text)

            return target

        @override
        def deserialize(
            self,
            source: str | HTMLFile,
            scope: TransportScope = Open(),
            realm: Realm | None = None,
            annotation: TypeDef[Any] | None = None,
        ) -> BeautifulSoup:
            """Deserialize a soup from HTML text or from an HTML file.

            Args:
                source: The file or text to deserialize from.
                scope: The serialization scope to use for the deserialization.
                realm: The realm to deserialize from.
                annotation: The declared type of the object, if any.

            Returns:
                The deserialized soup.
            """
            return check_type_and_cast(BeautifulSoup(_read_text(source), "html.parser"), annotation)

        @override
        def estimate_serialized_size(
            self,
            instance: BeautifulSoup,
            target_type: TypeDef[Any] | None = None,
            lossless: bool = True,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
        ) -> int:
            """Measure the serialized soup without writing it."""
            return _html_estimate(instance)

    _html_serializer: type[Serializer[Any, Any, Any]] | None = HTMLSerializer
except ImportError:
    # BeautifulSoup is not installed, so no serializer is registered for HTML.
    _html_serializer = None


# Pickle files

type PickleFile = File[Literal["application/pickle"], BinaryIO]


class PickleSerializer(CommonSerializer[Any, PickleFile]):
    """Serializer for Pickle files.

    Pickle reproduces any object exactly, but loading a pickle executes arbitrary
    code, so it is only safe for data from a trusted source.
    """

    __lossless_mimes__ = frozenset({"application/pickle"})

    @override
    def serialize(
        self,
        instance: Any,
        target_type: TypeDef[Any] | None = None,
        lossless: bool = True,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
        realm: Realm | None = None,
        file_factory: Callable[[], File] | None = None,
    ) -> PickleFile:
        """Serialize any object to a Pickle file.

        Args:
            instance: The object to serialize.
            target_type: The target type or format to serialize to.
            lossless: Whether to require a lossless serialization.
            scope: The serialization scope to use for the serialization.
            annotation: The declared type of the object, if any.
            realm: The realm to serialize to.
            file_factory: An optional factory function to create new files.

        Returns:
            The written file.
        """
        target = self._file_target(target_type, file_factory)

        with target.open("w") as stream:
            pickle.dump(instance, cast(BinaryIO, stream))

        return target

    @override
    def deserialize(
        self,
        source: PickleFile,
        scope: TransportScope = Open(),
        realm: Realm | None = None,
        annotation: TypeDef[Any] | None = None,
    ) -> Any:
        """Deserialize any object from a Pickle file.

        Args:
            source: The file to deserialize from.
            scope: The serialization scope to use for the deserialization.
            realm: The realm to deserialize from.
            annotation: The declared type of the object, if any.

        Returns:
            The deserialized object.
        """
        file = _require_file(source, type(self).__name__)

        with file.open("r") as stream:
            obj = pickle.load(stream)  # noqa: S301

        return check_type_and_cast(obj, annotation)

    @override
    def estimate_serialized_size(
        self,
        instance: Any,
        target_type: TypeDef[Any] | None = None,
        lossless: bool = True,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
    ) -> int:
        """Measure the pickle payload without writing it.

        The pickler is C and about eighty times faster than summing
        :func:`sys.getsizeof` over the object graph in Python, and no cheaper proxy
        bounds the payload, which depends on the pickler's own opcode stream.
        """
        return len(pickle.dumps(instance))


# Dates and times


class DateTimeSerializer(PrimitiveSerializer[dt.datetime, str]):
    """Serializer for datetimes, as ISO 8601 text."""

    @override
    def _encode(self, instance: dt.datetime) -> str:
        return instance.isoformat()

    @override
    def _decode(self, source: str, annotation: TypeDef[Any] | None = None) -> dt.datetime:
        return dt.datetime.fromisoformat(source)


class DateSerializer(PrimitiveSerializer[dt.date, str]):
    """Serializer for dates, as ISO 8601 text."""

    @override
    def _encode(self, instance: dt.date) -> str:
        return instance.isoformat()

    @override
    def _decode(self, source: str, annotation: TypeDef[Any] | None = None) -> dt.date:
        return dt.date.fromisoformat(source)


class TimeSerializer(PrimitiveSerializer[dt.time, str]):
    """Serializer for times, as ISO 8601 text."""

    @override
    def _encode(self, instance: dt.time) -> str:
        return instance.isoformat()

    @override
    def _decode(self, source: str, annotation: TypeDef[Any] | None = None) -> dt.time:
        return dt.time.fromisoformat(source)


class TimedeltaSerializer(PrimitiveSerializer[dt.timedelta, float]):
    """Serializer for durations, as a number of seconds."""

    @override
    def _encode(self, instance: dt.timedelta) -> float:
        return instance.total_seconds()

    @override
    def _decode(self, source: float, annotation: TypeDef[Any] | None = None) -> dt.timedelta:
        return dt.timedelta(seconds=source)


class ComplexSerializer(PrimitiveSerializer[complex, list[float]]):
    """Serializer for complex numbers, as a two-element ``[real, imaginary]`` list."""

    @override
    def _encode(self, instance: complex) -> list[float]:
        return [instance.real, instance.imag]

    @override
    def _decode(self, source: list[float], annotation: TypeDef[Any] | None = None) -> complex:
        real, imaginary = source

        return complex(real, imaginary)


class DecimalSerializer(PrimitiveSerializer[Decimal, str]):
    """Serializer for decimals, as their exact decimal text."""

    @override
    def _encode(self, instance: Decimal) -> str:
        return str(instance)

    @override
    def _decode(self, source: str, annotation: TypeDef[Any] | None = None) -> Decimal:
        return Decimal(source)


class UuidSerializer(PrimitiveSerializer[uuid.UUID, str]):
    """Serializer for UUIDs, as their canonical text."""

    @override
    def _encode(self, instance: uuid.UUID) -> str:
        return str(instance)

    @override
    def _decode(self, source: str, annotation: TypeDef[Any] | None = None) -> uuid.UUID:
        return uuid.UUID(source)


class PathSerializer(PrimitiveSerializer[Path, str]):
    """Serializer for filesystem paths, as text."""

    @override
    def _encode(self, instance: Path) -> str:
        return str(instance)

    @override
    def _decode(self, source: str, annotation: TypeDef[Any] | None = None) -> Path:
        return Path(source)


class BytesSerializer(PrimitiveSerializer[bytes, str]):
    """Serializer for binary data, as base64 text."""

    @override
    def _encode(self, instance: bytes) -> str:
        return base64.b64encode(instance).decode("ascii")

    @override
    def _decode(self, source: str, annotation: TypeDef[Any] | None = None) -> bytes:
        return base64.b64decode(source.encode("ascii"))


class EnumSerializer(PrimitiveSerializer[Enum, JSONScalar]):
    """Serializer for enum members, as their value."""

    @override
    def _encode(self, instance: Enum) -> JSONScalar:
        return cast(JSONScalar, instance.value)

    @override
    def _decode(self, source: JSONScalar, annotation: TypeDef[Any] | None = None) -> Enum:
        if annotation is None:
            raise ValueError("Deserializing an enum requires the enum class as annotation.")

        enum_type = TypeRef(annotation).concrete_type

        if not (isinstance(enum_type, type) and issubclass(enum_type, Enum)):
            raise TypeError(f"{annotation!r} is not an enum type.")

        return enum_type(source)


# Registration
#
# Order matters twice over: selection keeps the registration order between
# serializers that declare a target at the same position, and a serializer for a
# broad type must not shadow a serializer for a narrower one. The pickle
# serializer accepts any object, so it comes last.

for _serializer in (
    JSONSerializer,
    XMLSerializer,
    _pyd_serializer,
    _pd_serializer,
    PlSerializer,
    _np_serializer,
    _sql_serializer,
    _image_serializer,
    _html_serializer,
    DateTimeSerializer,
    DateSerializer,
    TimeSerializer,
    TimedeltaSerializer,
    ComplexSerializer,
    DecimalSerializer,
    UuidSerializer,
    PathSerializer,
    BytesSerializer,
    EnumSerializer,
    PickleSerializer,
):
    if _serializer is not None:
        common_serializers.append(_serializer())
