"""Common serializers for common types."""

from __future__ import annotations

import base64
import copyreg
import datetime as dt
import io
import json
import pickle
import uuid
import xml.etree.ElementTree as xml  # noqa: N813
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import fields, is_dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Final,
    Literal,
    TextIO,
    TypeVar,
    cast,
    final,
    get_origin,
    override,
)

import polars as pl
import sqlparse
import yaml
from typing_extensions import Sentinel

from based_datatypes.files import File
from based_datatypes.types import DataclassInstance, TypeDef
from based_datatypes.typing import TypeRef, has_type, is_subtype, validated

from .base import (
    Backend,
    Memory,
    Open,
    Scalar,
    Serialized,
    SerializedDict,
    Serializer,
    Tagged,
    TransportScope,
    UnsupportedTargetError,
    deserialize,
    register_serializer,
    serialize,
)

V = TypeVar("V")
T = TypeVar("T", bound=Serialized)


# Helper functions:



type Container = Mapping[Any, Any] | Sequence[Any] | Set[Any] | DataclassInstance
"""Everything the container serializer handles as a whole."""


def _container_payload(value: Any, scope: TransportScope, realm: Backend | None) -> SerializedValue:
    """Turn one value into the JSON-shaped payload a container target can carry.

    Scalar values, files and containers are handled here; anything else is
    serialized on its own, which is how a datetime, a UUID or a dataclass ends up
    as JSON data. A value that serializes to itself cannot be carried by a
    container and is refused.

    Args:
        value: The value to convert.
        scope: The serialization scope the operation requests.
        realm: The realm of the operation.

    Returns:
        The JSON-shaped payload, with files kept for a value target.
    """
    if isinstance(value, File) or value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, bytes | bytearray):
        return base64.b64encode(bytes(value)).decode("ascii")

    if isinstance(value, Mapping):
        return {str(key): _container_payload(item, scope, realm) for key, item in value.items()}

    if isinstance(value, list | tuple):
        return [_container_payload(item, scope, realm) for item in value]

    if isinstance(value, set | frozenset):
        # A set has no order of its own; sorting by ``repr`` keeps the payload
        # stable between runs.
        return [_container_payload(item, scope, realm) for item in sorted(value, key=repr)]

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _container_payload(getattr(value, field.name), scope, realm)
            for field in fields(value)
        }

    converted = serialize(value, scope=scope, realm=realm)
    if converted is value:
        raise UnsupportedTargetError(
            f"{type(value).__name__} serializes to itself, so a container cannot carry it."
        )

    return _container_payload(converted, scope, realm)


class JSONSerializer(Serializer[JSONDict, JSONDict | JSONFile, Open]):
    """Serializer for JSON-shaped dictionaries."""

    __lossless_mimes__ = frozenset({"application/json", "application/yaml"})

    @override
    def serialize(
        self,
        instance: JSONDict,
        target_type: TypeDef[SerializedDict | JSONFile] | None = None,
        allow_lossy: bool = False,
        scope: TransportScope = Open(),
        annotation: TypeDef[JSONDict] | None = None,
        backend: Backend = Memory(),
    ) -> SerializedDict | JSONFile:
        if target_type is None or not is_subtype(target_type, File):
            return cast(SerializedDict, instance)

        target = backend.get_writable_file(target_type)
        _write_text(target, _serialize_json(instance, target.default_mime))

        return cast(SerializedDict | JSONFile, target)

    @override
    def deserialize(
        self,
        source: JSONDict | JSONFile,
        annotation: TypeDef[JSONDict] | None = None,
        scope: TransportScope = Open(),
    ) -> JSONDict:
        parsed = _read_json_value(source)
        assert has_type(parsed, JSONDict), (
            f"Expected a JSON-shaped dictionary, got {type(parsed).__name__}."
        )
        return parsed

    @override
    def estimate_serialized_size(
        self,
        instance: JSONDict,
        target_type: TypeDef[SerializedDict | JSONFile] | None = None,
        allow_lossy: bool = False,
        scope: TransportScope = Open(),
        annotation: TypeDef[JSONDict] | None = None,
    ) -> int:
        """Measure the encoded dictionary.

        The JSON encoder is C, so measuring costs less than any Python size walk
        of the same data; a YAML target is measured from the compact JSON payload
        instead, because the YAML dumper is pure Python and far slower.
        """
        mimes = target_mimes(target_type)

        return _json_estimate(instance, mimes[0] if len(mimes) > 0 else "application/json")


class ContainerSerializer(Serializer[Container, SerializedDict | list[SerializedValue], Open]):
    """Serializer for mappings, sequences, sets and dataclass instances.

    Every item is serialized on its own, so a container of datetimes, UUIDs,
    dataclasses or further containers becomes JSON-shaped data as long as each
    item can be serialized; an item that serializes to a file is nested in the
    payload as that file, which is what a directory tree stores. All containers
    reduce to a dictionary or a list, so this driver only declares those value
    targets — writing JSON or YAML files stays with :class:`JSONSerializer`.
    """

    @override
    def serialize(
        self,
        instance: Container,
        target_type: TypeDef[Any] | None = None,
        allow_lossy: bool = False,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
        backend: Backend = Memory(),
    ) -> SerializedDict | list[SerializedValue]:
        if _is_file_hint(target_type):
            raise UnsupportedTargetError(
                f"{type(self).__name__} reduces a container to a value, not to a file; "
                "serialize the payload to a file target through its own serializer."
            )

        return cast(
            SerializedDict | list[SerializedValue],
            _container_payload(instance, scope, backend),
        )

    @override
    def deserialize(
        self,
        source: SerializedDict | list[SerializedValue],
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
    ) -> Any:
        return check_type_and_cast(source, annotation)

    @override
    def estimate_serialized_size(
        self,
        instance: Container,
        target_type: TypeDef[Any] | None = None,
        allow_lossy: bool = False,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
    ) -> int:
        """Measure the encoded container.

        The payload is built first, because translating nested items is part of
        the cost; the encoded size is then measured with the C-level JSON encoder.
        """
        payload = _container_payload(instance, scope, None)
        mimes = target_mimes(target_type)

        return _json_estimate(payload, mimes[0] if len(mimes) > 0 else "application/json")


# XML

type XMLFile = File[Literal["application/xml"], TextIO] | File[Literal["text/xml"], TextIO]


def _serialize_xml(instance: xml.ElementTree) -> str:
    """Serialize a tree, keeping comments and processing instructions intact."""
    root = instance.getroot()
    if root is None:
        return ""

    return xml.tostring(root, encoding="unicode")


class XMLSerializer(Serializer[xml.ElementTree, str | XMLFile, Open]):
    """Serializer for XML element trees."""

    __lossless_mimes__ = frozenset({"application/xml", "text/xml"})

    @override
    def serialize(
        self,
        instance: xml.ElementTree,
        target_type: TypeDef[Any] | None = None,
        allow_lossy: bool = False,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
        backend: Backend = Memory(),
    ) -> str | XMLFile:
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
        annotation: TypeDef[Any] | None = None,
    ) -> xml.ElementTree:
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
        allow_lossy: bool = False,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
    ) -> int:
        """Measure the serialized tree without writing it.

        ``xml.tostring`` is a C-level pass over the tree, so it is cheaper than
        walking the elements in Python — which would also under-count the markup
        the serializer emits.
        """
        return len(_serialize_xml(instance).encode("utf-8"))


# Pydantic models
try:
    import pydantic as pyd

    class PydSerializer(Serializer[pyd.BaseModel, SerializedDict | JSONFile, Open]):
        """Serializer for Pydantic models."""

        @override
        def serialize(
            self,
            instance: pyd.BaseModel,
            target_type: TypeDef[Any] | None = None,
            allow_lossy: bool = False,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
            backend: Backend = Memory(),
        ) -> SerializedDict | JSONFile:
            """Serialize a model to a JSON or YAML file, or return its dump.

            ``mode="json"`` converts dates, UUIDs and decimals to JSON-native
            values, so a value the encoder rejects cannot truncate an existing
            file and YAML's safe dumper accepts it. Aliases and excluded fields
            do not survive the dump, which is why the format is not lossless.

            Args:
                instance: The model to serialize.
                target_type: The target type or format to serialize to.
                allow_lossy: Whether to require a lossless serialization.
                scope: The serialization scope to use for the serialization.
                annotation: The declared type of the object, if any.
                backend: The backend to serialize to.

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
            annotation: TypeDef[Any] | None = None,
        ) -> pyd.BaseModel:
            if annotation is None:
                raise ValueError("Deserializing a Pydantic model requires the model as annotation.")

            return pyd.TypeAdapter(annotation).validate_python(_read_json_value(source))

        @override
        def estimate_serialized_size(
            self,
            instance: pyd.BaseModel,
            target_type: TypeDef[Any] | None = None,
            allow_lossy: bool = False,
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

    class PdSerializer(Serializer[pd.DataFrame, DfFile, Open]):
        """Serializer for Pandas DataFrames."""

        __lossless_mimes__ = frozenset({"application/vnd.apache.parquet"})

        @override
        def serialize(
            self,
            instance: pd.DataFrame,
            target_type: TypeDef[Any] | None = None,
            allow_lossy: bool = False,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
            backend: Backend = Memory(),
        ) -> DfFile:
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
            annotation: TypeDef[Any] | None = None,
        ) -> pd.DataFrame:
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
            allow_lossy: bool = False,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
        ) -> int:
            """Estimate the encoded size, sampling CSV output by row."""
            if target_mimes(target_type)[:1] == ("text/csv",):
                total = len(instance)
                if total == 0:
                    return 0

                sample = instance.iloc[:: max(total // 100, 1)]
                return int(len(sample.to_csv(index=False).encode("utf-8")) * total / len(sample))

            return int(instance.memory_usage(deep=True).sum())

    _pd_serializer: type[Serializer[Any, Any, Any]] | None = PdSerializer
except ImportError:
    _pd_serializer = None


# Polars


class PlSerializer(Serializer[pl.DataFrame, DfFile, Open]):
    """Serializer for Polars DataFrames."""

    __lossless_mimes__ = frozenset({"application/vnd.apache.parquet"})

    @override
    def serialize(
        self,
        instance: pl.DataFrame,
        target_type: TypeDef[Any] | None = None,
        allow_lossy: bool = False,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
        backend: Backend = Memory(),
    ) -> DfFile:
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
        annotation: TypeDef[Any] | None = None,
    ) -> pl.DataFrame:
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
        allow_lossy: bool = False,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
    ) -> int:
        """Estimate the encoded size, sampling CSV output by row."""
        if target_mimes(target_type)[:1] == ("text/csv",):
            total = instance.height
            if total == 0:
                return 0

            sample = instance.gather_every(max(total // 100, 1))
            return int(len(sample.write_csv().encode("utf-8")) * total / sample.height)

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

    class NpSerializer(Serializer[np.ndarray, NpFile, Open]):
        """Serializer for NumPy arrays."""

        __lossless_mimes__ = frozenset({"application/x-npy", "application/x-npz"})

        @override
        def serialize(
            self,
            instance: np.ndarray,
            target_type: TypeDef[Any] | None = None,
            allow_lossy: bool = False,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
            backend: Backend = Memory(),
        ) -> NpFile:
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
            annotation: TypeDef[Any] | None = None,
        ) -> np.ndarray:
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
            allow_lossy: bool = False,
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

    class SQLSerializer(Serializer[SQLStatement, str | SQLFile, Open]):
        """Serializer for SQL statements."""

        # The text target and the ``.sql`` file hold the same reformatted
        # statements, so losslessness is declared once for both.
        __lossless_values__ = False

        @override
        def serialize(
            self,
            instance: SQLStatement,
            target_type: TypeDef[Any] | None = None,
            allow_lossy: bool = False,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
            backend: Backend = Memory(),
        ) -> str | SQLFile:
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
            annotation: TypeDef[Any] | None = None,
        ) -> SQLStatement:
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
            allow_lossy: bool = False,
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

            sample = statements[:: max(total // _SQL_ESTIMATE_SAMPLE, 1)]
            formatted = len(
                ";\n\n".join(
                    sqlparse.format(str(statement), reindent=True, keyword_case="upper")
                    for statement in sample
                ).encode("utf-8")
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

    class ImageSerializer(Serializer[Image.Image, ImageFile, Open]):
        """Serializer for PIL images."""

        __lossless_mimes__ = frozenset({"image/png", "image/tiff", "image/bmp"})

        @override
        def serialize(
            self,
            instance: Image.Image,
            target_type: TypeDef[Any] | None = None,
            allow_lossy: bool = False,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
            backend: Backend = Memory(),
        ) -> ImageFile:
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
            annotation: TypeDef[Any] | None = None,
        ) -> Image.Image:
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
            allow_lossy: bool = False,
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
                size += len(str(node).encode("utf-8"))

        return size

    class HTMLSerializer(Serializer[BeautifulSoup, str | HTMLFile, Open]):
        """Serializer for BeautifulSoup HTML."""

        # The text target and the ``.html`` file hold the same re-rendered
        # markup, so losslessness is declared once for both.
        __lossless_values__ = True

        @override
        def serialize(
            self,
            instance: BeautifulSoup,
            target_type: TypeDef[Any] | None = None,
            allow_lossy: bool = False,
            scope: TransportScope = Open(),
            annotation: TypeDef[Any] | None = None,
            backend: Backend = Memory(),
        ) -> str | HTMLFile:
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
            annotation: TypeDef[Any] | None = None,
        ) -> BeautifulSoup:
            return check_type_and_cast(BeautifulSoup(_read_text(source), "html.parser"), annotation)

        @override
        def estimate_serialized_size(
            self,
            instance: BeautifulSoup,
            target_type: TypeDef[Any] | None = None,
            allow_lossy: bool = False,
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

type ReduceTuple = tuple[Callable[..., Any], tuple[Any, ...]]
"""A pickle reduce value: (rebuild_callable, args)."""

# Types the default pickler handles natively. Only the pure-Python pickler
# exposes this table; the C-accelerated Pickler hides it.
_DISPATCH: Final[dict[type[Any], Callable[[pickle.Pickler, Any], None]]] = pickle._Pickler.dispatch  # type: ignore[attr-defined]  # noqa: SLF001


class SerializePickler(pickle.Pickler):
    """Pickler that routes unpicklable leaf objects to `fallback`."""

    def __init__(  # noqa: D107
        self,
        file: BinaryIO,
        protocol: int | None = None,
        fix_imports: bool = True,
        buffer_callback: Callable[[pickle.PickleBuffer], Any] | None = None,
    ) -> None:
        super().__init__(file, protocol, fix_imports, buffer_callback)
        # Per-type probe verdicts. Kept per-instance: probes use the default
        # protocol, so verdicts must not leak across picklers.
        self._picklable_types: set[type[Any]] = set()
        self._unpicklable_types: set[type[Any]] = set()

    def serialize(self, obj: object) -> ReduceTuple:
        """Handle an unpicklable leaf object via `serialize`."""
        return (partial(deserialize, annotation=type(obj)), (serialize(obj),))

    @override
    def reducer_override(self, obj: Any, /) -> Any:
        t: type[Any] = type(obj)
        # 1. Natively handled -> never probe, never touch.
        if t in _DISPATCH or t in copyreg.dispatch_table:
            return NotImplemented
        # 2. Cached per-type verdicts.
        if t in self._picklable_types:
            return NotImplemented
        if t in self._unpicklable_types:
            return self.serialize(obj)
        # 3. Decomposable (custom reduce / __dict__) -> let pickle recurse;
        #    failing contents are caught individually, deeper down.
        if (
            t.__reduce_ex__ is not object.__reduce_ex__
            or t.__reduce__ is not object.__reduce__
            or hasattr(obj, "__dict__")
        ):
            return NotImplemented
        # 4. Opaque leaf of unknown type -> probe once per type, cache verdict.
        if self._probe(obj):
            self._picklable_types.add(t)
            return NotImplemented
        self._unpicklable_types.add(t)
        return self.serialize(obj)

    @staticmethod
    def _probe(obj: object) -> bool:
        """Return True if the default pickler can serialize `obj`.

        Catches Exception broadly on purpose: any failure mode (TypeError,
        PicklingError, AttributeError from a broken __reduce__, ...) means
        the object needs the fallback.
        """
        try:
            pickle.dumps(obj)
        except Exception:
            return False
        return True


class PickleSerializer(Serializer[Any, PickleFile, Open]):
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
        allow_lossy: bool = False,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
        backend: Backend = Memory(),
    ) -> PickleFile:
        """Serialize any object to a Pickle file.

        Args:
            instance: The object to serialize.
            target_type: The target type or format to serialize to.
            allow_lossy: Whether to require a lossless serialization.
            scope: The serialization scope to use for the serialization.
            annotation: The declared type of the object, if any.
            backend: The backend to serialize to.

        Returns:
            The written file.
        """
        target = self._file_target(target_type, file_factory)

        with target.open("w") as stream:
            SerializePickler(stream).dump(instance)

        return target

    @override
    def deserialize(
        self,
        source: PickleFile,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
    ) -> Any:
        file = _require_file(source, type(self).__name__)

        with file.open("r") as stream:
            obj = pickle.load(stream)  # noqa: S301

        return check_type_and_cast(obj, annotation)

    @override
    def estimate_serialized_size(
        self,
        instance: Any,
        target_type: TypeDef[Any] | None = None,
        allow_lossy: bool = False,
        scope: TransportScope = Open(),
        annotation: TypeDef[Any] | None = None,
    ) -> int:
        """Measure the pickle payload without writing it.

        The pickler is C and about eighty times faster than summing
        :func:`sys.getsizeof` over the object graph in Python, and no cheaper proxy
        bounds the payload, which depends on the pickler's own opcode stream.
        """
        return len(pickle.dumps(instance))


# Primitive types

# def scalar_to_json(scalar: Scalar) -> JSONValue:  # noqa: C901
#     """Convert a scalar to a JSON-supported value, or a JSON-shaped dictionary."""
#     match scalar:
#         case str() | int() | float() | bool() | None:
#             return scalar
#         case Decimal():
#             return {"$dec": str(scalar)}
#         case Fraction():
#             return {"$frac": [scalar.numerator, scalar.denominator]}
#         case complex():
#             return {"$complex": [scalar.real, scalar.imag]}
#         case bytes():
#             return {"$bytes": base64.b64encode(bytes(scalar)).decode("ascii")}
#         case bytearray():
#             return {"$bytearray": base64.b64encode(bytes(scalar)).decode("ascii")}
#         case dt.datetime():
#             return {"$dt": scalar.isoformat()}
#         case dt.date():
#             return {"$date": scalar.isoformat()}
#         case dt.time():
#             return {"$time": scalar.isoformat()}
#         case dt.timedelta():
#             return {"$timedelta": scalar.total_seconds()}
#         case uuid.UUID():
#             return {"$uuid": str(scalar)}
#         case Path():
#             return {"$path": str(scalar)}


# Registration
#
# Order matters twice over: selection keeps the registration order between
# serializers that declare a target at the same position, and a serializer for a
# broad type must not shadow a serializer for a narrower one. The container
# serializer comes after every specialized one it could otherwise shadow, and the
# pickle serializer accepts any object, so it comes last.

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
    ContainerSerializer,
    PickleSerializer,
):
    if _serializer is not None:
        register_serializer(_serializer())
