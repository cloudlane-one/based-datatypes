"""Common transport drivers for common types."""

from __future__ import annotations

import json
import pickle
import xml.etree.ElementTree as xml  # noqa: N813
from collections.abc import Iterable, Sequence
from typing import Any, BinaryIO, Literal, TextIO, TypeVar, cast, get_origin, override

import polars as pl
import sqlparse
import yaml

from based_datatypes.descriptors import cached_method
from based_datatypes.files import File
from based_datatypes.types import TypeDef
from based_datatypes.typing import TypeNode, TypeRef

from .base import (
    V2,
    JSONDict,
    Open,
    Realm,
    SerializationTarget,
    Serializer,
    TransportScope,
    current_ser_ctx,
)

common_serializers: list[Serializer] = []


# Simple, stringifiable types

type TextFile = File[Literal["text/plain", "text/csv"], TextIO]
TF = TypeVar(
    "TF",
    bound=TextFile,
    default=TextFile,
)

type StrScalar = str | int | float | complex | bool | None
type Stringifiable = StrScalar | list[StrScalar]


def check_type_and_cast[T](obj: Any, annotation: TypeNode[T] | None = None) -> T:
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


# JSON / YAML dict

type JSONFile = (
    File[Literal["application/yaml"], TextIO] | File[Literal["application/json"], TextIO]
)
JF = TypeVar(
    "JF",
    bound=JSONFile,
    default=JSONFile,
)

J = TypeVar("J", bound=JSONDict, default=JSONDict)


class JSONSerializer(Serializer[JSONDict, JSONDict | JSONFile]):
    """Serializer for JSON and YAML dictionaries."""

    @override
    @cached_method
    def serialization_targets(self) -> list[SerializationTarget[JSONDict | JSONFile, Any, Any]]:
        """Serialization targets supported by this transport driver."""
        return [SerializationTarget(type=JSONDict | JSONFile, scope=Open(), lossless=True)]

        @classmethod
        def import_object(
            cls: type[Serializer[V2, JSONFile]],
            source: JSONFile,
            realm: Any,
            annotation: TypeNode[V2] | None = None,
        ) -> V2:
            """Parse a dictionary from a JSON or YAML file."""
            dct: dict

            with source.open("r") as f:
                if source.mime == "application/yaml":
                    dct = yaml.load(f, Loader=yaml.CSafeLoader)
                else:
                    dct = json.load(f)

            return check_type_and_cast(dct, annotation)

        @classmethod
        def export_object(
            cls,
            instance: JSONDict,
            target: JSONFile,
            realm: Any,
            annotation: TypeNode[JSONDict] | None = None,
        ) -> None:
            """Convert a dictionary to a JSON or YAML file."""
            text = _serialize_json(instance, target.mime)

            with target.open("w") as f:
                f.write(text)

    @override
    def serialize(
        self,
        source: JSONDict | JSONFile,
        scope: TransportScope = Open(),
        realm: Realm | None = None,
        annotation: TypeDef[V2] | None = None,
    ) -> V2:
        """Import an object from the source type or format.

        Args:
            source: The source object to import from.
            scope: The transport scope to use for the import.
            realm: The realm to import from.
            annotation: The declared type of the object, if any.

        Returns:
            An instance of the object, imported from the source.
        """
        ...

    @override
    def serialize(
        self: Serializer[V2, T2],
        instance: V2,
        target_type: TypeDef[T2],
        lossless: bool = True,
        scope: TransportScope = Open(),
        realm: Realm | None = None,
        target: T2 | None = None,
        annotation: TypeDef[Serializable[T2]] | None = None,
    ) -> T2:
        """Export an object to the target type or format.

        Args:
            instance: The object to export.
            target_type: The target type or format to export to.
            lossless: Whether to require a lossless export.
            scope: The transport scope to use for the export.
            realm: The realm to export to.
            target: An optional pre-allocated target object to export into.
            annotation: The declared type of the object, if any.

        Returns:
            The exported target object.
        """
        ...

    @override
    def estimate_serialized_size(
        self,
        instance: JSONDict,
        target_type: TypeDef[JSONDict | JSONFile],
        lossless: bool = True,
        scope: TransportScope = Open(),
        realm: Realm | None = None,
        target: JSONDict | JSONFile | None = None,
        annotation: TypeDef[JSONDict] | None = None,
    ) -> int:
        return _encoded_size(
            _serialize_json(
                instance, target.default_mime if target is not None else "application/json"
            )
        )


common_serializers[JSONDict] = JSONSerializer


# XML etree

XMLFile = File[Literal["text/xml", "application/xml"], TextIO]

XMLF = TypeVar(
    "XMLF",
    bound=XMLFile,
    default=XMLFile,
)


def _serialize_xml(instance: xml.ElementTree) -> str:
    """Serialize a tree, keeping comments and processing instructions intact."""
    root = instance.getroot()
    if root is None:
        return ""

    return xml.tostring(root, encoding="unicode")


class XMLTransporter(Serializer[xml.ElementTree, XMLFile]):
    """Transporter for XML ElementTree."""

    @classmethod
    def __lossless_formats__(cls) -> frozenset[str]:
        """The serialized tree preserves every element and attribute."""
        return frozenset({"application/xml", "text/xml"})

    @classmethod
    def estimate_serialized_size(
        cls,
        instance: xml.ElementTree,
        target: XMLFile,
        realm: Any,
        annotation: TypeNode[xml.ElementTree] | None = None,
    ) -> int:
        """Measure the serialized tree without writing it."""
        return _encoded_size(_serialize_xml(instance))

    @classmethod
    def serialize(
        cls: type[Serializer[V2, XMLFile]],
        source: XMLFile,
        realm: Any,
        annotation: TypeNode[V2] | None = None,
    ) -> V2:
        """Parse an XML ElementTree from a file."""
        if source.default_mime not in {"application/xml", "text/xml"}:
            raise ValueError(f"Unsupported MIME type: {source.default_mime}")

        # The default builder drops comments and processing instructions, which
        # would break the lossless round trip this driver declares.
        parser = xml.XMLParser(target=xml.TreeBuilder(insert_comments=True, insert_pis=True))
        with source.open("r") as f:
            tree = xml.parse(f, parser=parser)

        return check_type_and_cast(tree, annotation)

    @classmethod
    def serialize(
        cls,
        instance: xml.ElementTree,
        target: XMLFile,
        realm: Any,
        annotation: TypeNode[xml.ElementTree] | None = None,
    ) -> None:
        """Convert an XML ElementTree to a file."""
        assert target.default_mime in {"application/xml", "text/xml"}
        text = _serialize_xml(instance)

        with target.open("w") as f:
            f.write(text)


common_serializers[xml.ElementTree] = XMLTransporter


# Pydantic models
try:
    import pydantic as pyd

    class PydTransporter(Serializer[pyd.BaseModel, JSONFile]):
        """Transporter for Pydantic models."""

        @classmethod
        def estimate_serialized_size(
            cls,
            instance: pyd.BaseModel,
            target: JSONFile,
            realm: Any,
            annotation: TypeNode[pyd.BaseModel] | None = None,
        ) -> int:
            """Measure the serialized model without writing it."""
            return _encoded_size(
                _serialize_json(instance.model_dump(mode="json"), target.default_mime)
            )

        @classmethod
        def serialize(
            cls: type[Serializer[V2, JSONFile]],
            source: JSONFile,
            realm: Any,
            annotation: TypeNode[V2] | None = None,
        ) -> V2:
            """Parse a Pydantic model from a JSON or YAML file."""
            with source.open("r") as f:
                if source.default_mime == "application/yaml":
                    data = yaml.load(f, Loader=yaml.CSafeLoader)
                else:
                    data = json.load(f)

            if annotation is None:
                raise ValueError("Annotation must be provided for Pydantic model parsing.")

            adapter = pyd.TypeAdapter(annotation)
            return adapter.validate_python(data)

        @classmethod
        def serialize(
            cls,
            instance: pyd.BaseModel,
            target: JSONFile,
            realm: Any,
            annotation: TypeNode[pyd.BaseModel] | None = None,
        ) -> None:
            """Convert a Pydantic model to a JSON or YAML file."""
            # ``mode="json"`` converts dates, UUIDs and decimals to JSON-native
            # values before the target is opened, so a value the encoder rejects
            # cannot truncate an existing file and YAML's safe dumper accepts it.
            text = _serialize_json(instance.model_dump(mode="json"), target.default_mime)

            with target.open("w") as f:
                f.write(text)

    common_serializers[pyd.BaseModel] = PydTransporter
except ImportError:
    pass

# DataFrames

type DfFile = (
    File[Literal["application/vnd.ms-excel", "application/vnd.apache.parquet"], BinaryIO]
    | File[Literal["text/csv"], TextIO]
)


# Pandas
try:
    import pandas as pd

    class PdTransporter(Serializer[pd.DataFrame, DfFile]):
        """Transporter for Pandas DataFrames."""

        @classmethod
        def __lossless_formats__(cls) -> frozenset[str]:
            """Parquet keeps dtypes and values; CSV and Excel do not."""
            return frozenset({"application/vnd.apache.parquet"})

        @classmethod
        def estimate_serialized_size(
            cls,
            instance: pd.DataFrame,
            target: DfFile,
            realm: Any,
            annotation: TypeNode[pd.DataFrame] | None = None,
        ) -> int:
            """Estimate the encoded size, sampling CSV output by row."""
            if target.default_mime == "text/csv":
                total = len(instance)
                if total == 0:
                    return 0

                sample = instance.iloc[:: _sample_step(total)]
                return int(_encoded_size(sample.to_csv(index=False)) * total / len(sample))

            return int(instance.memory_usage(deep=True).sum())

        @classmethod
        def serialize(
            cls: type[Serializer[V2, DfFile]],
            source: DfFile,
            realm: Any,
            annotation: TypeNode[V2] | None = None,
        ) -> V2:
            """Parse a DataFrame from a CSV or Parquet file."""
            with source.open("r") as f:
                if source.default_mime == "text/csv":
                    res = pd.read_csv(f)
                elif source.default_mime == "application/vnd.apache.parquet":
                    f = cast(BinaryIO, f)
                    res = pd.read_parquet(f)
                elif source.default_mime == "application/vnd.ms-excel":
                    res = pd.read_excel(f)
                else:
                    raise ValueError(f"Unsupported MIME type: {source.default_mime}")

            return check_type_and_cast(res, annotation)

        @classmethod
        def serialize(
            cls,
            instance: pd.DataFrame,
            target: DfFile,
            realm: Any,
            annotation: TypeNode[pd.DataFrame] | None = None,
        ) -> None:
            """Convert a DataFrame to a CSV or Parquet file."""
            with target.open("w") as f:
                if target.default_mime == "text/csv":
                    instance.to_csv(f, index=False)
                elif target.default_mime == "application/vnd.apache.parquet":
                    f = cast(BinaryIO, f)
                    instance.to_parquet(f)
                elif target.default_mime == "application/vnd.ms-excel":
                    f = cast(BinaryIO, f)
                    instance.to_excel(f, index=False)
                else:
                    raise ValueError(f"Unsupported MIME type: {target.default_mime}")

    common_serializers[pd.DataFrame] = PdTransporter
except ImportError:
    pass

# Polars


class PlTransporter(Serializer[pl.DataFrame, DfFile]):
    """Transporter for Polars DataFrames."""

    @classmethod
    def __lossless_formats__(cls) -> frozenset[str]:
        """Parquet keeps dtypes and values; CSV and Excel do not."""
        return frozenset({"application/vnd.apache.parquet"})

    @classmethod
    def estimate_serialized_size(
        cls,
        instance: pl.DataFrame,
        target: DfFile,
        realm: Any,
        annotation: TypeNode[pl.DataFrame] | None = None,
    ) -> int:
        """Estimate the encoded size, sampling CSV output by row."""
        if target.default_mime == "text/csv":
            total = instance.height
            if total == 0:
                return 0

            sample = instance.gather_every(_sample_step(total))
            return int(_encoded_size(sample.write_csv()) * total / sample.height)

        return int(instance.estimated_size())

    @classmethod
    def serialize(
        cls: type[Serializer[V2, DfFile]],
        source: DfFile,
        realm: Any,
        annotation: TypeNode[V2] | None = None,
    ) -> V2:
        """Parse a Polars DataFrame from a CSV or Parquet file."""
        with source.open("r") as f:
            if source.default_mime == "text/csv":
                res = pl.read_csv(f)
            elif source.default_mime == "application/vnd.apache.parquet":
                f = cast(BinaryIO, f)
                res = pl.read_parquet(f)
            elif source.default_mime == "application/vnd.ms-excel":
                f = cast(BinaryIO, f)
                res = pl.read_excel(f)
            else:
                raise ValueError(f"Unsupported MIME type: {source.default_mime}")

        return check_type_and_cast(res, annotation)

    @classmethod
    def serialize(
        cls,
        instance: pl.DataFrame,
        target: DfFile,
        realm: Any,
        annotation: TypeNode[pl.DataFrame] | None = None,
    ) -> None:
        """Convert a Polars DataFrame to a CSV or Parquet file."""
        with target.open("w") as f:
            if target.default_mime == "text/csv":
                instance.write_csv(f)
            elif target.default_mime == "application/vnd.apache.parquet":
                f = cast(BinaryIO, f)
                instance.write_parquet(f)
            elif target.default_mime == "application/vnd.ms-excel":
                f = cast(BinaryIO, f)
                instance.write_excel(f)
            else:
                raise ValueError(f"Unsupported MIME type: {target.default_mime}")


common_serializers[pl.DataFrame] = PlTransporter


# NumPy arrays
try:
    import numpy as np

    type NpFile = (
        File[Literal["application/x-npy", "application/x-npz"], BinaryIO]
        | File[Literal["text/csv"], TextIO]
    )

    def _require_native_dtype(instance: np.ndarray) -> None:
        """Refuse arrays whose native-format payload would need unpickling.

        ``np.save`` pickles object arrays, but loading them back requires
        ``allow_pickle=True``, which the import side deliberately never enables:
        that would bypass the pickle driver's Open/IPC scope restriction.
        """
        if instance.dtype.hasobject:
            raise TypeError(
                "NumPy arrays with object dtype cannot be exported to npy/npz: the "
                "native formats would require pickling, which is restricted to the "
                "Open and IPC scopes. Convert the array to a concrete dtype first."
            )

    class NpTransporter(Serializer[np.ndarray, NpFile]):
        """Transporter for NumPy arrays."""

        @classmethod
        def __lossless_formats__(cls) -> frozenset[str]:
            """The native formats keep dtype, shape and values; CSV does not."""
            return frozenset({"application/x-npy", "application/x-npz"})

        @classmethod
        def estimate_serialized_size(
            cls,
            instance: np.ndarray,
            target: NpFile,
            realm: Any,
            annotation: TypeNode[np.ndarray] | None = None,
        ) -> int:
            """Estimate the encoded size from the array's memory footprint."""
            if target.default_mime in {"application/x-npy", "application/x-npz"}:
                _require_native_dtype(instance)

            if target.default_mime == "application/x-npy":
                # The ``.npy`` format prepends a fixed 128-byte header.
                return instance.nbytes + 128

            return instance.nbytes

        @classmethod
        def serialize(
            cls: type[Serializer[V2, NpFile]],
            source: NpFile,
            realm: Any,
            annotation: TypeNode[V2] | None = None,
        ) -> V2:
            """Parse a NumPy array from a file."""
            with source.open("r") as f:
                if source.default_mime == "application/x-npy":
                    f = cast(BinaryIO, f)
                    res = np.load(f, allow_pickle=False)
                elif source.default_mime == "application/x-npz":
                    f = cast(BinaryIO, f)
                    res = np.load(f, allow_pickle=False)["arr_0"]
                elif source.default_mime == "text/csv":
                    res = np.loadtxt(f, delimiter=",")
                else:
                    raise ValueError(f"Unsupported MIME type: {source.default_mime}")

            return check_type_and_cast(res, annotation)

        @classmethod
        def serialize(
            cls,
            instance: np.ndarray,
            target: NpFile,
            realm: Any,
            annotation: TypeNode[np.ndarray] | None = None,
        ) -> None:
            """Convert a NumPy array to a file."""
            if target.default_mime in {"application/x-npy", "application/x-npz"}:
                _require_native_dtype(instance)

            with target.open("w") as f:
                if target.default_mime == "application/x-npy":
                    f = cast(BinaryIO, f)
                    np.save(f, instance)
                elif target.default_mime == "application/x-npz":
                    f = cast(BinaryIO, f)
                    np.savez(f, arr_0=instance)
                elif target.default_mime == "text/csv":
                    np.savetxt(f, instance, delimiter=",")
                else:
                    raise ValueError(f"Unsupported MIME type: {target.default_mime}")

    common_serializers[np.ndarray] = NpTransporter
except ImportError:
    pass

# SQL statements
try:
    import sqlalchemy as sqla  # pyright: ignore[reportMissingImports]

    type SQLFile = File[Literal["application/sql"], TextIO]

    def _format_statements(
        instance: sqla.Executable | Iterable[sqla.Executable],
    ) -> list[str]:
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

    class SQLTransporter(Serializer[sqla.Executable | Iterable[sqla.Executable], SQLFile]):
        """Transporter for SQL statements."""

        @classmethod
        def estimate_serialized_size(
            cls,
            instance: sqla.Executable | Iterable[sqla.Executable],
            target: SQLFile,
            realm: Any,
            annotation: TypeNode[sqla.Executable | Iterable[sqla.Executable]] | None = None,
        ) -> int:
            """Measure the formatted statements without writing them."""
            return _encoded_size(";\n\n".join(_format_statements(instance)))

        @classmethod
        def serialize(
            cls: type[Serializer[V2, SQLFile]],
            source: SQLFile,
            realm: Any,
            annotation: TypeNode[V2] | None = None,
        ) -> V2:
            """Parse an SQL statement from a file."""
            with source.open("r") as f:
                sql_content = f.read()
                parsed_sql = sqlparse.parse(sql_content)

            stmts = [sqla.text(str(stmt)) for stmt in parsed_sql]

            # Rebuild the container the annotation asks for; the statement count
            # alone cannot tell a scalar import from a one-element list.
            origin = get_origin(annotation) if annotation is not None else None
            shaped: object
            if origin is list or origin is Sequence:
                shaped = list(stmts)
            elif origin is tuple:
                shaped = tuple(stmts)
            elif len(stmts) == 1:
                shaped = stmts[0]
            else:
                shaped = stmts

            return check_type_and_cast(shaped, annotation)

        @classmethod
        def serialize(
            cls,
            instance: sqla.Executable | Iterable[sqla.Executable],
            target: SQLFile,
            realm: Any,
            annotation: TypeNode[sqla.Executable | Iterable[sqla.Executable]] | None = None,
        ) -> None:
            """Convert an SQL statement to a file."""
            formatted = _format_statements(instance)

            with target.open("w") as f:
                f.write(";\n\n".join(formatted))

    # Registered for the concrete containers rather than ``Iterable``: the key is
    # matched by concrete type, and a bare ``Iterable`` would claim mappings too.
    common_serializers[sqla.Executable | list[sqla.Executable] | tuple[sqla.Executable, ...]] = (
        SQLTransporter
    )
except ImportError:
    pass

# PIL images, if PIL is installed
try:
    from PIL import Image  # pyright: ignore[reportMissingImports]

    type ImageFile = File[
        Literal[
            "image/png",
            "image/jpeg",
            "image/gif",
            "image/bmp",
            "image/tiff",
            "image/webp",
        ],
        BinaryIO,
    ]

    ImageF = TypeVar("ImageF", bound=ImageFile, default=ImageFile)

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
        """
        Image.init()
        for key, value in Image.MIME.items():
            if key == mime:
                return value
            if value == mime:
                return key

        return None

    class ImageTransporter(Serializer[Image.Image, ImageFile]):
        """Transporter for PIL Images."""

        @classmethod
        def __lossless_formats__(cls) -> frozenset[str]:
            """PNG, TIFF and BMP are lossless; JPEG, GIF and WebP are not."""
            return frozenset({"image/png", "image/tiff", "image/bmp"})

        @classmethod
        def estimate_serialized_size(
            cls,
            instance: Image.Image,
            target: ImageFile,
            realm: Any,
            annotation: TypeNode[Image.Image] | None = None,
        ) -> int:
            """Estimate the uncompressed pixel payload.

            Container overhead and compression are not modelled, so the result
            can be smaller than a tiny compressed file.
            """
            bytes_per_band = _BYTES_PER_BAND.get(instance.mode, 1)
            return int(instance.width * instance.height * len(instance.getbands()) * bytes_per_band)

        @classmethod
        def serialize(
            cls: type[Serializer[V2, ImageFile]],
            source: ImageFile,
            realm: Any,
            annotation: TypeNode[V2] | None = None,
        ) -> V2:
            """Parse an image from a file."""
            with source.open("r") as f:
                img = Image.open(f)
                img.load()

            return check_type_and_cast(img, annotation)

        @classmethod
        def serialize(
            cls,
            instance: Image.Image,
            target: ImageFile,
            realm: Any,
            annotation: TypeNode[Image.Image] | None = None,
        ) -> None:
            """Convert an image to a file."""
            # The encoder comes from the target format, never from the source
            # image: a JPEG loaded from disk must still be written as PNG when
            # the target is a PNG, or the payload contradicts the declared loss.
            image_format = _image_format_for_mime(target.default_mime)
            if image_format is None:
                raise ValueError(f"Unsupported image MIME type: {target.default_mime}")

            with target.open("w") as f:
                instance.save(f, format=image_format)

    common_serializers[Image.Image] = ImageTransporter

except ImportError:
    # PIL is not installed, so we won't register any converters for images
    pass


# BeautifulSoup HTML, if BeautifulSoup is installed
try:
    from bs4 import BeautifulSoup

    type HTMLFile = File[Literal["text/html"], TextIO]

    HTMLF = TypeVar("HTMLF", bound=HTMLFile, default=HTMLFile)

    class HTMLTransporter(Serializer[BeautifulSoup, HTMLFile]):
        """Transporter for BeautifulSoup HTML."""

        @classmethod
        def estimate_serialized_size(
            cls,
            instance: BeautifulSoup,
            target: HTMLFile,
            realm: Any,
            annotation: TypeNode[BeautifulSoup] | None = None,
        ) -> int:
            """Measure the serialized soup without writing it."""
            return _encoded_size(str(instance))

        @classmethod
        def serialize(
            cls: type[Serializer[V2, HTMLFile]],
            source: HTMLFile,
            realm: Any,
            annotation: TypeNode[V2] | None = None,
        ) -> V2:
            """Parse HTML from a file."""
            with source.open("r") as f:
                html_content = f.read()
                soup = BeautifulSoup(html_content, "html.parser")

            return check_type_and_cast(soup, annotation)

        @classmethod
        def serialize(
            cls,
            instance: BeautifulSoup,
            target: HTMLFile,
            realm: Any,
            annotation: TypeNode[BeautifulSoup] | None = None,
        ) -> None:
            """Convert BeautifulSoup object to an HTML file."""
            with target.open("w") as f:
                f.write(str(instance))

    common_serializers[BeautifulSoup] = HTMLTransporter

except ImportError:
    # BeautifulSoup is not installed, so we won't register any converters for HTML
    pass


# Pickle files

type PickleFile = File[Literal["application/pickle"], BinaryIO]


class PickleTransporter(Serializer[Any, PickleFile]):
    """Transporter for Pickle files."""

    @classmethod
    def __transport_scopes__(cls) -> frozenset[type[TransportScope]]:
        """Pickles are machine-specific and unsafe to load from the network."""
        return frozenset({Open, IPC})

    @classmethod
    def __lossless_formats__(cls) -> frozenset[str]:
        """Pickle reproduces the object exactly."""
        return frozenset({"application/pickle"})

    @classmethod
    def _require_supported_scope(cls) -> None:
        """Refuse payloads that must cross a process or network boundary."""
        scope = current_ser_ctx.get()
        if scope.matches(cls.__transport_scopes__()):
            return

        raise ValueError(
            f"Refusing to transport pickle data under transport scope {scope!r}; "
            "pickle is only supported for Open and IPC"
        )

    @classmethod
    def estimate_serialized_size(
        cls,
        instance: Any,
        target: PickleFile,
        realm: Any,
        annotation: TypeNode[Any] | None = None,
    ) -> int:
        """Measure the pickle payload without writing it."""
        return len(pickle.dumps(instance))

    @classmethod
    def serialize(
        cls: type[Serializer[V2, PickleFile]],
        source: PickleFile,
        realm: Any,
        annotation: TypeNode[V2] | None = None,
    ) -> V2:
        """Parse an object from a Pickle file."""
        # Unpickling executes arbitrary code, so it is deliberately restricted to
        # the Open and IPC scopes (see ``__transport_scopes__``) and re-checked
        # here for callers that bypass driver selection.
        PickleTransporter._require_supported_scope()

        with source.open("r") as f:
            obj = pickle.load(f)

        return check_type_and_cast(obj, annotation)

    @classmethod
    def serialize(
        cls,
        instance: Any,
        target: PickleFile,
        realm: Any,
        annotation: TypeNode[Any] | None = None,
    ) -> None:
        """Convert an object to a Pickle file."""
        PickleTransporter._require_supported_scope()

        with target.open("w") as f:
            pickle.dump(instance, f)


common_serializers[object] = PickleTransporter
