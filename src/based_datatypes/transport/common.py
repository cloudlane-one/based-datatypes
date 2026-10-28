"""Common transport drivers for common types."""

from __future__ import annotations

import json
import pickle
import xml.etree.ElementTree as xml  # noqa: N813
from collections.abc import Iterable
from types import UnionType
from typing import Any, BinaryIO, Literal, TextIO, TypeVar, cast

import polars as pl
import sqlparse
import yaml

from based_datatypes.files import File
from based_datatypes.typing import SingleTypeDef, has_type

from .base import V2, TransportDriver

common_drivers: dict[SingleTypeDef | UnionType, type[TransportDriver]] = {}


# Simple, stringifiable types

type TextFile = File[Literal["text/plain", "text/csv"], TextIO]
TF = TypeVar(
    "TF",
    bound=TextFile,
    default=TextFile,
)

type StrScalar = str | int | float | complex | bool | None
type Stringifiable = StrScalar | list[StrScalar]


type JSONFile = (
    File[Literal["application/yaml"], TextIO] | File[Literal["application/json"], TextIO]
)
JF = TypeVar(
    "JF",
    bound=JSONFile,
    default=JSONFile,
)

# JSON / YAML dict

type JSONScalar = str | int | float | bool | None
type JSONDict = dict[str, JSONDict | list[JSONScalar]]

J = TypeVar("J", bound=JSONDict, default=JSONDict)


def check_type_and_cast[T](obj: Any, annotation: SingleTypeDef[T] | None = None) -> T:
    """Check if the object matches the annotation type and cast it."""
    if annotation is not None and not has_type(obj, annotation):
        raise TypeError(f"Object does not match expected type: {annotation}")
    return cast(T, obj)


class JSONTransporter(TransportDriver[JSONDict, JSONFile]):
    """Transporter for JSON and YAML dictionaries."""

    @classmethod
    def import_object(
        cls: type[TransportDriver[V2, JSONFile]],
        source: JSONFile,
        realm: Any,
        annotation: SingleTypeDef[V2] | None = None,
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
        annotation: SingleTypeDef[JSONDict] | None = None,
    ) -> None:
        """Convert a dictionary to a JSON or YAML file."""
        with target.open("w") as f:
            if target.mime == "application/yaml":
                yaml.safe_dump(instance, allow_unicode=True, stream=f)
            else:
                f.write(json.dumps(instance, indent=2))


common_drivers[JSONDict] = JSONTransporter


# XML etree

XMLFile = File[Literal["text/xml", "application/xml"], TextIO]

XMLF = TypeVar(
    "XMLF",
    bound=XMLFile,
    default=XMLFile,
)


class XMLTransporter(TransportDriver[xml.ElementTree, XMLFile]):
    """Transporter for XML ElementTree."""

    @classmethod
    def import_object(
        cls: type[TransportDriver[V2, XMLFile]],
        source: XMLFile,
        realm: Any,
        annotation: SingleTypeDef[V2] | None = None,
    ) -> V2:
        """Parse an XML ElementTree from a file."""
        if source.mime not in {"application/xml", "text/xml"}:
            raise ValueError(f"Unsupported MIME type: {source.mime}")

        with source.open("r") as f:
            tree = xml.parse(f)

        return check_type_and_cast(tree, annotation)

    @classmethod
    def export_object(
        cls,
        instance: xml.ElementTree,
        target: XMLFile,
        realm: Any,
        annotation: SingleTypeDef[xml.ElementTree] | None = None,
    ) -> None:
        """Convert an XML ElementTree to a file."""
        assert target.mime in {"application/xml", "text/xml"}
        with target.open("w") as f:
            instance.write(f, encoding="unicode")


common_drivers[xml.ElementTree] = XMLTransporter


# Pydantic models
try:
    import pydantic as pyd

    class PydTransporter(TransportDriver[pyd.BaseModel, JSONFile]):
        """Transporter for Pydantic models."""

        @classmethod
        def import_object(
            cls: type[TransportDriver[V2, JSONFile]],
            source: JSONFile,
            realm: Any,
            annotation: SingleTypeDef[V2] | None = None,
        ) -> V2:
            """Parse a Pydantic model from a JSON or YAML file."""
            with source.open("r") as f:
                if source.mime == "application/yaml":
                    data = yaml.load(f, Loader=yaml.CSafeLoader)
                else:
                    data = json.load(f)

            if annotation is None:
                raise ValueError("Annotation must be provided for Pydantic model parsing.")

            adapter = pyd.TypeAdapter(annotation)
            return adapter.validate_python(data)

        @classmethod
        def export_object(
            cls,
            instance: pyd.BaseModel,
            target: JSONFile,
            realm: Any,
            annotation: SingleTypeDef[pyd.BaseModel] | None = None,
        ) -> None:
            """Convert a Pydantic model to a JSON or YAML file."""
            with target.open("w") as f:
                if target.mime == "application/yaml":
                    yaml.safe_dump(instance.model_dump(), allow_unicode=True, stream=f)
                else:
                    json.dump(instance.model_dump(), f, indent=2)

    common_drivers[pyd.BaseModel] = PydTransporter
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

    class PdTransporter(TransportDriver[pd.DataFrame, DfFile]):
        """Transporter for Pandas DataFrames."""

        @classmethod
        def import_object(
            cls: type[TransportDriver[V2, DfFile]],
            source: DfFile,
            realm: Any,
            annotation: SingleTypeDef[V2] | None = None,
        ) -> V2:
            """Parse a DataFrame from a CSV or Parquet file."""
            with source.open("r") as f:
                if source.mime == "text/csv":
                    res = pd.read_csv(f)
                elif source.mime == "application/vnd.apache.parquet":
                    f = cast(BinaryIO, f)
                    res = pd.read_parquet(f)
                elif source.mime == "application/vnd.ms-excel":
                    res = pd.read_excel(f)
                else:
                    raise ValueError(f"Unsupported MIME type: {source.mime}")

            return check_type_and_cast(res, annotation)

        @classmethod
        def export_object(
            cls,
            instance: pd.DataFrame,
            target: DfFile,
            realm: Any,
            annotation: SingleTypeDef[pd.DataFrame] | None = None,
        ) -> None:
            """Convert a DataFrame to a CSV or Parquet file."""
            with target.open("w") as f:
                if target.mime == "text/csv":
                    instance.to_csv(f, index=False)
                elif target.mime == "application/vnd.apache.parquet":
                    f = cast(BinaryIO, f)
                    instance.to_parquet(f)
                elif target.mime == "application/vnd.ms-excel":
                    f = cast(BinaryIO, f)
                    instance.to_excel(f, index=False)
                else:
                    raise ValueError(f"Unsupported MIME type: {target.mime}")

    common_drivers[pd.DataFrame] = PdTransporter
except ImportError:
    pass

# Polars


class PlTransporter(TransportDriver[pl.DataFrame, DfFile]):
    """Transporter for Polars DataFrames."""

    @classmethod
    def import_object(
        cls: type[TransportDriver[V2, DfFile]],
        source: DfFile,
        realm: Any,
        annotation: SingleTypeDef[V2] | None = None,
    ) -> V2:
        """Parse a Polars DataFrame from a CSV or Parquet file."""
        with source.open("r") as f:
            if source.mime == "text/csv":
                res = pl.read_csv(f)
            elif source.mime == "application/vnd.apache.parquet":
                f = cast(BinaryIO, f)
                res = pl.read_parquet(f)
            elif source.mime == "application/vnd.ms-excel":
                f = cast(BinaryIO, f)
                res = pl.read_excel(f)
            else:
                raise ValueError(f"Unsupported MIME type: {source.mime}")

        return check_type_and_cast(res, annotation)

    @classmethod
    def export_object(
        cls,
        instance: pl.DataFrame,
        target: DfFile,
        realm: Any,
        annotation: SingleTypeDef[pl.DataFrame] | None = None,
    ) -> None:
        """Convert a Polars DataFrame to a CSV or Parquet file."""
        with target.open("w") as f:
            if target.mime == "text/csv":
                instance.write_csv(f)
            elif target.mime == "application/vnd.apache.parquet":
                f = cast(BinaryIO, f)
                instance.write_parquet(f)
            elif target.mime == "application/vnd.ms-excel":
                f = cast(BinaryIO, f)
                instance.write_excel(f)
            else:
                raise ValueError(f"Unsupported MIME type: {target.mime}")


common_drivers[pl.DataFrame] = PlTransporter


# NumPy arrays
try:
    import numpy as np

    type NpFile = (
        File[Literal["application/x-npy", "application/x-npz"], BinaryIO]
        | File[Literal["text/csv"], TextIO]
    )

    class NpTransporter(TransportDriver[np.ndarray, NpFile]):
        """Transporter for NumPy arrays."""

        @classmethod
        def import_object(
            cls: type[TransportDriver[V2, NpFile]],
            source: NpFile,
            realm: Any,
            annotation: SingleTypeDef[V2] | None = None,
        ) -> V2:
            """Parse a NumPy array from a file."""
            with source.open("r") as f:
                if source.mime == "application/x-npy":
                    f = cast(BinaryIO, f)
                    res = np.load(f)
                elif source.mime == "application/x-npz":
                    f = cast(BinaryIO, f)
                    res = np.load(f)["arr_0"]
                elif source.mime == "text/csv":
                    res = np.loadtxt(f, delimiter=",")
                else:
                    raise ValueError(f"Unsupported MIME type: {source.mime}")

            return check_type_and_cast(res, annotation)

        @classmethod
        def export_object(
            cls,
            instance: np.ndarray,
            target: NpFile,
            realm: Any,
            annotation: SingleTypeDef[np.ndarray] | None = None,
        ) -> None:
            """Convert a NumPy array to a file."""
            with target.open("w") as f:
                if target.mime == "application/x-npy":
                    f = cast(BinaryIO, f)
                    np.save(f, instance)
                elif target.mime == "application/x-npz":
                    f = cast(BinaryIO, f)
                    np.savez(f, arr_0=instance)
                elif target.mime == "text/csv":
                    np.savetxt(f, instance, delimiter=",")
                else:
                    raise ValueError(f"Unsupported MIME type: {target.mime}")

    common_drivers[np.ndarray] = NpTransporter
except ImportError:
    pass

# SQL statements
try:
    import sqlalchemy as sqla  # pyright: ignore[reportMissingImports]

    type SQLFile = File[Literal["application/sql"], TextIO]

    class SQLTransporter(TransportDriver[sqla.Executable | Iterable[sqla.Executable], SQLFile]):
        """Transporter for SQL statements."""

        @classmethod
        def import_object(
            cls: type[TransportDriver[V2, SQLFile]],
            source: SQLFile,
            realm: Any,
            annotation: SingleTypeDef[V2] | None = None,
        ) -> V2:
            """Parse an SQL statement from a file."""
            with source.open("r") as f:
                sql_content = f.read()
                parsed_sql = sqlparse.parse(sql_content)

            stmts = [sqla.text(str(stmt)) for stmt in parsed_sql]

            return check_type_and_cast(stmts[0] if len(stmts) == 1 else stmts, annotation)

        @classmethod
        def export_object(
            cls,
            instance: sqla.Executable | Iterable[sqla.Executable],
            target: SQLFile,
            realm: Any,
            annotation: SingleTypeDef[sqla.Executable | Iterable[sqla.Executable]] | None = None,
        ) -> None:
            """Convert an SQL statement to a file."""
            if isinstance(instance, sqla.Executable):
                instance = [instance]

            formatted = [
                sqlparse.format(str(stmt), reindent=True, keyword_case="upper") for stmt in instance
            ]

            with target.open("w") as f:
                f.write(";\n\n".join(formatted))

    common_drivers[sqla.Executable | Iterable[sqla.Executable]] = SQLTransporter
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

    class ImageTransporter(TransportDriver[Image.Image, ImageFile]):
        """Transporter for PIL Images."""

        @classmethod
        def import_object(
            cls: type[TransportDriver[V2, ImageFile]],
            source: ImageFile,
            realm: Any,
            annotation: SingleTypeDef[V2] | None = None,
        ) -> V2:
            """Parse an image from a file."""
            with source.open("r") as f:
                img = Image.open(f)
                img.load()

            return check_type_and_cast(img, annotation)

        @classmethod
        def export_object(
            cls,
            instance: Image.Image,
            target: ImageFile,
            realm: Any,
            annotation: SingleTypeDef[Image.Image] | None = None,
        ) -> None:
            """Convert an image to a file."""
            image_format = instance.format or Image.registered_extensions().get(
                target.extension.lower()
            )
            if image_format is None:
                raise ValueError(f"Unsupported image MIME type: {target.mime}")

            with target.open("w") as f:
                instance.save(f, format=image_format)

    common_drivers[Image.Image] = ImageTransporter

except ImportError:
    # PIL is not installed, so we won't register any converters for images
    pass


# BeautifulSoup HTML, if BeautifulSoup is installed
try:
    from bs4 import BeautifulSoup

    type HTMLFile = File[Literal["text/html"], TextIO]

    HTMLF = TypeVar("HTMLF", bound=HTMLFile, default=HTMLFile)

    class HTMLTransporter(TransportDriver[BeautifulSoup, HTMLFile]):
        """Transporter for BeautifulSoup HTML."""

        @classmethod
        def import_object(
            cls: type[TransportDriver[V2, HTMLFile]],
            source: HTMLFile,
            realm: Any,
            annotation: SingleTypeDef[V2] | None = None,
        ) -> V2:
            """Parse HTML from a file."""
            with source.open("r") as f:
                html_content = f.read()
                soup = BeautifulSoup(html_content, "html.parser")

            return check_type_and_cast(soup, annotation)

        @classmethod
        def export_object(
            cls,
            instance: BeautifulSoup,
            target: HTMLFile,
            realm: Any,
            annotation: SingleTypeDef[BeautifulSoup] | None = None,
        ) -> None:
            """Convert BeautifulSoup object to an HTML file."""
            with target.open("w") as f:
                f.write(str(instance))

    common_drivers[BeautifulSoup] = HTMLTransporter

except ImportError:
    # BeautifulSoup is not installed, so we won't register any converters for HTML
    pass


# Pickle files

type PickleFile = File[Literal["application/pickle"], BinaryIO]


class PickleTransporter(TransportDriver[Any, PickleFile]):
    """Transporter for Pickle files."""

    @classmethod
    def import_object(
        cls: type[TransportDriver[V2, PickleFile]],
        source: PickleFile,
        realm: Any,
        annotation: SingleTypeDef[V2] | None = None,
    ) -> V2:
        """Parse an object from a Pickle file."""
        with source.open("r") as f:
            obj = pickle.load(f)

        return check_type_and_cast(obj, annotation)

    @classmethod
    def export_object(
        cls,
        instance: Any,
        target: PickleFile,
        realm: Any,
        annotation: SingleTypeDef[Any] | None = None,
    ) -> None:
        """Convert an object to a Pickle file."""
        with target.open("w") as f:
            pickle.dump(instance, f)


common_drivers[object] = PickleTransporter
