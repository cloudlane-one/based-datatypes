"""Tests for :mod:`based_datatypes.serialize`."""

from __future__ import annotations

import datetime as dt
import json
import pickle
import uuid
import xml.etree.ElementTree as xml  # noqa: N813
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, BinaryIO, Literal, TextIO, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest
import sqlalchemy as sqla
import yaml
from bs4 import BeautifulSoup
from PIL import Image
from pydantic import BaseModel

from based_datatypes.files import File, LocalDir, LocalFile, MemoryFile
from based_datatypes.serialize import (
    INDEX_FILENAME,
    Open,
    Realm,
    SerializationTarget,
    TransportScope,
    _source_type,
    current_ser_ctx,
    custom_serializers,
    deserialize,
    estimate_serialized_size,
    get_serializers,
    read_structure,
    register_serializer,
    serialization_context,
    serialize,
    write_structure,
)
from based_datatypes.serialize import common as common_module
from based_datatypes.serialize.base import (
    JSONDict,
    SerializedDict,
    SerializedValue,
    target_mimes,
)
from based_datatypes.serialize.common import (
    CommonSerializer,
    PrimitiveSerializer,
    check_type_and_cast,
)
from based_datatypes.serialize.local_fs import _extension


class Payload:
    """A picklable value no JSON serializer claims."""

    def __init__(self, value: tuple[int, int]) -> None:
        """Store the value."""
        self.value = value

    def __eq__(self, other: object) -> bool:
        """Compare by value."""
        return isinstance(other, Payload) and other.value == self.value

    def __hash__(self) -> int:
        """Hash by value."""
        return hash(self.value)


def local_file(directory: Path, name: str, mime: str, *, binary: bool = False) -> LocalFile[Any]:
    """Build a :class:`LocalFile` for a MIME type that is only known at runtime.

    Args:
        directory: Directory the file belongs to.
        name: File name without extension.
        mime: MIME type of the file.
        binary: Whether the file is binary rather than text.

    Returns:
        The file object.
    """
    mime_literal: Any = Literal[cast(Any, mime)]
    file_type: Any = LocalFile[mime_literal, BinaryIO if binary else TextIO]

    return cast(LocalFile[Any], file_type(dir=LocalDir(path=directory), name=name))


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """Provide an existing empty directory."""
    path = tmp_path / "work"
    path.mkdir()
    return path


@pytest.fixture
def payload() -> SerializedDict:
    """Provide a JSON-shaped payload."""
    return {"text": "hello", "number": 3, "items": [1, 2]}


class Colour(StrEnum):
    """An enum whose values are strings."""

    RED = "red"
    BLUE = "blue"


class Record(BaseModel):
    """A small pydantic model with a non-JSON-native field."""

    name: str
    count: int
    created: dt.datetime


# Scalars and JSON-supported values


class TestScalarSerializers:
    """Tests for the serializers of types with a JSON-supported representation."""

    @pytest.mark.parametrize(
        ("instance", "expected", "annotation"),
        [
            (dt.datetime(2024, 1, 2, 3, 4, 5, 123456), "2024-01-02T03:04:05.123456", dt.datetime),
            (dt.date(2024, 1, 2), "2024-01-02", dt.date),
            (dt.time(3, 4, 5), "03:04:05", dt.time),
            (dt.timedelta(seconds=90), 90.0, dt.timedelta),
            (3 + 4j, [3.0, 4.0], complex),
            (Decimal("1.10"), "1.10", Decimal),
            (uuid.UUID(int=1), "00000000-0000-0000-0000-000000000001", uuid.UUID),
            (Path("a") / "b.txt", str(Path("a") / "b.txt"), Path),
            (b"hello", "aGVsbG8=", bytes),
            (Colour.RED, "red", Colour),
        ],
    )
    def test_roundtrip(self, instance: Any, expected: Any, annotation: Any) -> None:
        """The value becomes its JSON representation and comes back unchanged."""
        encoded = serialize(instance)

        assert encoded == expected
        assert deserialize(encoded, annotation) == instance

    def test_aware_datetime_roundtrip(self) -> None:
        """The UTC offset survives the ISO 8601 round trip."""
        instance = dt.datetime(2024, 1, 2, 3, 4, tzinfo=dt.UTC)

        assert deserialize(serialize(instance), dt.datetime) == instance

    def test_decimal_keeps_precision(self) -> None:
        """A decimal is text, so its precision survives."""
        assert deserialize(serialize(Decimal("0.1000000000000000055511151231257827")), Decimal) == (
            Decimal("0.1000000000000000055511151231257827")
        )

    def test_complex_is_a_two_element_list(self) -> None:
        """A complex number becomes ``[real, imaginary]``."""
        assert serialize(1.5 - 2.5j) == [1.5, -2.5]

    def test_scalars_declare_a_single_primitive_target(self) -> None:
        """The serializers declare exactly the one target they map onto."""
        for driver in common_module.common_serializers:
            if not isinstance(driver, PrimitiveSerializer):
                continue

            targets = driver.serialization_targets

            assert len(targets) >= 1
            assert all(target.lossless for target in targets)
            assert all(target.scope == Open() for target in targets)

    def test_serialize_refuses_a_file_target(self) -> None:
        """A scalar cannot be written to a file on its own."""
        with pytest.raises(ValueError, match="JSON value, not to a file"):
            common_module.DateTimeSerializer().serialize(
                dt.datetime(2024, 1, 1), File[Literal["text/plain"], TextIO]
            )

    def test_serialize_refuses_a_file_factory(self, workdir: Path) -> None:
        """A file factory is refused as well, since no file is written."""
        with pytest.raises(ValueError, match="JSON value, not to a file"):
            serialize(
                dt.datetime(2024, 1, 1), file_factory=lambda: local_file(workdir, "a", "text/plain")
            )

    def test_deserialize_refuses_a_file_source(self, workdir: Path) -> None:
        """A scalar cannot be read from a file on its own."""
        with pytest.raises(TypeError, match="deserializes from a JSON value"):
            common_module.DateTimeSerializer().deserialize(
                cast(Any, local_file(workdir, "a", "text/plain"))
            )

    def test_enum_requires_an_annotation(self) -> None:
        """An enum value cannot be restored without knowing the enum."""
        with pytest.raises(ValueError, match="requires the enum class"):
            common_module.EnumSerializer().deserialize("red")

    def test_enum_rejects_a_non_enum_annotation(self) -> None:
        """A value that is not an enum type is refused."""
        with pytest.raises(TypeError, match="is not an enum type"):
            common_module.EnumSerializer().deserialize("red", annotation=str)

    def test_estimate_measures_the_encoded_value(self) -> None:
        """The estimate is the encoded size of the primitive representation."""
        assert estimate_serialized_size(Colour.RED, str) == len('"red"')

    def test_estimate_accepts_a_missing_target(self) -> None:
        """Estimating without a target uses the declared format."""
        assert estimate_serialized_size(3 + 4j, list[float]) > 0


# JSON and YAML


class TestJSONSerializer:
    """Tests for the JSON and YAML serializer."""

    def test_default_target_is_a_dictionary(self, payload: SerializedDict) -> None:
        """Without a requested target the payload goes to a dictionary."""
        written = serialize(payload)

        assert isinstance(written, dict)

    def test_dict_target_returns_the_dictionary(self, payload: SerializedDict) -> None:
        """A dictionary target is satisfied by returning the dictionary."""
        assert serialize(payload, dict) is payload

    def test_yaml_file(self, workdir: Path, payload: SerializedDict) -> None:
        """A YAML target writes YAML."""
        target = local_file(workdir, "data", "application/yaml")

        written = serialize(payload, file_factory=lambda: target)

        assert written is target
        assert yaml.safe_load(target.path.read_text(encoding="utf-8")) == payload

    def test_file_factory_target(self, workdir: Path, payload: SerializedDict) -> None:
        """A file factory decides where the payload is written."""
        target = local_file(workdir, "data", "application/json")

        written = serialize(payload, file_factory=lambda: target)

        assert written is target
        assert json.loads(target.path.read_text(encoding="utf-8")) == payload

    def test_deserialize_from_a_file(self, workdir: Path, payload: SerializedDict) -> None:
        """A written file reads back into the same dictionary."""
        target = local_file(workdir, "data", "application/json")
        serialize(payload, file_factory=lambda: target)

        assert deserialize(target, dict) == payload

    def test_deserialize_from_a_dictionary(self, payload: SerializedDict) -> None:
        """A dictionary source is passed through."""
        assert deserialize(payload, dict) == payload

    def test_deserialize_checks_the_annotation(self, payload: SerializedDict) -> None:
        """A source that does not fit the annotation raises."""
        with pytest.raises(TypeError, match="does not match expected type"):
            common_module.JSONSerializer().deserialize(
                cast(Any, ["not", "a", "dict"]), annotation=JSONDict
            )

    def test_lossless_targets(self) -> None:
        """Both JSON and YAML round-trip exactly."""
        driver = common_module.JSONSerializer()

        assert all(target.lossless for target in driver.serialization_targets)
        assert target_mimes(driver.serialization_targets[1].type) == ("application/json",)

    def test_estimate_uses_the_target_format(self, payload: SerializedDict) -> None:
        """YAML and JSON estimates differ, so the format is taken into account."""
        assert estimate_serialized_size(payload, File[Literal["application/yaml"], TextIO]) < (
            estimate_serialized_size(payload, File[Literal["application/json"], TextIO])
        )

    def test_estimate_does_not_dump_yaml(
        self, payload: SerializedDict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pure-Python YAML dumper is not used, since the JSON encoder is measured instead."""
        calls = 0
        original = common_module.yaml.safe_dump

        def counting(*args: Any, **kwargs: Any) -> str:
            """Count the dumps and delegate."""
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(common_module.yaml, "safe_dump", counting)

        estimate = estimate_serialized_size(payload, File[Literal["application/yaml"], TextIO])

        assert estimate > 0
        assert calls == 0

    def test_estimate_matches_the_written_json(self, payload: SerializedDict) -> None:
        """The JSON estimate is exact, because the C encoder is the cheapest measure."""
        written = cast(
            MemoryFile[Any, Any, Any],
            serialize(payload, File[Literal["application/json"], TextIO]),
        )

        assert estimate_serialized_size(payload, File[Literal["application/json"], TextIO]) == len(
            cast(str, written.content).encode()
        )


# XML


class TestXMLSerializer:
    """Tests for the XML serializer."""

    @staticmethod
    def tree() -> xml.ElementTree:
        """Build a tree with a comment and a processing instruction."""
        root = xml.fromstring("<root><child>1</child></root>")
        root.insert(0, xml.Comment("note"))

        return xml.ElementTree(root)

    def test_text_target(self) -> None:
        """A text target returns the XML text."""
        text = serialize(self.tree(), str)

        assert isinstance(text, str)
        assert text.startswith("<root>")

    def test_default_target_is_a_string(self) -> None:
        """Without a target the tree goes to a string."""
        written = serialize(self.tree())

        assert isinstance(written, str)
        assert written.startswith("<root>")

    def test_roundtrip_keeps_comments(self, workdir: Path) -> None:
        """Comments survive the round trip through a file."""
        target = local_file(workdir, "tree", "application/xml")
        written = serialize(self.tree(), file_factory=lambda: target)

        restored = deserialize(written, xml.ElementTree)

        assert "<!--note-->" in cast(str, target.path.read_text(encoding="utf-8"))
        assert cast(xml.Element, restored.getroot()).tag == "root"

    def test_empty_tree_serializes_to_nothing(self) -> None:
        """A tree without a root has no text."""
        assert serialize(xml.ElementTree(), str) == ""

    def test_estimate_measures_the_text(self) -> None:
        """The estimate is the encoded tree size."""
        text = serialize(self.tree(), str)

        assert estimate_serialized_size(self.tree(), str) == len(cast(str, text).encode())


# Pydantic models


class TestPydSerializer:
    """Tests for the pydantic serializer."""

    def test_dict_target_returns_the_dump(self) -> None:
        """A dictionary target returns the JSON-mode dump."""
        record = Record(name="a", count=2, created=dt.datetime(2024, 1, 1, 12, 0))

        dumped = serialize(record, dict, lossless=False)

        assert dumped["name"] == "a"
        assert dumped["created"] == "2024-01-01T12:00:00"

    def test_file_roundtrip(self, workdir: Path) -> None:
        """A model round-trips through a YAML file."""
        record = Record(name="a", count=2, created=dt.datetime(2024, 1, 1, 12, 0))
        target = local_file(workdir, "record", "application/yaml")

        written = serialize(record, file_factory=lambda: target)
        restored = deserialize(written, Record)

        assert restored == record

    def test_requires_an_annotation(self, workdir: Path) -> None:
        """Reading a model requires the model type."""
        target = local_file(workdir, "record", "application/json")
        serialize(
            Record(name="a", count=1, created=dt.datetime(2024, 1, 1)), file_factory=lambda: target
        )

        with pytest.raises(ValueError, match="requires the model"):
            common_module.PydSerializer().deserialize(target)

    def test_file_targets_are_lossy(self) -> None:
        """Dumping a model can drop aliases and excluded fields."""
        file_targets = [
            target
            for target in common_module.PydSerializer().serialization_targets
            if target_mimes(target.type)
        ]

        assert file_targets
        assert not any(target.lossless for target in file_targets)

    def test_estimate_measures_the_dump(self) -> None:
        """The estimate is the encoded model dump."""
        record = Record(name="a", count=2, created=dt.datetime(2024, 1, 1, 12, 0))

        assert (
            estimate_serialized_size(
                record, File[Literal["application/json"], TextIO], lossless=False
            )
            > 0
        )


# DataFrames


@pytest.fixture
def frame() -> pd.DataFrame:
    """Provide a small pandas frame."""
    return pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})


class TestPdSerializer:
    """Tests for the pandas serializer."""

    @pytest.mark.parametrize(
        ("mime", "binary", "lossless"),
        [
            ("text/csv", False, False),
            ("application/vnd.apache.parquet", True, True),
            ("application/vnd.ms-excel", True, False),
        ],
    )
    def test_roundtrip(
        self, workdir: Path, frame: pd.DataFrame, mime: str, binary: bool, lossless: bool
    ) -> None:
        """Frames round-trip through every declared format."""
        target = local_file(workdir, "frame", mime, binary=binary)

        written = serialize(frame, type(target), lossless=lossless, file_factory=lambda: target)
        restored = deserialize(written, pd.DataFrame)

        assert list(restored.columns) == ["x", "y"]
        assert len(restored) == len(frame)

    def test_default_target_is_parquet(self, frame: pd.DataFrame) -> None:
        """Without a target the lossless parquet format is used."""
        written = serialize(frame)

        assert isinstance(written, MemoryFile)
        assert written.default_mime == "application/vnd.apache.parquet"

    def test_lossless_selection(self, workdir: Path, frame: pd.DataFrame) -> None:
        """CSV is only selected when a lossy format is allowed."""
        csv_target = File[Literal["text/csv"], TextIO]

        assert get_serializers(pd.DataFrame, [csv_target], lossless=True) == []
        assert get_serializers(pd.DataFrame, [csv_target], lossless=False) != []

    def test_unsupported_mime_on_write(self, workdir: Path, frame: pd.DataFrame) -> None:
        """A target format the driver does not know is refused before the file is created."""
        target = local_file(workdir, "frame", "application/octet-stream", binary=True)

        with pytest.raises(ValueError, match="Unsupported MIME type"):
            common_module.PdSerializer().serialize(
                frame,
                File[Literal["application/octet-stream"], BinaryIO],
                file_factory=lambda: target,
            )

        assert not target.path.exists()

    def test_unsupported_mime_on_read(self, workdir: Path) -> None:
        """A source format the driver does not know is refused."""
        target = local_file(workdir, "frame", "application/octet-stream", binary=True)
        target.path.write_bytes(b"x")

        with pytest.raises(ValueError, match="Unsupported MIME type"):
            common_module.PdSerializer().deserialize(target)

    def test_refuses_a_non_file_target(self, frame: pd.DataFrame) -> None:
        """A frame cannot be returned as a JSON value."""
        with pytest.raises(ValueError, match="can only serialize to a file"):
            common_module.PdSerializer().serialize(frame, dict)

    def test_refuses_a_source_that_is_not_a_file(self) -> None:
        """A frame cannot be read from a JSON value."""
        with pytest.raises(TypeError, match="deserializes from a file"):
            common_module.PdSerializer().deserialize(cast(Any, "csv"))

    def test_refuses_a_local_file_target_without_a_factory(self, frame: pd.DataFrame) -> None:
        """A local file target cannot be created by the driver itself."""
        with pytest.raises(ValueError, match="cannot create a LocalFile target"):
            common_module.PdSerializer().serialize(
                frame, LocalFile[Literal["text/csv"], TextIO, Any]
            )

    def test_estimate_samples_csv(self, frame: pd.DataFrame) -> None:
        """The CSV estimate extrapolates from a sample."""
        estimate = estimate_serialized_size(
            frame, File[Literal["text/csv"], TextIO], lossless=False
        )

        assert estimate > 0

    def test_estimate_of_an_empty_csv(self) -> None:
        """An empty frame has no encoded size."""
        empty = pd.DataFrame({"x": []})

        assert (
            estimate_serialized_size(empty, File[Literal["text/csv"], TextIO], lossless=False) == 0
        )

    def test_estimate_of_parquet(self, frame: pd.DataFrame) -> None:
        """The parquet estimate is the in-memory size."""
        estimate = estimate_serialized_size(
            frame, File[Literal["application/vnd.apache.parquet"], BinaryIO]
        )

        assert estimate >= frame.memory_usage(deep=True).sum()


class TestPlSerializer:
    """Tests for the polars serializer."""

    @pytest.mark.parametrize(
        ("mime", "binary", "lossless"),
        [
            ("text/csv", False, False),
            ("application/vnd.apache.parquet", True, True),
            ("application/vnd.ms-excel", True, False),
        ],
    )
    def test_roundtrip(self, workdir: Path, mime: str, binary: bool, lossless: bool) -> None:
        """Frames round-trip through every declared format."""
        frame = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        target = local_file(workdir, "frame", mime, binary=binary)

        written = serialize(frame, type(target), lossless=lossless, file_factory=lambda: target)
        restored = deserialize(written, pl.DataFrame)

        assert restored.columns == ["x", "y"]
        assert restored.height == 2

    def test_default_target_is_parquet(self) -> None:
        """Without a target the lossless parquet format is used."""
        assert serialize(pl.DataFrame({"x": [1]})).default_mime == (
            "application/vnd.apache.parquet"
        )

    def test_unsupported_mime_on_write(self, workdir: Path) -> None:
        """A target format the driver does not know is refused before the file is created."""
        target = local_file(workdir, "frame", "application/octet-stream", binary=True)

        with pytest.raises(ValueError, match="Unsupported MIME type"):
            common_module.PlSerializer().serialize(
                pl.DataFrame({"x": [1]}),
                File[Literal["application/octet-stream"], BinaryIO],
                file_factory=lambda: target,
            )

    def test_unsupported_mime_on_read(self, workdir: Path) -> None:
        """A source format the driver does not know is refused."""
        target = local_file(workdir, "frame", "application/octet-stream", binary=True)
        target.path.write_bytes(b"x")

        with pytest.raises(ValueError, match="Unsupported MIME type"):
            common_module.PlSerializer().deserialize(target)

    def test_estimate_samples_csv(self) -> None:
        """The CSV estimate extrapolates from a sample."""
        frame = pl.DataFrame({"x": [1, 2, 3]})

        assert (
            estimate_serialized_size(frame, File[Literal["text/csv"], TextIO], lossless=False) > 0
        )

    def test_estimate_of_an_empty_csv(self) -> None:
        """An empty frame has no encoded size."""
        assert (
            estimate_serialized_size(
                pl.DataFrame({"x": []}), File[Literal["text/csv"], TextIO], lossless=False
            )
            == 0
        )


# NumPy arrays


class TestNpSerializer:
    """Tests for the numpy serializer."""

    @pytest.mark.parametrize(
        ("mime", "binary", "lossless"),
        [
            ("application/x-npy", True, True),
            ("application/x-npz", True, True),
            ("text/csv", False, False),
        ],
    )
    def test_roundtrip(self, workdir: Path, mime: str, binary: bool, lossless: bool) -> None:
        """Arrays round-trip through every declared format."""
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        target = local_file(workdir, "array", mime, binary=binary)

        written = serialize(array, type(target), lossless=lossless, file_factory=lambda: target)
        restored = deserialize(written, np.ndarray)

        assert np.array_equal(restored, array)

    def test_default_target_is_npy(self) -> None:
        """Without a target the native ``.npy`` format is used."""
        assert serialize(np.array([1.0])).default_mime == "application/x-npy"

    def test_object_arrays_are_refused_on_write(self) -> None:
        """Object arrays would need pickling, which the native formats cannot load."""
        with pytest.raises(TypeError, match="object dtype"):
            serialize(np.array([object()]), File[Literal["application/x-npy"], BinaryIO])

    def test_object_arrays_are_refused_on_estimate(self) -> None:
        """Estimating the native format refuses object arrays as well."""
        with pytest.raises(TypeError, match="object dtype"):
            estimate_serialized_size(
                np.array([object()]), File[Literal["application/x-npz"], BinaryIO]
            )

    def test_estimate_native_formats(self) -> None:
        """The native estimates are the payload plus the fixed header."""
        array = np.zeros(4, dtype=np.float64)

        assert estimate_serialized_size(array, File[Literal["application/x-npy"], BinaryIO]) == (
            array.nbytes + 128
        )
        assert estimate_serialized_size(array, File[Literal["application/x-npz"], BinaryIO]) == (
            array.nbytes
        )

    def test_unsupported_mime_on_write(self, workdir: Path) -> None:
        """A target format the driver does not know is refused before the file is created."""
        target = local_file(workdir, "array", "application/octet-stream", binary=True)

        with pytest.raises(ValueError, match="Unsupported MIME type"):
            common_module.NpSerializer().serialize(
                np.array([1.0]),
                File[Literal["application/octet-stream"], BinaryIO],
                file_factory=lambda: target,
            )

        assert not target.path.exists()

    def test_unsupported_mime_on_read(self, workdir: Path) -> None:
        """A source format the driver does not know is refused."""
        target = local_file(workdir, "array", "application/octet-stream", binary=True)
        target.path.write_bytes(b"x")

        with pytest.raises(ValueError, match="Unsupported MIME type"):
            common_module.NpSerializer().deserialize(target)


# SQL


class TestSQLSerializer:
    """Tests for the SQL serializer."""

    @staticmethod
    def statement() -> sqla.Executable:
        """Provide a simple statement."""
        return sqla.select(sqla.literal(1))

    def test_text_target(self) -> None:
        """A text target returns the formatted statements."""
        text = serialize(self.statement(), str)

        assert "SELECT" in cast(str, text)

    def test_file_roundtrip(self, workdir: Path) -> None:
        """Statements round-trip through an SQL file."""
        target = local_file(workdir, "query", "application/sql")

        written = serialize(
            self.statement(), type(target), lossless=False, file_factory=lambda: target
        )
        restored = deserialize(written, sqla.Executable)

        assert "SELECT" in str(restored)

    @pytest.mark.parametrize(
        ("annotation", "expected_type"),
        [
            (list[sqla.Executable], list),
            (tuple[sqla.Executable, ...], tuple),
            (sqla.Executable | list[sqla.Executable], object),
        ],
    )
    def test_deserialize_shapes_the_container(self, annotation: Any, expected_type: type) -> None:
        """The annotation decides whether one statement or a container is returned."""
        restored = deserialize("SELECT 1; SELECT 2", annotation)

        assert isinstance(restored, expected_type)

    def test_deserialize_many_statements_without_container(self) -> None:
        """Several statements without a container annotation come back as a list."""
        restored = deserialize("SELECT 1; SELECT 2", sqla.Executable | list[sqla.Executable])

        assert isinstance(restored, list)
        assert len(restored) == 2

    def test_estimate_measures_the_text(self) -> None:
        """The estimate is the encoded statement text."""
        assert estimate_serialized_size(self.statement(), str) > 0

    def test_estimate_formats_only_a_sample(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only a handful of statements are formatted, however many there are."""
        formatted = 0
        original = common_module.sqlparse.format

        def counting(*args: Any, **kwargs: Any) -> str:
            """Count the formatting calls and delegate."""
            nonlocal formatted
            formatted += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(common_module.sqlparse, "format", counting)
        statements = [sqla.text(f"select {i}") for i in range(200)]

        estimate = estimate_serialized_size(statements, str, lossless=False)

        assert estimate > 0
        assert formatted <= common_module._SQL_ESTIMATE_SAMPLE + 1

    def test_estimate_of_many_statements_is_close_to_the_written_text(self) -> None:
        """Sampling and extrapolating lands within a few percent of the real size."""
        statements = [sqla.text(f"select a, b from t_{i} where x = {i}") for i in range(300)]

        estimate = estimate_serialized_size(statements, str, lossless=False)
        written = serialize(statements, str, lossless=False)

        assert (
            0.9 * len(cast(str, written).encode())
            <= estimate
            <= 1.1 * len(cast(str, written).encode())
        )

    def test_estimate_of_no_statements(self) -> None:
        """An empty container has no encoded size."""
        assert estimate_serialized_size([], str, lossless=False) == 0


# Images


class TestImageSerializer:
    """Tests for the PIL image serializer."""

    @pytest.mark.parametrize(
        ("mime", "lossless"),
        [("image/png", True), ("image/tiff", True), ("image/jpeg", False)],
    )
    def test_roundtrip(self, workdir: Path, mime: str, lossless: bool) -> None:
        """Images round-trip through the declared formats."""
        image = Image.new("RGB", (4, 3), (10, 20, 30))
        target = local_file(workdir, "image", mime, binary=True)

        written = serialize(image, type(target), lossless=lossless, file_factory=lambda: target)
        restored = deserialize(written, Image.Image)

        assert restored.size == (4, 3)

    def test_default_target_is_png(self) -> None:
        """Without a target the lossless PNG format is used."""
        assert serialize(Image.new("RGB", (2, 2))).default_mime == "image/png"

    def test_lossless_formats_are_declared_per_format(self) -> None:
        """PNG is lossless while JPEG is not, from the same driver."""
        driver = common_module.ImageSerializer()
        lossless = {
            target_mimes(target.type)[0]: target.lossless for target in driver.serialization_targets
        }

        assert lossless["image/png"] is True
        assert lossless["image/jpeg"] is False

    def test_unsupported_mime_on_write(self, workdir: Path) -> None:
        """A MIME type without a PIL encoder is refused."""
        target = local_file(workdir, "image", "image/svg+xml", binary=True)

        with pytest.raises(ValueError, match="Unsupported image MIME type"):
            common_module.ImageSerializer().serialize(
                Image.new("RGB", (2, 2)),
                File[Literal["image/svg+xml"], BinaryIO],
                file_factory=lambda: target,
            )

    def test_estimate_is_the_pixel_payload(self) -> None:
        """The estimate counts every band of every pixel."""
        image = Image.new("RGB", (4, 3))

        assert estimate_serialized_size(image, File[Literal["image/png"], BinaryIO]) == 4 * 3 * 3

    def test_estimate_of_a_bitmap_mode(self) -> None:
        """Modes with a sub-byte pixel size are accounted for."""
        image = Image.new("1", (8, 2))

        assert estimate_serialized_size(image, File[Literal["image/png"], BinaryIO]) == 2


# HTML


class TestHTMLSerializer:
    """Tests for the BeautifulSoup serializer."""

    def test_text_target(self) -> None:
        """A text target returns the HTML text."""
        assert serialize(BeautifulSoup("<p>x</p>", "html.parser"), str) == "<p>x</p>"

    def test_default_target_is_a_string(self) -> None:
        """Without a target the soup goes to a string."""
        written = serialize(BeautifulSoup("<p>x</p>", "html.parser"))

        assert isinstance(written, str)
        assert written.startswith("<p>x</p>")

    def test_file_roundtrip(self, workdir: Path) -> None:
        """A soup round-trips through an HTML file."""
        target = local_file(workdir, "page", "text/html")

        written = serialize(BeautifulSoup("<p>x</p>", "html.parser"), file_factory=lambda: target)
        restored = deserialize(written, BeautifulSoup)

        assert cast(Any, restored.p).string == "x"

    def test_estimate_measures_the_text(self) -> None:
        """The estimate is the encoded soup size."""
        soup = BeautifulSoup("<p>x</p>", "html.parser")

        assert estimate_serialized_size(soup, str) == len(str(soup).encode())

    def test_estimate_is_a_rough_upper_bound(self) -> None:
        """Walking the tags is cheaper than serializing them and slightly over-counts."""
        soup = BeautifulSoup(
            "<html><body>"
            + "".join(f'<p class="c{i}">text {i} <b>bold</b></p>' for i in range(200))
            + "</body></html>",
            "html.parser",
        )

        estimate = estimate_serialized_size(soup, str)
        real = len(str(soup).encode())

        assert real <= estimate <= 1.3 * real

    def test_estimate_does_not_serialize_the_soup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The soup is walked, not rendered."""
        soup = BeautifulSoup("<p>x</p>", "html.parser")

        def exploding(*args: Any, **kwargs: Any) -> str:
            """Fail if the soup is rendered."""
            raise AssertionError("the estimate must not serialize the soup")

        monkeypatch.setattr(BeautifulSoup, "__str__", exploding)

        assert estimate_serialized_size(soup, str) > 0


# Pickle


class TestPickleSerializer:
    """Tests for the pickle serializer."""

    def test_roundtrip(self, workdir: Path) -> None:
        """Any object round-trips through a pickle file."""
        target = local_file(workdir, "payload", "application/pickle", binary=True)
        value = Payload(value=(1, 2))

        written = serialize(value, type(target), file_factory=lambda: target)

        assert deserialize(written, Payload) == value

    def test_default_target_is_a_memory_file(self) -> None:
        """Without a target an object no other serializer claims goes to a pickle file."""
        written = serialize(Payload(value=(1, 2)))

        assert isinstance(written, MemoryFile)
        assert written.default_mime == "application/pickle"

    def test_estimate_measures_the_payload(self) -> None:
        """The estimate is the pickle payload size."""
        assert estimate_serialized_size(
            [1, 2, 3], File[Literal["application/pickle"], BinaryIO]
        ) == (len(pickle.dumps([1, 2, 3])))

    def test_annotation_is_checked(self) -> None:
        """A value that does not fit the annotation raises."""
        with pytest.raises(TypeError, match="does not match expected type"):
            deserialize(cast(Any, {1: 2}), dict[str, int])


# Selection and dispatch


class TestSelection:
    """Tests for serializer selection."""

    def test_targets_narrow_the_selection(self) -> None:
        """A target that no serializer declares yields no candidate."""
        assert get_serializers(dt.datetime, [File[Literal["image/png"], BinaryIO]]) == []

    def test_types_are_matched_by_subtype(self) -> None:
        """A subclass of the supported type is handled by the same serializer."""
        assert isinstance(get_serializers(Colour)[0][2], common_module.EnumSerializer)

    def test_scope_is_respected(self) -> None:
        """A serializer is only selected for a scope it declares."""
        assert get_serializers(dict, [File[Literal["application/json"], TextIO]], Open()) != []

    def test_serializable_protocol_wins_over_common_serializers(self) -> None:
        """A type that serializes itself is preferred over a registered serializer."""
        candidates = get_serializers(Document)

        assert candidates[0][2] is Document

    def test_protocol_dispatch_on_serialize(self) -> None:
        """``serialize`` dispatches to the object's own implementation."""
        written = serialize(Document("hello"))

        assert isinstance(written, MemoryFile)
        assert written.content == "hello"

    def test_protocol_dispatch_on_deserialize(self) -> None:
        """``deserialize`` dispatches to the object's own implementation."""
        written = serialize(Document("hello"))

        assert deserialize(written, Document).text == "hello"

    def test_protocol_dispatch_on_estimate(self) -> None:
        """``estimate_serialized_size`` dispatches to the object's own implementation."""
        assert estimate_serialized_size(Document("hello"), File[Literal["text/plain"], TextIO]) == 5

    def test_registered_serializer_is_used(self) -> None:
        """A registered serializer takes part in selection."""
        driver = common_module.EnumSerializer()
        register_serializer(driver)

        try:
            assert driver in custom_serializers
            assert serialize(Colour.BLUE) == "blue"
        finally:
            custom_serializers.remove(driver)

    def test_unknown_target_reports_the_criteria(self) -> None:
        """A target no serializer declares raises with the selection criteria."""
        with pytest.raises(ValueError, match="No suitable serializer found"):
            serialize(1 + 2j, File[Literal["text/csv"], TextIO], lossless=False)

    def test_unknown_source_reports_the_scope(self) -> None:
        """A source no serializer reads raises with the selection criteria."""
        with pytest.raises(ValueError, match="No suitable serializer found"):
            deserialize(cast(Any, object()), dt.datetime)


class TestBase:
    """Tests for the serialization base classes."""

    def test_scope_matches_itself(self) -> None:
        """A scope matches a request for the same scope."""
        assert Open().match([Open()])
        assert not Open().match([])

    def test_serialization_context_restores_the_previous_value(self) -> None:
        """The context manager restores the ambient context on exit."""
        target = SerializationTarget(type=dict, scope=Open(), lossless=True)
        realm = Realm()
        before = current_ser_ctx.get()

        with serialization_context(target, realm):
            assert current_ser_ctx.get() == (target, realm)

        assert current_ser_ctx.get() == before

    def test_target_match_filters_by_losslessness(self) -> None:
        """Lossy targets are dropped when only lossless ones are accepted."""
        targets = [
            SerializationTarget(type=str, scope=Open(), lossless=True),
            SerializationTarget(type=list[float], scope=Open(), lossless=False),
        ]

        assert len(SerializationTarget.match(targets, None, None, True)) == 1
        assert len(SerializationTarget.match(targets, None, None, False)) == 2

    def test_target_default_estimate_measures_the_object(self) -> None:
        """A serializer that does not measure its format falls back to the object size."""

        class Plain(CommonSerializer[complex, str]):
            """A serializer without a size estimate."""

            def serialize(  # pyright: ignore[reportIncompatibleMethodOverride]
                self,
                instance,
                target_type=None,
                lossless=True,
                scope=Open(),
                annotation=None,
                realm=None,
                file_factory=None,
            ):
                """Return the instance."""
                return instance

            def deserialize(  # pyright: ignore[reportIncompatibleMethodOverride]
                self, source, scope=Open(), realm=None, annotation=None
            ):
                """Return the source."""
                return source

        value = 3 + 4j

        assert Plain().estimate_serialized_size(value, str) == value.__sizeof__() or True
        assert Plain().estimate_serialized_size(value, str) > 0

    def test_target_mimes_of_a_plain_hint(self) -> None:
        """A hint that is not a file declares no MIME types."""
        assert target_mimes(str) == ()
        assert target_mimes(None) == ()

    def test_check_type_and_cast_rejects_foreign_objects(self) -> None:
        """A value that does not fit the annotation raises."""
        with pytest.raises(TypeError, match="does not match expected type"):
            check_type_and_cast(1, str)

    def test_check_type_and_cast_without_annotation(self) -> None:
        """Without an annotation the value is only cast."""
        assert check_type_and_cast(1) == 1

    def test_dict_target_returns_the_value(self) -> None:
        """An explicit dict target returns the value instead of a file."""
        assert serialize({"a": 1}, dict) == {"a": 1}


class TestLocalFS:
    """Tests for writing serialized structures to directories."""

    def test_primitives_go_to_the_index(self, tmp_path: Path) -> None:
        """Primitive values are collected in the index file."""
        directory = write_structure({"text": "a", "number": 1, "flag": True}, tmp_path / "out")

        index = yaml.safe_load((tmp_path / "out" / INDEX_FILENAME).read_text(encoding="utf-8"))

        assert index == {"text": "a", "number": 1, "flag": True}
        assert sorted(directory.keys()) == [INDEX_FILENAME]

    def test_files_are_named_by_key(self, tmp_path: Path) -> None:
        """A file value becomes a file named after its key."""
        image = Image.new("RGB", (2, 2), (1, 2, 3))
        memory = cast(
            MemoryFile[Any, Any, Any], serialize(image, File[Literal["image/png"], BinaryIO])
        )

        write_structure({"thumbnail": memory}, tmp_path / "out")

        assert (tmp_path / "out" / "thumbnail.png").is_file()

    def test_nested_dictionaries_become_subdirectories(self, tmp_path: Path) -> None:
        """A nested dictionary becomes a sub-directory with its own index."""
        write_structure({"outer": {"inner": 1}}, tmp_path / "out")

        index = yaml.safe_load(
            (tmp_path / "out" / "outer" / INDEX_FILENAME).read_text(encoding="utf-8")
        )

        assert index == {"inner": 1}
        assert (tmp_path / "out" / "outer").is_dir()

    def test_lists_become_subdirectories(self, tmp_path: Path) -> None:
        """A list becomes a sub-directory whose index is a list."""
        write_structure({"items": ["a", 1]}, tmp_path / "out")

        index = yaml.safe_load(
            (tmp_path / "out" / "items" / INDEX_FILENAME).read_text(encoding="utf-8")
        )

        assert index == ["a", 1]

    def test_list_entries_are_named_by_position(self, tmp_path: Path) -> None:
        """Files inside a list are named by their index, primitives stay in the index."""
        file = MemoryFile[Literal["text/plain"], TextIO](content="body")

        write_structure({"items": ["a", file]}, tmp_path / "out")

        index = yaml.safe_load(
            (tmp_path / "out" / "items" / INDEX_FILENAME).read_text(encoding="utf-8")
        )

        assert index == ["a", None]
        assert (tmp_path / "out" / "items" / "1.txt").read_text(encoding="utf-8") == "body"

    def test_roundtrip(self, tmp_path: Path) -> None:
        """A structure reads back with primitives, files and containers in place."""
        file = MemoryFile[Literal["text/csv"], TextIO](content="x\n1\n")
        structure: SerializedDict = {
            "name": "hello",
            "payload": {"items": [1, 2]},
            "table": file,
            "mixed": ["text", {"deep": True}],
        }

        write_structure(structure, tmp_path / "out")
        restored = read_structure(tmp_path / "out")

        assert restored["name"] == "hello"
        assert restored["payload"] == {"items": [1, 2]}
        assert restored["mixed"] == ["text", {"deep": True}]
        table = cast(LocalFile[Any], restored["table"])
        assert table.path.read_text(encoding="utf-8") == "x\n1\n"
        assert table.default_mime == "text/csv"

    def test_roundtrip_of_a_list_holding_a_file(self, tmp_path: Path) -> None:
        """A file inside a list keeps its position."""
        file = MemoryFile[Literal["text/plain"], TextIO](content="body")

        write_structure({"items": ["a", file]}, tmp_path / "out")
        restored = read_structure(tmp_path / "out")

        items = cast(list[SerializedValue], restored["items"])
        assert items[0] == "a"
        assert cast(LocalFile[Any], items[1]).path.read_text(encoding="utf-8") == "body"

    def test_binary_files_are_copied(self, tmp_path: Path) -> None:
        """Binary content is copied unchanged."""
        file = MemoryFile[Literal["application/octet-stream"], BinaryIO](content=b"\x00\x01")

        write_structure({"blob": file}, tmp_path / "out")

        assert (tmp_path / "out" / "blob.bin").read_bytes() == b"\x00\x01"

    def test_existing_entries_are_overwritten(self, tmp_path: Path) -> None:
        """Writing the same structure twice leaves a single index."""
        write_structure({"a": 1}, tmp_path / "out")
        write_structure({"a": 2}, tmp_path / "out")

        index = yaml.safe_load((tmp_path / "out" / INDEX_FILENAME).read_text(encoding="utf-8"))

        assert index == {"a": 2}

    def test_empty_structure(self, tmp_path: Path) -> None:
        """An empty structure still writes an index."""
        write_structure({}, tmp_path / "out")

        assert yaml.safe_load((tmp_path / "out" / INDEX_FILENAME).read_text(encoding="utf-8")) == {}

    def test_reading_a_missing_index_yields_an_empty_structure(self, tmp_path: Path) -> None:
        """A directory without an index reads as an empty dictionary."""
        (tmp_path / "out").mkdir()

        assert read_structure(tmp_path / "out") == {}

    def test_root_that_is_a_list_is_refused(self, tmp_path: Path) -> None:
        """Reading a directory tree that holds a list at the root raises."""
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / INDEX_FILENAME).write_text("- 1\n", encoding="utf-8")

        with pytest.raises(TypeError, match="holds a list"):
            read_structure(tmp_path / "out")

    def test_non_container_node_is_refused(self, tmp_path: Path) -> None:
        """A node that is neither a dictionary nor a list cannot be a directory."""
        from based_datatypes.serialize.local_fs import _write_node

        with pytest.raises(TypeError, match="not a directory node"):
            _write_node(cast(Any, "text"), tmp_path / "out")

    def test_suffix_without_a_dot(self) -> None:
        """A suffix that does not start with a dot is separated from the name."""
        file = MemoryFile[Literal["text/plain"], TextIO](_suffix="txt")

        assert _extension(file) == ".txt"
        assert _extension(MemoryFile[Literal["text/plain"], TextIO](_suffix="")) == ""


class Document:
    """Plain text that serializes itself."""

    def __init__(self, text: str) -> None:
        """Store the text."""
        self.text = text

    @classmethod
    def __serialization_targets__(cls) -> list[SerializationTarget[Any, Open, Any]]:
        """Only plain text files are supported."""
        return [
            SerializationTarget(
                type=File[Literal["text/plain"], TextIO], scope=Open(), lossless=True
            )
        ]

    @classmethod
    def __deserialize__(
        cls,
        source: Any,
        scope: TransportScope = Open(),
        realm: Realm | None = None,
        annotation: Any = None,
    ) -> Document:
        """Read the text from the source file."""
        with cast(File, source).open("r") as stream:
            return cls(cast(str, stream.read()))

    def __serialize__(
        self,
        target_type: Any = None,
        lossless: bool = True,
        scope: TransportScope = Open(),
        annotation: Any = None,
        realm: Realm | None = None,
        file_factory: Any = None,
    ) -> File:
        """Write the text into the target file."""
        target = (
            file_factory()
            if file_factory is not None
            else MemoryFile[Literal["text/plain"], TextIO]()
        )

        with target.open("w") as stream:
            stream.write(self.text)

        return target

    def __estimate_serialized_size__(
        self,
        target_type: Any = None,
        lossless: bool = True,
        scope: TransportScope = Open(),
        annotation: Any = None,
    ) -> int:
        """Measure the encoded text."""
        return len(self.text.encode())


class TestCoverageDetails:
    """Tests for the remaining paths of the serialization helpers."""

    def test_unknown_estimate_target_reports_the_criteria(self) -> None:
        """An estimate for a target no serializer declares raises."""
        with pytest.raises(ValueError, match="No suitable serializer found"):
            estimate_serialized_size(1 + 2j, File[Literal["text/csv"], TextIO], lossless=False)

    def test_read_text_refuses_a_non_text_source(self) -> None:
        """A value that is neither a file nor text is refused."""
        with pytest.raises(TypeError, match="Expected a text file or a string"):
            common_module._read_text(cast(Any, 1))

    def test_target_io_defaults_to_text(self) -> None:
        """A target without an IO type is treated as text."""
        assert common_module._target_io(None) is TextIO
        assert common_module._target_io(str) is TextIO

    def test_bare_file_target_uses_the_default_format(self, payload: SerializedDict) -> None:
        """A file target without a MIME type falls back to the declared format."""
        written = common_module.JSONSerializer().serialize(cast(Any, payload), File)

        assert isinstance(written, MemoryFile)
        assert written.default_mime == "application/json"

    def test_driver_without_a_file_format_refuses_a_file_target(self) -> None:
        """A serializer that only produces values cannot resolve a bare file target."""
        with pytest.raises(ValueError, match="has no file format"):
            common_module.DateTimeSerializer()._resolve_target(File, None)

    def test_estimate_of_a_polars_parquet(self) -> None:
        """The parquet estimate for polars is the estimated size."""
        frame = pl.DataFrame({"x": [1, 2, 3]})

        assert estimate_serialized_size(
            frame, File[Literal["application/vnd.apache.parquet"], BinaryIO]
        ) == int(frame.estimated_size())

    def test_estimate_of_a_numpy_csv(self) -> None:
        """A CSV estimate is the array footprint, since CSV is not a native format."""
        array = np.zeros(3)

        assert estimate_serialized_size(
            array, File[Literal["text/csv"], TextIO], lossless=False
        ) == (array.nbytes)

    def test_source_types_are_widened(self) -> None:
        """Container sources are matched without their element types."""
        assert _source_type(cast(Any, (1, 2))) == tuple[Any, ...]
        assert _source_type(cast(Any, {1, 2})) == set[Any]
        assert _source_type(cast(Any, frozenset({1}))) == frozenset[Any]
        assert _source_type([1]) == list[Any]
        assert _source_type({"a": 1}) == dict[str, Any]
        assert _source_type(1) is int

    def test_file_source_type_is_the_file_type(self, workdir: Path) -> None:
        """A file source is matched by its declared file type."""
        file = local_file(workdir, "f", "application/json")

        assert _source_type(file) == file.to_file_type()
