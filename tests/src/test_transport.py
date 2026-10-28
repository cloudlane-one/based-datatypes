"""Tests for the :mod:`based_datatypes.transport` subpackage."""

import xml.etree.ElementTree as xml  # noqa: N813
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, BinaryIO, Literal, TextIO, cast, get_args

import pytest
from pydantic import BaseModel

from based_datatypes import transport
from based_datatypes.files import File, LocalDir, LocalFile
from based_datatypes.transport import (
    export_object,
    get_transport_driver,
    get_transport_types,
    import_object,
    register_transport_driver,
)
from based_datatypes.transport.base import (
    IPC,
    LAN,
    V2,
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
from based_datatypes.transport.common import (
    JSONFile,
    JSONTransporter,
    PickleTransporter,
    PlTransporter,
    PydTransporter,
    XMLFile,
    check_type_and_cast,
)
from based_datatypes.typing import SingleTypeDef


class Record(BaseModel):
    """Small pydantic model used for transport round trips."""

    name: str
    count: int


class Point:
    """Small, non-pydantic object with a custom driver."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Point) and (self.x, self.y) == (other.x, other.y)


type PointFile = File[Literal["text/plain"], TextIO]
type JSONMime = Literal["application/json"]


class PointDriver(TransportDriver[Point, PointFile]):
    """Custom driver writing points as ``x,y``."""

    @classmethod
    def import_object(
        cls: type[TransportDriver[V2, PointFile]],
        source: PointFile,
        realm: Any,
        annotation: SingleTypeDef[V2] | None = None,
    ) -> V2:
        """Parse a point from a text file."""
        with source.open("r") as stream:
            x, y = stream.read().split(",")

        return cast(V2, Point(float(x), float(y)))

    @classmethod
    def export_object(
        cls: type[TransportDriver[Point, PointFile]],
        instance: Point,
        target: PointFile,
        realm: Any,
        annotation: SingleTypeDef[Point] | None = None,
    ) -> None:
        """Write a point to a text file."""
        with target.open("w") as stream:
            stream.write(f"{instance.x},{instance.y}")


class Doc:
    """Object implementing the transport protocol itself."""

    def __init__(self, text: str) -> None:
        self.text = text

    @classmethod
    def __transport_types__(cls) -> TransportTypes[TextIO]:
        """Declare plain text as the only transport format."""
        return TransportTypes((File[Literal["text/plain"], TextIO],))

    @classmethod
    def __import__(
        cls,
        source: File[Literal["text/plain"], TextIO],
        realm: Realm,
        annotation: "SingleTypeDef[Doc] | None" = None,
    ) -> "Doc":
        """Read the document from text."""
        with source.open("r") as stream:
            return cls(stream.read())

    def __export__(
        self,
        target: File[Literal["text/plain"], TextIO],
        realm: Realm,
        annotation: SingleTypeDef[Transportable[TextIO]] | None = None,
    ) -> None:
        """Write the document as text."""
        with target.open("w") as stream:
            stream.write(self.text)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """Provide an existing empty directory."""
    path = tmp_path / "work"
    path.mkdir()
    return path


def file_for(
    directory: Path,
    name: str,
    mime: str,
    *,
    binary: bool = False,
) -> LocalFile[Any]:
    """Build a :class:`LocalFile` that is not on disk yet.

    Args:
        directory: Directory the file belongs to.
        name: File name without extension.
        mime: MIME type of the target format.
        binary: Whether the format is binary rather than text.

    Returns:
        The file object.
    """
    return LocalFile(
        dir=LocalDir(path=directory),
        name=name,
        mime=mime,
        io_type=BinaryIO if binary else TextIO,
    )


def hint_mimes(hint: object) -> tuple[str, ...]:
    """Return the MIME literals declared by a ``File[...]`` hint.

    Args:
        hint: The transport type hint.

    Returns:
        The declared MIME types, or an empty tuple if none are declared.
    """
    args = get_args(hint)
    return tuple(str(mime) for mime in get_args(args[0])) if args else ()


class TestTransportScopes:
    """Tests for the transport scope hierarchy."""

    def test_scope_hierarchy(self) -> None:
        """Each scope extends the narrower one."""
        lan = LAN(private_subnet="192.168.1.0/24")
        wan = WAN(private_subnet="192.168.1.0/24", source_region="eu-west-1")

        assert isinstance(lan, IPC)
        assert isinstance(wan, LAN)
        assert isinstance(IPC(), TransportScope)
        assert not isinstance(IPC(), LAN)

    def test_wan_region_is_optional(self) -> None:
        """The WAN region defaults to ``None``."""
        assert WAN(private_subnet="10.0.0.0/8").source_region is None

    def test_scopes_are_immutable(self) -> None:
        """Scopes are frozen dataclasses."""
        lan = LAN(private_subnet="10.0.0.0/8")

        with pytest.raises(FrozenInstanceError):
            lan.private_subnet = "other"  # pyright: ignore[reportAttributeAccessIssue]

    def test_scopes_compare_by_value(self) -> None:
        """Equal scopes compare equal."""
        assert LAN(private_subnet="10.0.0.0/8") == LAN(private_subnet="10.0.0.0/8")
        assert IPC() != Open()


class TestSerializationPurpose:
    """Tests for :func:`serialization_purpose`."""

    def test_default_is_open(self) -> None:
        """Without a context the purpose is open."""
        assert isinstance(current_ser_purpose.get(), Open)

    def test_sets_and_restores(self) -> None:
        """The purpose is scoped to the context."""
        purpose = LAN(private_subnet="10.0.0.0/8")

        with transport_scope(purpose):
            assert current_ser_purpose.get() is purpose

        assert isinstance(current_ser_purpose.get(), Open)

    def test_nested_contexts(self) -> None:
        """Nested contexts restore the previous purpose."""
        outer = IPC()
        inner = WAN(private_subnet="10.0.0.0/8")

        with transport_scope(outer):
            with transport_scope(inner):
                assert current_ser_purpose.get() is inner
            assert current_ser_purpose.get() is outer

    def test_resets_after_exception(self) -> None:
        """The purpose is restored when the body raises."""
        with pytest.raises(RuntimeError), transport_scope(IPC()):
            raise RuntimeError("boom")

        assert isinstance(current_ser_purpose.get(), Open)


class TestTransportTypes:
    """Tests for :class:`TransportTypes`."""

    def test_normalizes_to_tuple(self) -> None:
        """A sequence is stored as a tuple."""
        types = TransportTypes([File[Literal["text/plain"], TextIO]])

        assert isinstance(types.types, tuple)
        assert len(types.types) == 1

    def test_matches_file_by_mime(self, workdir: Path) -> None:
        """A file instance is matched on its MIME type."""
        json_types = JSONTransporter.transport_types()

        assert json_types.match_targets([file_for(workdir, "a", "application/json")])
        assert json_types.match_targets([file_for(workdir, "a", "application/yaml")])
        assert (
            json_types.match_targets([file_for(workdir, "a", "application/pickle", binary=True)])
            == []
        )

    def test_returns_the_matching_hint(self, workdir: Path) -> None:
        """Only the declared hint for the target's MIME type is returned."""
        matched = JSONTransporter.transport_types().match_targets(
            [file_for(workdir, "a", "application/yaml")]
        )

        assert len(matched) == 1
        assert hint_mimes(matched[0]) == ("application/yaml",)

    def test_rejects_file_with_wrong_io_type(self, workdir: Path) -> None:
        """A matching MIME does not make a binary file a text transport."""
        types = TransportTypes([File[Literal["application/json"], TextIO]])

        assert types.match_targets([file_for(workdir, "a", "application/json", binary=True)]) == []

    def test_matches_file_with_aliased_mime_hint(self, workdir: Path) -> None:
        """MIME hints use the shared typing utility's alias resolution."""
        types = TransportTypes([File[JSONMime, TextIO]])

        assert types.match_targets([file_for(workdir, "a", "application/json")])
        assert types.match_targets([file_for(workdir, "a", "application/yaml")]) == []

    def test_matches_parameterized_file_targets(self) -> None:
        """Parameterized file targets retain both MIME and IO arguments."""
        hint = File[Literal["application/json"], TextIO]
        types = TransportTypes([hint])

        assert types.match_targets([hint]) == [hint]
        assert types.match_targets([File[Literal["application/yaml"], TextIO]]) == []
        assert types.match_targets([File[Literal["application/json"], BinaryIO]]) == []

    def test_matches_parameterized_non_file_targets(self) -> None:
        """Generic subtype checks also apply to non-file transport hints."""
        types = TransportTypes([list[int]])

        assert types.match_targets([list[int]]) == [list[int]]
        assert types.match_targets([list[str]]) == []

    def test_rejects_unrelated_type(self) -> None:
        """A target of an unrelated class does not match."""
        assert JSONTransporter.transport_types().match_targets([dict]) == []
        assert JSONTransporter.transport_types().match_targets([xml.ElementTree]) == []

    def test_generic_file_hint_accepts_any_mime(self, workdir: Path) -> None:
        """A hint without MIME literals matches every file of that class."""
        types = TransportTypes([File[Any, Any]])

        assert types.match_targets([file_for(workdir, "a", "text/plain")])
        assert types.match_targets([file_for(workdir, "a", "application/pickle", binary=True)])


class TestDriverIntrospection:
    """Tests for the type metadata a driver derives from its declaration."""

    def test_obj_type(self) -> None:
        """The handled object type is read from the driver's type arguments."""
        pl = pytest.importorskip("polars")

        assert PlTransporter.obj_type() is pl.DataFrame

    def test_transport_hints_keep_declared_mimes(self) -> None:
        """Declared MIME literals survive in the transport types."""
        hints = PickleTransporter.transport_types().types

        assert len(hints) == 1
        assert hint_mimes(hints[0]) == ("application/pickle",)

    def test_transport_hints_keep_text_and_binary(self) -> None:
        """A driver may declare several formats with different IO types."""
        hints = PlTransporter.transport_types().types
        mimes = {mime for hint in hints for mime in hint_mimes(hint)}

        assert mimes == {
            "text/csv",
            "application/vnd.apache.parquet",
            "application/vnd.ms-excel",
        }

    def test_abstract_methods_are_required(self) -> None:
        """A driver without implementations cannot be instantiated."""
        with pytest.raises(TypeError):
            TransportDriver()  # pyright: ignore[reportAbstractUsage]

    def test_batch_transfers_are_optional(self, workdir: Path) -> None:
        """A driver that does not support batches says so."""
        pl = pytest.importorskip("polars")
        source = file_for(workdir, "batch", "application/json")

        with pytest.raises(NotImplementedError):
            JSONTransporter.batch_import(cast(Any, source), Realm(), dict)
        with pytest.raises(NotImplementedError):
            JSONTransporter.batch_export(pl.Series([1]), cast(Any, source), Realm(), dict)


class TestGetTransportDriver:
    """Tests for :func:`get_transport_driver`."""

    def test_picks_driver_by_mime(self, workdir: Path) -> None:
        """The target's MIME type disambiguates between drivers."""
        assert (
            get_transport_driver({}, [file_for(workdir, "a", "application/json")])
            is JSONTransporter
        )
        assert (
            get_transport_driver({}, [file_for(workdir, "a", "application/pickle", binary=True)])
            is PickleTransporter
        )

    def test_accepts_the_object_type(self, workdir: Path) -> None:
        """A type instead of an instance is accepted."""
        assert (
            get_transport_driver(dict, [file_for(workdir, "a", "application/json")])
            is JSONTransporter
        )

    def test_no_target_returns_driver_for_the_type(self) -> None:
        """Omitting targets skips the medium check."""
        assert get_transport_driver({}) is JSONTransporter

    def test_returns_none_without_a_match(self, workdir: Path) -> None:
        """An unsupported medium yields no driver."""
        assert get_transport_driver({}, [file_for(workdir, "a", "text/html")]) is None
        assert (
            get_transport_driver(complex(1, 2), [file_for(workdir, "a", "application/json")])
            is None
        )

    def test_accepts_a_type_hint(self, workdir: Path) -> None:
        """A generic type hint is resolved through its typeset."""
        assert (
            get_transport_driver(dict[str, int], [file_for(workdir, "a", "application/json")])
            is JSONTransporter
        )


class TestGetTransportTypes:
    """Tests for :func:`get_transport_types`."""

    def test_known_type(self) -> None:
        """The driver's declared formats are returned."""
        assert get_transport_types(dict).types == JSONTransporter.transport_types().types

    def test_catch_all_type(self) -> None:
        """A type only the pickle driver covers reports the pickle format."""
        assert get_transport_types(int).types == PickleTransporter.transport_types().types

    def test_protocol_type(self) -> None:
        """A type implementing the protocol reports its own formats."""
        hints = get_transport_types(Doc).types

        assert len(hints) == 1
        assert hint_mimes(hints[0]) == ("text/plain",)

    def test_unknown_type_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without any registered driver the query fails loudly."""
        monkeypatch.setattr(transport, "common_drivers", {})

        with pytest.raises(TypeError, match="No transport types"):
            get_transport_types(int)


class TestCheckTypeAndCast:
    """Tests for :func:`check_type_and_cast`."""

    def test_without_annotation(self) -> None:
        """No annotation means no check."""
        value: object = {"a": 1}

        assert check_type_and_cast(value) is value

    def test_matching_annotation(self) -> None:
        """A matching value passes."""
        assert check_type_and_cast({"a": 1}, dict) == {"a": 1}

    def test_mismatched_annotation(self) -> None:
        """A non-matching value raises ``TypeError``."""
        with pytest.raises(TypeError, match="does not match"):
            check_type_and_cast({}, list[int])


class TestJSONTransporter:
    """Round trips through the JSON/YAML driver."""

    def test_json_roundtrip(self, workdir: Path) -> None:
        """A dictionary survives a JSON round trip."""
        target = file_for(workdir, "data", "application/json")
        payload = {"a": 1, "b": [1, 2], "nested": {"c": None}}

        export_object(payload, target, Realm())

        assert target.path.suffix == ".json"
        assert import_object(target, Realm(), dict) == payload

    def test_yaml_roundtrip_preserves_unicode(self, workdir: Path) -> None:
        """A dictionary survives a YAML round trip."""
        target = file_for(workdir, "data", "application/yaml")
        payload = {"text": "héllo", "items": [1, 2]}

        export_object(payload, target, Realm())

        assert target.path.suffix == ".yaml"
        assert import_object(target, Realm(), dict) == payload

    def test_import_checks_the_annotation(self, workdir: Path) -> None:
        """Importing as an incompatible type raises."""
        target = file_for(workdir, "data", "application/json")
        export_object({"a": 1}, target, Realm())

        with pytest.raises(TypeError, match="does not match"):
            JSONTransporter.import_object(cast(JSONFile, target), Realm(), cast(Any, list[int]))

    def test_export_rejects_unsupported_medium(self, workdir: Path) -> None:
        """No driver means a clear ``TypeError``."""
        with pytest.raises(TypeError, match="No driver found"):
            export_object({"a": 1}, file_for(workdir, "a", "text/html"), Realm())

    def test_import_rejects_unsupported_medium(self, workdir: Path) -> None:
        """No parser means a clear ``TypeError``."""
        target = file_for(workdir, "a", "text/html")

        with pytest.raises(TypeError, match="No parser found"):
            import_object(target, Realm(), dict)


class TestPickleTransporter:
    """Round trips through the pickle driver."""

    def test_roundtrip(self, workdir: Path) -> None:
        """An arbitrary object survives a pickle round trip."""
        target = file_for(workdir, "data", "application/pickle", binary=True)
        payload = {"a": 1, "b": (1, 2)}

        export_object(payload, target, Realm())

        assert target.path.suffix == ".pkl"
        assert import_object(target, Realm(), dict) == payload


class TestPydanticTransporter:
    """Round trips through the pydantic driver."""

    def test_json_roundtrip(self, workdir: Path) -> None:
        """A model survives a JSON round trip."""
        target = file_for(workdir, "record", "application/json")
        record = Record(name="a", count=2)

        export_object(record, target, Realm())

        assert import_object(target, Realm(), Record) == record

    def test_yaml_roundtrip(self, workdir: Path) -> None:
        """A model survives a YAML round trip."""
        target = file_for(workdir, "record", "application/yaml")
        record = Record(name="b", count=3)

        export_object(record, target, Realm())

        assert import_object(target, Realm(), Record) == record

    def test_import_requires_an_annotation(self, workdir: Path) -> None:
        """Without an annotation the driver refuses to guess."""
        target = file_for(workdir, "record", "application/json")
        export_object(Record(name="a", count=1), target, Realm())

        with pytest.raises(ValueError, match="Annotation must be provided"):
            PydTransporter.import_object(cast(JSONFile, target), Realm(), None)


class TestPolarsTransporter:
    """Round trips through the polars driver."""

    def test_csv_roundtrip(self, workdir: Path) -> None:
        """A polars frame survives a CSV round trip."""
        pl = pytest.importorskip("polars")
        target = file_for(workdir, "frame", "text/csv")
        frame = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})

        export_object(frame, target, Realm())
        restored = import_object(target, Realm(), pl.DataFrame)

        assert restored.equals(frame)

    def test_parquet_roundtrip(self, workdir: Path) -> None:
        """A polars frame survives a parquet round trip."""
        pl = pytest.importorskip("polars")
        target = file_for(workdir, "frame", "application/vnd.apache.parquet", binary=True)
        frame = pl.DataFrame({"x": [1, 2]})

        export_object(frame, target, Realm())
        restored = import_object(target, Realm(), pl.DataFrame)

        assert restored.equals(frame)

    def test_unsupported_mime(self, workdir: Path) -> None:
        """An unknown MIME type is refused."""
        pl = pytest.importorskip("polars")
        from based_datatypes.transport.common import PlTransporter

        target = file_for(workdir, "frame", "text/html")
        target.path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported MIME type"):
            PlTransporter.import_object(cast(Any, target), Realm(), pl.DataFrame)


class TestPandasTransporter:
    """Round trips through the pandas driver."""

    def test_csv_roundtrip(self, workdir: Path) -> None:
        """A pandas frame survives a CSV round trip."""
        pd = pytest.importorskip("pandas")
        target = file_for(workdir, "frame", "text/csv")
        frame = pd.DataFrame({"x": [1, 2]})

        export_object(frame, target, Realm())
        restored = import_object(target, Realm(), pd.DataFrame)

        assert restored.equals(frame)

    def test_unsupported_mime(self, workdir: Path) -> None:
        """An unknown MIME type is refused."""
        pd = pytest.importorskip("pandas")
        from based_datatypes.transport.common import PdTransporter

        target = file_for(workdir, "frame", "text/html")
        target.path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported MIME type"):
            PdTransporter.import_object(cast(Any, target), Realm(), pd.DataFrame)


class TestNumpyTransporter:
    """Round trips through the numpy driver."""

    def test_npy_roundtrip(self, workdir: Path) -> None:
        """An array survives an npy round trip."""
        np = pytest.importorskip("numpy")
        target = file_for(workdir, "array", "application/x-npy", binary=True)
        array = np.array([1.0, 2.0])

        export_object(array, target, Realm())
        restored = import_object(target, Realm(), np.ndarray)

        assert np.array_equal(restored, array)

    def test_npz_roundtrip(self, workdir: Path) -> None:
        """An array survives an npz round trip."""
        np = pytest.importorskip("numpy")
        target = file_for(workdir, "array", "application/x-npz", binary=True)
        array = np.array([[1, 2], [3, 4]])

        export_object(array, target, Realm())
        restored = import_object(target, Realm(), np.ndarray)

        assert np.array_equal(restored, array)

    def test_csv_roundtrip(self, workdir: Path) -> None:
        """An array survives a CSV round trip."""
        np = pytest.importorskip("numpy")
        target = file_for(workdir, "array", "text/csv")
        array = np.array([[1.0, 2.0], [3.0, 4.0]])

        export_object(array, target, Realm())
        restored = import_object(target, Realm(), np.ndarray)

        assert np.array_equal(restored, array)

    def test_unsupported_mime(self, workdir: Path) -> None:
        """An unknown MIME type is refused."""
        np = pytest.importorskip("numpy")
        from based_datatypes.transport.common import NpTransporter

        target = file_for(workdir, "array", "text/html")
        target.path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported MIME type"):
            NpTransporter.import_object(cast(Any, target), Realm(), np.ndarray)


class TestImageTransporter:
    """Round trips through the PIL driver."""

    def test_png_roundtrip(self, workdir: Path) -> None:
        """An image without a format is saved according to the target MIME type."""
        image_module = pytest.importorskip("PIL.Image")
        target = file_for(workdir, "image", "image/png", binary=True)
        image = image_module.new("RGB", (3, 2), "red")

        export_object(image, target, Realm())
        restored = import_object(target, Realm(), image_module.Image)

        assert restored.size == (3, 2)
        assert restored.mode == "RGB"

    def test_unsupported_target_mime(self, workdir: Path) -> None:
        """A format that cannot be inferred from the target raises."""
        image_module = pytest.importorskip("PIL.Image")
        from based_datatypes.transport.common import ImageFile, ImageTransporter

        target = file_for(workdir, "image", "application/octet-stream", binary=True)

        with pytest.raises(ValueError, match="Unsupported image MIME type"):
            ImageTransporter.export_object(
                image_module.new("RGB", (1, 1)), cast(ImageFile, target), Realm()
            )


class TestXMLTransporter:
    """Round trips through the XML driver."""

    def test_roundtrip(self, workdir: Path) -> None:
        """An element tree survives a round trip."""
        target = file_for(workdir, "tree", "application/xml")
        root = xml.Element("root")
        xml.SubElement(root, "child").text = "value"

        export_object(xml.ElementTree(root), target, Realm())
        restored = import_object(target, Realm(), xml.ElementTree)

        restored_root = restored.getroot()
        assert restored_root is not None
        assert xml.tostring(restored_root, encoding="unicode") == (
            "<root><child>value</child></root>"
        )

    def test_import_rejects_wrong_mime(self, workdir: Path) -> None:
        """The driver validates the MIME type it is given."""
        from based_datatypes.transport.common import XMLTransporter

        target = file_for(workdir, "tree", "application/json")

        with pytest.raises(ValueError, match="Unsupported MIME type"):
            XMLTransporter.import_object(cast(XMLFile, target), Realm(), xml.ElementTree)


class TestHTMLTransporter:
    """Round trips through the BeautifulSoup driver."""

    def test_roundtrip(self, workdir: Path) -> None:
        """A soup survives a round trip."""
        bs4 = pytest.importorskip("bs4")
        target = file_for(workdir, "page", "text/html")
        soup = bs4.BeautifulSoup("<p>hi</p>", "html.parser")

        export_object(soup, target, Realm())
        restored = import_object(target, Realm(), bs4.BeautifulSoup)

        assert str(restored) == "<p>hi</p>"


class TestSQLTransporter:
    """Round trips through the SQL driver."""

    def test_single_statement(self, workdir: Path) -> None:
        """One statement is written and read back as a single statement."""
        sqla = pytest.importorskip("sqlalchemy")
        target = file_for(workdir, "query", "application/sql")

        export_object(sqla.text("select 1"), target, Realm())
        restored = import_object(target, Realm(), sqla.TextClause)

        assert isinstance(restored, sqla.TextClause)
        assert "SELECT 1" in str(restored)

    def test_multiple_statements(self, workdir: Path) -> None:
        """Several statements stay separated and in order."""
        sqla = pytest.importorskip("sqlalchemy")
        target = file_for(workdir, "queries", "application/sql")
        statements = [sqla.text("select 1"), sqla.text("delete from t")]

        export_object(statements, target, Realm())
        restored = import_object(target, Realm(), list[sqla.TextClause])

        def normalize(statement: object) -> str:
            return " ".join(str(statement).rstrip(";").upper().split())

        assert [normalize(statement) for statement in restored] == [
            normalize(statement) for statement in statements
        ]


class TestTransportableProtocol:
    """Tests for objects implementing the transport protocol."""

    def test_runtime_protocol(self) -> None:
        """The protocol is runtime checkable."""
        assert isinstance(Doc("x"), Transportable)
        assert not isinstance(Point(1, 2), Transportable)

    def test_export_and_import(self, workdir: Path) -> None:
        """The object's own methods handle the transfer."""
        target = file_for(workdir, "doc", "text/plain")

        export_object(Doc("hello"), target, Realm())
        restored = import_object(target, Realm(), Doc)

        assert isinstance(restored, Doc)
        assert restored.text == "hello"


class TestCustomDrivers:
    """Tests for :func:`register_transport_driver`."""

    @pytest.fixture(autouse=True)
    def _isolated_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep custom registrations local to the test."""
        monkeypatch.setattr(transport, "custom_drivers", {})

    def test_registration_enables_selection(self, workdir: Path) -> None:
        """A registered driver is used for its object type."""
        target = file_for(workdir, "point", "text/plain")

        assert get_transport_driver(Point(1, 2), [target]) is None

        register_transport_driver(Point, PointDriver)

        assert get_transport_driver(Point(1, 2), [target]) is PointDriver
        assert get_transport_types(Point).types == PointDriver.transport_types().types

    def test_roundtrip_through_custom_driver(self, workdir: Path) -> None:
        """The public functions dispatch to the registered driver."""
        register_transport_driver(Point, PointDriver)
        target = file_for(workdir, "point", "text/plain")

        export_object(Point(1.5, 2.5), target, Realm())
        restored = import_object(target, Realm(), Point)

        assert restored == Point(1.5, 2.5)
