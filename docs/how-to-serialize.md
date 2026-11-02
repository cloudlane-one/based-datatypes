# Serialization

`based_datatypes.serialize` turns Python objects into two families of output — files and
JSON-supported values — without the caller wiring up a library per format. You describe *what*
you have and *what you want*; the package picks a serializer that understands both, tells you
whether the format is lossless, and can estimate how large the result would be before you
produce it.

## When to use it

- Serialize a dict, model, dataframe, array, image, XML tree or SQL statement without importing
  a different library for each format.
- Convert values JSON cannot hold — `datetime`, `complex`, `Decimal`, `UUID`, `bytes`, enums —
  into something a JSON document can carry, and back again.
- Read content back while declaring the type you expect, so a mismatched source fails loudly.
- Ask for a lossless format when round-tripping matters, and accept a lossy one when it does not.
- Budget an export before producing it.
- Lay a serialized dictionary of files and primitives out as a directory tree.

## Quick start

```python exec="on" session="serialize"
import tempfile
from pathlib import Path
from typing import Literal, TextIO

from based_datatypes.files import File, LocalDir, LocalFile
from based_datatypes.serialize import deserialize, serialize

workdir = Path(tempfile.mkdtemp())
payload = {"hello": "world", "items": [1, 2]}

# Without a target the serializer picks its preferred format: an in-memory file.
written = serialize(payload)
print(type(written).__name__, written.default_mime)
print(written.content)

# A file target is created by a factory, so the caller decides where it lands.
json_target = LocalFile[Literal["application/json"], TextIO](
    dir=LocalDir(path=workdir), name="payload"
)
serialize(payload, File[Literal["application/json"], TextIO], file_factory=lambda: json_target)
print(json_target.path.read_text())
print(deserialize(json_target, dict))
```

The serializer comes from the *pair*: the object's type and the requested target. A `dict` going
to `application/json` picks the JSON serializer; a `pandas.DataFrame` to `text/csv` picks the
dataframe serializer; a `datetime` picks the ISO 8601 serializer, because no file is involved.

## What can be serialized

| Object | Targets |
| --- | --- |
| `dict` of JSON-shaped data | JSON, YAML, itself |
| `pydantic.BaseModel` | JSON, YAML, `dict` of its dump |
| `pandas.DataFrame`, `polars.DataFrame` | CSV, parquet, Excel |
| `numpy.ndarray` | `.npy`, `.npz`, CSV |
| `PIL.Image.Image` | PNG, TIFF, BMP, JPEG, GIF, WebP |
| `xml.etree.ElementTree` | XML text, XML file |
| `bs4.BeautifulSoup` | HTML text, HTML file |
| `sqlalchemy` statements | SQL text, SQL file |
| `datetime`, `date`, `time` | ISO 8601 text |
| `timedelta` | a number of seconds |
| `complex` | a two-element `[real, imaginary]` list |
| `Decimal`, `uuid.UUID`, `pathlib.Path` | text |
| `bytes` | base64 text |
| `enum.Enum` | its value |
| anything picklable | pickle file |

Optional formats need the `common-serializers` extra:

```bash
uv add "based-datatypes[common-serializers] @ git+https://github.com/cloudlane-one/based-datatypes.git"
```

It installs the serializer implementations (`numpy`, `pandas`+`pyarrow`+`openpyxl`, `polars`
and its Excel backends, `pillow`, `beautifulsoup4`, `sqlalchemy`). Formats whose library is
missing simply have no serializer, so selection reports no candidate instead of failing inside
the serializer.

## Values JSON cannot hold

Scalars serialize to the primitive that carries them, so they can be embedded in a JSON
document and read back with the annotation you expect:

```python exec="on" session="serialize"
import datetime as dt
import uuid
from decimal import Decimal

from based_datatypes.serialize import deserialize, serialize

for value in (dt.datetime(2024, 1, 2, 3, 4, 5), dt.date(2024, 1, 2), 3 + 4j, Decimal("1.10")):
    print(f"{value!r} -> {serialize(value)!r}")

print(deserialize("2024-01-02T03:04:05", dt.datetime))
print(deserialize([3.0, 4.0], complex))
print(serialize(uuid.UUID(int=1)))
print(serialize(b"hello"))
```

These serializers produce values, not files: a scalar has nothing to put in a file of its own.
Embed it in a dictionary or list when you want it on disk.

## Reading back with an expected type

`deserialize(source, annotation)` validates what it parsed against the annotation. A source that
does not fit raises `TypeError`; a pair of source and annotation that no serializer handles
raises `ValueError`:

```python exec="on" session="serialize"
import datetime as dt

import pydantic

from based_datatypes.serialize import deserialize, serialize


class Record(pydantic.BaseModel):
    """A small pydantic model."""

    name: str
    created: dt.datetime


record_file = LocalFile[Literal["application/yaml"], TextIO](
    dir=LocalDir(path=workdir), name="record"
)
serialize(Record(name="a", created=dt.datetime(2024, 1, 1)), file_factory=lambda: record_file)
print(record_file.path.read_text())

print(deserialize(record_file, Record))

try:
    deserialize(record_file, list[int])
except ValueError as error:
    print("refused:", error)
```

Model fields are converted with pydantic's JSON mode first, so `datetime`, `UUID` and `Decimal`
values serialize instead of failing halfway through the write.

## Choosing a format

`get_serializers(typ, targets, scope, lossless)` returns the `(index, target, driver)` triples
that can handle an object, ordered by how well the target matches. It is what `serialize` and
`deserialize` use internally, and it is the way to ask *which* serializer would be picked:

```python exec="on" session="serialize"
from typing import BinaryIO

from based_datatypes.serialize import get_serializers

print(", ".join(type(driver).__name__ for _, _, driver in get_serializers(dt.datetime)))
print(get_serializers(dt.datetime, [File[Literal["image/png"], BinaryIO]]))
print(type(get_serializers(dict)[0][2]).__name__, "handles dicts")
```

## Lossless versus lossy formats

Some formats reproduce the object exactly, others keep only a reduced representation. A
serializer declares which of its MIME types round-trip, because one serializer can cover both:
parquet keeps a dataframe's dtypes and values, CSV does not; PNG is lossless, JPEG is not.

Pass `lossless=True` (the default) to require an exact round trip, `lossless=False` to accept a
reduced one:

```python exec="on" session="serialize"
import pandas as pd

from based_datatypes.serialize import get_serializers

csv_target = File[Literal["text/csv"], TextIO]
parquet_target = File[Literal["application/vnd.apache.parquet"], BinaryIO]


def drivers(targets, lossless):
    """Return the names of the serializers that accept the targets."""
    return ", ".join(
        type(driver).__name__
        for _, _, driver in get_serializers(pd.DataFrame, targets, lossless=lossless)
    )


print(drivers([csv_target], lossless=True))
print(drivers([csv_target], lossless=False))
print(drivers([parquet_target], lossless=True))
```

Reading is not restricted this way: losslessness describes what an export can drop, so
`deserialize` reads lossy formats as readily as lossless ones.

## Estimating the export size

`estimate_serialized_size(obj, target_type)` reports how many bytes the serialization would
produce, without writing anything. Speed wins over accuracy: an estimate only performs a full
serialization when that serialization *is* the cheap route — a single C-level pass — and
otherwise samples or bounds the payload.

| Format | Basis |
| --- | --- |
| JSON | exact, measured with the C encoder |
| YAML | the compact JSON payload, a rough upper bound (the YAML dumper is pure Python and ~150x slower) |
| pydantic dumps | the JSON-mode dump, measured like JSON |
| XML | exact, measured with the C serializer |
| HTML | a tag walk, about 10 % above the real size (rendering the soup is ~10x slower) |
| SQL | a strided sample of statements, formatted and extrapolated |
| pickle | exact, measured with the C pickler |
| pandas/polars parquet | in-memory / estimated size |
| pandas, polars CSV | sampled every n-th row and extrapolated |
| `.npy`, `.npz` | array payload plus the npy header |
| images | uncompressed pixel payload |

```python exec="on" session="serialize"
from based_datatypes.serialize import estimate_serialized_size

payload = {"a": 1, "b": [1, 2]}
written = serialize(payload, File[Literal["application/json"], TextIO])

print("estimated", estimate_serialized_size(payload, File[Literal["application/json"], TextIO]))
print("written", len(written.content.encode()))
```

## Writing a serialized dictionary as a directory tree

`write_structure` lays a dictionary of primitives and files out on disk:

- primitives are collected in `__index__.yaml` inside the directory,
- files are written next to it, named after their key,
- nested dictionaries and lists become sub-directories named after their key,
- the `__index__.yaml` of a list directory holds a YAML list, with `None` at the positions taken
  by files and sub-directories, whose files are named after their position.

```python exec="on" session="serialize"
from PIL import Image

from based_datatypes.serialize import read_structure, write_structure

image = serialize(Image.new("RGB", (2, 2), (10, 20, 30)), File[Literal["image/png"], BinaryIO])

structure = {
    "name": "thumbnail",
    "size": [2, 2],
    "image": image,
    "meta": {"author": "me", "tags": ["a", "b"]},
}

directory = write_structure(structure, workdir / "bundle")
print(sorted(directory.keys()))
print((workdir / "bundle" / "__index__.yaml").read_text())
print((workdir / "bundle" / "meta" / "__index__.yaml").read_text())

restored = read_structure(workdir / "bundle")
print(restored["name"], restored["size"], restored["meta"])
print(restored["image"].path.name)
```

Files read back as `LocalFile` handles, so they can be passed straight to `deserialize`.

## Writing a serializer for your own type

Subclass `Serializer[YourType, TargetHint]`, implement `serialize` and `deserialize`, and
register an instance. The class's type arguments are what selection reads, and each member of
the target union becomes a declared format:

```python exec="on" session="serialize"
from based_datatypes.serialize import Serializer, deserialize, register_serializer, serialize
from based_datatypes.serialize.base import Open, Realm, TransportScope
from based_datatypes.serialize.common import CommonSerializer


class Point:
    """A point in the plane."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"


type PointFile = File[Literal["text/plain"], TextIO]


class PointSerializer(CommonSerializer[Point, str | PointFile]):
    """Read and write points as ``x,y``."""

    __lossless_mimes__ = frozenset({"text/plain"})

    def serialize(
        self,
        instance: Point,
        target_type: object = None,
        lossless: bool = True,
        scope: TransportScope = Open(),
        annotation: object = None,
        realm: Realm | None = None,
        file_factory: object = None,
    ) -> str | PointFile:
        """Write the point as text, or return the text for a text target."""
        text = f"{instance.x},{instance.y}"
        target = self._resolve_target(target_type, file_factory)  # type: ignore[arg-type]

        if target is None:
            return text

        with target.open("w") as stream:
            stream.write(text)

        return target

    def deserialize(
        self,
        source: str | PointFile,
        scope: TransportScope = Open(),
        realm: Realm | None = None,
        annotation: object = None,
    ) -> Point:
        """Read a point from text or from a text file."""
        text = source
        if isinstance(source, File):
            with source.open("r") as stream:
                text = stream.read()

        x, y = text.split(",")
        return Point(float(x), float(y))


register_serializer(PointSerializer())

print(serialize(Point(1.5, 2.5), str))
print(serialize(Point(1.5, 2.5)))
print(deserialize("1.5,2.5", Point))
```

`CommonSerializer` supplies the target handling: `_resolve_target` turns a target type or a
`file_factory` into the file to write to, and returns `None` when the target is a plain value;
`__lossless_mimes__` declares which of the declared MIME types round-trip exactly.

## The `Serializable` protocol

When the object itself knows how to serialize, implement the protocol instead of a serializer:
`__serialization_targets__`, `__deserialize__`, `__serialize__` and
`__estimate_serialized_size__`. `serialize`, `deserialize` and `estimate_serialized_size` then
dispatch to the object:

```python exec="on" session="serialize"
from based_datatypes.files import MemoryFile
from based_datatypes.serialize import (
    Open,
    SerializationTarget,
    deserialize,
    estimate_serialized_size,
    serialize,
)
from based_datatypes.serialize.base import Realm, TransportScope


class Document:
    """Plain text that serializes itself."""

    def __init__(self, text: str) -> None:
        self.text = text

    @classmethod
    def __serialization_targets__(cls) -> list[SerializationTarget]:
        """Only plain text is supported."""
        return [
            SerializationTarget(
                type=File[Literal["text/plain"], TextIO], scope=Open(), lossless=True
            )
        ]

    @classmethod
    def __deserialize__(
        cls,
        source: File,
        scope: TransportScope = Open(),
        realm: Realm | None = None,
        annotation: object = None,
    ) -> "Document":
        """Read the text from the source file."""
        with source.open("r") as stream:
            return cls(stream.read())

    def __serialize__(
        self,
        target_type: object = None,
        lossless: bool = True,
        scope: TransportScope = Open(),
        annotation: object = None,
        realm: Realm | None = None,
        file_factory: object = None,
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
        target_type: object = None,
        lossless: bool = True,
        scope: TransportScope = Open(),
        annotation: object = None,
    ) -> int:
        """Measure the encoded text."""
        return len(self.text.encode())


document = serialize(Document("hello"))
print(document.content)
print(deserialize(document, Document).text)
print(estimate_serialized_size(Document("hello"), File[Literal["text/plain"], TextIO]))
```

## Traps worth knowing

- **A target names its format.** A bare `File` hint matches nothing: ask for
  `File[Literal["application/json"], TextIO]`, or pass a `file_factory` and let the serializer
  pick its default format. The factory decides *where* the bytes go, the target type decides
  *what* they are.
- **A failed encode does not truncate an existing file.** Text payloads are built before the
  target is opened, so a value the serializer rejects leaves the previous content intact.
- **Object arrays are refused for `.npy`/`.npz`.** Writing them would require pickle and loading
  them would require `allow_pickle=True`; convert the array to a concrete dtype first.
- **Losslessness is per format, not per serializer.** Asking for `lossless=True` with a CSV
  target has no candidate even though the same serializer writes lossless parquet.
- **A missing pair raises `ValueError`.** `serialize` and `deserialize` name the object type, the
  target and the active scope in the message.
