# Transport and serialization

`based_datatypes.transport` moves Python objects through concrete formats — files first —
without the caller wiring up a library per format. You describe *what* you have and *where*
it should go; the layer picks a driver that understands both, checks that the format is
allowed for the current transport scope, and can tell you how large the export would be
before you write it.

## When to use it

- Write a dict, model, dataframe, array, image, XML tree or SQL statement to a file without
  importing a different library for each.
- Read content back while declaring the type you expect, so a mismatched file fails loudly.
- Keep "this may only travel within this machine / network / anywhere" as a first-class rule
  instead of a comment.
- Ask for a lossless format when round-tripping matters, and accept a lossy one when it does
  not.
- Budget an export before writing it.

## Quick start

```python exec="on" session="transport"
import tempfile
from pathlib import Path
from typing import BinaryIO, TextIO

from based_datatypes.files import LocalDir, LocalFile
from based_datatypes.transport import Realm, export_object, import_object

tmp = tempfile.TemporaryDirectory()
workdir = Path(tmp.name)


def target(name: str, mime: str, *, binary: bool = False) -> LocalFile:
    """Build a file handle for a new file in the temporary directory."""
    return LocalFile(
        dir=LocalDir(path=workdir),
        name=name,
        mime=mime,
        io_type=BinaryIO if binary else TextIO,
    )


realm = Realm()
payload = {"hello": "world", "items": [1, 2]}

json_file = target("data", "application/json")
export_object(payload, json_file, realm)
print(json_file.path.read_text())
print(import_object(json_file, realm, dict))
```

The driver comes from the *pair*: the object's type and the target's MIME type. A `dict` going
to `application/json` picks the JSON driver; the same dict to `application/pickle` picks the
pickle driver; a `datetime`-bearing pydantic model picks the model driver.

## What can be transported

| Object | Formats |
| --- | --- |
| `dict` of JSON-shaped data | JSON, YAML, pickle |
| pydantic `BaseModel` | JSON, YAML |
| `pandas.DataFrame` | CSV, parquet, Excel |
| `polars.DataFrame` | CSV, parquet, Excel |
| `numpy.ndarray` | `.npy`, `.npz`, CSV |
| `PIL.Image.Image` | PNG, JPEG, GIF, BMP, TIFF, WebP |
| `xml.etree.ElementTree` | XML |
| `bs4.BeautifulSoup` | HTML |
| `sqlalchemy` statements | SQL text |
| anything picklable | pickle (offline and same-machine scopes only) |

Optional formats need the `common-transport` extra:

```bash
uv add "based-datatypes[common-transport] @ git+https://github.com/cloudlane-one/based-datatypes.git"
```

It installs the serializer implementations (`numpy`, `pandas`+`pyarrow`+`openpyxl`, `polars`
and its Excel backends, `pillow`, `beautifulsoup4`, `sqlalchemy`). Formats whose library is
missing simply have no driver, so selection reports an unsupported medium instead of failing
inside the serializer.

## Reading back with an expected type

`import_object(source, realm, annotation)` validates what it parsed against the annotation and
raises `TypeError` when the file does not match:

```python exec="on" session="transport"
import pydantic

from based_datatypes.transport import export_object, import_object


class Record(pydantic.BaseModel):
    """A small pydantic model."""

    name: str
    count: int


record_file = target("record", "application/yaml")
export_object(Record(name="a", count=2), record_file, realm)
print(record_file.path.read_text())
print(import_object(record_file, realm, Record))

try:
    import_object(record_file, realm, list[int])
except TypeError as error:
    print("refused:", error)
```

Model fields are converted with pydantic's JSON mode first, so `datetime`, `UUID` and
`Decimal` values serialize instead of failing halfway through the write.

## Selecting a driver yourself

`get_transport_driver(instance_or_type, targets)` returns the driver class, or `None` when
nothing fits. Pass the target *instance* to match on its MIME type and IO type; pass only the
object to ask which driver would handle it in general.

```python exec="on" session="transport"
from based_datatypes.transport import get_transport_driver, get_transport_types

pickle_file = target("payload", "application/pickle", binary=True)

print(get_transport_driver({}, [json_file]).__name__)
print(get_transport_driver({}, [pickle_file]).__name__)
print(get_transport_driver({}, [target("page", "text/html")]))
print(len(get_transport_types(dict).types), "declared formats for dict")
```

## Lossless versus lossy formats

Some formats reproduce the object exactly, others keep only a reduced representation. A driver
declares which of its formats are lossless per MIME type, because one driver can cover both:
parquet keeps a dataframe's dtypes and values, CSV does not; PNG is lossless, JPEG is not.
Pass `lossless=True` to require an exact round trip, `lossless=False` when you explicitly want
a reduced one.

```python exec="on" session="transport"
import pandas as pd

frame = pd.DataFrame({"x": [1, 2]})
csv_file = target("frame", "text/csv")
parquet_file = target("frame", "application/vnd.apache.parquet", binary=True)

print(get_transport_driver(frame, [csv_file], lossless=True))
print(get_transport_driver(frame, [csv_file], lossless=False).__name__)
print(get_transport_driver(frame, [parquet_file], lossless=True).__name__)
```

Lossless formats: JSON, YAML, XML, parquet, `.npy`/`.npz`, PNG/TIFF/BMP, pickle. Lossy: CSV,
Excel, JPEG/GIF/WebP, HTML, SQL text, and pydantic dumps (aliases and excluded fields can be
lost).

## Transport scopes

A scope says how far the bytes may travel: `Open` (offline only), `IPC` (another process on the
same machine), `LAN` (local network), `WAN` (the internet). The classes nest — `WAN` is a `LAN`,
which is an `IPC` — so a driver declared for WAN can also serve an IPC request: it is able to
travel further than asked. An `Open` request imposes no restriction, so every driver satisfies
it.

Set the scope for a block with `transport_scope`; drivers and estimation respect it
automatically:

```python exec="on" session="transport"
from based_datatypes.transport import IPC, LAN, export_object, transport_scope

payload_file = target("payload", "application/pickle", binary=True)
export_object({"a": 1}, payload_file, realm)

with transport_scope(LAN(private_subnet="10.0.0.0/8")):
    try:
        export_object({"a": 1}, payload_file, realm)
    except TypeError as error:
        print("refused:", str(error).split(" under ")[0])
    print("json still fine:", get_transport_driver({}, [json_file]).__name__)

with transport_scope(IPC()):
    print("ipc pickle:", get_transport_driver({}, [payload_file]).__name__)
```

Pickle is restricted to `Open` and `IPC`: unpickling executes arbitrary code, so it never
crosses a network boundary here. That restriction is enforced twice — once when selecting a
driver, and again inside the pickle driver for callers that bypass selection.

## Estimating the export size

`estimate_bytes(obj, target, realm)` reports how many bytes the export would produce without
writing to the target. The number is exact where measuring is cheap and an approximation where
it is not:

| Format | Basis |
| --- | --- |
| JSON, YAML, XML, HTML, SQL, pickle | exact encoded payload |
| pydantic, pandas/polars parquet | exact model dump / in-memory size |
| pandas, polars CSV | sampled every n-th row and extrapolated |
| `.npy`, `.npz` | array payload plus the npy header |
| images | uncompressed pixel payload |

```python exec="on" session="transport"
from based_datatypes.transport import estimate_bytes

payload = {"a": 1, "b": [1, 2]}
budget_file = target("budget", "application/json")

estimated = estimate_bytes(payload, budget_file, realm)
export_object(payload, budget_file, realm)
print("estimated", estimated, "written", budget_file.path.stat().st_size)
```

## Writing a driver for your own type

Subclass `TransportDriver[YourType, FileHint]`, implement `import_object` and `export_object`,
and register it. The class's type arguments are what selection reads, so declare the file
formats you accept there.

```python exec="on" session="transport"
from typing import Literal

from based_datatypes.files import File
from based_datatypes.transport import (
    TransportDriver,
    export_object,
    import_object,
    register_transport_driver,
)


class Point:
    """A point in the plane."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


type PointFile = File[Literal["text/plain"], TextIO]


class PointDriver(TransportDriver[Point, PointFile]):
    """Read and write points as ``x,y``."""

    @classmethod
    def export_object(cls, instance, target, realm, annotation=None):
        with target.open("w") as stream:
            stream.write(f"{instance.x},{instance.y}")

    @classmethod
    def import_object(cls, source, realm, annotation=None):
        with source.open("r") as stream:
            x, y = stream.read().split(",")
        return Point(float(x), float(y))


register_transport_driver(Point, PointDriver)

point_file = target("point", "text/plain")
export_object(Point(1.5, 2.5), point_file, realm)
restored = import_object(point_file, realm, Point)
print(point_file.path.read_text(), restored.x, restored.y)
```

A driver can also override `__transport_scopes__` to narrow where its format may go, and
`__lossless_formats__` to declare which MIME types round-trip exactly. `estimate_bytes` defaults
to `sys.getsizeof`; override it to measure the encoded payload instead.

## The `Transportable` protocol

When the object itself knows how to serialize, implement the protocol instead of a driver:
`__transport_types__`, `__transport_scopes__`, `__import__`, `__export__` and
`__estimate_bytes__`. `export_object`, `import_object` and `estimate_bytes` then dispatch to the
object, including the scope check.

```python exec="on" session="transport"
from based_datatypes.transport import (
    Open,
    TransportTypes,
    estimate_bytes,
    export_object,
    import_object,
)


class Document:
    """Plain text that serializes itself."""

    def __init__(self, text: str) -> None:
        self.text = text

    @classmethod
    def __transport_types__(cls) -> TransportTypes[TextIO]:
        return TransportTypes((File[Literal["text/plain"], TextIO],))

    @classmethod
    def __transport_scopes__(cls) -> frozenset[type]:
        return frozenset({Open})

    @classmethod
    def __import__(cls, source, realm, annotation=None):
        with source.open("r") as stream:
            return cls(stream.read())

    def __export__(self, target, realm, annotation=None):
        with target.open("w") as stream:
            stream.write(self.text)

    def __estimate_bytes__(self, target, realm) -> int:
        return len(self.text.encode())


doc_file = target("doc", "text/plain")
export_object(Document("hello"), doc_file, realm)
print(import_object(doc_file, realm, Document).text)
print(estimate_bytes(Document("hello"), doc_file, realm))
```

## Traps worth knowing

- **A failed encode does not truncate an existing file.** Text payloads are built before the
  target is opened, so a value the serializer rejects leaves the previous content intact.
- **Object arrays are refused for `.npy`/`.npz`.** Writing them would require pickle, and
  loading them would require `allow_pickle=True`, which would bypass the scope restriction.
  Convert the array to a concrete dtype first.
- **Rates are per format, not per driver.** Asking for `lossless=True` with a CSV target
  returns `None` even though the same driver writes lossless parquet.
- **An unsupported pair raises `TypeError`.** `export_object` and `import_object` name the
  object type, the target and the active scope in the message, because "no driver found" alone
  rarely explains which of the three criteria failed.
