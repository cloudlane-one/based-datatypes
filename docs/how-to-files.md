# Files and directories

`based_datatypes.files` gives files and directories one small interface, so storage code does
not care whether a file lives on the local disk or behind object storage later. A file knows
its MIME type and whether it carries text or bytes, and it hands out an open stream on
request; a directory behaves like a mapping of names to files and sub-directories.

## When to use it

- Pass a file around (to a serializer, a job, a test fixture) without passing paths and
  reopening them everywhere.
- Derive a MIME type and file extension for content you are about to write.
- Walk, count, or remove a directory tree with the mapping protocol instead of `os` calls.

## Quick start

```python exec="on" session="files"
import tempfile
from pathlib import Path

from based_datatypes.files import LocalFile

tmp = tempfile.TemporaryDirectory()
path = Path(tmp.name) / "notes.txt"
path.write_text("hello", encoding="utf-8")

file = LocalFile.from_path(path)

print(file.mime, file.io_type.__name__, file.detected_format, file.extension)
with file.open("r") as stream:
    print(stream.read())
```

`from_path` reads the suffix and the file's tags, so a JSON file becomes
`application/json` with a text stream and a `.json` extension, while a PNG becomes
`image/png` with a binary stream.

## Reading and writing

`open(mode)` takes `"r"`, `"w"` or `"rw"` and closes the stream when the block ends.
Text streams are always UTF-8 and never translate line endings, so the bytes on disk are the
same on every platform.

```python exec="on" session="files"
from typing import TextIO

target = LocalFile(
    dir=file.dir,
    name="out",
    mime="application/json",
    io_type=TextIO,
)
with target.open("w") as stream:
    stream.write('{"a": 1}')

print(target.path.name, target.path.read_bytes())
```

`get_io(mode)` returns the stream without a context manager when you need to hand ownership
to another library; you are then responsible for closing it.

## MIME type and extension

`mimetypes` cannot name every format a data library cares about, so the module keeps a small
supplementary table (`.parquet`, `.npy`, `.npz`, `.pkl`, `.yaml`, `.xlsx`, `.sql`, ...). The
table is consulted first, and the extension is derived from the MIME type when a file is
constructed from scratch:

```python exec="on" session="files"
from typing import TextIO

from based_datatypes.files import additional_mime, additional_mime_inv

print(additional_mime[".parquet"])
print(additional_mime_inv["application/yaml"])

synthetic = LocalFile(dir=file.dir, name="frame", mime="text/csv", io_type=TextIO)
print(synthetic.extension)
```

A file created by `from_path` keeps the suffix it actually has; a file you construct yourself
gets the canonical extension for its MIME type unless you pass `_suffix`.

## Directories

`LocalDir` is a mapping: `keys()`, `len()`, `in`, `[]`, `del` and iteration over its entries.
Indexing a sub-directory returns another `LocalDir`, indexing a file returns a `LocalFile`.

```python exec="on" session="files"
from based_datatypes.files import LocalDir

sub = Path(tmp.name) / "sub"
sub.mkdir()
(sub / "a.csv").write_text("x\n1\n", encoding="utf-8")

directory = LocalDir(path=tmp.name)
print(sorted(directory.keys()), len(directory))

print(type(directory["sub"]).__name__, type(directory["notes.txt"]).__name__)
print(directory.file("notes.txt").mime)
print([name for name, item in directory.items()])
```

`directory.file(name)` and `directory.subdir(name)` are typed accessors that assert the kind
of entry. Deleting a sub-directory removes the whole tree:

```python exec="on" session="files"
del directory["sub"]
print(directory.keys())
```

Creating a `LocalDir` for a path that does not exist yet is allowed: the directory is created
when you first read or write through it.

## Text or binary

Every file carries an `io_type` (`TextIO` or `BinaryIO`) and a `detected_format` of
`"text"` or `"binary"`. `from_path` decides from the file's tags, and falls back to the MIME
type for a path that does not exist yet. Transport drivers use this to reject a text target
for a binary format before any bytes are written.
