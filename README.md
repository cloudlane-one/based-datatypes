# Based Datatypes

Shared Python data types, typing helpers and data utilities for the
[`cloudlane-one`](https://github.com/cloudlane-one) Python projects.

The package exists so that recurring questions are answered once instead of in every project:
how to make an enum print readably, how to resolve a type hint at runtime, how to serialise an
arbitrary object to a file format, how to hash a dataframe deterministically. It is a library
only — no CLI and no services; the file and serialization helpers touch the disk only when
asked to.

## Install

The package is not published on PyPI yet; install it from the Git remote:

```bash
uv add "based-datatypes @ git+https://github.com/cloudlane-one/based-datatypes.git"
```

Requires Python 3.13 or newer.

Some serializers need third-party format backends (parquet, Excel, images, HTML, SQL). They are
optional; install the extra when you need them:

```bash
uv add "based-datatypes[common-serializers] @ git+https://github.com/cloudlane-one/based-datatypes.git"
```

Formats whose backend is missing simply have no serializer, so serializer selection reports no
candidate instead of failing halfway through a write.

## Quick start

```python
from based_datatypes.hashing import gen_str_hash
from based_datatypes.typing import TypeRef, is_subtype

print(is_subtype(list[int], list))
print(TypeRef(list[int]).resolved_type_def)
print(TypeRef(int | None).typeset)
print(gen_str_hash({"host": "localhost", "port": 8080}))
```

## What is in the box

| Module | What it gives you | Guide |
| --- | --- | --- |
| `based_datatypes.enums` | String enums that render as their value in docs and logs. | [Enums](docs/how-to-enums.md) |
| `based_datatypes.types` | Nominal string types, state markers and structural protocols. | [Types and protocols](docs/how-to-types.md) |
| `based_datatypes.context` | Scoped `ContextVar` values that restore themselves. | [Context variables](docs/how-to-context.md) |
| `based_datatypes.runtime` | Calling module, call arguments, return types and subclass trees. | [Runtime reflection](docs/how-to-runtime.md) |
| `based_datatypes.typing` | Runtime type-hint resolution, comparison and generic type maps. | [Type reflection](docs/how-to-typing.md) |
| `based_datatypes.hashing` | Stable hashes of containers, dataclasses, functions and dataframes. | [Hashing](docs/how-to-hashing.md) |
| `based_datatypes.data` | Dataclass copying, container conversion and mapping flattening. | [Data utilities](docs/how-to-data.md) |
| `based_datatypes.files` | One interface for single files and directory trees, with MIME types and typed streams. | [Files and directories](docs/how-to-files.md) |
| `based_datatypes.serialize` | Move objects through JSON, YAML, parquet, arrays, images and more. | [Serialization](docs/how-to-serialize.md) |
| `based_datatypes.descriptors` | Typed, cacheable properties and dual-dispatch methods. | [Descriptors](docs/how-to-descriptors.md) |
| `based_datatypes.ref` | Persistent, URL-serialisable references to Python objects. | [Object references](docs/how-to-refs.md) |
| `based_datatypes.dist` | Distribution and Git-repository lookup for modules. | [Distributions](docs/how-to-dist.md) |

## Documentation

The full documentation lives in [`/docs`](docs/index.md):

- [Documentation home](docs/index.md) — the entry point for every guide.
- The usage guides linked in the table above — one task-oriented page per module, with
  runnable examples.
- [Python API reference](docs/ref-python-api.md) — generated from the docstrings, covering
  every public module, class, function and attribute.
- [Directory structure](docs/ref-directory-structure.md) — how the repository is laid out.
- [Code conventions and tooling](docs/ref-code-conventions-tooling.md) — formatting, linting,
  type checking and testing conventions.
- [How to contribute](docs/how-to-contribute.md) — contribution types, pull requests and
  commit conventions.

## Development

Setup, checks and the documentation build are described in
[Local setup](docs/how-to-setup-locally.md) and
[How to check quality](docs/how-to-check-quality.md). In short:

```bash
uv sync                                         # install the development environment
uv run pytest                                   # run the test suite
uv run properdocs build -f docs/.config.yml     # build the documentation into dist/docs
uv run properdocs serve -f docs/.config.yml     # serve it locally with auto-reload
```
