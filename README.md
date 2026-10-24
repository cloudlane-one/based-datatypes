# Based Datatypes

Shared Python data types, typing helpers and data utilities for the
`cloudlane-one/based-datatypes` project.

## Install

The package is not published on PyPI yet; install it from the Git remote:

```bash
uv add "based-datatypes @ git+https://github.com/cloudlane-one/based-datatypes.git"
```

Requires Python 3.13 or newer.

## What is in the box

| Module | What it gives you |
| --- | --- |
| `based_datatypes.enums` | String enums that render as their value in docs and logs. |
| `based_datatypes.types` | Nominal string types, state markers, structural protocols. |
| `based_datatypes.context` | Scoped `ContextVar` values that restore themselves. |
| `based_datatypes.runtime` | Calling module, call arguments, return types, subclass trees. |
| `based_datatypes.typing` | Runtime type-hint resolution, comparison and generic type maps. |
| `based_datatypes.hashing` | Stable hashes of containers, dataclasses, functions and dataframes. |
| `based_datatypes.data` | Dataclass copying, container conversion, mapping flattening. |
| `based_datatypes.descriptors` | Typed, cacheable properties and dual-dispatch methods. |
| `based_datatypes.ref` | Persistent, URL-serialisable references to Python objects. |
| `based_datatypes.dist` | Distribution and Git-repository lookup for modules. |

## Quick start

```python
from based_datatypes.hashing import gen_str_hash
from based_datatypes.typing import TypeRef, is_subtype

print(is_subtype(list[int], list))
print(TypeRef(list[int]).typeform)
print(gen_str_hash({"host": "localhost", "port": 8080}))
```

## Documentation

See [`/docs`](docs/index.md) for usage guides per module, contribution guides, project
references and the generated API documentation.
