# Stable hashes of Python objects

`based_datatypes.hashing` computes integers and short hex strings from Python values. The
hash depends on the *value*, not on `id()`, so it survives pickling, process restarts and
re-serialisation of the same data.

## When to use it

Use it to compare or key data that is not hashable (lists, dicts, dataclasses, dataframes),
to cache derived results, or to derive short, readable identifiers for test cases and
config snapshots.

Do **not** use it as a cryptographic digest or as a replacement for `hash()` in hot paths:
it is deterministic, not secret, and slower than `hash`.

## Quick start

```python exec="on" session="hashing"
from based_datatypes.hashing import gen_int_hash, gen_str_hash

first = {"name": "ada", "roles": ["admin", "user"]}
second = {"roles": ["admin", "user"], "name": "ada"}

print(gen_int_hash(first) == gen_int_hash(second))
print(gen_str_hash(first))
print(gen_str_hash(first, length=6))
```

## API

| Name | Purpose |
| --- | --- |
| `gen_int_hash(obj)` | Stable non-negative integer hash of a supported object. |
| `gen_str_hash(obj, length=10)` | Left-truncated, zero-padded hex representation of `gen_int_hash`. |

## What is supported

| Input | Behaviour |
| --- | --- |
| `None` | `0`. |
| Modules | Hash of the module name. |
| Classes and named aliases | Hash of `(module, name)`. |
| Named functions | Hash of the function's `PyObjectRef` URL (module + qualname). |
| Generic aliases and unions | Hash of the origin and the arguments. |
| `int`, `float`, `bool`, `str`, `bytes`, dates and times | Hash of the type and value. |
| `tuple`, `list` | Ordered composition; a tuple and a list with equal items hash alike. |
| `dict` | Key/value pairs, independent of insertion order. |
| `set` | Members, independent of iteration order. |
| pandas `DataFrame`, `Series`, `Index` | `pandas.util.hash_pandas_object`. |
| Any other hashable object | Its own `__hash__` (see the caveat below). |
| Unhashable dataclass | Hash of its field mapping. |
| Unhashable object with `__getstate__` | Hash of the state mappings. |
| Unhashable object with `__getnewargs__(_ex)` | Hash of the constructor arguments. |

Anything else raises `ValueError`.

```python exec="on" session="hashing"
from dataclasses import dataclass

from based_datatypes.hashing import gen_int_hash


class Unsupported:
    """Unhashable and without pickling hooks."""

    __hash__ = None  # type: ignore[assignment]


try:
    gen_int_hash(Unsupported())
except ValueError:
    print("unsupported")
```

## Structured data

```python exec="on" session="hashing"
from dataclasses import dataclass

from based_datatypes.hashing import gen_int_hash


@dataclass
class Config:
    """Unhashable dataclass; fields are hashed structurally."""

    host: str
    port: int


print(gen_int_hash(Config("localhost", 80)) == gen_int_hash({"host": "localhost", "port": 80}))
```

Cycles terminate: a container that references itself contributes `0` at the repeated
reference instead of recursing forever.

```python exec="on" session="hashing"
from based_datatypes.hashing import gen_int_hash

cyclic: list[object] = [1]
cyclic.append(cyclic)

print(gen_int_hash(cyclic) == gen_int_hash(cyclic))
```

## Pandas

```python exec="on" session="hashing"
import pandas as pd

from based_datatypes.hashing import gen_int_hash

frame = pd.DataFrame({"a": [1, 2, 3]})

print(gen_int_hash(frame) == gen_int_hash(pd.DataFrame({"a": [1, 2, 3]})))
print(gen_int_hash(frame) != gen_int_hash(pd.DataFrame({"a": [3, 2, 1]})))
```

!!! warning "Identity hashes are not portable"
    Python's built-in `hash()` for `str` is randomised per process, and for most objects it
    is derived from `id()`. Because the fallback branch uses `__hash__`, hashing an object
    that is neither one of the supported types nor structurally hashable is only stable
    *within* a process. Supported types, dataclasses, pandas objects and picklable objects
    are stable across processes.

!!! note "`Hashable` wins over pickling hooks"
    The fallback order is deliberate: any hashable object uses its `__hash__` before the
    dataclass and `__getstate__` branches. Those branches apply to unhashable objects. A
    class that defines `__getstate__` and is also hashable will use `__hash__`.

## Short identifiers

`gen_str_hash` returns the low-order hex digits of the integer hash, left-padded with zeros
to the requested length. Truncation makes collisions more likely, so keep the default width
unless the identifier is display-only.

```python exec="on" session="hashing"
from based_datatypes.hashing import gen_str_hash

print(repr(gen_str_hash(None)))
print(repr(gen_str_hash("value", length=0)))
print(gen_str_hash("value", length=16))
```
