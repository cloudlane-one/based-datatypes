# Persistent references to Python objects

`based_datatypes.ref` turns a live Python object into a small, serialisable record that
says *what* the object is and *where it came from*: module, qualified name, repository (or
package index), version range, and the Python version it was resolved under.

The record can be rendered as a URL, parsed back, and used later to import and resolve the
same object again — in another process, on another machine, or in a generated document.

## When to use it

Use it to build registries, documentation links, experiment provenance, or plugin
manifests that must survive a restart. Do not use it as a general serialisation format for
object state; it stores a *pointer*, not the object.

## Quick start

```python exec="on" session="refs"
from based_datatypes.ref import PyObjectRef

reference = PyObjectRef.reference(int)

print(reference)
print(reference.url)
print(reference.object)
print(reference.module.__name__)
```

## API

| Name | Purpose |
| --- | --- |
| `PyObjectRef.reference(obj, version=None, version_strategy="major")` | Build a reference from a live object. |
| `PyObjectRef.from_url(text, obj_type)` | Parse a reference URL back into a record. |
| `PyObjectRef.url` | URL rendering of the record (cached). |
| `PyObjectRef.module` / `.dist` / `.object` | Import, verify, and resolve on demand (cached). |
| `PyObjectRef.docs_url` | Deep link into the package's documentation, when discoverable. |
| `stref(obj)` | Short `module.qualname` string. |
| `get_all_module_dependencies(module)` | External package names and internal modules a module uses. |
| `version_diff`, `version_to_range`, `semver_range_to_spec` | Version arithmetic used by the record. |

## What a reference records

`reference` inspects the object's module and distribution:

- Built-in objects get no repository and no version.
- Objects installed from a distribution get the distribution's version (as a semver
  range by default) and the PyPI simple index as their repository.
- Objects belonging to a Git checkout (editable installs, or code that is not packaged)
  get the checkout's remote URL and either the installed version or the tag/branch name.

```python exec="on" session="refs"
import pandas

from based_datatypes.ref import PyObjectRef

reference = PyObjectRef.reference(pandas.DataFrame)
print(reference.module_name, reference.qualname)
print(reference.repo_url, reference.version)
```

!!! note "Versions that are not semver"
    If the distribution (or tag) is not a valid semantic version, the reference carries no
    version unless `version_strategy="exact"` is used, in which case it pins `==<version>`.

## Rendering and parsing

The URL uses a `py:` scheme. `py+<scheme>` records where the code lives, and the query
carries the version constraints. The fragment is `module:qualname`.

```python exec="on" session="refs"
from based_datatypes.ref import PyObjectRef

original = PyObjectRef(
    object_type=int,
    qualname="Parser.parse",
    module_name="my_package.parsing",
    repo_url="https://github.com/example/project",
    version="^1.4",
    py_version="~3.12",
)

text = str(original.url)
print(text)

restored = PyObjectRef.from_url(text, int)
print(restored.repo_url, restored.version, restored.py_version)
print(restored.module_name, restored.qualname)
```

!!! warning "The fragment is not URL-decoded for display"
    Nested qualnames (`Outer.method`) and dots survive the round trip because the fragment
    is split on the first colon only. Use `from_url` rather than parsing the string by hand.

## Resolving later

`module` imports the module, `dist` verifies that the installed distribution satisfies the
recorded version constraint, and `object` walks the qualified name and checks the resulting
type.

```python exec="on" session="refs"
from based_datatypes.ref import PyObjectRef

reference = PyObjectRef.from_url("py:?python=~3.12#builtins:int", type)

print(reference.module.__name__)
print(reference.object)
```

A wrong `object_type` is reported, not silently accepted:

```python exec="on" session="refs"
from based_datatypes.ref import PyObjectRef

reference = PyObjectRef(object_type=str, qualname="int", module_name="builtins")

try:
    reference.object
except TypeError as error:
    print(type(error).__name__, error)
```

!!! note "Built-ins skip the distribution check"
    A reference without a repository (a built-in) has no distribution to verify, so
    `module`/`object` work without an installed package of the same name.

## Documentation links

`docs_url` looks up the distribution's `Project-URL` entries. If a `Documentation` URL is
present, its Sphinx `objects.inv` is searched for the object and the exact anchor is
returned when found (otherwise the documentation root). Only when no `Documentation` URL is
declared does it probe an `API Reference` URL for a matching page. Both discovery steps
perform HTTP requests, so cache the result or catch failures at the call site.

## Short names

```python exec="on" session="refs"
from based_datatypes.ref import PyObjectRef, stref

print(stref(int))
print(stref(PyObjectRef))
```

## Version arithmetic

The helpers are plain functions and can be used independently of references.

```python exec="on" session="refs"
from packaging.version import Version

from based_datatypes.ref import (
    semver_range_to_spec,
    version_diff,
    version_to_range,
)

print(version_diff(Version("2.3.4"), Version("1.2.3")))
print(version_diff(Version("1.0.0"), Version("2.0.0")))
print(version_to_range(Version("1.2.3"), "minor"))
print(semver_range_to_spec("^1.2.3"))
print(Version("1.9") in semver_range_to_spec("^1.2.3"))
```

`semver_range_to_spec` accepts `^` (caret), `~` (tilde), the comparison operators
`>=`, `<=`, `>`, `<`, `==`/`=` and a bare version.

## Dependencies of a module

`get_all_module_dependencies` walks the members of a module, classifying each module they
come from as external (part of an installed distribution) or internal, and recurses into
the internal ones. Cycles are handled, and the module itself is never reported as its own
dependency.

```python exec="on" session="refs"
from based_datatypes import enums, ref

external, internal = ref.get_all_module_dependencies(enums)
print("external:", sorted(external))
print(enums in internal)
```
