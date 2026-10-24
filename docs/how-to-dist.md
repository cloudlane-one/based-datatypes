# Reflecting installed distributions

`based_datatypes.dist` answers questions about the Python environment: which distributions
are installed, which distribution a module belongs to, which Git repository a module lives
in, and what a distribution's project metadata points at.

It is the low-level half of `based_datatypes.ref`; use it directly when you need
environment facts without building an object reference.

## When to use it

- Map a module (or file) back to the distribution that owns it.
- Resolve a module to its Git work tree.
- Read `Project-URL` metadata (documentation, repository, changelog).
- Download and parse a Sphinx `objects.inv` inventory.

## Quick start

```python exec="on" session="dist"
import pandas

from based_datatypes.dist import get_module_distribution

distribution = get_module_distribution(pandas)

print(distribution is not None and distribution.metadata["Name"])
print(distribution.version)
```

## API

| Name | Purpose |
| --- | --- |
| `get_distributions()` | Cached mapping of distribution name → `importlib.metadata` distribution. |
| `get_module_file(module)` | `Path` of a module's source file, or `None` for built-ins. |
| `get_module_distribution(module)` | Distribution that owns the module, or `None`. |
| `get_file_repo(path)` / `get_module_repo(module)` | GitPython `Repo` containing the file/module, or `None`. |
| `get_project_urls(dist, key)` | Values of the distribution's `Project-URL` entries with that key. |
| `get_py_inventory(docs_url)` | Cached mapping of Python object names to inventory items. |

## Installed distributions

`get_distributions` indexes every installed distribution by its metadata name. Import
metadata can be unreadable for a broken installation; those entries are skipped rather than
raising.

```python exec="on" session="dist"
from based_datatypes.dist import get_distributions

distributions = get_distributions()

print(len(distributions) > 0)
print("pandas" in distributions)
print(distributions["pandas"].version == __import__("pandas").__version__)
```

## Which distribution owns a module

`get_module_distribution` compares the module's file path with every distribution's
location. The **most specific** match wins: an editable install points at the repository
root, which also contains the virtual environment, so a naive first match would claim every
module in `site-packages`.

```python exec="on" session="dist"
import pandas

from based_datatypes import dist

module_file = dist.get_module_file(pandas)
owner = dist.get_module_distribution(pandas)

print(module_file is not None and module_file.name == "__init__.py")
print(owner is not None and owner.metadata["Name"] == "pandas")
print(dist.get_module_distribution(__import__("builtins")))
```

## Repository lookup

`get_file_repo` searches parent directories for a Git work tree, so a module deep inside a
package still resolves. `get_module_repo` is the module-level convenience.

```python exec="on" session="dist"
import tempfile
from pathlib import Path

from git import Repo

from based_datatypes.dist import get_file_repo

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    Repo.init(root)
    module_file = root / "module.py"
    module_file.write_text("", encoding="utf-8")

    repository = get_file_repo(module_file)
    print(repository is not None)
    print(Path(repository.working_dir).resolve() == root.resolve())
```

A file outside any repository returns `None`; no exception is raised.

## Project URLs

`get_project_urls` reads the `Project-URL` metadata entries and matches the key exactly.
Keys are conventionally `Documentation`, `Repository`, `Changelog`, `API Reference`.

```python exec="on" session="dist"
import importlib.metadata as metadata

from based_datatypes.dist import get_project_urls

pandas_metadata = metadata.distribution("pandas")

print("repository:", get_project_urls(pandas_metadata, "Repository"))
print("missing key:", get_project_urls(pandas_metadata, "Not A Real Key"))
```

## Object inventories

`get_py_inventory` downloads `<docs_url>/objects.inv`, parses it with Sphinx, and merges the
`py:` namespaces into one mapping. It is cached per URL, because inventories are large and
rarely change during a process lifetime.

```python exec="on" session="dist"
from based_datatypes.dist import get_py_inventory

# Performs an HTTP request; cache the result yourself if the URL is user-supplied.
# inventory = get_py_inventory("https://docs.python.org/3")
# print(inventory["builtins.int"].uri)
print("see the commented example")
```

!!! warning "HTTP errors propagate"
    `get_py_inventory` calls `raise_for_status()`, so a 404 or a network failure raises
    instead of returning an empty mapping. Wrap the call when the documentation URL comes
    from untrusted input.

!!! note "Version checks and outdated dependencies live in `ref`"
    `get_versions_on_pypi`, `get_outdated_deps` and `version_diff` are in
    `based_datatypes.ref`; they combine distribution metadata with PyPI requests and are
    documented in [Object references](how-to-refs.md).
