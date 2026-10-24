---
hide:
- navigation
---
# Based Datatypes Docs

Welcome! 👋

This documentation covers the `based-datatypes` project: a small, dependency-light library of
shared Python types, typing helpers and data utilities.

## Install

The package is not published on PyPI yet; install it from the Git remote:

```bash
uv add "based-datatypes @ git+https://github.com/cloudlane-one/based-datatypes.git"
```

Requires Python 3.13 or newer.

## Usage guides

Each module has a task-oriented guide with runnable examples:

<div class="grid cards" markdown>

- :material-format-letter-case: **[Enums](how-to-enums.md)**
  String enums that render as their value in docs and logs.

- :material-shape: **[Types and protocols](how-to-types.md)**
  Nominal strings, state markers, structural protocols and shared aliases.

- :material-swap-horizontal: **[Context variables](how-to-context.md)**
  Scoped context values with guaranteed restoration.

- :material-magnify: **[Runtime reflection](how-to-runtime.md)**
  Calling module, call arguments, return types and subclass trees.

- :material-code-braces: **[Type reflection](how-to-typing.md)**
  Resolve, compare and map type hints — including generic classes that remember their
  arguments.

- :material-pound: **[Hashing](how-to-hashing.md)**
  Stable hashes of dataclasses, containers, functions and dataframes.

- :material-database: **[Data utilities](how-to-data.md)**
  Dataclass copying, container conversion and mapping flattening.

- :material-cogs: **[Descriptors](how-to-descriptors.md)**
  Typed, cacheable properties and dual-dispatch methods.

- :material-link-variant: **[Object references](how-to-refs.md)**
  Persistent, URL-serialisable pointers to Python objects.

- :material-package-variant: **[Distributions](how-to-dist.md)**
  Which distribution or Git repository a module belongs to.

</div>

## Development

- [How to contribute](how-to-contribute.md)
- [Local setup](how-to-setup-locally.md)
- [Adding dependencies](how-to-add-dependencies.md)
- [Checking quality](how-to-check-quality.md)

## Reference

- [Python API](ref-python-api.md)
- [Directory structure](ref-directory-structure.md)
- [Code conventions and tooling](ref-code-conventions-tooling.md)
