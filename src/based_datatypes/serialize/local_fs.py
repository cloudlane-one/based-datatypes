"""Store serialized dictionaries as local directory trees.

A serialized dictionary mixes JSON-supported primitives with files and nested
containers. This module turns such a dictionary into a directory tree and back:

* primitives are collected in a ``__index__.yaml`` file inside the directory,
* files are written next to it, named by their key,
* nested dictionaries and lists become sub-directories named by their key,
* the ``__index__.yaml`` of a list directory holds a YAML list, with ``None`` at
  the positions taken by files or sub-directories, and the files in that
  directory are named by their position.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from based_datatypes.files import File, LocalDir, LocalFile, ensure_dir_exists

from .base import SerializedDict, SerializedValue

INDEX_FILENAME = "__index__.yaml"
"""Name of the YAML file that holds the primitive values of a directory."""


def _index_path(directory: Path) -> Path:
    """Return the path of the index file of a directory."""
    return directory / INDEX_FILENAME


def _read_index(directory: Path) -> Any:
    """Read the index file of a directory."""
    path = _index_path(directory)

    if not path.exists():
        return {}

    # ``safe_load`` refuses to construct arbitrary Python objects.
    with path.open("r", encoding="utf-8", newline="") as stream:
        return yaml.safe_load(stream)


def _write_index(directory: Path, index: Any) -> None:
    """Write the index file of a directory."""
    text = yaml.safe_dump(index, allow_unicode=True)

    with _index_path(directory).open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)


def _write_file(source: File, destination: Path) -> None:
    """Copy the content of a serialized file to a local path."""
    ensure_dir_exists(destination.parent)

    with source.open("r") as stream:
        content = stream.read()

    if isinstance(content, str):
        with destination.open("w", encoding="utf-8", newline="") as target:
            target.write(content)
    else:
        with destination.open("wb") as target:
            target.write(cast(bytes, content))


def _extension(file: File) -> str:
    """Return the suffix a serialized file is stored under, including the dot."""
    extension = file.extension

    return extension if extension.startswith(".") or extension == "" else f".{extension}"


def _write_node(node: SerializedValue, directory: Path) -> None:
    """Write one dictionary or list as a directory."""
    ensure_dir_exists(directory)

    if isinstance(node, dict):
        _write_mapping(node, directory)
    elif isinstance(node, list):
        _write_sequence(node, directory)
    else:
        raise TypeError(f"{type(node).__name__} is not a directory node; only dicts and lists are.")


def _write_mapping(mapping: SerializedDict, directory: Path) -> None:
    """Write a dictionary as a directory, keeping primitives in the index file."""
    index: dict[str, Any] = {}

    for key, value in mapping.items():
        if isinstance(value, File):
            _write_file(value, directory / f"{key}{_extension(value)}")
        elif isinstance(value, dict | list):
            _write_node(value, directory / key)
        else:
            index[key] = value

    _write_index(directory, index)


def _write_sequence(items: list[SerializedValue], directory: Path) -> None:
    """Write a list as a directory, naming files and sub-directories by position."""
    index: list[Any] = []

    for position, value in enumerate(items):
        if isinstance(value, File):
            _write_file(value, directory / f"{position}{_extension(value)}")
            index.append(None)
        elif isinstance(value, dict | list):
            _write_node(value, directory / str(position))
            index.append(None)
        else:
            index.append(value)

    _write_index(directory, index)


def write_structure(structure: SerializedDict, path: Path | str) -> LocalDir:
    """Write a serialized dictionary into a local directory tree.

    The directory is created when it does not exist. Entries are overwritten by
    key, but entries that the structure no longer mentions are left in place, so
    writing an updated structure over an existing tree does not delete anything.

    Args:
        structure: The serialized dictionary to write.
        path: The directory to write into.

    Returns:
        The directory that was written.
    """
    directory = ensure_dir_exists(path)
    _write_node(structure, directory)

    return LocalDir(path=directory)


def _load_entry(path: Path) -> SerializedValue:
    """Load one file or sub-directory of a directory tree."""
    if path.is_dir():
        return _read_node(path)

    return LocalFile.from_path(path)


def _read_node(directory: Path) -> SerializedValue:
    """Read one directory back into a dictionary or a list."""
    index = _read_index(directory)

    if isinstance(index, list):
        items = list(index)

        for entry in directory.iterdir():
            if entry.name == INDEX_FILENAME:
                continue
            items[int(entry.stem)] = _load_entry(entry)

        return items

    mapping: SerializedDict = dict(index) if isinstance(index, dict) else {}

    for entry in directory.iterdir():
        if entry.name == INDEX_FILENAME:
            continue
        mapping[entry.stem] = _load_entry(entry)

    return mapping


def read_structure(path: Path | str) -> SerializedDict:
    """Read a local directory tree back into a serialized dictionary.

    Primitive values come from ``__index__.yaml``, files are returned as
    :class:`LocalFile` handles, and sub-directories are read recursively.

    Args:
        path: The directory to read.

    Returns:
        The serialized dictionary stored in the directory.
    """
    structure = _read_node(Path(path).absolute())

    if not isinstance(structure, dict):
        raise TypeError(f"{path} holds a list, not a dictionary.")

    return structure


__all__ = ["INDEX_FILENAME", "read_structure", "write_structure"]
