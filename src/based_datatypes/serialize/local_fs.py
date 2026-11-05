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

import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

import yaml

from based_datatypes.files import File, LocalDir, LocalFile, ensure_dir_exists

from .base import SerializedDict, SerializedValue

INDEX_FILENAME = "__index__.yaml"
"""Name of the YAML file that holds the primitive values of a directory."""


def _entry_name(key: str) -> str:
    """Validate a mapping key as the name of one directory entry.

    Keys come from serialized data, so they decide where a file is written. A
    key that is not a single path component (``..``, ``../outside``, an absolute
    path, a drive) would write outside the requested tree, and the index file
    itself must stay the index. Both are refused before anything is created.

    Args:
        key: The mapping key a file or sub-directory is stored under.

    Returns:
        The key, unchanged, when it is a valid entry name.

    Raises:
        ValueError: If the key cannot name one entry of the directory.
    """
    windows = PureWindowsPath(key)

    if (
        not key
        or key in (".", "..")
        or "/" in key
        or "\\" in key
        or "\x00" in key
        or key == "__index__"
        or PurePosixPath(key).name != key
        or windows.name != key
        or bool(windows.drive)
    ):
        raise ValueError(
            f"{key!r} is not a valid entry name; a key must be one path component "
            "and must not be '__index__'."
        )

    return key


def _remove_previous(directory: Path, key: str, *, keep_directory: bool = False) -> None:
    """Remove entries a previous write left for a key that this write replaces.

    A file stored under ``<key><extension>`` and a sub-directory named ``<key>``
    both read back as ``key``, so a leftover of the other shape would be overlaid
    on the value written now. Entries for keys this write does not mention are
    left alone.

    Args:
        directory: The directory the key belongs to.
        key: The key written now.
        keep_directory: Keep an existing sub-directory named ``key``, because the
            value written now is itself a container.
    """
    for entry in directory.iterdir():
        if entry.name == INDEX_FILENAME:
            continue

        if entry.is_dir():
            if entry.name == key and not keep_directory:
                shutil.rmtree(entry)
        elif entry.stem == key:
            entry.unlink()


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
        name = _entry_name(key)

        if isinstance(value, File):
            _remove_previous(directory, name)
            _write_file(value, directory / f"{name}{_extension(value)}")
        elif isinstance(value, dict | list):
            _remove_previous(directory, name, keep_directory=True)
            _write_node(value, directory / name)
        else:
            _remove_previous(directory, name)
            index[name] = value

    _write_index(directory, index)


def _write_sequence(items: list[SerializedValue], directory: Path) -> None:
    """Write a list as a directory, naming files and sub-directories by position."""
    index: list[Any] = []

    for position, value in enumerate(items):
        if isinstance(value, File):
            _remove_previous(directory, str(position))
            _write_file(value, directory / f"{position}{_extension(value)}")
            index.append(None)
        elif isinstance(value, dict | list):
            _remove_previous(directory, str(position), keep_directory=True)
            _write_node(value, directory / str(position))
            index.append(None)
        else:
            _remove_previous(directory, str(position))
            index.append(value)

    _write_index(directory, index)


def write_structure(structure: SerializedDict, path: Path | str) -> LocalDir:
    """Write a serialized dictionary into a local directory tree.

    The directory is created when it does not exist. Entries are overwritten by
    key, and an entry whose key changes shape (a file replaced by a
    sub-directory, or the other way round) replaces the old artifact instead of
    being overlaid on it. Entries that the structure no longer mentions at all
    are left in place, so writing an updated structure over an existing tree
    does not delete anything.

    Args:
        structure: The serialized dictionary to write.
        path: The directory to write into.

    Returns:
        The directory that was written.

    Raises:
        ValueError: If a key is not a valid entry name, or would take the name of
            the directory index.
    """
    directory = ensure_dir_exists(path)
    _write_node(structure, directory)

    return LocalDir(path=directory)


def _load_entry(path: Path) -> SerializedValue:
    """Load one file or sub-directory of a directory tree."""
    if path.is_dir():
        return _read_node(path)

    return LocalFile.from_path(path)


def _position(entry: Path) -> int | None:
    """Return the position a list entry is stored at, or ``None`` when it has none."""
    try:
        return int(entry.stem)
    except ValueError:
        return None


def _read_node(directory: Path) -> SerializedValue:
    """Read one directory back into a dictionary or a list."""
    index = _read_index(directory)

    if isinstance(index, list):
        items = list(index)

        for entry in directory.iterdir():
            if entry.name == INDEX_FILENAME:
                continue

            position = _position(entry)

            # Entries a shorter rewrite left behind are not part of the list.
            if position is None or position >= len(items):
                continue

            items[position] = _load_entry(entry)

        return items

    mapping: SerializedDict = dict(index) if isinstance(index, dict) else {}

    for entry in directory.iterdir():
        if entry.name == INDEX_FILENAME:
            continue

        # A sub-directory is named by its whole key, a file by its key plus the
        # suffix of its format, so only the file name loses one component.
        key = entry.name if entry.is_dir() else entry.stem

        # The index holds the primitives written last: an entry under a key the
        # index already values is a leftover of a replaced shape.
        if key in mapping:
            continue

        mapping[key] = _load_entry(entry)

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
