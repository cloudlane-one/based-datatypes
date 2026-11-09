"""Store serialized dictionaries as local directory trees.

A serialized dictionary mixes JSON-supported primitives with files and nested
containers. This module turns such a dictionary into a directory tree and back:

* primitives are collected in a ``.index.yaml`` file inside the directory,
* files are written next to it, named by their key,
* nested dictionaries and lists become sub-directories named by their key,
* the ``.index.yaml`` of a list directory holds a YAML list, with ``None`` at
  the positions taken by files or sub-directories, and the files in that
  directory are named by their position.
"""

from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, final

import yaml

from based_datatypes.files import File, LocalDir, LocalFile, ensure_dir_exists
from based_datatypes.typing import has_type

from .base import Primitive, PrimitiveContainer, Scalar, Serialized, SerializedContainer
from .common import YamlDumper, YamlLoader


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
        or key == ".index"
        or PurePosixPath(key).name != key
        or windows.name != key
        or bool(windows.drive)
    ):
        raise ValueError(
            f"{key!r} is not a valid entry name; a key must be one path component "
            "and must not be '.index'."
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
        if entry.name == ".index.yaml":
            continue

        if entry.is_dir():
            if entry.name == key and not keep_directory:
                shutil.rmtree(entry)
        elif entry.stem == key:
            entry.unlink()


def _read_index(directory: Path) -> PrimitiveContainer:
    """Read the index file of a directory."""
    index_path = directory / ".index.yaml"

    if not index_path.exists():
        return {}

    # ``safe_load`` refuses to construct arbitrary Python objects.
    with index_path.open("r", encoding="utf-8", newline="") as stream:
        return yaml.load(stream, Loader=YamlLoader)


def _write_index(directory: Path, index: PrimitiveContainer) -> None:
    """Write the index file of a directory."""
    with (directory / ".index.yaml").open("w", encoding="utf-8", newline="") as stream:
        yaml.dump(index, stream, Dumper=YamlDumper, default_flow_style=False, allow_unicode=True)


@final
class Written:
    """Signal that a value was written to disk."""


def _write_node(value: Serialized, name: str, directory: LocalDir) -> Scalar | Written:
    if has_type(value, SerializedContainer):
        _remove_previous(directory.abs_path, name, keep_directory=True)
        _write_container(value, LocalDir(directory.abs_path / name))
        return Written()
    elif has_type(value, LocalFile):
        # A LocalFile is already on disk, so we can just copy it to the right place.
        _remove_previous(directory.abs_path, name)
        shutil.copy(value.path, directory.abs_path / name)
        return Written()
    elif has_type(value, File):
        content = value.read()
        _remove_previous(directory.abs_path, name)
        LocalFile(directory, name, mime=value.default_mime, encoding=value.default_encoding).write(
            content
        )
        return Written()

    _remove_previous(directory.abs_path, name)
    return value


def _write_container(container: Serialized, directory: LocalDir) -> None:
    """Write one dictionary or list/tuple/set/frozenset as a directory."""
    ensure_dir_exists(directory.abs_path)

    if has_type(container, dict[str, Serialized]):
        _write_mapping(container, directory)
    elif has_type(container, SerializedContainer):
        _write_sequence(container, directory)
    else:
        raise TypeError(
            f"{type(container).__name__} is not a directory node; only dicts and lists are."
        )


def _write_mapping(mapping: dict[str, Serialized], directory: LocalDir) -> None:
    """Write a dictionary as a directory, keeping primitives in the index file."""
    index: dict[str, Primitive] = {}

    for key, value in mapping.items():
        name = _entry_name(key)
        res = _write_node(value, name, directory)
        if not isinstance(res, Written):
            index[name] = res

    _write_index(directory.abs_path, index)


def _write_sequence(
    items: list[Serialized] | tuple[Serialized, ...] | set[Serialized] | frozenset[Serialized],
    directory: LocalDir,
) -> None:
    """Write a list as a directory, naming files and sub-directories by position."""
    index: list[Primitive] = []

    for position, value in enumerate(items):
        res = _write_node(value, str(position), LocalDir(directory.abs_path / str(position)))
        if not isinstance(res, Written):
            index.append(res)
        else:
            index.append(None)

    _write_index(directory.abs_path, index)


def write_structure(structure: dict[str, Serialized], path: Path | str) -> LocalDir:
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
    _write_container(structure, LocalDir(directory))

    return LocalDir(path=directory)


def _load_entry(path: Path) -> Serialized:
    """Load one file or sub-directory of a directory tree."""
    if path.is_dir():
        return _read_dir(path)

    return LocalFile.from_path(path)


def _position(entry: Path) -> int | None:
    """Return the position a list entry is stored at, or ``None`` when it has none."""
    try:
        return int(entry.stem)
    except ValueError:
        return None


def _read_dir(directory: Path) -> Serialized:
    """Read one directory back into a dictionary or a list."""
    index = _read_index(directory)

    if isinstance(index, list | tuple | set | frozenset):
        items: list = list(index)

        for entry in directory.iterdir():
            if entry.name == ".index.yaml":
                continue

            position = _position(entry)

            # Entries a shorter rewrite left behind are not part of the list.
            if position is None or position >= len(items):
                continue

            items[position] = _load_entry(entry)

        return items

    mapping: dict[str, Any] = dict(index) if isinstance(index, dict) else {}

    for entry in directory.iterdir():
        if entry.name == ".index.yaml":
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


def read_structure(path: Path | str) -> dict[str, Serialized]:
    """Read a local directory tree back into a serialized dictionary.

    Primitive values come from ``.index.yaml``, files are returned as
    :class:`LocalFile` handles, and sub-directories are read recursively.

    Args:
        path: The directory to read.

    Returns:
        The serialized dictionary stored in the directory.
    """
    structure = _read_dir(Path(path).absolute())

    if not isinstance(structure, dict):
        raise TypeError(f"{path} holds a list, not a dictionary.")

    return structure
