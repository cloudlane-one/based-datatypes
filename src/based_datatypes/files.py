"""Helper functions for handling files and directories."""

from __future__ import annotations

import io
import shutil
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from mimetypes import guess_extension, guess_type
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Generic,
    Literal,
    LiteralString,
    TextIO,
    TypeVar,
    cast,
    final,
    overload,
    override,
)

from identify.identify import tags_from_filename, tags_from_path

from based_datatypes.descriptors import prop
from based_datatypes.types import TypeDef
from based_datatypes.typing import TypeAwareClass, is_subtype


def ensure_dir_exists(path: Path | str) -> Path:
    """Make sure the given path exists and is a directory.

    Create dir if necessary. Then return the path unchanged.

    Args:
        path: Path to ensure exists.

    Returns:
        The given path.
    """
    path = Path(path).absolute()
    p = Path(path.parts[0])
    for d in path.parts[1:]:
        p = p / d
        if not p.is_dir():
            if not p.exists():
                p.mkdir()
            else:
                raise NotADirectoryError()

    return path


additional_mime = {
    ".csv": "text/csv",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".parquet": "application/vnd.apache.parquet",
    ".xlsx": "application/vnd.ms-excel",
    ".xls": "application/vnd.ms-excel",
    ".npy": "application/x-npy",
    ".pkl": "application/pickle",
    ".npz": "application/x-npz",
    ".sql": "application/sql",
}
additional_mime_inv: dict[str, list[str]] = {}
for _extension, _mime in additional_mime.items():
    additional_mime_inv.setdefault(_mime, []).append(_extension)


@final
class Read:
    """A marker class for read-only file."""


@final
class Write:
    """A marker class for write-only file."""


M = TypeVar("M", bound=LiteralString | str, covariant=True)

I = TypeVar(  # noqa: E741
    "I",
    bound=TextIO | BinaryIO,
    default=Any,
)

R = TypeVar("R", bound=Read | Write, contravariant=True, default=Read | Write)


@dataclass(kw_only=True)
class File(ABC, TypeAwareClass, Generic[M, I, R]):
    """A generic, single file (may be within a file system or object storage)."""

    _suffix: str | None = None
    """Explicit file suffix; ``None`` derives it from the MIME type."""

    @prop(cached=True, mode="dual")
    def mimes(self) -> list[M]:
        """MIME types of the file."""
        return cast(list[M], list(self.typeref.scalar_typevar_map[M].literal_values))

    @prop(cached=True, mode="dual")
    def io_type(self) -> TypeDef[I]:
        """The IO type of the file."""
        typeref = self.typeref.scalar_typevar_map[I]
        return cast(type[I], typeref.resolved_type_def)

    @prop(cached=True, mode="dual")
    def read_write(self) -> Literal["rw", "r", "w"]:
        """The read/write mode of the file."""
        typeref = self.typeref.scalar_typevar_map[R]
        return (
            "rw"
            if typeref.is_supertype(Read | Write)
            else "r" if typeref.is_supertype(Read) else "w"
        )

    @property
    def default_mime(self) -> M:
        """Default MIME type of the file."""
        return self.mimes[0]

    @cached_property
    def extension(self) -> str:
        """Return the file extension, from the suffix or the MIME type."""
        if self._suffix is not None:
            return self._suffix

        ext = additional_mime_inv.get(self.default_mime, [None])[0]
        if ext is None:
            ext = guess_extension(self.default_mime, strict=False)

        if ext is None:
            ext = ".txt" if is_subtype(self.io_type, TextIO) else ".bin"

        assert ext is not None, "Could not determine file extension"
        return ext

    @abstractmethod
    def get_io(self, mode: Literal["rw", "r", "w"] = "rw") -> I:
        """Return the file's content as an IO stream."""
        ...

    @overload
    def open(self: File[Any, Any, Read], mode: Literal["r"] = ...) -> I: ...

    @overload
    def open(self: File[Any, Any, Write], mode: Literal["w"] = ...) -> I: ...

    @overload
    def open(self: File[Any, Any, Read | Write], mode: Literal["rw"] = ...) -> I: ...

    @contextmanager
    def open(self, mode: Literal["rw", "r", "w"] = "rw") -> Generator[I]:
        """Open the file and return its IO stream."""
        io = self.get_io(mode)
        try:
            yield io
        finally:
            io.close()

    def to_file_type(self) -> TypeDef[File[M, I, R]]:
        """Return a TypeDef for this file type."""
        return File[
            cast(type[M], Literal[self.default_mime]),
            cast(type[I], self.io_type),
            Any,
        ]


@dataclass
class LocalFile(File[M, I, R]):
    """A generic, single file (may be within a file system or object storage)."""

    dir: LocalDir
    name: str

    @staticmethod
    def from_path(path: Path | str) -> LocalFile:
        """Create a LocalFile from a given path."""
        path = Path(path).absolute()

        dir_path = path.parent
        name = path.stem
        # ``additional_mime`` covers the formats ``mimetypes`` cannot name
        # (parquet, npy, pickle, ...), so it takes precedence.
        mime = (
            additional_mime.get(path.suffix.lower())
            or guess_type(path)[0]
            or "application/octet-stream"
        )
        tags = tags_from_path(str(path)) if path.exists() else tags_from_filename(str(path))
        io_type = TextIO if "text" in tags else BinaryIO

        return LocalFile[cast(type[LiteralString], Literal[mime]), io_type, Any](
            dir=LocalDir(path=dir_path),
            name=name,
            _suffix=path.suffix,
        )

    @cached_property
    def path(self) -> Path:
        """Return the absolute path of the file."""
        return (self.dir.abs_path / (self.name + self.extension)).absolute()

    @cached_property
    def detected_format(self) -> Literal["text", "binary"]:
        """Default encoding based on filename and content."""
        path = Path(self.path)
        if not path.exists():
            return "text" if is_subtype(self.io_type, TextIO) else "binary"

        file_tags = tags_from_path(str(path))
        return "text" if "text" in file_tags else "binary"

    def get_file_mode(
        self, mode: Literal["rw", "r", "w"], encoding: Literal["text", "binary"]
    ) -> str:
        """Map the file mode to a string representation."""
        if mode == "r":
            return "rb" if encoding == "binary" else "r"
        elif mode == "w":
            return "wb" if encoding == "binary" else "w"
        else:
            return "wb+" if encoding == "binary" else "w+"

    @override
    def get_io(self, mode: Literal["rw", "r", "w"] = "rw") -> I:
        """Return the file's content as a text IO stream.

        Text streams always use UTF-8 with newline translation disabled, so the
        bytes on disk do not depend on the platform or locale and match what the
        transport estimators measure.
        """
        if is_subtype(self.io_type, TextIO):
            return cast(
                I,
                open(
                    self.path,
                    mode=self.get_file_mode(mode, "text"),
                    encoding="utf-8",
                    newline="",
                ),
            )

        return cast(I, open(self.path, mode=self.get_file_mode(mode, "binary")))


class _TextMemoryStream(io.StringIO):
    """A text stream that stores its content back into a :class:`MemoryFile`."""

    def __init__(self, file: MemoryFile[Any, Any, Any], initial: str, *, writable: bool) -> None:
        super().__init__(initial)
        self._file = file
        self._writable = writable

    @override
    def writable(self) -> bool:
        """Report whether the stream accepts writes."""
        return self._writable

    @override
    def write(self, data: str) -> int:
        """Write text, unless the stream was opened read-only."""
        if not self._writable:
            raise io.UnsupportedOperation("not writable")
        return super().write(data)

    @override
    def close(self) -> None:
        """Store the content in the file, then close the stream."""
        if not self.closed and self._writable:
            self._file.content = self.getvalue()
        super().close()


class _BinaryMemoryStream(io.BytesIO):
    """A binary stream that stores its content back into a :class:`MemoryFile`."""

    def __init__(self, file: MemoryFile[Any, Any, Any], initial: bytes, *, writable: bool) -> None:
        super().__init__(initial)
        self._file = file
        self._writable = writable

    @override
    def writable(self) -> bool:
        """Report whether the stream accepts writes."""
        return self._writable

    @override
    def write(self, data: bytes) -> int:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Write bytes, unless the stream was opened read-only."""
        if not self._writable:
            raise io.UnsupportedOperation("not writable")
        return super().write(data)

    @override
    def close(self) -> None:
        """Store the content in the file, then close the stream."""
        if not self.closed and self._writable:
            self._file.content = self.getvalue()
        super().close()


@dataclass
class MemoryFile(File[M, I, R]):
    """A file whose content lives in memory instead of on a storage medium.

    Useful for tests, for staging content before it is written somewhere, and
    for handing a file to code that only knows the :class:`File` interface.
    """

    content: bytes | str | None = None
    """Current content; ``None`` starts empty and is normalized by IO type."""

    name: str | None = None
    """Optional file name, used for display only."""

    def __post_init__(self) -> None:
        """Normalize the content to the type implied by ``io_type``."""
        if self.content is None:
            self.content = "" if is_subtype(self.io_type, TextIO) else b""
        elif is_subtype(self.io_type, TextIO):
            if not isinstance(self.content, str):
                raise TypeError("A text file requires str content.")
        elif not isinstance(self.content, bytes):
            raise TypeError("A binary file requires bytes content.")

    @cached_property
    def detected_format(self) -> Literal["text", "binary"]:
        """Return the format implied by the declared IO type."""
        return "text" if is_subtype(self.io_type, TextIO) else "binary"

    @override
    def get_io(self, mode: Literal["rw", "r", "w"] = "rw") -> I:
        """Return a stream over the file's content.

        Writes are stored back into :attr:`content` when the stream is closed.
        ``"w"`` and ``"rw"`` start from empty content, like a local file, while
        ``"r"`` starts from the current content and rejects writes.
        """
        initial = self.content if mode == "r" else None
        writable = mode != "r"

        if is_subtype(self.io_type, TextIO):
            return cast(
                I,
                _TextMemoryStream(
                    self,
                    cast(str, initial) if initial is not None else "",
                    writable=writable,
                ),
            )

        return cast(
            I,
            _BinaryMemoryStream(
                self,
                cast(bytes, initial) if initial is not None else b"",
                writable=writable,
            ),
        )


F = TypeVar("F", bound="File[Any, TextIO | BinaryIO]", covariant=True)
D = TypeVar("D", bound="Dir | None", covariant=True)


@dataclass(kw_only=True)
class Dir(ABC, Generic[F, D]):  # noqa: UP046
    """A directory of files or sub-directories."""

    @abstractmethod
    def keys(self) -> Sequence[str]:
        """Iterate over the items in the directory."""
        ...

    @abstractmethod
    def __getitem__(self, name: str) -> F | D:
        """Get an item from the directory."""
        ...

    @abstractmethod
    def __delitem__(self, name: str) -> None:
        """Delete an item from the directory."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the directory."""
        ...

    def values(self) -> Sequence[F | D]:
        """Iterate over the items in the directory."""
        return [self[key] for key in self.keys()]

    def items(self) -> Iterator[tuple[str, F | D]]:
        """Iterate over the items in the directory as (name, item) pairs."""
        for key in self.keys():
            yield key, self[key]

    def __iter__(self) -> Iterator[F | D]:
        """Iterate over the items in the directory."""
        return iter(self.values())

    def __contains__(self, name: str) -> bool:
        """Check if an item exists in the directory."""
        return name in self.keys()

    def file(self, name: str) -> F:
        """Get a file from the directory."""
        item = self[name]
        assert isinstance(item, File)
        return item

    def subdir(self, name: str) -> D:
        """Get a sub-directory from the directory."""
        item = self[name]
        assert isinstance(item, Dir)
        return item


@dataclass
class LocalDir(Dir[LocalFile, "LocalDir"]):
    """A local directory of files or sub-directories."""

    path: Path | str

    _exists: Literal[True] | None = None

    @cached_property
    def abs_path(self) -> Path:
        """Return the absolute path of the directory."""
        return Path(self.path).absolute()

    def _ensure_exists(self) -> None:
        if not self._exists:
            ensure_dir_exists(self.abs_path)
            self._exists = True

    @override
    def keys(self) -> Sequence[str]:
        """Iterate over the items in the directory."""
        self._ensure_exists()
        return [item.name for item in self.abs_path.iterdir() if item.is_file() or item.is_dir()]

    @override
    def __getitem__(self, item: str) -> LocalDir | LocalFile:
        """Get an item from the directory."""
        self._ensure_exists()
        item_path = self.abs_path / item
        if item_path.is_dir():
            return LocalDir(path=item_path)
        elif item_path.is_file():
            return LocalFile.from_path(item_path)

        raise KeyError(f"Item '{item}' not found in directory '{self.abs_path}'.")

    @override
    def __delitem__(self, key: str) -> None:
        """Delete an item from the directory."""
        self._ensure_exists()
        item_path = self.abs_path / key
        if item_path.is_file():
            item_path.unlink()
        elif item_path.is_dir():
            shutil.rmtree(item_path)
        else:
            raise KeyError(f"Item '{key}' not found in directory '{self.abs_path}'.")

    @override
    def __len__(self) -> int:
        """Return the number of items in the directory."""
        self._ensure_exists()
        return len(list(self.abs_path.iterdir()))
