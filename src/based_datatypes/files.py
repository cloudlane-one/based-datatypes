"""Helper functions for handling files and directories."""

from __future__ import annotations

import io
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterator, Sequence, Set
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from functools import cached_property
from mimetypes import guess_extension, guess_type
from pathlib import Path
from types import NoneType
from typing import (
    IO,
    Any,
    BinaryIO,
    Generic,
    Literal,
    LiteralString,
    Self,
    TextIO,
    TypeVar,
    TypeVarTuple,
    cast,
    overload,
    override,
)

from identify.identify import tags_from_filename, tags_from_path
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from based_datatypes.descriptors import prop
from based_datatypes.types import TypeDef
from based_datatypes.typing import Runtyped, TypeRef, is_subtype


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


type Read = Literal["r"]
type Write = Literal["w"]

type UTF8 = Literal["utf-8"]
type AutoEncoding = Literal["auto"]

M = TypeVar("M", bound=LiteralString | str, covariant=True)

E = TypeVar(
    "E",
    bound=LiteralString | None,
    default=None,
    covariant=True,
)

R = TypeVar("R", bound=Read | Write, contravariant=True, default=Read | Write)


@dataclass(kw_only=True)
class File(Runtyped, Generic[M, E, R]):
    """A generic, single file (may be within a file system or object storage)."""

    @staticmethod
    def get_all_mimes(typedef: TypeDef[File]) -> list[str]:
        """Return all MIME types of the file typedef."""
        mimes = []
        typeref = TypeRef(typedef)

        for m in typeref.members:
            member_typeref = TypeRef(m)
            if M in member_typeref.typevar_map:
                for mime in member_typeref.typevar_map[M].literal_values:
                    if mime not in mimes:
                        mimes.append(mime)

        return mimes

    @prop(cached=True, mode="class")
    @classmethod
    def mimes(cls) -> list[M]:
        """MIME types of the file."""
        return cast(list[M], list(cls._typeref.typevar_map[M].literal_values))

    @prop(cached=True, mode="class")
    @classmethod
    def default_mime(cls) -> M:
        """Default MIME type of the file."""
        return cls.mimes[0]

    @prop(cached=True, mode="class")
    @classmethod
    def encodings(cls) -> list[E]:
        """The IO type of the file."""
        typeref = cls._typeref.typevar_map[E]
        if is_subtype(typeref.resolved_type_def, NoneType):
            return []
        return cast(list[E], list(typeref.literal_values))

    @prop(cached=True, mode="class")
    @classmethod
    def default_encoding(cls) -> E | None:
        """Default IO type of the file."""
        encodings = cls.encodings
        return encodings[0] if encodings else None

    @prop(cached=True, mode="class")
    @classmethod
    def readable(cls) -> bool:
        """The read/write mode of the file."""
        return "r" in cls._typeref.typevar_map[R].literal_values

    @prop(cached=True, mode="class")
    @classmethod
    def writable(cls) -> bool:
        """The read/write mode of the file."""
        return "w" in cls._typeref.typevar_map[R].literal_values

    @classmethod
    @override
    def narrow_runtype(cls, init_args: dict[str, Any]) -> dict[TypeVar | TypeVarTuple, TypeDef]:
        rw = []
        if cls.readable:
            rw.append("r")
        if cls.writable:
            rw.append("w")

        inst_rw = init_args.get("rw")
        if isinstance(inst_rw, str):
            rw = {inst_rw}

        return {
            M: Literal[init_args.get("mime") or cls.default_mime],
            E: Literal[init_args.get("encoding") or cls.default_encoding],
            R: Literal[*(inst_rw or rw)],
        }

    @classmethod
    def _deserialize(cls, state: dict[str, Any]) -> Self:
        """Reconstitute the file from pickled data."""
        instance = object.__new__(cls)
        content = state.pop("content", None)

        for f in fields(cls):
            if f.name in state:
                setattr(instance, f.name, state[f.name])
            else:
                setattr(instance, f.name, None)

        if content is not None:
            if not instance.writable:
                raise TypeError("Cannot set content on a read-only file.")

            cast(File[Any, Any, Write], instance).write(content)

        return instance

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type[Any], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Return the Pydantic core schema for the file."""
        assert issubclass(source, cls)
        return core_schema.no_info_after_validator_function(
            cls._deserialize,
            core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls._serialize,
                info_arg=False,
                return_schema=core_schema.str_schema(),
            ),
        )

    mime: M | None = None
    """Explicit MIME type."""

    encoding: E | None = None
    """Explicit IO type."""

    rw: Set[R] | R | None = None
    """Explicit read/write mode."""

    @cached_property
    def extension(self) -> str:
        """Return the file extension, from the suffix or the MIME type."""
        ext = additional_mime_inv.get(self.default_mime, [None])[0]
        if ext is None:
            ext = guess_extension(self.default_mime, strict=False)

        if ext is None:
            ext = ".txt" if self.encoding is not None else ".bin"

        assert ext is not None, "Could not determine file extension"
        return ext

    @abstractmethod
    def get_io(self) -> IO[Any]:
        """Return the file's content as an IO stream."""
        ...

    @overload
    def open(self: File[Any, Any, Read]) -> TextIO: ...

    @overload
    def open(self: File[Any, Any, Write], mode: Literal["w"] = ...) -> BinaryIO: ...

    @overload
    def open(
        self: File[Any, Any, Read | Write], mode: Literal["rw"] = ...
    ) -> TextIO | BinaryIO: ...

    @contextmanager
    def open(self) -> Generator[TextIO | BinaryIO]:
        """Open the file and return its IO stream."""
        io = self.get_io()
        try:
            yield cast(TextIO | BinaryIO, io)
        finally:
            io.close()

    @overload
    def read(self: File[Any, None, Read]) -> bytes: ...

    @overload
    def read(self: File[Any, LiteralString, Read]) -> str: ...

    @overload
    def read(self: File[Any, LiteralString | None, Read]) -> bytes | str: ...

    def read(self) -> bytes | str:
        """Read the file and return its content."""
        with self.get_io() as io:
            assert (
                self.encoding is None
                and isinstance(io, BinaryIO)
                or self.encoding is not None
                and isinstance(io, TextIO)
            )
            return io.read()

    @overload
    def write(self: File[Any, None, Write], content: bytes) -> None: ...

    @overload
    def write(self: File[Any, LiteralString, Write], content: str) -> None: ...

    def write(self, content: bytes | str) -> None:
        """Write content to the file."""
        with self.get_io() as io:
            assert (
                self.encoding is None
                and isinstance(io, BinaryIO)
                or self.encoding is not None
                and isinstance(io, TextIO)
            )
            io.write(content)  # pyright: ignore[reportArgumentType, reportCallIssue]

    def __get_state__(self: File[Any, Any, Any]) -> dict[str, Any]:
        """Return the state of the file for serialization."""
        state = self.__dict__.copy()

        if self.readable:
            state["content"] = self.read()

        return state

    def _serialize(self) -> dict[str, Any]:
        """Serialize the file to a dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def __reduce__(self) -> tuple[Callable[[dict[str, Any]], Self], tuple[dict[str, Any]]]:
        """Return a tuple for pickling the file."""
        return (self.__class__._deserialize, (self._serialize(),))


@dataclass
class LocalFile(File[M, E, R]):
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

        encoding = "utf-8" if "text" in tags else None

        return LocalFile[Any, Any, Any](
            dir=LocalDir(path=dir_path),
            name=name,
            encoding=encoding,
            mime=mime,
        )

    @cached_property
    def path(self) -> Path:
        """Return the absolute path of the file."""
        return (self.dir.abs_path / (self.name + self.extension)).absolute()

    @cached_property
    def resolved_format(self) -> Literal["text", "binary"]:
        """Default encoding based on filename and content."""
        path = Path(self.path)
        if self.default_encoding != "auto":
            return "text" if self.default_encoding is not None else "binary"

        file_tags = tags_from_path(str(path))
        return "text" if "text" in file_tags else "binary"

    @cached_property
    def file_mode(self) -> str:
        """Map the file mode to a string representation."""
        if self.readable and self.writable:
            return "wb+" if self.resolved_format == "binary" else "w+"
        elif self.writable:
            return "wb" if self.resolved_format == "binary" else "w"
        else:
            return "rb" if self.resolved_format == "binary" else "r"

    @override
    def get_io(self) -> IO[Any]:
        """Return the file's content as a text IO stream.

        Text streams always use UTF-8 with newline translation disabled, so the
        bytes on disk do not depend on the platform or locale and match what the
        serialization size estimators measure.
        """
        if self.default_encoding is not None:
            return open(
                self.path,
                mode=self.file_mode,
                encoding=self.default_encoding,
                newline="",
            )

        return open(self.path, mode=self.file_mode)


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
class MemoryFile(File[M, E, R]):
    """A file whose content lives in memory instead of on a storage medium.

    Useful for tests, for staging content before it is written somewhere, and
    for handing a file to code that only knows the :class:`File` interface.
    """

    content: bytes | str | None = None
    """Current content; ``None`` starts empty and is normalized by IO type."""

    name: str | None = None
    """Optional file name, used for display only."""

    def __post_init__(self) -> None:
        """Normalize the content to the type implied by ``encoding``."""
        if self.content is None:
            self.content = "" if self.encoding is not None else b""
        elif self.encoding is not None:
            if not isinstance(self.content, str):
                raise TypeError("A text file requires str content.")
        elif not isinstance(self.content, bytes):
            raise TypeError("A binary file requires bytes content.")

    @cached_property
    def detected_format(self) -> Literal["text", "binary"]:
        """Return the format implied by the declared IO type."""
        return "text" if self.encoding is not None else "binary"

    @override
    def get_io(self) -> IO[Any]:
        """Return a stream over the file's content.

        Writes are stored back into :attr:`content` when the stream is closed.
        ``"w"`` and ``"rw"`` start from empty content, like a local file, while
        ``"r"`` starts from the current content and rejects writes.
        """
        initial = self.content if self.readable else None

        if self.encoding is not None:
            return _TextMemoryStream(
                self,
                cast(str, initial) if initial is not None else "",
                writable=self.writable,
            )

        return _BinaryMemoryStream(
            self,
            cast(bytes, initial) if initial is not None else b"",
            writable=self.writable,
        )


F = TypeVar("F", bound="File", covariant=True)
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
