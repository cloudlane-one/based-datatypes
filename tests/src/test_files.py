"""Tests for :mod:`based_datatypes.files`."""

import io
from pathlib import Path
from typing import Any, BinaryIO, Literal, TextIO, cast

import pytest

from based_datatypes.files import (
    Dir,
    File,
    LocalDir,
    LocalFile,
    additional_mime,
    additional_mime_inv,
    ensure_dir_exists,
)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """Provide an existing empty directory."""
    path = tmp_path / "work"
    path.mkdir()
    return path


def make_file(
    directory: Path,
    name: str,
    mime: str,
    *,
    binary: bool = False,
    suffix: str | None = None,
) -> LocalFile[Any]:
    """Build a :class:`LocalFile` that is not on disk yet.

    Args:
        directory: Directory the file belongs to.
        name: File name without extension.
        mime: MIME type.
        binary: Whether the file is binary rather than text.
        suffix: Explicit file suffix, or ``None`` to derive it from ``mime``.

    Returns:
        The file object.
    """
    return LocalFile(
        dir=LocalDir(path=directory),
        name=name,
        mime=mime,
        io_type=BinaryIO if binary else TextIO,
        _suffix=suffix,
    )


class TestEnsureDirExists:
    """Tests for :func:`ensure_dir_exists`."""

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        """Every missing component is created."""
        target = tmp_path / "a" / "b" / "c"

        result = ensure_dir_exists(target)

        assert result == target.absolute()
        assert target.is_dir()
        assert result.is_absolute()

    def test_returns_existing_directory(self, workdir: Path) -> None:
        """An existing directory is returned unchanged."""
        assert ensure_dir_exists(workdir) == workdir.absolute()

    def test_accepts_string(self, tmp_path: Path) -> None:
        """A string path is accepted."""
        assert ensure_dir_exists(str(tmp_path / "made")) == (tmp_path / "made").absolute()

    def test_rejects_file_in_path(self, tmp_path: Path) -> None:
        """A file where a directory is needed raises."""
        blocker = tmp_path / "file.txt"
        blocker.write_text("x", encoding="utf-8")

        with pytest.raises(NotADirectoryError):
            ensure_dir_exists(blocker / "child")


class TestAdditionalMime:
    """Tests for the supplementary MIME tables."""

    @pytest.mark.parametrize(
        ("extension", "mime"),
        [
            (".csv", "text/csv"),
            (".yaml", "application/yaml"),
            (".yml", "application/yaml"),
            (".parquet", "application/vnd.apache.parquet"),
            (".xlsx", "application/vnd.ms-excel"),
            (".npy", "application/x-npy"),
            (".npz", "application/x-npz"),
            (".pkl", "application/pickle"),
            (".sql", "application/sql"),
        ],
    )
    def test_extension_to_mime(self, extension: str, mime: str) -> None:
        """Formats ``mimetypes`` cannot name are covered."""
        assert additional_mime[extension] == mime

    def test_inverse_groups_all_extensions(self) -> None:
        """Every extension is listed under its MIME type."""
        assert additional_mime_inv["application/yaml"] == [".yaml", ".yml"]
        assert additional_mime_inv["application/vnd.ms-excel"] == [".xlsx", ".xls"]

        for extension, mime in additional_mime.items():
            assert extension in additional_mime_inv[mime]


class TestLocalFileFromPath:
    """Tests for :meth:`LocalFile.from_path`."""

    def test_text_file(self, workdir: Path) -> None:
        """A text file keeps its stem, MIME type, and IO type."""
        path = workdir / "note.txt"
        path.write_text("hello", encoding="utf-8")

        file = LocalFile.from_path(path)

        assert file.name == "note"
        assert file.mime == "text/plain"
        assert file.io_type is TextIO
        assert file.extension == ".txt"
        assert file.path == path
        assert file.detected_format == "text"
        assert file.dir.abs_path == workdir.absolute()

    def test_accepts_string_path(self, workdir: Path) -> None:
        """A string path is accepted."""
        path = workdir / "a.txt"
        path.write_text("a", encoding="utf-8")

        assert LocalFile.from_path(str(path)).path == path

    @pytest.mark.parametrize(
        ("filename", "mime", "extension"),
        [
            ("data.csv", "text/csv", ".csv"),
            ("data.json", "application/json", ".json"),
            ("data.yaml", "application/yaml", ".yaml"),
            ("data.yml", "application/yaml", ".yml"),
            ("data.xlsx", "application/vnd.ms-excel", ".xlsx"),
            ("data.parquet", "application/vnd.apache.parquet", ".parquet"),
            ("data.npy", "application/x-npy", ".npy"),
            ("data.pkl", "application/pickle", ".pkl"),
            ("data.sql", "application/sql", ".sql"),
        ],
    )
    def test_extension_prefers_supplementary_table(
        self, workdir: Path, filename: str, mime: str, extension: str
    ) -> None:
        """The declared MIME table and the real suffix win over ``mimetypes``."""
        path = workdir / filename
        path.write_text("content", encoding="utf-8")

        file = LocalFile.from_path(path)

        assert file.mime == mime
        assert file.extension == extension
        assert file.path == path

    def test_binary_file(self, workdir: Path) -> None:
        """A binary file is detected as such."""
        path = workdir / "image.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n")

        file = LocalFile.from_path(path)

        assert file.io_type is BinaryIO
        assert file.detected_format == "binary"
        assert file.extension == ".png"
        assert file.mime == "image/png"

    def test_no_suffix_keeps_name(self, workdir: Path) -> None:
        """A file without a suffix has no derived extension."""
        path = workdir / "README"
        path.write_text("text", encoding="utf-8")

        file = LocalFile.from_path(path)

        assert file.extension == ""
        assert file.path == path

    def test_missing_path_falls_back_to_extension_rules(self, workdir: Path) -> None:
        """A path that does not exist yet is still described."""
        file = LocalFile.from_path(workdir / "later.csv")

        assert file.mime == "text/csv"
        assert file.io_type is TextIO
        assert file.path == workdir / "later.csv"


class TestFileExtension:
    """Tests for :attr:`File.extension`."""

    @pytest.mark.parametrize(
        ("mime", "expected"),
        [
            ("application/json", ".json"),
            ("text/csv", ".csv"),
            ("application/yaml", ".yaml"),
            ("application/vnd.ms-excel", ".xlsx"),
            ("application/vnd.apache.parquet", ".parquet"),
            ("application/pickle", ".pkl"),
            ("image/png", ".png"),
        ],
    )
    def test_derived_from_mime(self, workdir: Path, mime: str, expected: str) -> None:
        """Known MIME types map to a canonical extension."""
        assert make_file(workdir, "f", mime).extension == expected

    @pytest.mark.parametrize(("binary", "expected"), [(False, ".txt"), (True, ".bin")])
    def test_fallback_by_io_type(self, workdir: Path, binary: bool, expected: str) -> None:
        """An unknown MIME type falls back to the IO type."""
        file = make_file(workdir, "f", "application/x-unknown", binary=binary)

        assert file.extension == expected

    def test_explicit_suffix_wins(self, workdir: Path) -> None:
        """A supplied suffix is used verbatim, including an empty one."""
        assert make_file(workdir, "f", "application/json", suffix=".txt").extension == ".txt"
        assert make_file(workdir, "f", "application/json", suffix="").extension == ""

    def test_is_cached(self, workdir: Path) -> None:
        """The extension is computed once and reused."""
        file = make_file(workdir, "f", "application/json")

        assert file.extension == ".json"
        assert file.extension == ".json"


class TestGetFileMode:
    """Tests for :meth:`LocalFile.get_file_mode`."""

    @pytest.mark.parametrize(
        ("mode", "encoding", "expected"),
        [
            ("r", "text", "r"),
            ("r", "binary", "rb"),
            ("w", "text", "w"),
            ("w", "binary", "wb"),
            ("rw", "text", "w+"),
            ("rw", "binary", "wb+"),
        ],
    )
    def test_mapping(self, workdir: Path, mode: str, encoding: str, expected: str) -> None:
        """Every mode maps to the matching builtin mode."""
        file = make_file(workdir, "f", "text/plain")

        assert file.get_file_mode(mode, encoding) == expected  # pyright: ignore[reportArgumentType]


class TestLocalFileIO:
    """Tests for reading and writing through a :class:`LocalFile`."""

    def test_text_read(self, workdir: Path) -> None:
        """A text file is read as text."""
        path = workdir / "note.txt"
        path.write_text("hello", encoding="utf-8")

        with cast(TextIO, LocalFile.from_path(path).get_io("r")) as io:
            assert io.read() == "hello"

    def test_text_write(self, workdir: Path) -> None:
        """A text file is written as text."""
        file = make_file(workdir, "note", "text/plain", suffix=".txt")

        with cast(TextIO, file.get_io("w")) as io:
            io.write("written")

        assert file.path.read_text(encoding="utf-8") == "written"

    def test_binary_roundtrip(self, workdir: Path) -> None:
        """A binary file is written and read as bytes."""
        file = make_file(workdir, "blob", "application/octet-stream", binary=True)

        with cast(BinaryIO, file.get_io("w")) as io:
            io.write(b"\x00\x01")

        with cast(BinaryIO, file.get_io("r")) as io:
            assert io.read() == b"\x00\x01"

    def test_read_write_mode(self, workdir: Path) -> None:
        """``rw`` opens for both reading and writing."""
        file = make_file(workdir, "note", "text/plain", suffix=".txt")

        with cast(TextIO, file.open("rw")) as io:
            io.write("abc")
            io.seek(0)
            assert io.read() == "abc"

    def test_open_read_does_not_truncate(self, workdir: Path) -> None:
        """``open("r")`` honours the mode instead of reopening for writing."""
        path = workdir / "note.txt"
        path.write_text("keep me", encoding="utf-8")
        file = LocalFile.from_path(path)

        with cast(TextIO, file.open("r")) as io:
            assert io.read() == "keep me"

        assert path.read_text(encoding="utf-8") == "keep me"

    def test_open_write_truncates(self, workdir: Path) -> None:
        """``open("w")`` truncates the existing content."""
        path = workdir / "note.txt"
        path.write_text("old content", encoding="utf-8")

        with cast(TextIO, LocalFile.from_path(path).open("w")) as io:
            io.write("new")

        assert path.read_text(encoding="utf-8") == "new"

    def test_open_closes_stream(self, workdir: Path) -> None:
        """The stream is closed when the context manager exits."""
        file = make_file(workdir, "note", "text/plain", suffix=".txt")

        with file.open("w") as io:
            pass

        assert io.closed


class TestLocalDir:
    """Tests for :class:`LocalDir`."""

    def test_ensure_exists_creates_directory(self, tmp_path: Path) -> None:
        """Listing a directory that does not exist creates it."""
        path = tmp_path / "made" / "here"
        directory = LocalDir(path=path)

        assert directory.keys() == []
        assert path.is_dir()

    def test_abs_path(self, workdir: Path) -> None:
        """``abs_path`` is absolute and does not create anything."""
        directory = LocalDir(path=workdir)

        assert directory.abs_path == workdir.absolute()
        assert directory.abs_path.is_absolute()

    def test_keys_len_and_contains(self, workdir: Path) -> None:
        """Entries are listed and counted."""
        (workdir / "a.txt").write_text("a", encoding="utf-8")
        (workdir / "sub").mkdir()
        directory = LocalDir(path=workdir)

        assert sorted(directory.keys()) == ["a.txt", "sub"]
        assert len(directory) == 2
        assert "a.txt" in directory
        assert "missing" not in directory

    def test_getitem_file(self, workdir: Path) -> None:
        """Reading a file entry returns a :class:`LocalFile`."""
        (workdir / "a.txt").write_text("a", encoding="utf-8")

        item = LocalDir(path=workdir)["a.txt"]

        assert isinstance(item, LocalFile)
        assert item.path == workdir / "a.txt"

    def test_getitem_subdirectory(self, workdir: Path) -> None:
        """Reading a directory entry returns a :class:`LocalDir`."""
        (workdir / "sub").mkdir()

        item = LocalDir(path=workdir)["sub"]

        assert isinstance(item, LocalDir)
        assert item.abs_path == (workdir / "sub").absolute()

    def test_getitem_missing(self, workdir: Path) -> None:
        """A missing entry raises ``KeyError``."""
        with pytest.raises(KeyError, match="missing"):
            LocalDir(path=workdir)["missing"]

    def test_values_items_and_iteration(self, workdir: Path) -> None:
        """Files and subdirectories are reachable through all views."""
        (workdir / "a.txt").write_text("a", encoding="utf-8")
        (workdir / "sub").mkdir()
        directory = LocalDir(path=workdir)

        values = directory.values()
        items = dict(directory.items())

        assert len(values) == 2
        assert sorted(items) == ["a.txt", "sub"]
        assert len(list(directory)) == 2
        assert all(isinstance(value, File | Dir) for value in values)

    def test_file_and_subdir_helpers(self, workdir: Path) -> None:
        """The typed accessors return the requested kind."""
        (workdir / "a.txt").write_text("a", encoding="utf-8")
        (workdir / "sub").mkdir()
        directory = LocalDir(path=workdir)

        assert isinstance(directory.file("a.txt"), LocalFile)
        assert isinstance(directory.subdir("sub"), LocalDir)
        with pytest.raises(AssertionError):
            directory.file("sub")
        with pytest.raises(AssertionError):
            directory.subdir("a.txt")

    def test_delete_file(self, workdir: Path) -> None:
        """Deleting a file removes it from disk."""
        (workdir / "a.txt").write_text("a", encoding="utf-8")
        directory = LocalDir(path=workdir)

        del directory["a.txt"]

        assert not (workdir / "a.txt").exists()
        assert directory.keys() == []

    def test_delete_subdirectory(self, workdir: Path) -> None:
        """Deleting a subdirectory removes it with its content."""
        nested = workdir / "sub" / "deep"
        nested.mkdir(parents=True)
        (nested / "a.txt").write_text("a", encoding="utf-8")
        directory = LocalDir(path=workdir)

        del directory["sub"]

        assert not (workdir / "sub").exists()
        assert directory.keys() == []

    def test_delete_missing(self, workdir: Path) -> None:
        """Deleting a missing entry raises ``KeyError``."""
        with pytest.raises(KeyError, match="missing"):
            del LocalDir(path=workdir)["missing"]


class TestAbstractBases:
    """Tests for the abstract bases."""

    def test_file_is_abstract(self, workdir: Path) -> None:
        """``File`` cannot be instantiated directly."""
        with pytest.raises(TypeError):
            File(mime="text/plain", io_type=TextIO)  # pyright: ignore[reportAbstractUsage]

    def test_dir_is_abstract(self, workdir: Path) -> None:
        """``Dir`` cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Dir()  # pyright: ignore[reportAbstractUsage]


class OtherFile(File[str, TextIO]):
    """A different file implementation with the same format."""

    def get_io(self, mode: Literal["rw", "r", "w"] = "rw") -> TextIO:
        """Return an in-memory stream."""
        return io.StringIO()


class TestFileMatch:
    """Tests for :meth:`File.match`."""

    def test_matches_mime_and_io_type(self, workdir: Path) -> None:
        """A hint matches when its MIME type and IO type agree."""
        file = make_file(workdir, "f", "application/json")

        assert file.match(File[Literal["application/json"], TextIO])
        assert not file.match(File[Literal["application/pickle"], TextIO])
        assert not file.match(File[Literal["application/json"], BinaryIO])

    def test_matches_a_bare_file_hint(self, workdir: Path) -> None:
        """A hint without format arguments accepts any file of that class."""
        assert make_file(workdir, "f", "text/plain").match(File)

    def test_checks_the_declared_class(self, workdir: Path) -> None:
        """A hint for a concrete file class excludes other implementations."""
        file = make_file(workdir, "f", "application/json")
        other = OtherFile(mime="application/json", io_type=TextIO)

        assert file.match(LocalFile[Literal["application/json"]])
        assert not other.match(LocalFile[Literal["application/json"]])
        assert other.match(File[Literal["application/json"], TextIO])

    def test_rejects_non_file_hints(self, workdir: Path) -> None:
        """A hint that is not a file never matches."""
        assert not make_file(workdir, "f", "text/plain").match(dict)


class TestTextStreams:
    """Tests for the text encoding and newline handling."""

    def test_writes_utf8_with_lf_newlines(self, workdir: Path) -> None:
        """Text is UTF-8 and keeps ``\\n`` on every platform."""
        file = make_file(workdir, "note", "text/plain", suffix=".txt")

        with cast(TextIO, file.open("w")) as stream:
            stream.write("héllo\nworld")

        assert file.path.read_bytes() == "héllo\nworld".encode()

    def test_reads_untouched_newlines(self, workdir: Path) -> None:
        """Reading does not translate line endings."""
        path = workdir / "note.txt"
        path.write_bytes(b"a\r\nb")

        with cast(TextIO, LocalFile.from_path(path).open("r")) as stream:
            assert stream.read() == "a\r\nb"
