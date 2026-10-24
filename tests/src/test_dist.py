"""Tests for :mod:`based_datatypes.dist`."""

import importlib.metadata as meta
import types
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from git import Repo

from based_datatypes import dist
from based_datatypes.dist import (
    _file_url_to_path,
    get_distributions,
    get_file_repo,
    get_module_distribution,
    get_module_file,
    get_module_repo,
    get_project_urls,
    get_py_inventory,
)
from tests.src.conftest import FakeDist


class FakeInventoryResponse:
    """Response stub carrying an ``objects.inv`` payload."""

    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        """Raise for error statuses, like ``requests`` does."""
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def make_inventory(entries: list[tuple[str, str, str]]) -> bytes:
    """Build a valid ``objects.inv`` payload.

    Args:
        entries: ``(name, type, uri)`` triples to include.

    Returns:
        The compressed Sphinx inventory file content.
    """
    header = (
        b"# Sphinx inventory version 2\n"
        b"# Project: demo\n"
        b"# Version: 1\n"
        b"# The remainder of this file is compressed using zlib.\n"
    )
    body = "".join(f"{name} {type} 1 {uri} -\n" for name, type, uri in entries).encode()
    return header + zlib.compress(body)


@pytest.fixture
def fake_module() -> types.ModuleType:
    """Return a module object with a plausible file path."""
    module = types.ModuleType("fake_module")
    module.__file__ = str(Path("/srv/lib/fake_module.py"))
    return module


def test_get_distributions_is_keyed_by_name() -> None:
    """Installed distributions are indexed by their metadata name."""
    distributions = get_distributions()

    assert "pandas" in distributions
    assert distributions["pandas"].metadata["Name"] == "pandas"


def test_get_distributions_is_cached() -> None:
    """Repeated calls return the same mapping object."""
    assert get_distributions() is get_distributions()


def test_get_module_file_of_a_module(tmp_path: Path) -> None:
    """A module with a file reports its path."""
    module = types.ModuleType("with_file")
    module.__file__ = str(tmp_path / "mod.py")

    assert get_module_file(module) == tmp_path / "mod.py"


def test_get_module_file_of_a_builtin() -> None:
    """A module without a file reports ``None``."""
    assert get_module_file(__import__("builtins")) is None


def test_get_module_file_with_a_none_file() -> None:
    """A module whose ``__file__`` is ``None`` reports ``None``."""
    module = types.ModuleType("no_file")
    module.__file__ = None

    assert get_module_file(module) is None


def test_file_url_to_path_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows file URLs drop the leading slash."""
    monkeypatch.setattr(dist.platform, "system", lambda: "Windows")

    assert _file_url_to_path("file:///C:/repos/pkg") == Path("C:/repos/pkg")


def test_file_url_to_path_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    """POSIX file URLs keep their absolute path."""
    monkeypatch.setattr(dist.platform, "system", lambda: "Linux")

    assert _file_url_to_path("file:///srv/repos/pkg") == Path("/srv/repos/pkg")


def test_get_module_distribution_matches_the_owning_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The distribution whose root contains the module is returned."""
    module_dir = tmp_path / "pkg"
    module_dir.mkdir()
    module = types.ModuleType("pkg")
    module.__file__ = str(module_dir / "__init__.py")
    owning = SimpleNamespace(
        locate_file=lambda name: module_dir,
        origin=None,
        metadata=SimpleNamespace(),
    )
    monkeypatch.setattr(dist, "get_distributions", lambda: {"pkg": owning})

    assert get_module_distribution(module) is owning


def test_get_module_distribution_prefers_the_deepest_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An editable install at the repository root does not shadow site-packages."""
    module_dir = tmp_path / "site-packages" / "pkg"
    module_dir.mkdir(parents=True)
    module = types.ModuleType("pkg")
    module.__file__ = str(module_dir / "__init__.py")
    editable = SimpleNamespace(
        locate_file=lambda name: tmp_path,
        origin=None,
        metadata=SimpleNamespace(),
    )
    owning = SimpleNamespace(
        locate_file=lambda name: module_dir,
        origin=None,
        metadata=SimpleNamespace(),
    )
    monkeypatch.setattr(dist, "get_distributions", lambda: {"editable": editable, "pkg": owning})

    assert get_module_distribution(module) is owning


def test_get_module_distribution_matches_a_file_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A distribution installed from a local path is matched by its URL."""
    module_dir = tmp_path / "repo" / "pkg"
    module_dir.mkdir(parents=True)
    module = types.ModuleType("pkg")
    module.__file__ = str(module_dir / "__init__.py")
    editable = SimpleNamespace(
        locate_file=lambda name: tmp_path / "elsewhere",
        origin=SimpleNamespace(url=(tmp_path / "repo").as_uri()),
        metadata=SimpleNamespace(),
    )
    monkeypatch.setattr(dist, "get_distributions", lambda: {"editable": editable})

    assert get_module_distribution(module) is editable


def test_get_module_distribution_without_a_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module outside every distribution reports ``None``."""
    module = types.ModuleType("loose")
    module.__file__ = str(tmp_path / "loose.py")
    other = SimpleNamespace(
        locate_file=lambda name: tmp_path / "nothing",
        origin=None,
        metadata=SimpleNamespace(),
    )
    monkeypatch.setattr(dist, "get_distributions", lambda: {"other": other})

    assert get_module_distribution(module) is None


def test_get_module_distribution_of_a_builtin() -> None:
    """A builtin has no distribution."""
    assert get_module_distribution(__import__("builtins")) is None


def test_get_module_distribution_of_an_installed_module() -> None:
    """A real installed module resolves to its own distribution."""
    import pandas

    resolved = get_module_distribution(pandas)

    assert resolved is not None
    assert resolved.metadata is not None
    assert resolved.metadata["Name"] == "pandas"


def test_get_file_repo_finds_the_repository(tmp_path: Path) -> None:
    """A file inside a Git work tree resolves to its repository."""
    Repo.init(tmp_path)
    target = tmp_path / "module.py"
    target.write_text("", encoding="utf-8")

    repo = get_file_repo(target)

    assert repo is not None
    assert Path(repo.working_dir).resolve() == tmp_path.resolve()


def test_get_file_repo_of_an_unversioned_directory(tmp_path: Path) -> None:
    """A directory outside any repository reports ``None``."""
    plain = tmp_path / "plain"
    plain.mkdir()
    target = plain / "module.py"
    target.write_text("", encoding="utf-8")

    assert get_file_repo(target) is None


def test_get_module_repo_of_a_versioned_module(tmp_path: Path) -> None:
    """A module inside a work tree resolves to its repository."""
    Repo.init(tmp_path)
    target = tmp_path / "module.py"
    target.write_text("", encoding="utf-8")
    module = types.ModuleType("versioned")
    module.__file__ = str(target)

    assert get_module_repo(module) is not None


def test_get_module_repo_of_a_builtin() -> None:
    """A module without a file has no repository."""
    assert get_module_repo(__import__("builtins")) is None


def test_get_project_urls_returns_matching_urls() -> None:
    """All URLs stored under the requested key are returned."""
    fake = FakeDist(
        "fake",
        "1.0",
        root=SimpleNamespace(),
        urls=[
            "Documentation, https://docs.example.com",
            "Repository, https://git.example.com",
            "Documentation, https://alt.example.com",
        ],
    )

    assert get_project_urls(fake, "Documentation") == [  # type: ignore[arg-type]
        "https://docs.example.com",
        "https://alt.example.com",
    ]


def test_get_project_urls_without_a_match() -> None:
    """An unknown key yields an empty list."""
    fake = FakeDist("fake", "1.0", root=SimpleNamespace(), urls=["Repository, https://x"])

    assert get_project_urls(fake, "Documentation") == []  # type: ignore[arg-type]


def test_get_project_urls_without_metadata() -> None:
    """A distribution without metadata yields an empty list."""
    fake = SimpleNamespace(metadata=None)

    assert get_project_urls(fake, "Documentation") == []  # type: ignore[arg-type]


def test_get_project_urls_without_project_urls() -> None:
    """A distribution without ``Project-URL`` entries yields an empty list."""
    fake = FakeDist("fake", "1.0", root=SimpleNamespace(), urls=None)

    assert get_project_urls(fake, "Documentation") == []  # type: ignore[arg-type]


def test_get_py_inventory_flattens_python_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only ``py:`` namespaces are merged, with absolute URIs."""
    payload = make_inventory(
        [
            ("mod", "py:module", "mod.html"),
            ("mod.value", "py:function", "mod.html#value"),
            ("index", "std:doc", "index.html"),
        ]
    )
    requests: list[str] = []

    def fake_get(url: str, **kwargs: Any) -> FakeInventoryResponse:
        requests.append(url)
        return FakeInventoryResponse(payload)

    monkeypatch.setattr(dist.requests, "get", fake_get)

    inventory = get_py_inventory("https://docs.example.com/base")

    assert set(inventory) == {"mod", "mod.value"}
    assert str(inventory["mod.value"].uri) == "https://docs.example.com/base/mod.html#value"
    assert requests == ["https://docs.example.com/base/objects.inv"]


def test_get_py_inventory_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parsed inventory is cached per documentation URL."""
    payload = make_inventory([("mod", "py:module", "mod.html")])
    calls: list[str] = []

    def fake_get(url: str, **kwargs: Any) -> FakeInventoryResponse:
        calls.append(url)
        return FakeInventoryResponse(payload)

    monkeypatch.setattr(dist.requests, "get", fake_get)
    url = "https://cached.example.com/docs"

    first = get_py_inventory(url)
    second = get_py_inventory(url)

    assert first is second
    assert len(calls) == 1


def test_get_py_inventory_propagates_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing inventory request raises."""
    monkeypatch.setattr(
        dist.requests,
        "get",
        lambda url, **kwargs: FakeInventoryResponse(b"", status_code=500),
    )

    with pytest.raises(RuntimeError, match="HTTP 500"):
        get_py_inventory("https://broken.example.com/docs")


def test_get_py_inventory_requires_python_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inventory without ``py:`` namespaces yields an empty mapping."""
    payload = make_inventory([("index", "std:doc", "index.html")])
    monkeypatch.setattr(
        dist.requests,
        "get",
        lambda url, **kwargs: FakeInventoryResponse(payload),
    )

    assert get_py_inventory("https://nopy.example.com/docs") == {}


def test_get_module_distribution_uses_real_metadata_types() -> None:
    """The returned object is a real ``importlib.metadata`` distribution."""
    import pytest as pytest_module

    resolved = get_module_distribution(pytest_module)

    assert isinstance(resolved, meta.Distribution)
