"""Shared fixtures for the library test suite."""

import importlib
import importlib.metadata as meta
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@dataclass
class FakeDist:
    """Minimal stand-in for ``importlib.metadata.Distribution``."""

    name: str
    version: str
    root: Any
    requires: list[str] | None = None
    urls: list[str] | None = None
    origin_url: str | None = None

    @property
    def metadata(self) -> "FakeMetadata":
        """Metadata object exposing the attributes the library reads."""
        return FakeMetadata(self.name, self.requires, self.urls)

    @property
    def origin(self) -> object | None:
        """Distribution origin, if the fake declares one."""
        if self.origin_url is None:
            return None
        return FakeOrigin(self.origin_url)

    def locate_file(self, path: str) -> Path:
        """Resolve a file relative to the fake distribution root."""
        return self.root / path


@dataclass
class FakeMetadata:
    """Metadata stand-in supporting ``get_all`` and item access."""

    name: str
    requires: list[str] | None = None
    urls: list[str] | None = None

    def __getitem__(self, key: str) -> str:
        """Return a metadata field by name."""
        if key == "Name":
            return self.name
        raise KeyError(key)

    def get_all(self, key: str) -> list[str] | None:
        """Return all values of a multi-valued metadata field."""
        if key == "Project-URL":
            return self.urls
        return None


@dataclass
class FakeOrigin:
    """Distribution origin stand-in."""

    url: str


@pytest.fixture
def make_dist(tmp_path: Path) -> Iterator[Callable[..., tuple[meta.Distribution, ModuleType]]]:
    """Create importable packages backed by real ``.dist-info`` metadata."""
    added_modules: list[str] = []

    def _make(
        name: str = "fake-pkg",
        version: str = "1.2.3",
        requires: tuple[str, ...] = (),
        project_urls: tuple[str, ...] = (),
    ) -> tuple[meta.Distribution, ModuleType]:
        module_name = name.replace("-", "_")
        info = tmp_path / f"{module_name}-{version}.dist-info"
        info.mkdir(parents=True, exist_ok=True)

        metadata = [
            "Metadata-Version: 2.1",
            f"Name: {name}",
            f"Version: {version}",
            *(f"Requires-Dist: {req}" for req in requires),
            *(f"Project-URL: {url}" for url in project_urls),
        ]
        (info / "METADATA").write_text("\n".join(metadata) + "\n", encoding="utf-8")
        (info / "RECORD").write_text("", encoding="utf-8")

        package = tmp_path / module_name
        package.mkdir(exist_ok=True)
        (package / "__init__.py").write_text(
            "VALUE = 1\n\ndef helper() -> int:\n    return VALUE\n",
            encoding="utf-8",
        )

        if str(tmp_path) not in sys.path:
            sys.path.insert(0, str(tmp_path))
        importlib.invalidate_caches()

        module = importlib.import_module(module_name)
        added_modules.append(module_name)

        return meta.Distribution.at(info), module

    yield _make

    for module_name in added_modules:
        sys.modules.pop(module_name, None)
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))
