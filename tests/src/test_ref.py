"""Tests for :mod:`based_datatypes.ref`."""

import importlib.metadata as meta
import types
from types import SimpleNamespace
from typing import Any

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import Specifier, SpecifierSet
from packaging.version import Version

from based_datatypes import ref
from based_datatypes.ref import (
    PyObjectRef,
    get_all_module_dependencies,
    get_dist_requirements,
    get_outdated_deps,
    get_versions_on_pypi,
    semver_range_to_spec,
    stref,
    version_diff,
    version_to_range,
)
from tests.src.conftest import FakeDist


class FakeResponse:
    """Stand-in for ``requests.Response``."""

    def __init__(self, status_code: int = 200, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        """Return the canned JSON body."""
        return self._payload

    def raise_for_status(self) -> None:
        """Raise for error statuses, like ``requests`` does."""
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class DistributionWithOrigin(meta.Distribution):
    """Distribution whose origin is not ``None`` (e.g. a local wheel)."""

    @property
    def origin(self) -> Any:  # type: ignore[override]
        """Report a non-``None`` origin."""
        return SimpleNamespace(url="file:///tmp/wheel.whl")

    def locate_file(self, path: Any) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Resolve a path against the distribution root."""
        return SimpleNamespace(path=path)

    def read_text(self, filename: str) -> str | None:
        """Provide a minimal metadata reader."""
        return None


def fake_git(
    tag: str | None = None,
    branch: str | None = "main",
    detached: bool = False,
) -> SimpleNamespace:
    """Build a GitPython repository stub."""
    commit = object()
    tags = [SimpleNamespace(name=tag, commit=commit)] if tag is not None else []
    head = SimpleNamespace(commit=commit, is_detached=detached)
    active_branch = SimpleNamespace(name=branch)
    return SimpleNamespace(
        tags=tags,
        head=head,
        active_branch=active_branch,
        remote=lambda: SimpleNamespace(url="https://git.example.com/repo.git"),
    )


def test_get_dist_requirements_lists_declared_packages() -> None:
    """Requirements are parsed from the distribution metadata."""
    dist = FakeDist("fake", "1.0", root=SimpleNamespace(), requires=["requests>=2", "yarl"])

    requirements = get_dist_requirements(dist)  # type: ignore[arg-type]

    assert requirements is not None
    assert [str(req) for req in requirements] == ["requests>=2", "yarl"]
    assert isinstance(requirements[0], Requirement)


def test_get_dist_requirements_without_declarations() -> None:
    """A distribution without requirements yields ``None``."""
    dist = FakeDist("fake", "1.0", root=SimpleNamespace(), requires=None)

    assert get_dist_requirements(dist) is None  # type: ignore[arg-type]


def test_get_versions_on_pypi_of_a_package_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Published versions are parsed and invalid releases are skipped."""
    payload = {"releases": {"1.0.0": [], "1.2.0": [], "not-a-version": []}}
    monkeypatch.setattr(ref.requests, "get", lambda url: FakeResponse(200, payload))

    assert get_versions_on_pypi("fake-pkg") == {Version("1.0.0"), Version("1.2.0")}


def test_get_versions_on_pypi_returns_empty_for_unknown_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 yields an empty version set."""
    monkeypatch.setattr(ref.requests, "get", lambda url: FakeResponse(404))

    assert get_versions_on_pypi("does-not-exist") == set()


def test_get_versions_on_pypi_queries_the_package_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The JSON endpoint of the distribution's name is queried."""
    requested: list[str] = []

    def fake_get(url: str) -> FakeResponse:
        requested.append(url)
        return FakeResponse(200, {"releases": {}})

    monkeypatch.setattr(ref.requests, "get", fake_get)

    get_versions_on_pypi("fake-pkg")

    assert requested == ["https://pypi.org/pypi/fake-pkg/json"]


def test_get_versions_on_pypi_skips_a_distribution_with_an_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A locally built distribution is not looked up on PyPI."""
    calls: list[str] = []
    monkeypatch.setattr(ref.requests, "get", lambda url: calls.append(url) or FakeResponse())

    assert get_versions_on_pypi(DistributionWithOrigin()) == set()
    assert calls == []


def test_version_diff_of_a_major_change() -> None:
    """A major difference carries the newer minor and micro versions."""
    assert version_diff(Version("2.3.4"), Version("1.2.3")) == Version("1.3.4")


def test_version_diff_of_a_minor_change() -> None:
    """A minor difference carries the newer micro version."""
    assert version_diff(Version("1.5.4"), Version("1.2.3")) == Version("0.3.4")


def test_version_diff_of_a_micro_change() -> None:
    """A micro difference keeps all three components."""
    assert version_diff(Version("1.2.9"), Version("1.2.3")) == Version("0.0.6")


def test_version_diff_of_equal_versions() -> None:
    """Equal versions have a zero difference."""
    assert version_diff(Version("1.2.3"), Version("1.2.3")) == Version("0.0.0")


def test_version_diff_of_an_older_version() -> None:
    """An older first argument has no difference."""
    assert version_diff(Version("1.0.0"), Version("2.0.0")) is None


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [("exact", "==1.2.3"), ("minor", "~1.2.3"), ("major", "^1.2.3")],
)
def test_version_to_range(strategy: str, expected: str) -> None:
    """Each strategy renders its own range operator."""
    assert version_to_range(Version("1.2.3"), strategy) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("range_text", "expected"),
    [
        ("^1.2.3", ">=1.2.3,<2"),
        ("~1.2.3", ">=1.2.3,<1.3"),
        (">=1.2.3", ">=1.2.3"),
        ("<=2.0", "<=2.0"),
        (">1.0", ">1.0"),
        ("<2.0", "<2.0"),
        ("==1.2.3", "==1.2.3"),
        ("=1.2.3", "==1.2.3"),
        ("1.2.3", "==1.2.3"),
        ("  ^1.2.3 ", ">=1.2.3,<2"),
    ],
)
def test_semver_range_to_spec(range_text: str, expected: str) -> None:
    """Semver ranges map onto PEP 440 specifiers."""
    assert str(semver_range_to_spec(range_text)) == str(SpecifierSet(expected))


def test_semver_range_to_spec_rejects_an_invalid_version() -> None:
    """A malformed range raises ``InvalidVersion``."""
    with pytest.raises(ValueError):
        semver_range_to_spec("not-a-version")


def test_semver_range_matches_versions() -> None:
    """The produced specifier set contains the expected versions."""
    spec = semver_range_to_spec("^1.2.3")

    assert Version("1.2.3") in spec
    assert Version("1.9.9") in spec
    assert Version("2.0.0") not in spec


def test_get_outdated_deps_reports_a_major_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dependency with a newer major release is reported."""
    dist = FakeDist("fake", "1.0", root=SimpleNamespace(), requires=["dep>=1.0,<2"])
    monkeypatch.setattr(ref.meta, "distribution", lambda name: dist)
    monkeypatch.setattr(
        ref,
        "get_versions_on_pypi",
        lambda package: {Version("1.5.0"), Version("3.5.0")},
    )

    outdated = get_outdated_deps(dist)  # type: ignore[arg-type]

    assert outdated == {"dep": (Version("1.5.0"), Version("3.5.0"))}


def test_get_outdated_deps_ignores_a_small_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gap within the allowed difference is ignored."""
    dist = FakeDist("fake", "1.0", root=SimpleNamespace(), requires=["dep>=1.0,<2"])
    monkeypatch.setattr(ref.meta, "distribution", lambda name: dist)
    monkeypatch.setattr(
        ref,
        "get_versions_on_pypi",
        lambda package: {Version("1.0.0"), Version("2.0.1")},
    )

    assert get_outdated_deps(dist) == {}  # type: ignore[arg-type]


def test_get_outdated_deps_honours_allowed_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stricter ``allowed_diff`` reports smaller gaps."""
    dist = FakeDist("fake", "1.0", root=SimpleNamespace(), requires=["dep>=1.0,<2"])
    monkeypatch.setattr(ref.meta, "distribution", lambda name: dist)
    monkeypatch.setattr(
        ref,
        "get_versions_on_pypi",
        lambda package: {Version("1.0.0"), Version("2.0.1")},
    )

    outdated = get_outdated_deps(dist, allowed_diff=Specifier("<0.0.1"))  # type: ignore[arg-type]

    assert outdated == {"dep": (Version("1.0.0"), Version("2.0.1"))}


def test_get_outdated_deps_skips_prereleases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-, post- and dev-releases never count as newer versions."""
    dist = FakeDist("fake", "1.0", root=SimpleNamespace(), requires=["dep>=1.0,<2"])
    monkeypatch.setattr(ref.meta, "distribution", lambda name: dist)
    monkeypatch.setattr(
        ref,
        "get_versions_on_pypi",
        lambda package: {
            Version("1.0.0"),
            Version("3.0.0rc1"),
            Version("3.0.0.post1"),
            Version("3.0.0.dev1"),
        },
    )

    assert get_outdated_deps(dist) == {}  # type: ignore[arg-type]


def test_get_outdated_deps_without_requirements() -> None:
    """A distribution without requirements reports nothing."""
    dist = FakeDist("fake", "1.0", root=SimpleNamespace(), requires=None)

    assert get_outdated_deps(dist) == {}  # type: ignore[arg-type]


def test_get_outdated_deps_accepts_a_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """A module is resolved to its distribution first."""
    dist = FakeDist("fake", "1.0", root=SimpleNamespace(), requires=None)
    monkeypatch.setattr(ref, "get_module_distribution", lambda module: dist)

    assert get_outdated_deps(types.ModuleType("fake")) == {}


def test_get_outdated_deps_rejects_a_module_without_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A module outside any distribution raises ``ValueError``."""
    monkeypatch.setattr(ref, "get_module_distribution", lambda module: None)

    with pytest.raises(ValueError, match="not part of a distribution"):
        get_outdated_deps(types.ModuleType("loose-module"))


def test_get_all_module_dependencies_finds_external_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Members belonging to distributions are reported by package name."""
    module = types.ModuleType("mod_a")
    module.json = __import__("json")  # type: ignore[attr-defined]
    dep_dist = FakeDist("json-dist", "1.0", root=SimpleNamespace())

    monkeypatch.setattr(
        ref,
        "get_module_distribution",
        lambda mod: dep_dist if mod is module.json else None,
    )

    external, internal = get_all_module_dependencies(module)

    assert external == {"json-dist"}
    assert internal == set()


def test_get_all_module_dependencies_follows_internal_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Internal dependencies are traversed and reported."""
    module_a = types.ModuleType("mod_a")
    module_b = types.ModuleType("mod_b")
    module_b.json = __import__("json")  # type: ignore[attr-defined]
    module_a.b = module_b  # type: ignore[attr-defined]
    dep_dist = FakeDist("json-dist", "1.0", root=SimpleNamespace())

    monkeypatch.setattr(
        ref,
        "get_module_distribution",
        lambda mod: dep_dist if mod is module_b.json else None,
    )

    external, internal = get_all_module_dependencies(module_a)

    assert external == {"json-dist"}
    assert internal == {module_b}


def test_get_all_module_dependencies_handles_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutually-referencing modules terminate."""
    module_a = types.ModuleType("mod_a")
    module_b = types.ModuleType("mod_b")
    module_a.b = module_b  # type: ignore[attr-defined]
    module_b.a = module_a  # type: ignore[attr-defined]
    monkeypatch.setattr(ref, "get_module_distribution", lambda mod: None)

    external, internal = get_all_module_dependencies(module_a)

    assert external == set()
    assert internal == {module_b}


def test_get_all_module_dependencies_of_an_empty_module() -> None:
    """A module without members has no dependencies."""
    assert get_all_module_dependencies(types.ModuleType("empty")) == (set(), set())


def test_reference_of_a_builtin() -> None:
    """Builtins carry no repository or version."""
    obj_ref = PyObjectRef.reference(int)

    assert obj_ref.repo_url is None
    assert obj_ref.version is None
    assert obj_ref.module_name == "builtins"
    assert obj_ref.qualname == "int"
    assert obj_ref.object is int


def test_reference_of_a_distribution_member(
    make_dist: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A member of a distribution points at the PyPI index and its version."""
    dist, module = make_dist("fake-pkg", "1.2.3")
    monkeypatch.setattr(ref, "get_module_distribution", lambda mod: dist)

    obj_ref = PyObjectRef.reference(module.helper)

    assert obj_ref.repo_url == "https://pypi.org/simple/fake-pkg"
    assert obj_ref.version == "^1.2.3"
    assert obj_ref.py_version is not None
    assert obj_ref.py_version.startswith("~")


def test_reference_with_an_exact_version_strategy(
    make_dist: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact strategy pins the distribution version."""
    dist, module = make_dist("fake-pkg", "1.2.3")
    monkeypatch.setattr(ref, "get_module_distribution", lambda mod: dist)

    obj_ref = PyObjectRef.reference(module.helper, version_strategy="exact")

    assert obj_ref.version == "==1.2.3"


def test_reference_with_an_explicit_version(
    make_dist: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit version is kept verbatim."""
    dist, module = make_dist("fake-pkg", "1.2.3")
    monkeypatch.setattr(ref, "get_module_distribution", lambda mod: dist)

    obj_ref = PyObjectRef.reference(module.helper, version=">=2.0")

    assert obj_ref.version == ">=2.0"


def test_reference_with_a_non_semver_version(
    make_dist: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-semver distribution version is only kept in exact mode."""
    dist, module = make_dist("fake-pkg", "not-a-version")
    monkeypatch.setattr(ref, "get_module_distribution", lambda mod: dist)

    assert PyObjectRef.reference(module.helper).version is None
    assert PyObjectRef.reference(module.helper, version_strategy="exact").version == (
        "==not-a-version"
    )


def test_reference_of_a_first_party_module(make_dist: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """An editable distribution is referenced by its checkout's remote."""
    _, module = make_dist("fake-pkg", "1.2.3")
    dist = SimpleNamespace(
        name="fake-pkg",
        version="1.2.3",
        origin=SimpleNamespace(url="file:///repo"),
    )
    monkeypatch.setattr(ref, "get_module_distribution", lambda mod: dist)
    monkeypatch.setattr(ref, "get_module_repo", lambda mod: fake_git(tag="9.9.9"))

    obj_ref = PyObjectRef.reference(module.helper)

    assert obj_ref.repo_url == "https://git.example.com/repo.git"
    # The installed distribution version wins over the checkout's tags.
    assert obj_ref.version == "^1.2.3"


def test_reference_tags_a_git_checkout(make_dist: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkout without a distribution uses the tag on the head commit."""
    _, module = make_dist("fake-pkg", "1.2.3")
    monkeypatch.setattr(ref, "get_module_distribution", lambda mod: None)
    monkeypatch.setattr(ref, "get_module_repo", lambda mod: fake_git(tag="1.5.0"))

    obj_ref = PyObjectRef.reference(module.helper)

    assert obj_ref.repo_url == "https://git.example.com/repo.git"
    assert obj_ref.version == "^1.5.0"


def test_reference_uses_the_active_branch_without_tags(
    make_dist: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a tag, a semver branch name supplies the version."""
    _, module = make_dist("fake-pkg", "1.2.3")
    monkeypatch.setattr(ref, "get_module_distribution", lambda mod: None)
    monkeypatch.setattr(ref, "get_module_repo", lambda mod: fake_git(branch="2.0.0"))

    assert PyObjectRef.reference(module.helper).version == "^2.0.0"


def test_reference_without_a_tag_or_branch_version(
    make_dist: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detached head without tags has no version."""
    _, module = make_dist("fake-pkg", "1.2.3")
    monkeypatch.setattr(ref, "get_module_distribution", lambda mod: None)
    monkeypatch.setattr(
        ref, "get_module_repo", lambda mod: fake_git(tag=None, branch=None, detached=True)
    )

    assert PyObjectRef.reference(module.helper).version is None


def test_reference_rejects_a_loose_non_builtin_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-builtin object outside any repository is refused."""
    monkeypatch.setattr(ref, "get_module_distribution", lambda mod: None)
    monkeypatch.setattr(ref, "get_module_repo", lambda mod: None)

    with pytest.raises(ValueError, match="must be associated with a repository"):
        PyObjectRef.reference(version_diff)


def test_reference_requires_a_qualname() -> None:
    """An object without ``__qualname__`` is refused."""
    with pytest.raises(ValueError, match="fully qualified name"):
        PyObjectRef.reference(object())


def test_reference_requires_a_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """An object without a module is refused."""
    monkeypatch.setattr(ref.inspect, "getmodule", lambda obj: None)

    with pytest.raises(ValueError, match="associated with a module"):
        PyObjectRef.reference(Outer)


def test_from_url_parses_a_repository_reference() -> None:
    """Scheme, query parameters and fragment are split into fields."""
    obj_ref = PyObjectRef.from_url(
        "py+https://github.com/x/y?version=^1.2&python=~3.12#mod:obj", int
    )

    assert obj_ref.repo_url == "https://github.com/x/y"
    assert obj_ref.version == "^1.2"
    assert obj_ref.py_version == "~3.12"
    assert obj_ref.module_name == "mod"
    assert obj_ref.qualname == "obj"
    assert obj_ref.object_type is int


def test_from_url_parses_a_builtin_reference() -> None:
    """A bare ``py:`` scheme means a built-in object."""
    obj_ref = PyObjectRef.from_url("py:?python=~3.12#builtins:int", type)

    assert obj_ref.repo_url is None
    assert obj_ref.version is None
    assert obj_ref.object is int


def test_from_url_keeps_nested_qualnames() -> None:
    """Only the first colon separates module and object."""
    obj_ref = PyObjectRef.from_url("py+https://h/x#mod:Outer.method", int)

    assert obj_ref.qualname == "Outer.method"


def test_from_url_rejects_a_fragment_without_an_object() -> None:
    """The fragment must name both a module and an object."""
    with pytest.raises(ValueError, match="<module>:<qualname>"):
        PyObjectRef.from_url("py+https://h/x#mod", int)


def test_url_of_a_hosted_reference() -> None:
    """A hosted reference renders as a ``py+`` URL with its constraints."""
    obj_ref = PyObjectRef(
        object_type=int,
        qualname="value",
        module_name="mod",
        repo_url="https://github.com/x/y",
        version="^1.2",
        py_version="~3.12",
    )

    url = obj_ref.url

    assert url.scheme == "py+https"
    assert url.host == "github.com"
    assert url.path == "/x/y"
    assert url.query["version"] == "^1.2"
    assert url.query["python"] == "~3.12"
    assert url.fragment == "mod:value"


def test_url_preserves_an_explicit_port() -> None:
    """An explicit port survives the round trip."""
    obj_ref = PyObjectRef(
        object_type=int,
        qualname="value",
        module_name="mod",
        repo_url="https://github.com:8443/x/y",
    )

    assert obj_ref.url.explicit_port == 8443


def test_url_of_a_builtin_reference() -> None:
    """A builtin reference has no host."""
    obj_ref = PyObjectRef(object_type=int, qualname="int", module_name="builtins")

    assert obj_ref.url.scheme == "py"
    assert obj_ref.url.fragment == "builtins:int"


def test_url_of_a_file_repository() -> None:
    """A ``file://`` repository keeps its path."""
    obj_ref = PyObjectRef(
        object_type=int,
        qualname="value",
        module_name="mod",
        repo_url="file:///srv/repo",
    )

    assert obj_ref.url.scheme == "py+file"
    assert obj_ref.url.path == "/srv/repo"


def test_url_round_trips_through_from_url() -> None:
    """Parsing a rendered URL reproduces the reference fields."""
    obj_ref = PyObjectRef(
        object_type=int,
        qualname="Outer.value",
        module_name="mod",
        repo_url="https://git.example.com/group/repo.git",
        version="^1.2.3",
        py_version="~3.12",
    )

    restored = PyObjectRef.from_url(str(obj_ref.url), int)

    assert restored.repo_url == obj_ref.repo_url
    assert restored.version == obj_ref.version
    assert restored.py_version == obj_ref.py_version
    assert restored.module_name == obj_ref.module_name
    assert restored.qualname == obj_ref.qualname


def test_url_is_cached() -> None:
    """The rendered URL is cached on the reference."""
    obj_ref = PyObjectRef(object_type=int, qualname="int", module_name="builtins")

    assert obj_ref.url is obj_ref.url


def test_module_property_imports_the_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module is imported on first use."""
    obj_ref = PyObjectRef(object_type=int, qualname="int", module_name="builtins")

    assert obj_ref.module.__name__ == "builtins"


def test_module_property_reports_a_missing_module() -> None:
    """An unimportable module raises ``ImportError``."""
    obj_ref = PyObjectRef(object_type=int, qualname="x", module_name="no_such_module")

    with pytest.raises(ImportError, match="Cannot import module"):
        obj_ref.module  # noqa: B018


def test_dist_property_returns_the_cached_distribution() -> None:
    """A cached distribution is returned without a lookup."""
    dist = FakeDist("fake", "1.0", root=SimpleNamespace())
    obj_ref = PyObjectRef(object_type=int, qualname="x", module_name="mod", _dist=dist)  # type: ignore[arg-type]

    assert obj_ref.dist is dist


def test_dist_property_reports_a_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A module outside any distribution raises ``ImportError``."""
    obj_ref = PyObjectRef(object_type=int, qualname="x", module_name="builtins")
    monkeypatch.setattr(ref, "get_module_distribution", lambda mod: None)

    with pytest.raises(ImportError, match="not available in this environment"):
        obj_ref.dist  # noqa: B018


def test_dist_property_checks_the_requested_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A distribution that does not satisfy the reference version is refused."""
    dist = FakeDist("fake", "2.0.0", root=SimpleNamespace())
    obj_ref = PyObjectRef(
        object_type=int,
        qualname="x",
        module_name="builtins",
        version="^1.0",
    )
    monkeypatch.setattr(ref, "get_module_distribution", lambda mod: dist)

    with pytest.raises(ImportError, match="correct version"):
        obj_ref.dist  # noqa: B018


def test_dist_property_accepts_a_matching_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A distribution inside the requested range is accepted."""
    dist = FakeDist("fake", "2.0.0", root=SimpleNamespace())
    obj_ref = PyObjectRef(
        object_type=int,
        qualname="x",
        module_name="builtins",
        version="^2.0",
    )
    monkeypatch.setattr(ref, "get_module_distribution", lambda mod: dist)

    assert obj_ref.dist is dist


def test_docs_url_uses_the_object_inventory() -> None:
    """A matching inventory entry is used directly."""
    dist = FakeDist(
        "fake",
        "1.0",
        root=SimpleNamespace(),
        urls=["Documentation, https://docs.example.com"],
    )
    obj_ref = PyObjectRef(
        object_type=int,
        qualname="value",
        module_name="mod",
        _dist=dist,  # type: ignore[arg-type]
    )
    inventory = {"mod.value": SimpleNamespace(uri="https://docs.example.com/mod.html#value")}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ref, "get_py_inventory", lambda url: inventory)
        assert str(obj_ref.docs_url) == "https://docs.example.com/mod.html#value"


def test_docs_url_falls_back_to_the_api_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a documentation URL, the API-reference link is probed."""
    dist = FakeDist(
        "fake",
        "1.0",
        root=SimpleNamespace(),
        urls=["API Reference, https://api.example.com"],
    )
    obj_ref = PyObjectRef(
        object_type=int,
        qualname="value",
        module_name="mod",
        _dist=dist,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(ref, "get_py_inventory", lambda url: {})
    monkeypatch.setattr(ref.requests, "get", lambda url: FakeResponse(200))

    assert str(obj_ref.docs_url) == "https://api.example.com/mod.html#value"


def test_docs_url_keeps_the_documentation_root_when_no_api_exists() -> None:
    """A documentation URL without an inventory match is returned as-is."""
    dist = FakeDist(
        "fake",
        "1.0",
        root=SimpleNamespace(),
        urls=["Documentation, https://docs.example.com"],
    )
    obj_ref = PyObjectRef(
        object_type=int,
        qualname="value",
        module_name="mod",
        _dist=dist,  # type: ignore[arg-type]
    )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ref, "get_py_inventory", lambda url: None)
        assert str(obj_ref.docs_url) == "https://docs.example.com"


def test_docs_url_without_project_urls() -> None:
    """A distribution without documentation URLs yields ``None``."""
    dist = FakeDist("fake", "1.0", root=SimpleNamespace(), urls=None)
    obj_ref = PyObjectRef(
        object_type=int,
        qualname="value",
        module_name="mod",
        _dist=dist,  # type: ignore[arg-type]
    )

    assert obj_ref.docs_url is None


def test_docs_url_without_a_distribution() -> None:
    """A reference without a distribution raises when asked for docs."""
    obj_ref = PyObjectRef(object_type=int, qualname="x", module_name="builtins")

    with pytest.raises(ImportError):
        obj_ref.docs_url  # noqa: B018


def test_object_resolves_a_nested_attribute() -> None:
    """The qualname is followed through attributes."""
    obj_ref = PyObjectRef(
        object_type=int,
        qualname="Outer.value",
        module_name=__name__,
    )

    assert obj_ref.object == 42


def test_object_rejects_a_wrong_type() -> None:
    """A resolved object of the wrong type raises ``TypeError``."""
    obj_ref = PyObjectRef(object_type=str, qualname="Outer.value", module_name=__name__)

    with pytest.raises(TypeError, match="must have type"):
        obj_ref.object  # noqa: B018


def test_object_verifies_the_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A module that comes from another repository is refused."""
    dist = FakeDist("fake", "1.0", root=SimpleNamespace(), origin_url="file:///elsewhere")
    obj_ref = PyObjectRef(
        object_type=int,
        qualname="Outer.value",
        module_name=__name__,
        repo_url="https://github.com/expected/repo",
        _dist=dist,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(ref, "get_module_repo", lambda mod: None)

    with pytest.raises(ImportError, match="Expected module"):
        obj_ref.object  # noqa: B018


def test_object_is_cached() -> None:
    """The resolved object is cached on the reference."""
    obj_ref = PyObjectRef(object_type=int, qualname="Outer.value", module_name=__name__)

    assert obj_ref.object is obj_ref.object


def test_stref_of_a_builtin() -> None:
    """Built-in objects render as ``module.qualname``."""
    assert stref(int) == "builtins.int"


def test_stref_of_a_library_function() -> None:
    """Library functions render with their defining module."""
    assert stref(version_diff) == "based_datatypes.ref.version_diff"


class Outer:
    """Container used to exercise nested attribute resolution."""

    value = 42
