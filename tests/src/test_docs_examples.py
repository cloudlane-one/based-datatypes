"""Execute every runnable example embedded in the documentation.

The documentation uses ``markdown-exec`` so that examples run during the docs build. This
test runs the same blocks under pytest, which keeps them correct even when the docs are not
built.
"""

import re
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).parents[2] / "docs"

FENCE_START = re.compile(r"^```python(?P<info>[^\n]*)$")
ATTR = re.compile(r'(?P<key>\w+)="(?P<value>[^"]*)"')

MODULE_PAGES = (
    "how-to-enums.md",
    "how-to-types.md",
    "how-to-context.md",
    "how-to-runtime.md",
    "how-to-typing.md",
    "how-to-hashing.md",
    "how-to-data.md",
    "how-to-descriptors.md",
    "how-to-refs.md",
    "how-to-dist.md",
)

PAGES = [DOCS_DIR / name for name in MODULE_PAGES]


def page_examples(path: Path) -> list[tuple[str, int, str, str]]:
    """Extract executable code blocks from a documentation page.

    Args:
        path: Documentation page to scan.

    Returns:
        ``(session, line, code, page name)`` tuples for every ``exec`` block.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[tuple[str, int, str, str]] = []
    index = 0

    while index < len(lines):
        match = FENCE_START.match(lines[index])
        if match is None:
            index += 1
            continue

        info = {m.group("key"): m.group("value") for m in ATTR.finditer(match.group("info"))}
        start_line = index + 1
        index += 1
        body: list[str] = []

        while index < len(lines) and not lines[index].startswith("```"):
            body.append(lines[index])
            index += 1

        index += 1

        if info.get("exec") in {"on", "true"}:
            session = info.get("session", f"{path.stem}:default")
            blocks.append((session, start_line, "\n".join(body), path.name))

    return blocks


def test_every_module_has_a_documentation_page() -> None:
    """Each documented module has an executable usage page."""
    assert all(page.is_file() for page in PAGES)


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.stem)
def test_documentation_examples_run(page: Path) -> None:
    """Every executable block on a page runs, in order, in its session."""
    blocks = page_examples(page)
    assert blocks, f"{page.name} has no executable examples"

    namespaces: dict[str, dict[str, object]] = {}
    for session, line, code, name in blocks:
        namespace = namespaces.setdefault(
            session, {"__name__": f"docs_{page.stem}_{session.replace(':', '_')}"}
        )
        try:
            exec(compile(code, f"{name}:{line}", "exec"), namespace)  # noqa: S102
        except Exception as error:  # pragma: no cover - surfaced with context
            raise AssertionError(f"{name}:{line} failed: {error}") from error
