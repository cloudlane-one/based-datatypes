"""Tests for :mod:`based_datatypes.context`."""

import contextvars

import pytest

from based_datatypes.context import set_ctx_var


def test_sets_the_value_inside_the_block() -> None:
    """The context variable holds the new value inside the block."""
    var: contextvars.ContextVar[int] = contextvars.ContextVar("var", default=0)

    with set_ctx_var(var, 5):
        assert var.get() == 5


def test_restores_the_previous_value_on_exit() -> None:
    """The previous value is restored after the block."""
    var: contextvars.ContextVar[int] = contextvars.ContextVar("var", default=0)

    with set_ctx_var(var, 5):
        pass

    assert var.get() == 0


def test_restores_a_previously_set_value() -> None:
    """A value set before the block comes back afterwards."""
    var: contextvars.ContextVar[str] = contextvars.ContextVar("var", default="default")
    var.set("outer")

    with set_ctx_var(var, "inner"):
        assert var.get() == "inner"

    assert var.get() == "outer"


def test_yields_the_value() -> None:
    """The context manager yields the value it set."""
    var: contextvars.ContextVar[str] = contextvars.ContextVar("var")

    with set_ctx_var(var, "value") as yielded:
        assert yielded == "value"


def test_nested_blocks_unwind_in_order() -> None:
    """Nested blocks restore level by level."""
    var: contextvars.ContextVar[int] = contextvars.ContextVar("var", default=0)

    with set_ctx_var(var, 1):
        with set_ctx_var(var, 2):
            assert var.get() == 2
        assert var.get() == 1

    assert var.get() == 0


def test_restores_on_exception() -> None:
    """The token is reset even when the block raises."""
    var: contextvars.ContextVar[int] = contextvars.ContextVar("var", default=7)

    with pytest.raises(RuntimeError):
        with set_ctx_var(var, 9):
            raise RuntimeError("boom")

    assert var.get() == 7


def test_generator_is_closed_after_the_block() -> None:
    """The generator cannot be advanced after the block completes."""
    var: contextvars.ContextVar[int] = contextvars.ContextVar("var", default=0)
    manager = set_ctx_var(var, 1)

    with manager as value:
        assert value == 1

    with pytest.raises(StopIteration):
        next(manager.gen)  # type: ignore[attr-defined]


def test_changes_do_not_leak_into_a_copied_context() -> None:
    """Context variables remain isolated per context."""
    var: contextvars.ContextVar[int] = contextvars.ContextVar("var", default=0)
    context = contextvars.copy_context()

    assert context.run(var.get) == 0
    with set_ctx_var(var, 3):
        assert context.run(var.get) == 0
