# Scoped context variables

`based_datatypes.context` provides `set_ctx_var`, a context manager that sets a
`contextvars.ContextVar` for a block and restores the previous value afterwards — including
when the block raises.

## When to use it

Use it instead of manual `token = var.set(...)` / `var.reset(token)` pairs whenever the
value is scoped to a `with` block: request identifiers, per-operation log context, feature
flags during a test.

## Quick start

```python exec="on" session="context"
from contextvars import ContextVar

from based_datatypes.context import set_ctx_var

request_id: ContextVar[str] = ContextVar("request_id", default="none")

with set_ctx_var(request_id, "req-42") as current:
    print("inside:", current, request_id.get())

print("outside:", request_id.get())
```

## Nesting and restoration

Blocks unwind in order, so the value of the enclosing scope returns when an inner block
exits.

```python exec="on" session="context"
from contextvars import ContextVar

from based_datatypes.context import set_ctx_var

level: ContextVar[int] = ContextVar("level", default=0)

with set_ctx_var(level, 1):
    with set_ctx_var(level, 2):
        print("inner:", level.get())
    print("middle:", level.get())
print("outer:", level.get())
```

An exception does not leave a stale value behind:

```python exec="on" session="context"
from contextvars import ContextVar

from based_datatypes.context import set_ctx_var

flag: ContextVar[bool] = ContextVar("flag", default=False)

try:
    with set_ctx_var(flag, True):
        raise RuntimeError("boom")
except RuntimeError:
    pass

print(flag.get())
```

## API

| Name | Purpose |
| --- | --- |
| `set_ctx_var(var, value)` | Context manager yielding `value`; restores the previous value on exit. |

## Contexts stay isolated

`ContextVar` values are per-context, and the helper only touches the context it runs in.
A copy taken before the block keeps the old value, which is what asyncio tasks rely on.

```python exec="on" session="context"
import contextvars

from based_datatypes.context import set_ctx_var

var: ContextVar[int] = ContextVar("var", default=0)
copied = contextvars.copy_context()

with set_ctx_var(var, 7):
    print("current:", var.get(), "copied:", copied.run(var.get))
```

!!! note "Generators keep their context"
    The helper is a generator-based context manager. Do not yield out of the `with` block
    inside a generator that outlives it; the value would remain set for the caller.
