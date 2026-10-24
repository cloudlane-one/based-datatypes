# Enums that read well in docs and logs

`based_datatypes.enums` provides one class, `StrEnum`, a drop-in replacement for
`enum.StrEnum`. Its only difference is presentation: a member renders as its value
everywhere, including in `repr`, so generated API docs, logs and tracebacks show
`'red'` instead of `<Color.red: 'red'>`.

## When to use it

Use it for every string enum in a project that documents its enums (mkdocstrings, Sphinx)
or logs enum values, and especially where members are parsed from configuration files.

## Quick start

```python exec="on" session="enums"
from based_datatypes.enums import StrEnum


class Color(StrEnum):
    """Colours supported by the example widget."""

    red = "red"
    green = "green"


color = Color.parse("red")
print(repr(color))
print(str(color))
print(f"{color}!")
```

## API

| Name | Purpose |
| --- | --- |
| `StrEnum` | `enum.StrEnum` subclass whose `repr` is the quoted value. |
| `StrEnum.parse(value)` | Classmethod returning the member for a value; raises `ValueError` otherwise. |

## Members behave like strings

A member is a `str` subclass, so it compares equal to its value, serialises as that value,
and can be used as a mapping key.

```python exec="on" session="enums"
import json

print(Color.red == "red")
print(json.dumps({"color": Color.red}))
print({Color.red: "#f00"}[Color("red")])
```

## Parsing configuration values

`parse` is the typed counterpart of calling the enum: it keeps the subclass type, so code
that receives a `Color` keeps receiving a `Color`.

```python exec="on" session="enums"
try:
    Color.parse("blue")
except ValueError as error:
    print(type(error).__name__, error)
```

!!! note "No fuzzy parsing"
    `parse` matches values exactly. Case folding, aliases and prefix matching are policy
    decisions and belong in the calling code, not in the enum base class.
