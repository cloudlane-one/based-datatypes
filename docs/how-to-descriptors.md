# Properties and dual-dispatch methods

`based_datatypes.descriptors` provides two descriptors:

- `prop`: a typed property that can be an instance property, a class property, or both, and
  that can cache its value.
- `dualmethod`: a method that works when called on an instance *and* on the class.

Both exist because the builtin `property` / `classmethod` combinations are hard to type and
easy to get subtly wrong.

## When to use `prop`

- A computed value that should behave like an attribute (`@prop`).
- A constant derived from the class, reachable as `Model.timeout` (`mode="class"`).
- A value computed per instance exactly once (`cached=True`).
- An attribute that needs a setter and/or deleter without boilerplate.

## Instance properties

`@prop` is the default, instance-mode form. Reading it on the class returns the descriptor
itself, just like `property`.

```python exec="on" session="descriptors"
from based_datatypes.descriptors import prop


class Circle:
    """A circle with a computed area."""

    def __init__(self, radius: float) -> None:
        self.radius = radius

    @prop
    def area(self) -> float:
        """Area of the circle."""
        return 3.141592653589793 * self.radius**2


circle = Circle(2)
print(round(circle.area, 3))
print(type(Circle.area).__name__)
print(Circle.area.name)
```

Setters and deleters use the familiar decorator pattern. Declaring `writable=True` or
`deletable=True` without the corresponding function fails at class creation, so a
half-implemented property cannot reach production.

```python exec="on" session="descriptors"
from based_datatypes.descriptors import prop


class Config:
    """Mutable configuration value."""

    def __init__(self) -> None:
        self._timeout = 5

    @prop
    def timeout(self) -> int:
        """Current timeout in seconds."""
        return self._timeout

    @timeout.setter
    def timeout(self, value: int) -> None:
        self._timeout = value


config = Config()
config.timeout = 10
print(config.timeout)
```

## Class and dual properties

`mode="class"` evaluates against the class (works from instances too). `mode="dual"` passes
whichever object was used to access it, so one implementation can serve both.

```python exec="on" session="descriptors"
from based_datatypes.descriptors import prop


class Endpoint:
    """Class-level default plus per-instance override."""

    base = "https://api.example.com"

    @prop(mode="class")
    @classmethod
    def base_url(cls) -> str:
        """Base URL of the service."""
        return cls.base

    @prop(mode="dual")
    def described_as(self: "Endpoint | type[Endpoint]") -> str:
        """Report whether the class or an instance was used."""
        return "class" if isinstance(self, type) else "instance"


print(Endpoint.base_url, Endpoint().base_url)
print(Endpoint.described_as, Endpoint().described_as)
```

## Caching

`cached=True` computes the value once and reuses it. Assigning or deleting through the
property drops the cache; with a setter the value is then recomputed from the new state.

```python exec="on" session="descriptors"
from based_datatypes.descriptors import prop


class Expensive:
    """Counts how often the cached value was computed."""

    def __init__(self) -> None:
        self.calls = 0
        self._value = 1

    @prop(cached=True)
    def value(self) -> int:
        """Computed once per instance."""
        self.calls += 1
        return self._value

    @value.setter
    def value(self, new_value: int) -> None:
        self._value = new_value


item = Expensive()
print(item.value, item.value, item.calls)

item.value = 7
print(item.value, item.calls)
```

!!! warning "The instance cache holds strong references"
    Cached instance values are stored per property, keyed by instance identity, and the
    entry keeps the instance alive. Do not use `cached=True` for a property on a class with
    a very large number of short-lived instances; use an explicit cache with a policy
    instead.

!!! note "Class caches are per class"
    A subclass parameterised differently gets its own cached class value.

## Dual-dispatch methods

`dualmethod` wraps a function whose first parameter is annotated `Self | type[Self]`. The
descriptor passes the instance when called through an instance and the class when called
through the class.

```python exec="on" session="descriptors"
from based_datatypes.descriptors import dualmethod


class Factory:
    """Constructs itself through a dual method."""

    def __init__(self, name: str) -> None:
        self.name = name

    @dualmethod
    def describe(self: "Factory | type[Factory]") -> str:
        """Describe the class or the instance."""
        if isinstance(self, type):
            return f"class {self.__name__}"
        return f"instance {self.name}"


print(Factory.describe())
print(Factory("primary").describe())
```

!!! note "Annotate the first parameter"
    The implementation receives either a class or an instance, so its first parameter must
    be annotated `ClassName | type[ClassName]` for the type checker to accept the
    descriptor. A plain `self: ClassName` annotation is rejected — the call would not be
    type-safe through the class.

## Marker classes

`Read`, `Write`, `Delete` and `RWD` are capability markers used in `prop`'s typing. They
form a small hierarchy (`Write` and `Delete` derive from `Read`, `RWD` from both) and are
exported for annotations in downstream code.
