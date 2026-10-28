"""Tests for :mod:`based_datatypes.descriptors`."""

import pytest

from based_datatypes.descriptors import Delete, Read, RWD, Write, cached_method, dualmethod, prop


class Dual:
    """Class using a dual-mode property."""

    @prop(mode="dual")
    def mode(self: "Dual | type[Dual]") -> str:
        return "class" if isinstance(self, type) else "instance"


class InstanceProp:
    """Class using the default instance-mode property."""

    def __init__(self, value: int = 1) -> None:
        self._value = value

    @prop
    def value(self) -> int:  # pyright: ignore[reportRedeclaration]
        return self._value

    @value.setter
    def value(self, new_value: int) -> None:  # pyright: ignore[reportRedeclaration]
        self._value = new_value


class ClassProp:
    """Class using a class-mode property."""

    factor = 3

    @prop(mode="class")
    @classmethod
    def tripled(cls) -> int:
        return cls.factor * 3


class CachedProp:
    """Class using a cached instance property."""

    def __init__(self) -> None:
        self.calls = 0
        self._value = 1

    @prop(cached=True)
    def value(self) -> int:
        self.calls += 1
        return self._value

    @prop(cached=True)
    def settable(self) -> int:  # pyright: ignore[reportRedeclaration]
        self.calls += 1
        return self._value

    @settable.setter
    def settable(self, new_value: int) -> None:  # pyright: ignore[reportRedeclaration]
        self._value = new_value


class CachedClassProp:
    """Class using a cached class property."""

    calls = 0

    @prop(cached=True, mode="class")
    @classmethod
    def value(cls) -> int:
        cls.calls += 1
        return cls.calls


class Deletable:
    """Class using a deletable property."""

    def __init__(self) -> None:
        self._value = 5

    @prop(deletable=True)
    def value(self) -> int:  # pyright: ignore[reportRedeclaration]
        return self._value

    @value.deleter
    def value(self) -> None:  # pyright: ignore[reportRedeclaration]
        self._value = 0


class CallableProp:
    """Class defining a property through the decorator-factory form."""

    @prop(cached=True)
    def value(self) -> int:
        return 42


def test_dualmethod_works_on_the_instance() -> None:
    """The descriptor passes the instance when accessed through one."""

    class Example:
        @dualmethod
        def describe(self: "Example | type[Example]") -> str:
            return f"instance:{type(self).__name__}"

    assert Example().describe() == "instance:Example"


def test_dualmethod_works_on_the_class() -> None:
    """The descriptor passes the class when accessed through it."""

    class Example:
        @dualmethod
        def describe(self: "Example | type[Example]") -> str:
            return f"class:{getattr(self, '__name__', '')}"

    assert Example.describe() == "class:Example"


def test_dualmethod_preserves_function_metadata() -> None:
    """The wrapper keeps the wrapped function's name."""

    class Example:
        @dualmethod
        def describe(self: "Example | type[Example]") -> str:
            return "x"

    assert Example().describe.__name__ == "describe"


def test_dualmethod_returns_itself_without_an_owner() -> None:
    """Accessing the descriptor without an owner yields the descriptor."""
    method = dualmethod(lambda self: "x")

    assert method.__get__(None, None) is method


def test_prop_reads_through_the_instance() -> None:
    """The getter receives the instance."""
    assert InstanceProp(4).value == 4


def test_prop_returns_the_descriptor_on_class_access() -> None:
    """Instance-mode properties are not evaluated on the class."""
    assert isinstance(InstanceProp.value, prop)


def test_prop_uses_the_attribute_name() -> None:
    """``__set_name__`` records the attribute name."""
    assert InstanceProp.value.name == "value"


def test_prop_records_function_metadata() -> None:
    """The descriptor mirrors the getter's name when not renamed."""
    descriptor = InstanceProp.__dict__["value"]

    assert getattr(descriptor, "__name__") == "value"
    assert descriptor.__qualname__ == "InstanceProp.value"


def test_prop_setter_updates_the_value() -> None:
    """The setter is invoked through the descriptor."""
    instance = InstanceProp(1)
    setattr(instance, "value", 9)

    assert instance.value == 9


def test_prop_deleter_runs_on_delete() -> None:
    """The deleter is invoked through the descriptor."""
    instance = Deletable()
    del instance.value

    assert instance.value == 0


def test_prop_delete_without_deleter_is_a_no_op() -> None:
    """Deleting a read-only value silently leaves it unchanged."""
    instance = InstanceProp(3)

    del instance.value

    assert instance.value == 3


def test_prop_mode_instance_on_the_class() -> None:
    """An instance-mode property is returned unevaluated by the class."""
    assert isinstance(InstanceProp.__dict__["value"], prop)
    assert isinstance(InstanceProp.value, prop)


def test_prop_mode_dual_from_instance_and_class() -> None:
    """A dual-mode property resolves both ways."""
    assert Dual().mode == "instance"
    assert Dual.mode == "class"


def test_prop_mode_class_from_class_and_instance() -> None:
    """A class-mode property resolves through the class in both cases."""
    assert ClassProp.tripled == 9
    assert ClassProp().tripled == 9


def test_prop_cached_instance_value_is_computed_once() -> None:
    """Cached instance properties compute their value once."""
    instance = CachedProp()

    assert instance.value == 1
    assert instance.value == 1
    assert instance.calls == 1


def test_prop_cached_values_are_per_instance() -> None:
    """Each instance owns its cached value."""
    first = CachedProp()
    second = CachedProp()
    second._value = 7

    assert first.value == 1
    assert second.value == 7
    assert first.calls == 1
    assert second.calls == 1


def test_prop_cached_setter_invalidates_the_cache() -> None:
    """Assigning through a cached property drops the stale value."""
    instance = CachedProp()
    assert instance.value == 1

    instance._value = 5
    assert instance.value == 1

    setattr(instance, "settable", 5)
    assert instance.settable == 5


def test_prop_cached_delete_invalidates_the_cache() -> None:
    """Deleting through a cached property drops the cached value."""
    instance = CachedProp()
    assert instance.settable == 1

    del instance.settable

    assert instance.settable == 1
    assert instance.calls == 2


def test_prop_cached_class_value_is_computed_once() -> None:
    """Cached class properties are computed once per class."""
    assert CachedClassProp.value == 1
    assert CachedClassProp.value == 1
    assert CachedClassProp.calls == 1


def test_prop_cached_class_values_are_per_class() -> None:
    """A subclass gets its own cached class value."""

    class Sub(CachedClassProp):
        pass

    assert CachedClassProp.value == 1
    assert Sub.value == 2
    assert Sub.value == 2


def test_prop_can_be_used_as_a_decorator_factory() -> None:
    """``@prop(cached=True)`` applies the getter through ``__call__``."""
    assert CallableProp().value == 42


def test_prop_rejects_a_second_getter() -> None:
    """A getter can only be defined once."""
    descriptor = prop(lambda self: 1)

    with pytest.raises(AssertionError, match="Getter already defined"):
        descriptor.getter(lambda self: 2)


def test_prop_rejects_a_second_setter() -> None:
    """A setter can only be defined once."""
    descriptor = prop(lambda self: 1)
    descriptor.setter(lambda self, value: None)  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(AssertionError, match="Setter already defined"):
        descriptor.setter(lambda self, value: None)  # pyright: ignore[reportAttributeAccessIssue]


def test_prop_rejects_a_second_deleter() -> None:
    """A deleter can only be defined once."""
    descriptor = prop(lambda self: 1)
    descriptor.deleter(lambda self: None)  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(AssertionError, match="Deleter already defined"):
        descriptor.deleter(lambda self: None)  # pyright: ignore[reportAttributeAccessIssue]


def test_prop_requires_a_setter_when_writable() -> None:
    """Declaring a writable property without a setter fails at class creation."""
    with pytest.raises(AssertionError, match="Setter must be defined"):

        class Broken:
            @prop(writable=True)
            def value(self) -> int:
                return 1


def test_prop_requires_a_deleter_when_deletable() -> None:
    """Declaring a deletable property without a deleter fails."""
    with pytest.raises(AssertionError, match="Deleter must be defined"):

        class Broken:
            @prop(deletable=True)
            def value(self) -> int:
                return 1


def test_prop_name_requires_a_getter_or_binding() -> None:
    """A descriptor without a getter has no name."""
    with pytest.raises(AssertionError, match="Property name is not set"):
        prop().name  # noqa: B018


def test_prop_missing_getter_fails_on_access() -> None:
    """A descriptor without a getter raises when read."""
    descriptor = prop()
    descriptor._name = "value"

    with pytest.raises(AssertionError, match="no getter"):
        descriptor.__get__(InstanceProp(), InstanceProp)


def test_prop_marks_itself_as_concrete() -> None:
    """The descriptor is not abstract."""
    assert prop(lambda self: 1).__isabstractmethod__ is False


def test_read_write_delete_hierarchy() -> None:
    """The capability markers build a small hierarchy."""
    assert issubclass(Write, Read)
    assert issubclass(Delete, Read)
    assert issubclass(RWD, Write)
    assert issubclass(RWD, Delete)


class CachedMethodUser:
    """Class caching the result of an instance method per argument set."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], dict[str, int]]] = []

    @cached_method
    def compute(self, value: int, *, factor: int = 1) -> int:
        """Compute a value, recording every real call."""
        self.calls.append(((value,), {"factor": factor}))
        return value * factor

    @cached_method
    def combine(self, *, a: int, b: int) -> int:
        """Combine two keyword arguments, recording every real call."""
        self.calls.append(((), {"factor": a + b}))
        return a + b


class CachedMethodChild(CachedMethodUser):
    """Subclass sharing the decorated method but not its cache."""


def test_cached_method_caches_per_instance() -> None:
    """Repeated calls with the same arguments only compute once."""
    user = CachedMethodUser()

    assert user.compute(2) == 2
    assert user.compute(2) == 2
    assert len(user.calls) == 1


def test_cached_method_distinguishes_arguments() -> None:
    """Positional and keyword arguments form distinct cache keys."""
    user = CachedMethodUser()

    assert user.compute(1) == 1
    assert user.compute(2) == 2
    assert user.compute(1, factor=3) == 3
    assert len(user.calls) == 3


def test_cached_method_ignores_keyword_order() -> None:
    """Keyword arguments are keyed regardless of order."""
    user = CachedMethodUser()

    assert user.combine(a=1, b=2) == 3
    assert user.combine(b=2, a=1) == 3
    assert len(user.calls) == 1


def test_cached_method_is_per_instance_not_per_class() -> None:
    """Each instance keeps its own cache."""
    first = CachedMethodUser()
    second = CachedMethodUser()

    assert first.compute(3) == 3
    assert second.compute(3) == 3
    assert len(first.calls) == 1
    assert len(second.calls) == 1


def test_cached_method_on_subclass_is_independent() -> None:
    """A subclass instance has its own cache."""
    child = CachedMethodChild()
    parent = CachedMethodUser()

    assert child.compute(4) == 4
    assert parent.compute(4) == 4
    assert len(child.calls) == 1
    assert len(parent.calls) == 1


def test_cached_method_preserves_function_metadata() -> None:
    """The wrapper keeps the wrapped name and docstring."""
    assert CachedMethodUser.compute.__name__ == "compute"
    assert CachedMethodUser.compute.__doc__ == "Compute a value, recording every real call."


def test_cached_method_rejects_unhashable_arguments() -> None:
    """An unhashable argument cannot form a cache key."""
    user = CachedMethodUser()

    with pytest.raises(TypeError):
        user.compute([1], factor=1)  # pyright: ignore[reportArgumentType]
