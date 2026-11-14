"""Base classes for the serialization of arbitrary types and data formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Any,
    Self,
)

import numpy as np
import rustworkx as rx

from based_datatypes.caching.base import cached
from based_datatypes.descriptors import dualmethod, prop
from based_datatypes.types import TypeDef
from based_datatypes.typing import Runtyped, get_inheritance_distance, is_subtype


@dataclass(frozen=True)
class Scope(ABC):
    """Base class for representing nested scopes via the type system."""

    @classmethod
    def depth(cls) -> int:
        """Return the depth of this scope in the hierarchy."""
        dist = get_inheritance_distance(cls, Scope)
        assert dist is not None, f"{cls.__name__} is not a subclass of Scope"
        return dist

    @dualmethod
    def match(self: Scope | type[Scope], scope: Scope | type[Scope]) -> bool:
        """Check whether this declared scope can serve the requested scope.

        Args:
            scope: The requested scope, as a class or value.

        Returns:
            ``True`` if the request is the declared scope or a narrower one.
        """
        self_cls = self if isinstance(self, type) else type(self)
        requested_cls = scope if isinstance(scope, type) else type(scope)
        return issubclass(requested_cls, self_cls)


@dataclass(frozen=True)
class OpenReach(Scope):
    """Unrestricted transport (online or offline)."""


@dataclass(frozen=True)
class InternetReach(OpenReach):
    """Transport over the public Internet."""


@dataclass(frozen=True)
class LocalNetReach(InternetReach):
    """Transport over a local network."""


@dataclass(frozen=True)
class LocalMachineReach(LocalNetReach):
    """Transport within a host machine."""


@dataclass(frozen=True)
class FullIntegrity(Scope):
    """Full, lossless copying."""


@dataclass(frozen=True)
class CoreIntegrity(FullIntegrity):
    """Represent the core aspects of object state."""


@dataclass(frozen=True)
class RenderIntegrity(FullIntegrity):
    """Retain all info relevant for rendering within a UI (no editing)."""


@dataclass(frozen=True)
class Conversion[S, T, R: OpenReach = OpenReach, I: FullIntegrity = FullIntegrity]:
    """A set of supported serialization formats."""

    src_type: TypeDef[S]
    """Source type of the conversion."""

    tgt_type: TypeDef[T]
    """Target type of the conversion."""

    reach: type[R] | R
    """The reach of the conversion."""

    integrity: type[I] | I
    """The integrity of the conversion."""


current_conversion_ctx: ContextVar[Conversion | None] = ContextVar(
    "current_conversion_ctx",
    default=None,
)
"""The current conversion context."""


@contextmanager
def conversion_context(conv: Conversion[Any, Any, Any, Any]) -> Generator[None]:
    """Context manager to set the current serialization context."""
    token = current_conversion_ctx.set(conv)
    try:
        yield
    finally:
        current_conversion_ctx.reset(token)


class Convertible[T, R: OpenReach = OpenReach, I: FullIntegrity = FullIntegrity](Runtyped):
    """Protocol for classes that can convert themselves."""

    @classmethod
    @abstractmethod
    def __conversions__(cls) -> Sequence[Conversion[Self, T, R, I]]:
        """Conversion targets supported by this class."""
        ...

    @classmethod
    @abstractmethod
    def __from__(
        cls,
        target: T,
        reach: R = OpenReach(),
        integrity: I = FullIntegrity(),
        annotation: TypeDef[Self] | None = None,
    ) -> Self:
        """Convert back from the target type or format to the original type.

        Args:
            target: The target object to convert from.
            reach: The reach of the conversion.
            integrity: The integrity of the conversion.
            annotation: The declared type of the object, if any.

        Returns:
            An instance of the class, converted from the source.
        """
        ...

    @abstractmethod
    def __to__(
        self,
        target_type: TypeDef[T],
        reach: R = OpenReach(),
        integrity: I = FullIntegrity(),
        annotation: TypeDef[Convertible[T, Any, Any]] | None = None,
    ) -> T:
        """Convert to the target type or format.

        Args:
            target_type: The target type or format to convert to.
            reach: The reach of the conversion.
            integrity: The integrity of the conversion.
            annotation: The declared type of the object, if any.

        Returns:
            The converted target object.
        """
        ...


@dataclass
class Converter[S, T, R: OpenReach = OpenReach, I: FullIntegrity = FullIntegrity](Runtyped):
    """A class representing a conversion driver for a specific type."""

    @prop(cached=True, mode="class")
    @classmethod
    def conversions(cls) -> Sequence[Conversion[S, T, R, I]]:
        """Targets supported by this converter (defaults to the declared typeargs)."""
        return [
            Conversion(
                src_type=cls._typeref.typevar_map[S].resolved_type_def,
                tgt_type=cls._typeref.typevar_map[T].resolved_type_def,
                reach=cls._typeref.typevar_map[R].common_type,
                integrity=cls._typeref.typevar_map[I].common_type,
            )
        ]

    @abstractmethod
    def convert_to(
        self,
        source: S,
        target_type: TypeDef[T] | None = None,
        reach: R = OpenReach(),
        integrity: I = FullIntegrity(),
        annotation: TypeDef[S] | None = None,
    ) -> T:
        """Convert to the target type.

        Args:
            source: The source object to convert.
            target_type: The target type or format to convert to.
            reach: The reach of the conversion.
            integrity: The integrity of the conversion.
            annotation: The declared type of the object, if any.

        Returns:
            The converted target object.
        """
        ...


@dataclass(frozen=True)
class ConversionEdge[S, T, R: OpenReach, I: FullIntegrity]:
    """A node in a conversion graph, representing a conversion step."""

    converter: Converter[S, T, R, I] | type[Convertible[T, R, I]]
    reach: type[R] | R
    integrity: type[I] | I

    def matches(self, conv: Conversion[Any, Any, R, I]) -> bool:
        """Check if this edge matches a given conversion."""
        return self.reach.match(conv.reach) and self.integrity.match(conv.integrity)


_type_dag = rx.PyDAG[TypeDef[Any]]()


def _single_source_distances(g: rx.PyDiGraph, source: int) -> np.ndarray:
    """Distance from `source` to every node, as int32; -1 means unreachable."""
    n = max(g.node_indices(), default=-1) + 1  # max index, not num_nodes(): holes
    dist = np.full(n, -1, dtype=np.int32)
    for depth, layer in enumerate(rx.digraph_bfs_layers(g, [source])):
        dist[np.fromiter(layer, dtype=np.int32, count=len(layer))] = depth
    return dist


@dataclass(frozen=True)
class ConversionNode:
    """A node in a conversion graph, representing a type."""

    type_def: TypeDef[Any]
    """The type definition of the node."""

    type_graph_node: int
    """The index of the node in the type DAG."""

    @cached_property
    def type_depth(self) -> int:
        """Get the depth of this node in the conversion graph."""
        return max(_single_source_distances(_type_dag, self.type_graph_node))

    def match_source(self, src_type: TypeDef) -> bool:
        """Check if this node matches a given conversion."""
        return is_subtype(src_type, self.type_def)

    def match_target(self, tgt_type: TypeDef) -> bool:
        """Check if this node matches a given conversion."""
        return is_subtype(self.type_def, tgt_type)


_conversion_graph: rx.PyDiGraph[ConversionNode, ConversionEdge] = rx.PyDiGraph()


@dataclass(frozen=True)
class ConversionStep:
    """A step in a conversion path, representing a conversion from one type to another."""

    source: ConversionNode
    """The source node of the conversion step."""

    edge: ConversionEdge[Any, Any, OpenReach, FullIntegrity]
    """The edge representing the conversion."""

    target: ConversionNode
    """The target node of the conversion step."""


def _get_matching_paths(
    conv: Conversion[Any, Any, Any, Any],
) -> list[tuple[ConversionStep, *tuple[ConversionStep, ...]]]:
    """Get all paths in the conversion graph that match the given conversion.

    Args:
        conv: The conversion to match.

    Returns:
        A list of paths, where each path is a list of direct conversion steps.
    """
    src_nodes = _conversion_graph.filter_nodes(lambda node: node.match_source(conv.src_type))
    tgt_nodes = _conversion_graph.filter_nodes(lambda node: node.match_target(conv.tgt_type))
    subgraph = _conversion_graph.subgraph(list(src_nodes) + list(tgt_nodes))

    paths = [
        path
        for sn in src_nodes
        for tn in tgt_nodes
        for path in rx.all_shortest_paths(subgraph, sn, tn)
    ]

    path_tuples = [
        tuple(
            ConversionStep(
                source=subgraph.get_node_data(path[i]),
                edge=subgraph.get_edge_data(path[i], path[i + 1]),
                target=subgraph.get_node_data(path[i + 1]),
            )
            for i in range(len(path) - 1)
        )
        for path in paths
    ]

    return [
        path for path in path_tuples if all(c.edge.matches(conv) for c in path) and len(path) > 0
    ]


def _sort_paths(
    paths: list[tuple[ConversionStep, *tuple[ConversionStep, ...]]],
) -> list[tuple[ConversionStep, *tuple[ConversionStep, ...]]]:
    """Sort paths by source, target, reach, and integrity.

    Narrower types are prioritized over broader types.
    """
    return sorted(
        paths,
        key=lambda path: (
            path[0].source.type_depth,
            path[-1].target.type_depth,
            path[-1].edge.reach.depth(),
            path[-1].edge.integrity.depth(),
        ),
    )


@cached(args_callback={"value": lambda v: id(v) if v is not None else None})
def get_conversion_path[S, T, R: OpenReach = OpenReach, I: FullIntegrity = FullIntegrity](
    source_type: TypeDef[S],
    target_type: TypeDef[T],
    reach: R = OpenReach(),
    integrity: I = FullIntegrity(),
) -> list[ConversionStep] | None:
    """Get the conversion path from source to target type.

    Args:
        source_type: The source type of the conversion.
        target_type: The target type of the conversion.
        reach: The reach of the conversion.
        integrity: The integrity of the conversion.

    Returns:
        A list of conversion steps representing the path from source to target type,
        or ``None`` if no path exists.
    """
    conv = Conversion(
        src_type=source_type,
        tgt_type=target_type,
        reach=reach,
        integrity=integrity,
    )
    paths = _get_matching_paths(conv)
    if not paths:
        return None

    sorted_paths = _sort_paths(paths)
    return list(sorted_paths[0])


def _assert_conversion_graph_determinism(
    new_converter: Converter[Any, Any, Any, Any] | type[Convertible[Any, Any, Any]],
    src_node: ConversionNode | None,
    tgt_node: ConversionNode | None,
) -> None:
    """Assert new converter does not take precedence over existing ones for same types & scopes."""
    raise NotImplementedError("Determinism check not implemented yet.")


def _add_converter_to_graph(
    conv: Converter[Any, Any, Any, Any] | type[Convertible[Any, Any, Any]],
    src_node: ConversionNode | None,
    tgt_node: ConversionNode | None,
) -> None:
    raise NotImplementedError("Adding converter to graph not implemented yet.")


def register_converter(
    conv: Converter[Any, Any, Any, Any] | type[Convertible[Any, Any, Any]],
) -> None:
    """Register a converter."""
    src_type_def = (
        conv._typeref.typevar_map[Converter._param_dict["S"]].resolved_type_def
        if isinstance(conv, Converter)
        else conv
    )
    tgt_type_def = (
        conv._typeref.typevar_map[Converter._param_dict["T"]].resolved_type_def
        if isinstance(conv, Converter)
        else conv._typeref.typevar_map[Convertible._param_dict["T"]].resolved_type_def
    )

    exact_src_nodes = _conversion_graph.filter_nodes(
        lambda node: bool(node.type_def == src_type_def)
    )
    exact_tgt_nodes = _conversion_graph.filter_nodes(
        lambda node: bool(node.type_def == tgt_type_def)
    )

    src_node = (
        sorted(
            [_conversion_graph.get_node_data(n) for n in exact_src_nodes],
            key=lambda n: n.type_depth,
        )[0]
        if exact_src_nodes
        else None
    )
    tgt_node = (
        sorted(
            [_conversion_graph.get_node_data(n) for n in exact_tgt_nodes],
            key=lambda n: n.type_depth,
        )[0]
        if exact_tgt_nodes
        else None
    )

    _assert_conversion_graph_determinism(conv, src_node, tgt_node)
    _add_converter_to_graph(conv, src_node, tgt_node)
