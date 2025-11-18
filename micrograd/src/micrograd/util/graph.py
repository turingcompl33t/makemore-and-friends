"""
Graph utilities.
"""

from __future__ import annotations

from typing import Iterable, Protocol


class Node(Protocol):
    """A thing that is topological-sortable."""

    # fmt: off
    @property
    def _prev(self) -> Iterable[Node]:
        ...
    # fmt: on


def topological_sort(n: Node) -> list[Node]:
    topo: list[Node] = []

    visited = set()

    def build_topo(v: Node):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(n)
    return topo
