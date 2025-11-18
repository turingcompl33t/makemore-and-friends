"""
Unit tests for graph utilities.
"""

import typing

from .graph import topological_sort


class MyNode:
    def __init__(self, data: str, _children=()) -> None:
        self._data = data
        self._prev = set(_children)


def test_topo0() -> None:
    """Test topological sort."""
    a = MyNode("a")
    b = MyNode("b")
    c = MyNode("c", (a, b))

    sorted = typing.cast(list[MyNode], topological_sort(c))
    assert len(sorted) == 3
    assert sorted[-1]._data == "c"


def test_topo1() -> None:
    """Test topological sort."""
    a = MyNode("a")
    b = MyNode("b")
    c = MyNode("c", (a, b))
    d = MyNode("d")
    e = MyNode("e", (c, d))

    sorted = typing.cast(list[MyNode], topological_sort(e))
    assert len(sorted) == 5
    assert sorted[-1]._data == "e"
    assert sorted[-2]._data in ["d", "c"]
