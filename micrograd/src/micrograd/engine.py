from __future__ import annotations

import math
import typing

from micrograd.util.graph import topological_sort


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        # the data maintained by this object
        self.data = data
        # the gradient of the output of the graph w.r.t this node
        self.grad = 0.0
        # a human-readable label for this node
        self.label = label

        # the function for computing the local gradient
        self._backward = lambda: None
        # the anscestors of this node in the graph
        self._prev = set(_children)
        # the operation used to compute this node
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: float | int | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Value | int | float) -> Value:
        return self + other

    def __mul__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Value | int | float) -> Value:
        return self * other

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value | int | float) -> Value:
        return self + (-other)

    def __pow__(self, other: int | float) -> Value:
        assert isinstance(other, (int, float)), "broken precondition"

        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1.0)) * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other: Value | int | float) -> Value:
        return self * other**-1

    def exp(self) -> Value:
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def tanh(self) -> Value:
        x = self.data
        t = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        x = self.data
        t = 0.0 if x <= 0.0 else x
        out = Value(t, (self,), "relu")

        def _backward():
            self.grad += (0.0 if x < 0.0 else 1.0) * out.grad

        out._backward = _backward

        return out

    def backward(self) -> None:
        self.grad = 1.0
        for node in reversed(topological_sort(self)):
            node = typing.cast(Value, node)
            node._backward()
