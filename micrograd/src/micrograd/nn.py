import random
import typing

from micrograd.engine import Value


class Module:
    def __init__(self) -> None:
        pass

    def parameters(self) -> list[Value]:
        return []

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0


class Neuron(Module):
    def __init__(self, nin: int) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: list[float] | list[Value]) -> Value:
        # input may be wrapped in Value already, or cast here
        _x = typing.cast(
            list[Value], x if isinstance(x[0], Value) else [Value(v) for v in x]
        )
        # w * x + b
        activation = sum((wi * xi for wi, xi in zip(self.w, _x)), self.b)
        return activation.tanh()

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin: int, nout: int) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: list[float] | list[Value]) -> list[Value] | Value:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, nin: int, nouts: list[int]) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x: list[float] | list[Value]) -> Value:
        _x = x if isinstance(x[0], Value) else [Value(v) for v in x]
        _x = typing.cast(list[Value], _x)

        for layer in self.layers:
            r = layer(_x)
            _x = [r] if isinstance(r, Value) else r

        assert len(_x) == 1, "broken invariant"
        return _x[0]

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
