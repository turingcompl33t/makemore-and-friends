"""
Unit tests for engine.
"""

import numpy as np
import torch

from .engine import Value


def test_radd() -> None:
    a = Value(3.0)
    b = 1.0 + a
    assert b.data == 4.0


def test_sub() -> None:
    a = Value(4.0)
    b = Value(2.0)
    c = a - b
    assert c.data == 2.0


def test_rmul() -> None:
    a = Value(3.0)
    b = 2.0 * a
    assert b.data == 6.0


def test_div() -> None:
    a = Value(2.0)
    b = Value(4.0)
    c = a / b
    assert c.data == 0.5


def test_backward() -> None:
    a = Value(3.0, label="a")
    b = a + a
    b.label = "b"
    b.backward()

    assert a.grad == 2.0


def test_backward_torch() -> None:
    """micrograd and pytorch produce equivalent results on the same computation graph"""

    # inputs x1, x2
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    # weights w1, w2
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b = Value(6.8813, label="b")
    # x1*w1 + x2*w2 + b
    x1w1 = x1 * w1
    x1w1.label = "x1*w1"
    x2w2 = x2 * w2
    x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w1 + x2*w2"
    n = x1w1x2w2 + b
    n.label = "n"
    o = n.tanh()
    o.backward()

    _x1 = torch.Tensor([2.0]).double()
    _x1.requires_grad = True
    _x2 = torch.Tensor([0.0]).double()
    _x2.requires_grad = True
    _w1 = torch.Tensor([-3.0]).double()
    _w1.requires_grad = True
    _w2 = torch.Tensor([1.0]).double()
    _w2.requires_grad = True
    _b = torch.Tensor([6.8813]).double()
    _b.requires_grad = True
    _n = _x1 * _w1 + _x2 * _w2 + _b
    _o = torch.tanh(_n)
    _o.backward()

    # forward pass
    assert np.allclose([_o.data.item()], [o.data])

    # backward pass
    assert (
        _x1.grad is not None
        and _x2.grad is not None
        and _w1.grad is not None
        and _w2.grad is not None
    )
    assert np.allclose([x1.grad], [_x1.grad.item()])
    assert np.allclose([x2.grad], [_x2.grad.item()])
    assert np.allclose([w1.grad], [_w1.grad.item()])
    assert np.allclose([w2.grad], [_w2.grad.item()])
