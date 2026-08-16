"""
Gradient-based training for character-level language models.
"""

from collections.abc import Callable, Iterable

import torch

from .base import CharLM
from .data import make_dataset


class SGD:
    """Stochastic gradient descent."""

    def __init__(self, params: Iterable[torch.Tensor], lr: float) -> None:
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        """Discard the gradients left by the previous backward pass."""
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        """Nudge every parameter opposite its gradient."""
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p -= self.lr * p.grad


def train(
    model: CharLM,
    words: list[str],
    opt: SGD,
    *,
    steps: int = 200,
    batch_size: int | None = None,
    lr_schedule: Callable[[int], float] | None = None,
    seed: int = 1337,
) -> list[float]:
    """Train a model on a corpus, returning the per-step loss history.

    A `batch_size` of None trains against the full dataset at each step.
    `lr_schedule` maps a step index to a learning rate; when omitted, the
    optimizer's learning rate is left alone.
    """
    X, Y = make_dataset(words, model.vocab, model.block_size)
    g = torch.Generator().manual_seed(seed)

    history: list[float] = []
    for step in range(steps):
        if lr_schedule is not None:
            opt.lr = lr_schedule(step)

        if batch_size is not None:
            ix = torch.randint(0, X.shape[0], (batch_size,), generator=g)
            xb, yb = X[ix], Y[ix]
        else:
            xb, yb = X, Y

        # forward
        loss = model.loss(xb, yb)

        # backward
        opt.zero_grad()
        loss.backward()

        # update
        opt.step()

        history.append(loss.item())

    return history
