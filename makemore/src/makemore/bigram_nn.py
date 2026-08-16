"""
A character bigram language model implemented as a single linear layer.
"""

import torch
import torch.nn.functional as F

from .base import CharLM
from .vocab import Vocab


class BigramNN(CharLM):
    """A bigram model consisting of a single (V, V) linear layer."""

    def __init__(self, vocab: Vocab, *, seed: int = 1337) -> None:
        super().__init__(vocab, block_size=1)

        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(
            (len(vocab), len(vocab)), generator=g, requires_grad=True
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        # (B, 1) -> (B, V) one-hot -> (B, V) logits
        xenc = F.one_hot(ctx[:, 0], num_classes=len(self.vocab)).float()
        return xenc @ self.W
