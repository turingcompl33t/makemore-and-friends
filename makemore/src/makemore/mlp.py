"""
A character bigram language model implemented via MLP (Bengio et al. 2003)
"""

import torch

from .base import CharLM
from .vocab import Vocab


class MLP(CharLM):
    """A simple MLP character-level language model."""

    def __init__(
        self,
        vocab: Vocab,
        *,
        embedding_dimension: int = 2,
        hidden_layer_size: int = 100,
        block_size: int = 3,
        seed: int = 1337,
    ) -> None:
        super().__init__(vocab, block_size=block_size)

        # the character embedding dimension
        self.embedding_dimension = embedding_dimension
        # the number of neurons in the hidden layer
        self.hidden_layer_size = hidden_layer_size

        g = torch.Generator().manual_seed(seed)

        # embedding layer
        self.C = torch.randn(
            (len(vocab), self.embedding_dimension),
            generator=g,
            requires_grad=True,
        )
        # hidden layer
        self.W1 = torch.randn(
            (
                self.block_size * self.embedding_dimension,
                self.hidden_layer_size,
            ),
            generator=g,
            requires_grad=True,
        )
        self.b1 = torch.randn(
            self.hidden_layer_size, generator=g, requires_grad=True
        )
        # output layer
        self.W2 = torch.randn(
            (self.hidden_layer_size, len(vocab)),
            generator=g,
            requires_grad=True,
        )
        self.b2 = torch.randn(len(vocab), generator=g, requires_grad=True)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        emb = self.C[ctx]
        # hidden layer activations
        h = torch.tanh(
            emb.view((-1, self.block_size * self.embedding_dimension)) @ self.W1
            + self.b1
        )
        # logits (activations from output layer)
        logits = h @ self.W2 + self.b2
        return logits
