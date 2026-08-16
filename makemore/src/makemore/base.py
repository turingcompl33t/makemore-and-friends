"""
The interface implemented by all gradient-trained character-level models.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from .data import make_dataset
from .vocab import Vocab


class CharLM(ABC):
    """A character-level language model."""

    def __init__(self, vocab: Vocab, block_size: int) -> None:
        self.vocab = vocab
        self.block_size = block_size

    @abstractmethod
    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """
        Compute next-token logits for a batch of contexts.
        :param ctx: an int64 tensor of shape (B, block_size)
        :return: afloat tensor of shape (B, V).
        """

    def __call__(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.forward(ctx)

    def parameters(self) -> list[torch.Tensor]:
        """All trainable tensors held by the model."""
        found: list[torch.Tensor] = []
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                found.append(value)
            elif isinstance(value, (list, tuple)):
                found.extend(v for v in value if isinstance(v, torch.Tensor))
        return [p for p in found if p.requires_grad]

    @property
    def num_parameters(self) -> int:
        """The total number of scalar parameters in the model."""
        return sum(p.nelement() for p in self.parameters())

    def loss(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """The mean negative log-likelihood of targets Y given contexts X."""
        logits = self.forward(X)
        return F.cross_entropy(logits.view(-1, len(self.vocab)), Y.view(-1))

    @torch.no_grad()
    def evaluate(self, words: list[str]) -> float:
        """Compute the model's loss over a corpus, as a reported metric."""
        X, Y = make_dataset(words, self.vocab, self.block_size)
        return self.loss(X, Y).item()

    @torch.no_grad()
    def sample(
        self, n: int, *, seed: int = 1337, max_len: int = 64
    ) -> list[str]:
        """Sample n words from the model."""
        g = torch.Generator().manual_seed(seed)
        return [self._sample_one(g, max_len) for _ in range(n)]

    def _sample_one(self, g: torch.Generator, max_len: int = 32) -> str:
        """Sample a single word from the model."""
        context = [0] * self.block_size
        out: list[int] = []

        # max_len guards against a model (an untrained one, especially)
        # that is slow to emit the boundary token
        for _ in range(max_len):
            probs = F.softmax(self.forward(torch.tensor([context])), dim=1)
            ix = int(torch.multinomial(probs, 1, generator=g).item())

            # index 0 is the boundary token, which terminates the word
            if ix == 0:
                break

            out.append(ix)
            context = context[1:] + [ix]

        return self.vocab.decode(out)
