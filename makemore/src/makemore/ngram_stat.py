"""
Character N-gram model based on statistical modeling.
"""

import itertools
import string
from typing import Generator

import torch

# the special token for start and stop
_TOKEN_DOT = "."


class StatisticalNGram:
    def __init__(self, n: int, *, smoothing: int = 0) -> None:
        # the size the ngrams
        self.n = n
        # the amount of smoothing to apply (fake counts)
        self.smoothing = smoothing

        # the alphabet -> index
        self.alphabet: dict[str, int] = {}
        self.ralphabet: dict[int, str] = {}

        # string-to-index
        self.stoi: dict[str, int] = {}
        # index-to-string
        self.itos: dict[int, str] = {}

        # the model (conditioning x prediction matrix)
        self.P: torch.Tensor | None = None

        self.N: torch.Tensor | None = None

    def train(self, words: list[str]) -> None:
        """Train the model on the provided set of words."""

        # initialize lookup tables
        self._build_lut()

        # initialize counts to 0
        N = torch.zeros((len(self.stoi), len(self.alphabet)), dtype=torch.int32)

        for c, p in self._extract_ngrams(words):
            il = self.stoi[c]
            ir = self.alphabet[p]
            N[il, ir] += 1

        self.N = N

        # normalize counts to probability distribution
        P = (N + self.smoothing).float()
        P /= P.sum(1, keepdim=True)
        self.P = P

    def sample(self, n: int, seed: int = 1337):
        """Sample n words from the model."""
        g = torch.Generator().manual_seed(seed)
        return [self._sample_one(g) for _ in range(n)]

    def _sample_one(self, g: torch.Generator) -> str:
        """Sample a single word from the model."""

        # compute the index of the start token
        rix = self.stoi["." * (self.n - 1)]
        # the word we grow
        word = ""
        while True:
            # sample an index from the distribution
            cix: int = torch.multinomial(
                self.P[rix, :].float(), num_samples=1, generator=g  # type: ignore
            ).item()

            # check if this is the stop token; stop token is added as final character of alphabet
            if cix == (len(self.alphabet) - 1):
                return word

            # add the character to the growing word
            word += self.ralphabet[cix]  # type: ignore

            # compute the new row index for the next iteration
            chars_prv = self.itos[rix][1:] if self.n > 2 else ""
            chars_new = self.ralphabet[cix]
            rix = self.stoi[chars_prv + chars_new]

    def loss(self, words: list[str]) -> float:
        """Compute loss with respect to the provided examples."""
        if self.P is None:
            raise RuntimeError("cannot compute loss for untrained model")

        # the number of ngrams evaluated
        n = 0

        log_likelihood = 0.0

        for c, p in self._extract_ngrams(words):
            ix0, ix1 = self.stoi[c], self.alphabet[p]
            log_likelihood += torch.log(self.P[ix0, ix1]).item()
            n += 1

        # invert to get negative log-likelihood
        nll = -log_likelihood
        # compute mean of nll
        return nll / n

    def _build_lut(self) -> None:
        """Construct internal lookup tables."""
        # identify all unique characters
        chars = list(string.ascii_lowercase) + ["."]
        # compute the alphabet size (+1 for special token)
        self.alphabet = {c: i for i, c in enumerate(chars)}
        self.ralphabet = {i: c for c, i in self.alphabet.items()}

        # compute string to index
        self.stoi = {
            "".join(c): i
            for i, c in enumerate(
                itertools.product(*(chars for _ in range(self.n - 1)))
            )
        }

        # compute index to string
        self.itos = {i: c for c, i in self.stoi.items()}

        assert len(self.stoi) == len(self.alphabet) ** (
            self.n - 1
        ), "broken invariant"

    def _extract_ngrams(
        self, words: list[str]
    ) -> Generator[tuple[str, str], None, None]:
        """Extract ngrams from a sequence of words."""
        yield from _extract_ngrams(self.n, words)


def _extract_ngrams(
    n: int, words: list[str]
) -> Generator[tuple[str, str], None, None]:
    """Extract ngrams from the provided sequence of words."""
    for word in words:
        yield from _extract_ngrams_one(n, word)


def _extract_ngrams_one(
    n: int, word: str
) -> Generator[tuple[str, str], None, None]:
    """Extract ngrams from a single word."""
    chs = [_TOKEN_DOT] * (n - 1) + list(word) + [_TOKEN_DOT] * (n - 1)
    for ngram in zip(*(chs[i:] for i in range(n))):
        yield "".join(ngram[:-1]), ngram[-1]
