"""
Construction of (context, target) training examples from a corpus.
"""

import random

import torch

from .vocab import TOKEN_BOUNDARY, Vocab


def make_dataset(
    words: list[str], vocab: Vocab, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the (X, Y) training tensors for a corpus.
    :return: X with shape (N, block_size) and Y with shape (N,), both of dtype int64.
    """
    xs: list[list[int]] = []
    ys: list[int] = []

    for word in words:
        context = [0] * block_size
        for ch in list(word) + [TOKEN_BOUNDARY]:
            ix = vocab.stoi[ch]
            xs.append(context)
            ys.append(ix)
            # roll the context forward by one token
            context = context[1:] + [ix]

    return torch.tensor(xs), torch.tensor(ys)


def split(
    words: list[str],
    *,
    seed: int = 1337,
    fractions: tuple[float, float] = (0.8, 0.1),
) -> tuple[list[str], list[str], list[str]]:
    """
    Shuffle a corpus and partition it into train / dev / test splits.

    `fractions` gives the proportion of the corpus allocated to the train
    and dev splits; the remainder becomes the test split.
    """
    f_train, f_dev = fractions
    if f_train + f_dev >= 1.0:
        raise ValueError("train and dev fractions must leave a remainder")

    shuffled = list(words)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(f_train * n)
    n_dev = n_train + int(f_dev * n)

    return shuffled[:n_train], shuffled[n_train:n_dev], shuffled[n_dev:]
