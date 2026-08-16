"""
Vocabulary and token/index mappings shared by all character-level models.
"""

from collections.abc import Iterable, Sequence

# the special token that marks the start and end of a word
TOKEN_BOUNDARY = "."


class Vocab:
    """A bidirectional mapping between characters and integer indices."""

    def __init__(self, chars: Sequence[str]) -> None:
        if TOKEN_BOUNDARY in chars:
            raise ValueError(
                f"boundary token {TOKEN_BOUNDARY!r} is implicit; it should "
                "not appear in the character set"
            )

        # index-to-string
        self.itos: list[str] = [TOKEN_BOUNDARY, *sorted(chars)]
        # string-to-index
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def from_words(cls, words: Iterable[str]) -> "Vocab":
        """Derive a vocabulary from the characters present in a corpus."""
        return cls(sorted(set("".join(words))))

    def __len__(self) -> int:
        return len(self.itos)

    def __repr__(self) -> str:
        return f"Vocab(size={len(self)})"

    def encode(self, s: str) -> list[int]:
        """Map a string to a list of token indices."""
        return [self.stoi[c] for c in s]

    def decode(self, indices: Iterable[int]) -> str:
        """Map a sequence of token indices back to a string."""
        return "".join(self.itos[i] for i in indices)
