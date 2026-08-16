"""
Unit tests for vocabulary helpers.
"""

from makemore.vocab import Vocab


def test_len() -> None:
    """Test expected vocab size."""
    v = Vocab(["a", "b", "c"])
    assert len(v) == 4


def test_encode_decode() -> None:
    """Encoding and decoding works as expected."""
    v = Vocab(["a", "b", "c"])
    e = v.encode("abba")
    assert v.decode(e) == "abba"
