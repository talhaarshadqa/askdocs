"""Split long text into overlapping chunks for retrieval."""
from __future__ import annotations
import re

_WORD = re.compile(r"\S+")


def chunk_text(text: str, *, size: int = 120, overlap: int = 30) -> list[str]:
    """Split text into ~`size`-word chunks that overlap by `overlap` words.

    Overlap keeps sentences that straddle a boundary retrievable from both sides.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = _WORD.findall(text)
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks
