"""Character-level sliding-window chunking for RAG documents.

Zero dependencies. Splitting on characters (not tokens) keeps this independent of
any tokenizer and works identically across embedding backends.
"""

from __future__ import annotations


def chunk_text(text: str, max_chars: int = 512, overlap: int = 64) -> list[str]:
    """Split ``text`` into overlapping windows of at most ``max_chars`` characters.

    Consecutive windows share ``overlap`` characters so context spanning a chunk
    boundary is not lost. Always returns at least one chunk for non-empty input;
    returns an empty list for empty/whitespace-only input.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be in [0, max_chars)")

    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    step = max_chars - overlap
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        chunk = text[start : start + max_chars].strip()
        if chunk:
            chunks.append(chunk)
        if start + max_chars >= n:
            break
        start += step
    return chunks
