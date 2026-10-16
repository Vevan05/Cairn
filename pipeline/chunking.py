"""Fixed-size word chunker with overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_word: int

    def to_dict(self) -> dict:
        return asdict(self)


_WS = re.compile(r"\s+")


def chunk_text(doc_id: str, text: str, size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Split text into word windows of `size` words with `overlap` words of context."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("require size > 0 and 0 <= overlap < size")
    words = _WS.split(text.strip())
    if not words or words == [""]:
        return []
    chunks: list[Chunk] = []
    step = size - overlap
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::chunk{i}",
                doc_id=doc_id,
                text=" ".join(window),
                start_word=start,
            )
        )
        if start + size >= len(words):
            break
    return chunks
