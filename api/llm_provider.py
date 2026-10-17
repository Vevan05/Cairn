"""LLM provider abstraction (RAG prompting with citations + streaming).

`LLMProvider` is a two-method interface — `generate(prompt, options)` and
`generate_stream(prompt, options)` — with a single implementation,
`StubProvider`, that composes a short answer from the retrieved chunks.

The app ships with the stub deliberately: it keeps the demo and the test suite
fully offline, reproducible, and free of API keys. Swapping in a hosted model is
a drop-in change — implement `generate` (and optionally `generate_stream`) and
return your provider from `api/main.py`; nothing else in the app changes.
"""
from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, options: dict | None = None) -> dict:
        """Generate text given prompt. Returns dict with text + citations + latency."""
        ...

    def generate_stream(self, prompt: str, options: dict | None = None) -> Iterator[str]:
        """Yield answer tokens incrementally. Default: non-streaming fallback."""
        text = self.generate(prompt, options)["text"]
        for token in text.split(" "):
            yield token + " "


class StubProvider(LLMProvider):
    """Deterministic provider used by the app.

    Reads the retrieved chunk ids out of `options` and produces a short answer
    that cites them, so the citation-highlighting UI has something real to point
    at. No network, no API key, fully reproducible.
    """

    def _compose(self, options: dict | None) -> str:
        top_chunk = (options or {}).get("top_chunk_id", "")
        citations = (options or {}).get("chunk_ids", [])
        cite = ", ".join(f"[{c}]" for c in (citations[:2] or ([top_chunk] if top_chunk else [])))
        return (
            "Based on the retrieved passages, the most relevant context is in "
            f"the cited chunk(s){(' ' + cite) if cite else ''}. "
            "This is a deterministic stub answer: retrieval, ranking and citation "
            "wiring are real, but no language model is called."
        )

    def generate(self, prompt: str, options: dict | None = None) -> dict:
        t0 = time.time()
        text = self._compose(options)
        return {
            "text": text,
            "citations": (options or {}).get("chunk_ids", [])[:2],
            "latency_ms": int((time.time() - t0) * 1000),
            "provider": "stub",
        }

    def generate_stream(self, prompt: str, options: dict | None = None) -> Iterator[str]:
        text = self._compose(options)
        # Split on word boundaries but keep punctuation attached, with a small
        # delay so the streaming UI is visibly incremental.
        for token in re.findall(r"\S+\s*", text):
            yield token
            time.sleep(0.02)


def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """Construct the RAG prompt with citation markers.

    Each chunk: {"chunk_id": ..., "text": ...}
    """
    context = "\n\n".join(f"[{c['chunk_id']}]: {c['text'][:400]}" for c in chunks)
    return f"""Answer the question using ONLY the context below.
Cite the chunk IDs you rely on in square brackets, exactly as shown, e.g. [doc_9::chunk0].
If the context does not contain the answer, say so briefly.

Context:
{context}

Question: {query}

Answer:"""


def extract_citations(text: str, known_ids: list[str]) -> list[str]:
    """Pull cited chunk IDs from answer text, preserving first-seen order."""
    bracketed = re.findall(r"\[[\w\-:.@]+\]", text)
    out: list[str] = []
    for b in bracketed:
        cid = b.strip("[]")
        if cid in known_ids and cid not in out:
            out.append(cid)
    return out
