"""LLM provider abstraction (prompting with citations + streaming).

LLMProvider interface: generate(prompt, options) and generate_stream(prompt, options).
Implementations:
- StubProvider  — deterministic, for tests and offline demos
- CachedProvider — wraps any provider with an in-memory answer cache
"""
from __future__ import annotations

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
    """Deterministic provider for tests and offline demos.

    Reads the context out of the prompt and produces a short answer that cites
    the first retrieved chunk, so the citation-highlighting UI has something real
    to point at.
    """

    def _compose(self, prompt: str, options: dict | None) -> str:
        top_chunk = (options or {}).get("top_chunk_id", "")
        citations = (options or {}).get("chunk_ids", [])
        cite = ", ".join(f"[{c}]" for c in (citations[:2] or ([top_chunk] if top_chunk else [])))
        answer = (
            "Based on the retrieved passages, the most relevant context is in "
            f"the cited chunk(s){(' ' + cite) if cite else ''}. "
            "This is a stub answer — plug in a real LLM provider to generate a grounded response."
        )
        return answer

    def generate(self, prompt: str, options: dict | None = None) -> dict:
        t0 = time.time()
        text = self._compose(prompt, options)
        return {
            "text": text,
            "citations": (options or {}).get("chunk_ids", [])[:2],
            "latency_ms": int((time.time() - t0) * 1000),
            "provider": "stub",
        }

    def generate_stream(self, prompt: str, options: dict | None = None) -> Iterator[str]:
        import re
        text = self._compose(prompt, options)
        # Split on word boundaries but keep punctuation attached
        for token in re.findall(r"\S+\s*", text):
            yield token
            time.sleep(0.02)  # simulate token latency so the UI streams visibly


class CachedProvider(LLMProvider):
    """Wraps another provider with an in-memory answer cache keyed on prompt hash."""

    def __init__(self, inner: LLMProvider, ttl_seconds: int = 3600):
        self.inner = inner
        self.ttl = ttl_seconds
        self.cache: dict = {}

    def _key(self, prompt: str) -> str:
        import hashlib
        return hashlib.sha256(prompt.encode()).hexdigest()

    def generate(self, prompt: str, options: dict | None = None) -> dict:
        key = self._key(prompt)
        now = time.time()
        if key in self.cache and now - self.cache[key]["ts"] < self.ttl:
            return {**self.cache[key]["data"], "cached": True}
        data = self.inner.generate(prompt, options)
        self.cache[key] = {"ts": now, "data": data}
        return data


def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """Construct the RAG prompt with citation markers.

    Each chunk: {"chunk_id": ..., "text": ...}
    """
    context = "\n\n".join(f"[{c['chunk_id']}]: {c['text'][:300]}" for c in chunks)
    return f"""You are answering based ONLY on the context below.

Context:
{context}

Question: {query}

Answer: (cite relevant chunk IDs in brackets like [doc_0::chunk1])"""


def extract_citations(text: str, known_ids: list[str]) -> list[str]:
    """Pull cited chunk IDs from answer text, preserving first-seen order."""
    import re
    bracketed = re.findall(r"\[[\w\-:.@]+\]", text)
    out: list[str] = []
    for b in bracketed:
        cid = b.strip("[]")
        if cid in known_ids and cid not in out:
            out.append(cid)
    return out
