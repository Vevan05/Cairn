"""LLM provider abstraction (Day 12: prompting with citations).

LLMProvider interface generate(prompt, options). Implementations:
- StubProvider for tests
- FreeTierProvider (swappable, duck-typed over any free API)
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, options: dict | None = None) -> dict:
        """Generate text given prompt. Returns dict with text, citations (chunk_ids), latency."""
        ...


class StubProvider(LLMProvider):
    """Stub for unit tests: returns canned answer citing first retrieved chunk."""
    def generate(self, prompt: str, options: dict | None = None) -> dict:
        top_chunk = (options or {}).get("top_chunk_id", "")
        ans = "Stubbed answer citing the provided chunk."
        return {"text": ans, "citations": [top_chunk], "latency_ms": 0, "provider": "stub"}


class CachedProvider(LLMProvider):
    """Wraps another provider with an in-memory answer cache keyed on query hash.

    Survives rate limits during demos per Plan.md §3.3.
    """
    def __init__(self, inner: LLMProvider, ttl_seconds: int = 3600):
        self.inner = inner
        self.ttl = ttl_seconds
        self.cache = {}

    def generate(self, prompt: str, options: dict | None = None) -> dict:
        import hashlib
        key = hashlib.sha256(prompt.encode()).hexdigest()
        now = time.time()
        if key in self.cache:
            entry = self.cache[key]
            if now - entry["ts"] < self.ttl:
                return {**entry["data"], "cached": True}
        data = self.inner.generate(prompt, options)
        self.cache[key] = {"ts": now, "data": data}
        return data


def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """Construct the RAG prompt with citations.

    Each chunk in the list has {"chunk_id": ..., "text": ...}
    """
    context = "\n\n".join(f"[{c['chunk_id']}]: {c['text'][:300]}" for c in chunks)
    return f"""You are answering based ONLY on the context below.

Context:
{context}

Question: {query}

Answer: (cite relevant chunk IDs in brackets like [doc_0::chunk1])"""


def extract_citations(text: str, known_ids: list[str]) -> list[str]:
    """Pull cited chunk IDs from answer text."""
    import re
    bracketed = re.findall(r"\[[\w\-:]+\]", text)
    out = []
    for b in bracketed:
        cid = b.strip("[]")
        if cid in known_ids and cid not in out:
            out.append(cid)
    return out
