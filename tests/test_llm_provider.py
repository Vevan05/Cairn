"""Tests for the answer provider (stub) and prompt/citation helpers."""
from __future__ import annotations

from api.llm_provider import (
    LLMProvider,
    StubProvider,
    build_rag_prompt,
    extract_citations,
)


def test_stub_generate_cites_retrieved_chunks():
    out = StubProvider().generate("prompt", {"chunk_ids": ["doc1::chunk0", "doc2::chunk0"], "top_chunk_id": "doc1::chunk0"})
    assert out["provider"] == "stub"
    assert "doc1::chunk0" in out["text"]
    assert out["citations"] == ["doc1::chunk0", "doc2::chunk0"]
    assert isinstance(out["latency_ms"], int)


def test_stub_generate_without_chunks_does_not_crash():
    out = StubProvider().generate("prompt", {"chunk_ids": []})
    assert "stub answer" in out["text"]
    assert out["citations"] == []


def test_stub_stream_reassembles_to_the_full_answer():
    options = {"chunk_ids": ["doc1::chunk0"]}
    expected = StubProvider().generate("prompt", options)["text"]
    streamed = "".join(StubProvider().generate_stream("prompt", options))
    assert streamed == expected


def test_llmprovider_default_stream_falls_back_to_generate():
    class Echo(LLMProvider):
        def generate(self, prompt, options=None):
            return {"text": "a b c", "citations": [], "latency_ms": 0, "provider": "echo"}

    assert "".join(Echo().generate_stream("p")) == "a b c "


def test_build_rag_prompt_includes_context_and_citation_hint():
    prompt = build_rag_prompt("What is X?", [{"chunk_id": "doc9::chunk0", "text": "X is a thing."}])
    assert "[doc9::chunk0]: X is a thing." in prompt
    assert "What is X?" in prompt
    assert "cite the chunk ids" in prompt.lower()


def test_extract_citations_orders_and_filters():
    text = "See [doc2::chunk0] and [unknown] then [doc1::chunk1] then [doc2::chunk0]."
    assert extract_citations(text, ["doc1::chunk1", "doc2::chunk0"]) == ["doc2::chunk0", "doc1::chunk1"]
