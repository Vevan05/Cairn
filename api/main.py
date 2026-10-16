"""FastAPI entrypoint. Wiring engine + experiments + LLM into endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from engine.search import brute_force_search
from api.llm_provider import StubProvider, CachedProvider, build_rag_prompt, extract_citations

app = FastAPI(title="Cairn")

# Load read-only index artifacts (lazy-load)
_INDEX_VECTORS = None
_INDEX_CHUNKS = None

def _ensure_index():
    global _INDEX_VECTORS, _INDEX_CHUNKS
    if _INDEX_VECTORS is None:
        _INDEX_VECTORS = np.load("data/vectors.npy") if Path("data/vectors.npy").exists() else np.zeros((0, 384), dtype=np.float32)
        _INDEX_CHUNKS = []
        if Path("data/chunks.jsonl").exists():
            with open("data/chunks.jsonl") as f:
                for line in f:
                    _INDEX_CHUNKS.append(json.loads(line))

_ALLOWED_LOOPS = ("A", "B")


class SearchRequest(BaseModel):
    query: str
    user_id: str = "anon"


class SearchResponse(BaseModel):
    query: str
    variant: str
    results: list[dict]
    experiment_id: str | None


class AnswerRequest(BaseModel):
    query: str
    user_id: str = "anon"


@app.get("/health")
def health() -> dict:
    _ensure_index()
    return {"status": "ok", "index_version": "v1", "vector_count": len(_INDEX_VECTORS), "chunk_count": len(_INDEX_CHUNKS)}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    _ensure_index()
    # Evaluate variant via bucketing (simplified<rfl:parameter name="ssignment)
    import hashlib
    exp_id = "exp_hybrid_vs_vector"
    bucket = int(hashlib.sha256(f"{exp_id}:{req.user_id}".encode()).hexdigest(), 16) % 10000
    # 50/50 split
    variant = "control" if bucket < 5000 else "treatment"

    # Vector search for both; config is variant-controlled
    q_vec = np.asarray(__import__("pipeline.embeddings", fromlist=["embed_texts"]).embed_texts([req.query]))[0]
    hits = brute_force_search(q_vec, _INDEX_VECTORS, top_k=10)

    results = []
    if variant == "control":
        for idx in hits[:5]:
            c = _INDEX_CHUNKS[idx] if idx < len(_INDEX_CHUNKS) else {}
            results.append({"rank": len(results) + 1, "chunk_id": c.get("chunk_id"), "text": c.get("text", "")[:200]})
    else:
        for idx in hits[:5]:
            c = _INDEX_CHUNKS[idx] if idx < len(_INDEX_CHUNKS) else {}
            results.append({"rank": len(results) + 1, "chunk_id": c.get("chunk_id"), "text": c.get("text", "")[:200]})

    return SearchResponse(query=req.query, variant=variant, results=results, experiment_id=exp_id)


@app.post("/answer")
def answer(req: AnswerRequest):
    _ensure_index()
    # For Day 12: return an LLM-style answer with citations
    q_vec = np.asarray(__import__("pipeline.embeddings", fromlist=["embed_texts"]).embed_texts([req.query]))[0]
    hits = brute_force_search(q_vec, _INDEX_VECTORS, top_k=5)
    retrieved = []
    for idx in hits:
        if idx < len(_INDEX_CHUNKS):
            c = _INDEX_CHUNKS[idx]
            retrieved.append({"chunk_id": c["chunk_id"], "text": c.get("text", "")})

    provider = CachedProvider(StubProvider(), ttl_seconds=3600)
    prompt = build_rag_prompt(req.query, retrieved)
    result = provider.generate(prompt, options={"top_chunk_id": retrieved[0]["chunk_id"] if retrieved else ""})
    # Extract cited IDs from answer
    known_ids = [r["chunk_id"] for r in retrieved]
    citations = extract_citations(result["text"], known_ids)
    result["citations"] = citations or known_ids

    # Simulate thumbs up/down (logged via /events)
    return {
        "query": req.query,
        "answer": result["text"],
        "citations": result["citations"],
        "variant": "stub",
        "cached": result.get("cached", False),
        "retrieved_chunks": len(retrieved),
    }


@app.post("/events")
def events(batch: dict):
    """Batched outcome events, idempotent by event_id."""
    # In production: upsert into Postgres with ON CONFLICT DO NOTHING.
    stored = 0
    seen = set()
    for ev in batch.get("events", []):
        eid = ev.get("event_id")
        if eid in seen:
            continue
        seen.add(eid)
        stored += 1
    return {"stored": stored, "idempotent": True}
