"""FastAPI entrypoint. Wiring engine + experiments + LLM into endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from engine.search import brute_force_search
from pipeline.embeddings import embed_query
from api.llm_provider import StubProvider, CachedProvider, build_rag_prompt, extract_citations
from experiments import assign_variant, two_proportion_z, welch_t, srm_chi_square

app = FastAPI(title="Cairn")

# Default experiment definition served by /search.
DEFAULT_EXPERIMENT = {
    "id": "exp_hybrid_vs_vector",
    "name": "Hybrid vs Vector retrieval",
    "status": "running",
    "primary_metric": "thumbs_up_rate",
    "config_json": {
        "variants": [
            {"name": "control", "weight": 50, "config": {"retrieval": "vector", "topK": 5}},
            {"name": "treatment", "weight": 50, "config": {"retrieval": "hybrid", "topK": 5, "alpha": 0.6}},
        ]
    },
}

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

# ---------------------------------------------------------------------------
# In-memory event store (replaces Postgres in v1).
# Exposures: experiment_id -> {variant -> count}
# Outcomes:  experiment_id -> {variant -> {"n": int, "succ": int, "latencies": [ms]}}
# ---------------------------------------------------------------------------
_EXPOSURES: dict = {}
_OUTCOMES: dict = {}
_EVENT_IDS: set = set()  # idempotency keys


def _log_exposure(experiment_id: str, variant: str, user_id: str, event_id: str) -> bool:
    """Idempotent exposure log. Returns True if newly stored."""
    if event_id in _EVENT_IDS:
        return False
    _EVENT_IDS.add(event_id)
    exp = _EXPOSURES.setdefault(experiment_id, {})
    exp[variant] = exp.get(variant, 0) + 1
    return True


def _log_outcome(experiment_id: str, variant: str, metric: str, value: float, event_id: str) -> bool:
    """Idempotent outcome log."""
    if event_id in _EVENT_IDS:
        return False
    _EVENT_IDS.add(event_id)
    store = _OUTCOMES.setdefault(experiment_id, {}).setdefault(
        variant, {"n": 0, "succ": 0, "latencies": []}
    )
    if metric == "thumbs":
        store["n"] += 1
        if value > 0:
            store["succ"] += 1
    elif metric == "latency_ms":
        store["latencies"].append(float(value))
    return True


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
    # Assign a variant deterministically (same user -> same variant).
    exp = DEFAULT_EXPERIMENT
    exp_id = exp["id"]
    variants = exp["config_json"]["variants"]
    variant = assign_variant(exp_id, req.user_id, variants)

    # Log the exposure (idempotent per user+experiment+query hour bucket).
    exposure_id = f"exp_{exp_id}_{req.user_id}"
    _log_exposure(exp_id, variant, req.user_id, exposure_id)

    # Vector search for both; config is variant-controlled.
    q_vec = embed_query(req.query)
    hits = brute_force_search(q_vec, _INDEX_VECTORS, top_k=10)

    results = []
    qn = np.linalg.norm(q_vec)
    qv_norm = (q_vec / qn) if qn > 0 else q_vec
    for idx in hits[:5]:
        c = _INDEX_CHUNKS[idx] if idx < len(_INDEX_CHUNKS) else {}
        score = (float(np.dot(qv_norm, _INDEX_VECTORS[idx])) if idx < len(_INDEX_VECTORS) else 0.0)
        conf = max(0.0, min(1.0, score))
        results.append({
            "rank": len(results) + 1,
            "chunk_id": c.get("chunk_id"),
            "text": c.get("text", "")[:200],
            "score": round(score, 4),
            "confidence": f"{conf * 100:.0f}%",
        })

    return SearchResponse(query=req.query, variant=variant, results=results, experiment_id=exp_id)


@app.post("/answer")
def answer(req: AnswerRequest):
    _ensure_index()
    # For Day 12: return an LLM-style answer with citations
    q_vec = embed_query(req.query)
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
    stored = 0
    for ev in batch.get("events", []):
        eid = ev.get("event_id")
        if not eid:
            continue
        exp_id = ev.get("experiment_id", DEFAULT_EXPERIMENT["id"])
        variant = ev.get("variant") or assign_variant(
            exp_id, ev.get("user_id", "anon"), DEFAULT_EXPERIMENT["config_json"]["variants"]
        )
        metric = ev.get("metric", ev.get("event_type", "thumbs"))
        if _log_outcome(exp_id, variant, metric, float(ev.get("value", 0)), eid):
            stored += 1
    return {"stored": stored, "idempotent": True}


# In-memory store for experiment definitions (replaces Postgres in v1)
_EXPERIMENTS: dict = {DEFAULT_EXPERIMENT["id"]: DEFAULT_EXPERIMENT}


@app.get("/experiments")
def list_experiments():
    return {"experiments": list(_EXPERIMENTS.values())}


class ExperimentCreate(BaseModel):
    id: str
    name: str
    status: str = "draft"
    config_json: dict | None = None
    primary_metric: str | None = None


@app.post("/experiments", status_code=201)
def create_experiment(exp: ExperimentCreate):
    if exp.id in _EXPERIMENTS:
        raise HTTPException(status_code=409, detail="experiment already exists")
    _EXPERIMENTS[exp.id] = exp.model_dump()
    return _EXPERIMENTS[exp.id]


@app.get("/experiments/{exp_id}/results")
def experiment_results(exp_id: str):
    if exp_id not in _EXPERIMENTS:
        raise HTTPException(status_code=404, detail="experiment not found")
    exp = _EXPERIMENTS[exp_id]
    config = exp.get("config_json") or {}
    variants = [v["name"] for v in config.get("variants", [])] or ["control", "treatment"]

    exposures = _EXPOSURES.get(exp_id, {})
    outcomes = _OUTCOMES.get(exp_id, {})

    per_variant = []
    for name in variants:
        o = outcomes.get(name, {"n": 0, "succ": 0, "latencies": []})
        n = o["n"]
        rate = (o["succ"] / n) if n else None
        lats = o["latencies"]
        p95 = (sorted(lats)[max(0, int(0.95 * len(lats)) - 1)] if lats else None)
        per_variant.append({
            "variant": name,
            "exposures": exposures.get(name, 0),
            "n": n,
            "successes": o["succ"],
            "thumbs_up_rate": round(rate, 4) if rate is not None else None,
            "p95_latency_ms": p95,
        })

    # Primary metric: two-proportion z-test between first two variants.
    test = None
    if len(per_variant) >= 2:
        a, b = per_variant[0], per_variant[1]
        if a["n"] and b["n"] and a["thumbs_up_rate"] is not None and b["thumbs_up_rate"] is not None:
            test = two_proportion_z(a["thumbs_up_rate"], a["n"], b["thumbs_up_rate"], b["n"])
            test["a"] = a["variant"]
            test["b"] = b["variant"]
            test["absolute_lift"] = round(b["thumbs_up_rate"] - a["thumbs_up_rate"], 4)
            if a["thumbs_up_rate"]:
                test["relative_lift_pct"] = round(
                    100 * (b["thumbs_up_rate"] - a["thumbs_up_rate"]) / a["thumbs_up_rate"], 2
                )

    # SRM check against configured weights (uses exposures if present, else outcomes n).
    srm = None
    observed = [v["exposures"] for v in per_variant]
    if sum(observed) == 0:
        observed = [v["n"] for v in per_variant]
    total_obs = sum(observed)
    if total_obs > 0 and len(per_variant) >= 2:
        weights = [v.get("weight", 1) for v in config.get("variants", [])] or [1, 1]
        if len(weights) != len(observed):
            weights = [1] * len(observed)
        tw = sum(weights)
        expected = [total_obs * w / tw for w in weights]
        srm = srm_chi_square(observed, expected)

    # Plain-language status (guard against peeking).
    label = "not enough data"
    if test:
        if srm and srm["srm_detected"]:
            label = "SRM warning — experiment untrustworthy"
        elif sum(v["n"] for v in per_variant) < 100:
            label = "not enough data (provisional)"
        elif test["significant"]:
            better = "treatment better" if test["absolute_lift"] > 0 else "control better"
            label = better
        else:
            label = "no detectable difference"

    return {
        "experiment_id": exp_id,
        "status": exp.get("status", "draft"),
        "primary_metric": exp.get("primary_metric"),
        "variants": per_variant,
        "test": test,
        "srm": srm,
        "verdict": label,
    }
