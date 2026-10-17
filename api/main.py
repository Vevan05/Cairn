"""FastAPI entrypoint. Wiring engine + experiments + eval + LLM into endpoints."""
from __future__ import annotations

import json
import os
import random
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from engine.search import brute_force_search, BM25, rrf_fuse
from pipeline.embeddings import embed_query
from api.llm_provider import StubProvider, build_rag_prompt, extract_citations
from experiments import assign_variant, two_proportion_z, welch_t, srm_chi_square
from eval import EvalConfig, run_eval


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the index and seed demo traffic at startup.

    Loading here (rather than lazily on first request) means a cold start is
    deterministic and a missing/broken artifact set fails fast, before the
    platform routes traffic.
    """
    _ensure_index()
    _seed_demo_data()
    yield


app = FastAPI(title="Cairn", lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# The frontend is deployed separately (e.g. Vercel) and calls this API
# cross-origin, so browsers require CORS headers. Set CORS_ORIGINS to a
# comma-separated allow-list in production; the default allows any origin,
# which is fine for a public, credential-less demo API.
_cors_origins = os.getenv("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _cors_origins == "*" else [o.strip() for o in _cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Default experiment definition
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Read-only index artifacts (lazy load)
# ---------------------------------------------------------------------------
_INDEX_VECTORS = None
_INDEX_CHUNKS = None
_BM25 = None


def _ensure_index():
    global _INDEX_VECTORS, _INDEX_CHUNKS, _BM25
    if _INDEX_VECTORS is None:
        _INDEX_VECTORS = (
            np.load("data/vectors.npy")
            if Path("data/vectors.npy").exists()
            else np.zeros((0, 384), dtype=np.float32)
        )
        _INDEX_CHUNKS = []
        if Path("data/chunks.jsonl").exists():
            with open("data/chunks.jsonl") as f:
                for line in f:
                    _INDEX_CHUNKS.append(json.loads(line))
        if _INDEX_CHUNKS:
            _BM25 = BM25([c["text"] for c in _INDEX_CHUNKS])


# ---------------------------------------------------------------------------
# Answer provider
# ---------------------------------------------------------------------------
# The app ships with the deterministic stub: no API key, no network, and the
# test suite stays offline. Swap in a hosted model by implementing LLMProvider
# and returning it here — nothing else changes.
def _provider() -> StubProvider:
    return StubProvider()


# Number of synthetic users seeded into the demo experiment at startup. Set to
# 0 to disable. See _seed_demo_data for why this exists.
DEMO_SEED_USERS = int(os.getenv("DEMO_SEED_USERS", "1200"))


# ---------------------------------------------------------------------------
# In-memory stores (replace Postgres in v1)
# ---------------------------------------------------------------------------
_EXPOSURES: dict = {}   # experiment_id -> {variant -> count}
_OUTCOMES: dict = {}    # experiment_id -> {variant -> {n, succ, latencies}}
_EVENT_IDS: set = set()
_EXPERIMENTS: dict = {DEFAULT_EXPERIMENT["id"]: DEFAULT_EXPERIMENT}
_ACTIVE_EXPERIMENT_ID: str = DEFAULT_EXPERIMENT["id"]
_EVAL_SETS: dict = {}   # id -> {"id", "name", "labeled": [...]}
_EVAL_RUNS: list = []   # history


def _active_experiment() -> dict:
    """The experiment currently serving traffic."""
    return _EXPERIMENTS.get(_ACTIVE_EXPERIMENT_ID, DEFAULT_EXPERIMENT)


def _log_exposure(experiment_id: str, variant: str, user_id: str, event_id: str) -> bool:
    if event_id in _EVENT_IDS:
        return False
    _EVENT_IDS.add(event_id)
    _EXPOSURES.setdefault(experiment_id, {})
    _EXPOSURES[experiment_id][variant] = _EXPOSURES[experiment_id].get(variant, 0) + 1
    return True


def _log_outcome(experiment_id: str, variant: str, metric: str, value: float,
                 event_id: str, user_id: str | None = None) -> bool:
    """Idempotent outcome log.

    For thumbs, the unit of analysis is the *user*: one vote per user per
    variant, so outcomes can never exceed exposures.
    """
    if event_id in _EVENT_IDS:
        return False
    _EVENT_IDS.add(event_id)
    store = _OUTCOMES.setdefault(experiment_id, {}).setdefault(
        variant, {"n": 0, "succ": 0, "latencies": [], "voters": set()}
    )
    if metric == "thumbs":
        key = user_id or event_id
        if key in store["voters"]:
            return False  # this user's vote is already counted
        store["voters"].add(key)
        store["n"] += 1
        if value > 0:
            store["succ"] += 1
    elif metric == "latency_ms":
        store["latencies"].append(float(value))
    return True


# ---------------------------------------------------------------------------
# Demo seeding
# ---------------------------------------------------------------------------
def _seed_demo_data() -> None:
    """Populate the default experiment with simulated traffic at startup.

    Render's free tier has no persistence, so experiment state is lost on every
    restart/sleep. Without this the A/B dashboard would be empty on each cold
    start. The data goes through the same bucketing/exposure/outcome helpers as
    live traffic, and the experiment is flagged `demo_seeded` so the UI can
    label it as simulated.
    """
    exp_id = DEFAULT_EXPERIMENT["id"]
    if DEMO_SEED_USERS <= 0 or exp_id not in _EXPERIMENTS:
        return
    if _EXPOSURES.get(exp_id):
        return  # already has traffic (e.g. a warm reload)
    simulate_traffic(
        exp_id,
        SimulateRequest(n_users=DEMO_SEED_USERS, control_rate=0.10, treatment_lift=0.06, seed=7),
    )
    _EXPERIMENTS[exp_id]["demo_seeded"] = True


# ---------------------------------------------------------------------------
# Retrieval (shared by /search, /answer, /eval)
# ---------------------------------------------------------------------------
def _retrieve(query: str, config: dict, k: int = 5, vector_only: bool = False) -> list[dict]:
    """Return ranked chunks for a variant config.

    config: {"retrieval": "vector"|"hybrid", "topK": int, "alpha": float}
    """
    _ensure_index()
    q_vec = embed_query(query)
    retrieval = "vector" if vector_only else config.get("retrieval", "vector")
    candidate_k = max(k * 3, k)

    if retrieval == "hybrid" and _BM25 is not None:
        vec_idx = brute_force_search(q_vec, _INDEX_VECTORS, top_k=candidate_k)
        bm_idx = _BM25.search(query, top_k=candidate_k)
        fused = rrf_fuse(vec_idx, bm_idx, k=60)
        order = fused[:k]
    else:
        order = brute_force_search(q_vec, _INDEX_VECTORS, top_k=k)

    qn = float(np.linalg.norm(q_vec))
    qv = q_vec / qn if qn > 0 else q_vec

    out = []
    for rank, idx in enumerate(order, start=1):
        if idx >= len(_INDEX_CHUNKS):
            continue
        c = _INDEX_CHUNKS[idx]
        score = float(np.dot(qv, _INDEX_VECTORS[idx])) if idx < len(_INDEX_VECTORS) else 0.0
        out.append({
            "rank": rank,
            "chunk_id": c.get("chunk_id"),
            "text": c.get("text", "")[:400],
            "score": round(score, 4),
            "confidence": f"{max(0.0, min(1.0, score)) * 100:.0f}%",
        })
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str
    user_id: str = "anon"


class AnswerRequest(BaseModel):
    query: str
    user_id: str = "anon"


class VariantSpec(BaseModel):
    name: str
    weight: int = 50
    config: dict = {}

    @field_validator("name")
    @classmethod
    def _name_ok(cls, v: str) -> str:
        if not v or not v.replace("_", "").isalnum():
            raise ValueError("variant name must be alphanumeric/underscore")
        return v

    @field_validator("weight")
    @classmethod
    def _weight_ok(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("weight must be positive")
        return v


class ExperimentCreate(BaseModel):
    id: str
    name: str
    status: str = "draft"
    primary_metric: str | None = "thumbs_up_rate"
    config_json: dict | None = None


class ExperimentUpdate(BaseModel):
    name: str | None = None
    status: str | None = None
    primary_metric: str | None = None
    config_json: dict | None = None


class EvalSetCreate(BaseModel):
    id: str
    name: str
    labeled: list[dict]  # [{"query": str, "relevant_chunk_ids": [str, ...]}]


class EvalRunRequest(BaseModel):
    set_id: str
    retrieval: str = "vector"
    top_k: int = 10
    alpha: float = 0.6
    compare: list[str] | None = None  # e.g. ["vector", "hybrid"]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    _ensure_index()
    return {
        "status": "ok",
        "index_version": "v1",
        "vector_count": len(_INDEX_VECTORS),
        "chunk_count": len(_INDEX_CHUNKS),
    }


# ---------------------------------------------------------------------------
# Search / Answer / Streaming
# ---------------------------------------------------------------------------
@app.post("/search")
def search(req: SearchRequest) -> dict:
    _ensure_index()
    exp = _active_experiment()
    exp_id = exp["id"]
    variants = exp["config_json"]["variants"]
    variant = assign_variant(exp_id, req.user_id, variants)
    _log_exposure(exp_id, variant, req.user_id, f"exp_{exp_id}_{req.user_id}")

    vconf = next((v["config"] for v in variants if v["name"] == variant), {})
    results = _retrieve(req.query, vconf, k=vconf.get("topK", 5))
    return {
        "query": req.query,
        "variant": variant,
        "results": results,
        "experiment_id": exp_id,
    }


@app.post("/answer")
def answer(req: AnswerRequest) -> dict:
    retrieved = _retrieve(req.query, {"retrieval": "hybrid"}, k=5)
    prompt = build_rag_prompt(req.query, [{"chunk_id": r["chunk_id"], "text": r["text"]} for r in retrieved])
    chunk_ids = [r["chunk_id"] for r in retrieved]
    result = _provider().generate(prompt, options={"top_chunk_id": chunk_ids[0] if chunk_ids else "", "chunk_ids": chunk_ids})
    citations = extract_citations(result["text"], chunk_ids) or chunk_ids[:1]
    return {
        "query": req.query,
        "answer": result["text"],
        "citations": citations,
        "chunks": retrieved,          # for citation highlighting
        "cached": result.get("cached", False),
        "retrieved_chunks": len(retrieved),
    }


@app.get("/answer/stream")
def answer_stream(query: str = Query(...), user_id: str = "anon"):
    """Server-Sent Events stream of a grounded answer, then citations."""
    retrieved = _retrieve(query, {"retrieval": "hybrid"}, k=5)
    chunk_ids = [r["chunk_id"] for r in retrieved]
    options = {"top_chunk_id": chunk_ids[0] if chunk_ids else "", "chunk_ids": chunk_ids}

    def event_stream():
        # 1) send the retrieved chunks first so the UI can render citations
        yield f"event: chunks\ndata: {json.dumps(retrieved)}\n\n"
        # 2) stream the answer tokens
        prompt = build_rag_prompt(query, [{"chunk_id": r["chunk_id"], "text": r["text"]} for r in retrieved])
        full = ""
        for token in _provider().generate_stream(prompt, options=options):
            full += token
            yield f"event: token\ndata: {json.dumps(token)}\n\n"
        # 3) done event with final citations
        citations = extract_citations(full, chunk_ids) or chunk_ids[:1]
        yield f"event: done\ndata: {json.dumps({'citations': citations, 'answer': full})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
@app.post("/events")
def events(batch: dict) -> dict:
    stored = 0
    for ev in batch.get("events", []):
        eid = ev.get("event_id")
        if not eid:
            continue
        exp_id = ev.get("experiment_id") or _ACTIVE_EXPERIMENT_ID
        exp = _EXPERIMENTS.get(exp_id, DEFAULT_EXPERIMENT)
        variants = exp["config_json"]["variants"]
        variant = ev.get("variant") or assign_variant(exp_id, ev.get("user_id", "anon"), variants)
        metric = ev.get("metric", ev.get("event_type", "thumbs"))
        if _log_outcome(exp_id, variant, metric, float(ev.get("value", 0)), eid, ev.get("user_id")):
            stored += 1
    return {"stored": stored, "idempotent": True}


# ---------------------------------------------------------------------------
# Experiments — admin console
# ---------------------------------------------------------------------------
@app.get("/experiments")
def list_experiments() -> dict:
    exps = [{**e, "active": e["id"] == _ACTIVE_EXPERIMENT_ID} for e in _EXPERIMENTS.values()]
    return {"experiments": exps, "active": _ACTIVE_EXPERIMENT_ID}


@app.get("/experiments/{exp_id}")
def get_experiment(exp_id: str) -> dict:
    if exp_id not in _EXPERIMENTS:
        raise HTTPException(status_code=404, detail="experiment not found")
    return _EXPERIMENTS[exp_id]


def _validate_config(config: dict) -> None:
    variants = (config or {}).get("variants")
    if not isinstance(variants, list) or len(variants) < 2:
        raise HTTPException(status_code=422, detail="config_json.variants must have >= 2 variants")
    names = [v.get("name") for v in variants]
    if len(set(names)) != len(names):
        raise HTTPException(status_code=422, detail="variant names must be unique")
    for v in variants:
        if not isinstance(v.get("weight", 0), int) or v.get("weight", 0) <= 0:
            raise HTTPException(status_code=422, detail="each variant needs a positive integer weight")


@app.post("/experiments", status_code=201)
def create_experiment(exp: ExperimentCreate) -> dict:
    if exp.id in _EXPERIMENTS:
        raise HTTPException(status_code=409, detail="experiment already exists")
    config = exp.config_json or {
        "variants": [
            {"name": "control", "weight": 50, "config": {"retrieval": "vector", "topK": 5}},
            {"name": "treatment", "weight": 50, "config": {"retrieval": "hybrid", "topK": 5, "alpha": 0.6}},
        ]
    }
    _validate_config(config)
    doc = {
        "id": exp.id,
        "name": exp.name,
        "status": exp.status,
        "primary_metric": exp.primary_metric,
        "config_json": config,
    }
    _EXPERIMENTS[exp.id] = doc
    return doc


@app.patch("/experiments/{exp_id}")
def update_experiment(exp_id: str, patch: ExperimentUpdate) -> dict:
    if exp_id not in _EXPERIMENTS:
        raise HTTPException(status_code=404, detail="experiment not found")
    doc = _EXPERIMENTS[exp_id]
    if patch.config_json is not None:
        _validate_config(patch.config_json)
        doc["config_json"] = patch.config_json
    for field in ("name", "status", "primary_metric"):
        val = getattr(patch, field)
        if val is not None:
            doc[field] = val
    return doc


@app.delete("/experiments/{exp_id}", status_code=200)
def delete_experiment(exp_id: str) -> dict:
    if exp_id not in _EXPERIMENTS:
        raise HTTPException(status_code=404, detail="experiment not found")
    if exp_id == DEFAULT_EXPERIMENT["id"]:
        raise HTTPException(status_code=400, detail="cannot delete the default experiment")
    del _EXPERIMENTS[exp_id]
    global _ACTIVE_EXPERIMENT_ID
    if _ACTIVE_EXPERIMENT_ID == exp_id:
        _ACTIVE_EXPERIMENT_ID = DEFAULT_EXPERIMENT["id"]
        _EXPERIMENTS[DEFAULT_EXPERIMENT["id"]]["status"] = "running"
    return {"deleted": exp_id}


@app.post("/experiments/{exp_id}/activate")
def activate_experiment(exp_id: str) -> dict:
    if exp_id not in _EXPERIMENTS:
        raise HTTPException(status_code=404, detail="experiment not found")
    global _ACTIVE_EXPERIMENT_ID
    if _ACTIVE_EXPERIMENT_ID in _EXPERIMENTS and _ACTIVE_EXPERIMENT_ID != exp_id:
        _EXPERIMENTS[_ACTIVE_EXPERIMENT_ID]["status"] = "paused"
    _ACTIVE_EXPERIMENT_ID = exp_id
    _EXPERIMENTS[exp_id]["status"] = "running"
    exps = [{**e, "active": e["id"] == _ACTIVE_EXPERIMENT_ID} for e in _EXPERIMENTS.values()]
    return {"active": exp_id, "experiments": exps}


class SimulateRequest(BaseModel):
    n_users: int = 800
    feedback_rate: float = 0.7
    control_rate: float = 0.10
    treatment_lift: float = 0.05
    seed: int | None = None


@app.post("/experiments/{exp_id}/simulate")
def simulate_traffic(exp_id: str, req: SimulateRequest) -> dict:
    """Generate synthetic users, exposures, and thumbs for an experiment.

    Useful for filling an experiment to statistical significance without real
    traffic. The first variant is treated as the control; others get +lift.
    Exposures and outcomes are logged through the same helpers as real traffic,
    so the stats and SRM checks run on them identically.
    """
    if exp_id not in _EXPERIMENTS:
        raise HTTPException(status_code=404, detail="experiment not found")
    exp = _EXPERIMENTS[exp_id]
    variants = exp["config_json"]["variants"]
    control_name = variants[0]["name"]
    rng = random.Random(req.seed)

    for i in range(max(0, req.n_users)):
        uid = f"sim_{exp_id}_{i}"
        variant = assign_variant(exp_id, uid, variants)
        _log_exposure(exp_id, variant, uid, f"sim_exp_{exp_id}_{i}")
        if rng.random() < req.feedback_rate:
            rate = req.control_rate + (0.0 if variant == control_name else req.treatment_lift)
            value = 1.0 if rng.random() < rate else -1.0
            _log_outcome(exp_id, variant, "thumbs", value, f"sim_ev_{exp_id}_{i}", uid)

    return {"simulated_users": req.n_users, "experiment_id": exp_id}


@app.get("/experiments/{exp_id}/results")
def experiment_results(exp_id: str) -> dict:
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

    label = "not enough data"
    if test:
        if srm and srm["srm_detected"]:
            label = "SRM warning — experiment untrustworthy"
        elif sum(v["n"] for v in per_variant) < 100:
            label = "not enough data (provisional)"
        elif test["significant"]:
            label = "treatment better" if test["absolute_lift"] > 0 else "control better"
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
        "demo_seeded": exp.get("demo_seeded", False),
    }


# ---------------------------------------------------------------------------
# Eval harness
# ---------------------------------------------------------------------------
@app.get("/eval/sets")
def list_eval_sets() -> dict:
    return {"sets": [{"id": s["id"], "name": s["name"], "n": len(s["labeled"])} for s in _EVAL_SETS.values()]}


@app.post("/eval/sets", status_code=201)
def create_eval_set(s: EvalSetCreate) -> dict:
    _ensure_index()
    if s.id in _EVAL_SETS:
        raise HTTPException(status_code=409, detail="eval set already exists")
    known = {c["chunk_id"] for c in _INDEX_CHUNKS}
    for item in s.labeled:
        if "query" not in item or "relevant_chunk_ids" not in item:
            raise HTTPException(status_code=422, detail="each item needs query + relevant_chunk_ids")
        unknown = [cid for cid in item["relevant_chunk_ids"] if cid not in known]
        if unknown:
            raise HTTPException(status_code=422, detail=f"unknown chunk ids: {unknown[:3]}")
    _EVAL_SETS[s.id] = {"id": s.id, "name": s.name, "labeled": s.labeled}
    return {"id": s.id, "name": s.name, "n": len(s.labeled)}


@app.post("/eval/sets/sample", status_code=201)
def seed_sample_eval_set(n: int = 40) -> dict:
    """Create a demo labeled set: each query is the opening of a real chunk,
    and the relevant id is that chunk. Self-supervised, so vector recall is high
    by construction — good for demonstrating the harness UI, not a benchmark."""
    _ensure_index()
    set_id = "sample_corpus"
    if not _INDEX_CHUNKS:
        raise HTTPException(status_code=503, detail="index not loaded")
    step = max(1, len(_INDEX_CHUNKS) // n)
    labeled = []
    for c in _INDEX_CHUNKS[::step][:n]:
        text = c.get("text", "")
        if len(text) < 40:
            continue
        labeled.append({"query": text[:80], "relevant_chunk_ids": [c["chunk_id"]]})
    _EVAL_SETS[set_id] = {"id": set_id, "name": "Sample (corpus-derived)", "labeled": labeled}
    return {"id": set_id, "name": "Sample (corpus-derived)", "n": len(labeled)}


@app.post("/eval/run")
def run_eval_endpoint(req: EvalRunRequest) -> dict:
    _ensure_index()
    if req.set_id not in _EVAL_SETS:
        raise HTTPException(status_code=404, detail="eval set not found")
    labeled = _EVAL_SETS[req.set_id]["labeled"]

    retrievals = req.compare or [req.retrieval]
    reports = []
    for r in retrievals:
        cfg = EvalConfig(retrieval=r, top_k=req.top_k, alpha=req.alpha)
        rep = run_eval(labeled, _INDEX_CHUNKS, _INDEX_VECTORS, _BM25, cfg, embed_query)
        row = {
            "retrieval": r,
            "n_queries": rep.n_queries,
            "recall_at_k": rep.recall_at_k,
            "mrr": rep.mrr,
            "ndcg_at_k": rep.ndcg_at_k,
        }
        reports.append(row)

    run = {"set_id": req.set_id, "top_k": req.top_k, "reports": reports}
    _EVAL_RUNS.append(run)
    return {"set_id": req.set_id, "top_k": req.top_k, "results": reports}


@app.get("/eval/runs")
def list_eval_runs() -> dict:
    return {"runs": _EVAL_RUNS[-20:]}
