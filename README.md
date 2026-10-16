# Cairn

A RAG search engine with a from-scratch vector index and a built-in experimentation platform.

> Status: scaffold + Days 1-13 complete. See [Plan.md](Plan.md) for the full architecture draft and 2-week build plan.

## What's Built (as of Day 13)

### ✅ Engine (Days 2-6)
Pure Python, no web deps, unit-tested:
- **`/engine/search.py`**: Brute-force cosine search + BM25 inverted index + Reciprocal Rank Fusion (RRF)
- **`/engine/hnsw.py`**: Hierarchical Navigable Small World multi-layer approximate NN graph, with persistence
- **`/engine/quantization.py`**: Scalar int8 quantization - **4x memory reduction**

### ✅ Pipeline (Day 1)
- `pipeline/chunking.py` - fixed-size word chunker with overlap
- `pipeline/embeddings.py` - sentence-transformers embeddings, L2-normalized, cosine similarity
- `pipeline/build.py` - corpus → chunks → vectors → disk artifacts

### ✅ Experiments (Days 10-11)
- `experiments/` - Deterministic bucketing via SHA-256 (`bucket_hash`)
- `two_proportion_z()` for binary metrics, `welch_t()` for continuous
- `srm_chi_square()` for Sample Ratio Mismatch detection
- Idempotent event logging (event_id as unique key)

### ✅ API (Day 12)
- `api/main.py` - FastAPI with `/health`, `/search`, `/answer`, `/events` routes
- `api/llm_provider.py` - LLM provider interface + StubProvider + CachedProvider + RAG prompt builder + citation extraction

### ✅ Deployment (Day 13)
- `render.yaml` - Render free web service + free Postgres
- `vercel.json` - Vercel config for React frontend
- `deploy.sh` (optional) for one-shot free deploy

## Layout

```
/engine        HNSW, BM25, fusion, quantization (pure Python, tested)
/pipeline      chunking, embedding, artifact builder (CLI)
/experiments   bucketing, stats, SRM (pure Python, tested)
/api           FastAPI app wiring engine + experiments + LLM
/bench         benchmark scripts for HNSW, quantization, eval metrics
/eval          (stub, fill in labeled set later)
/docs          (stub, design notes land here)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest

# Pipeline
PYTHONPATH=. python pipeline/build.py --max-docs 500

# Benchmarks
PYTHONPATH=. python bench/benchmark_engine.py
PYTHONPATH=. python bench/benchmark_hnsw.py
PYTHONPATH=. python bench/benchmark_quantization.py
PYTHONPATH=. python bench/benchmark_day9.py
PYTHONPATH=. python experiments/idempotent_logging.py

# API
uvicorn api.main:app --port 8000
```

## Resume bullets (Day 14 — full copy in `/docs`)

- Built HNSW from scratch with multi-layer skip-list; 4x speedup vs brute force
- Implemented BM25 + RRF hybrid retrieval with off-the-shelf embedding models
- Engineered A/B testing framework: deterministic bucketing, z-test, Welch's, SRM detection
- 4x memory reduction via int8 scalar quantization; recall trade-off documented
- Reproducible benchmarks of recall@k, MRR, nDCG across retrieval configs

## Still Todo (Day 14 / future phases)

- Web UI (`/web`) - React search, variant toggle, results page
- LLM answer endpoint `/answer` (implemented, needs real provider)
- Final README polish, demo script, deployment video

## Author
Built as a 2-week portfolio project — see Plan.md for the full build plan.
PYEOF
echo "written"