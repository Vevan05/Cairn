# Cairn

A retrieval-augmented search engine with a from-scratch vector index and an integrated A/B experimentation platform.

Cairn implements its own approximate-nearest-neighbour index (HNSW), keyword index (BM25), hybrid rank fusion, and scalar quantization in Python and NumPy, embeds text with a small ONNX model, then layers a statistically grounded experimentation framework on top. The system is designed to be deployable on free-tier infrastructure without a managed vector database.


## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key results](#key-results)
- [Components](#components)
- [API reference](#api-reference)
- [Experimentation framework](#experimentation-framework)
- [Offline evaluation](#offline-evaluation)
- [Answer generation](#answer-generation)
- [Benchmarks](#benchmarks)
- [Testing](#testing)
- [Local development](#local-development)
- [Deployment](#deployment)
- [Repository structure](#repository-structure)
- [Engineering notes](#engineering-notes)
- [Limitations and future work](#limitations-and-future-work)
- [License](#license)


## Overview

The project addresses two problems:

1. **Retrieval.** Given a query, identify the most relevant passages from a corpus. Cairn implements the retrieval stack directly — HNSW graph search, BM25 ranking, Reciprocal Rank Fusion, and int8 quantization — rather than depending on an external vector database.

2. **Measurement.** Determine, with statistical confidence, which retrieval configuration performs better. Cairn provides deterministic user bucketing, two-proportion and Welch's t-tests, Sample Ratio Mismatch detection, idempotent event logging, and an offline evaluation harness.

The application exposes four views:

| View | Function |
|---|---|
| Search | Ranked corpus passages by cosine similarity, with per-result score and confidence. No language model involved. |
| Answer | Streams a citation-grounded answer from the deterministic stub generator; selecting a citation highlights the corresponding source passage. |
| Experiments | A/B dashboard and administration console: create, activate, pause, and delete experiments; edit variants and weights; generate traffic; inspect hypothesis tests and SRM status. |
| Eval | Offline evaluation of retrieval configurations against a labelled set (recall@k, MRR, nDCG@k). |

A public landing page is served at `/` and a documentation page at `/docs`.


## Architecture

The system separates an **offline indexing path** from an **online serving path**.

**Offline path** (executed once, on a workstation or in CI):

```
documents → chunking → embedding (384-dim) → index artifacts (JSONL + NPY)
```

The shipped index is built from the full WikiText-2 train split: 14,324 paragraphs across 629 articles, chunked to 16,770 passages.

**Online path** (per request):

```
query → variant assignment → retrieval (vector and/or BM25) → rank fusion
      → results → outcome logging → aggregate statistics
```

Retrieval and the experimentation framework reside in separate Python packages with no web-framework dependencies (`engine/` uses only NumPy; `experiments/` is standard library), so both are independently unit-testable. The FastAPI application (`api/`) composes them, and the React frontend (`web/`) consumes the HTTP API.

At startup the service loads the precomputed artifacts (vectors + chunks) and builds the in-memory BM25 index; it **never builds the HNSW graph**. Online search is exact brute-force over the float32 vectors plus BM25, fused with RRF. HNSW is a benchmarked component (`bench/`, with save/load persistence) rather than the serving path.


## Key results

All figures are reproducible from the scripts referenced in the final column. Corpus: **16,770 chunks** (14,324 WikiText-2 paragraphs across 629 articles), 384 dimensions.

| Metric | Value | Source |
|---|---|---|
| int8 quantization, memory | 24.6 MB → 6.1 MB (4×) | `bench/benchmark_quantization.py` |
| int8 average cosine similarity (after dequantization) | 0.9999 | same |
| int8 vs float32, brute-force recall@10 | 0.99 overlap | same |
| HNSW recall@10 (M=32, efConstruction=200) | 0.96 @ efSearch 40 → 1.00 @ 80 | `bench/benchmark_hnsw.py` |
| HNSW query latency (16,770 vectors) | 0.6 ms (ef 10) … 3.0 ms (ef 100) | same |
| Brute-force query latency (16,770 vectors) | ~2 ms | same |
| Retrieval — vector: recall@10 / MRR / nDCG@10 | 0.970 / 0.884 / 0.905 | `bench/benchmark_day9.py` |
| Retrieval — BM25: recall@10 / MRR / nDCG@10 | 1.000 / 0.985 / 0.989 | same |
| Retrieval — hybrid: recall@10 / MRR / nDCG@10 | 1.000 / 0.948 / 0.961 | same |
| Bucketing split (20,000 users, 50/50) | 49.41% / 50.59% | `experiments/simulation.py` |
| Cross-experiment bucket correlation | +0.007 | same |
| A/A false-positive rate (α = 0.05) | 4.3% | same |
| A/B detection power (+2 pp, 3,000/arm) | 0.73 | same |
| End-to-end experiment | control 9.95% vs treatment 12.73% thumbs-up; +27.9% relative; z = −6.20 | same |
| SRM detection | 50/50 not flagged; 52/48, 60/40, 90/10 flagged | same |

HNSW recall responds to `efSearch` as expected: **0.68 → 0.96 → 1.00** at `efSearch` 10 / 40 / 80. Because graph traversal is Python-level — one NumPy dot product per visited node — the speedup over a vectorised NumPy brute force depends on how far it searches: roughly **3× at recall 0.68** (`efSearch` 10), near parity at recall 0.96, and slower at recall 1.0. The honest summary is that HNSW trades recall for latency, and at ~17k vectors the Python-level implementation only wins when some recall is sacrificed; at larger corpora the O(N) brute-force term grows and the crossover moves.


## Components

| Path | Responsibility |
|---|---|
| `pipeline/` | WikiText-2 loading, word-window chunking (200 words, 40-word overlap), embedding, and artifact generation. |
| `engine/` | Brute-force and HNSW nearest-neighbour search, BM25, Reciprocal Rank Fusion, int8 quantization. Python + NumPy, no web dependencies. |
| `experiments/` | Deterministic bucketing, hypothesis tests, SRM detection, idempotent logging, and simulation. Pure Python (standard library only). |
| `eval/` | Retrieval metrics (recall@k, MRR, nDCG) and the labelled-set harness. |
| `api/` | FastAPI application, request/response models, and the language-model provider interface. |
| `web/` | React, TypeScript, and Tailwind frontend built with Vite. |
| `bench/` | Reproducible benchmark scripts. |
| `tests/` | pytest suite (36 tests). |
| `scripts/` | PostgreSQL schema for future persistence (not required to run). |
| `docs/` | Design notes and the simulation guide. |

### Retrieval engine

- **Embeddings.** `BAAI/bge-small-en-v1.5` (384-dim) served through **ONNX Runtime** (`fastembed`), chosen over PyTorch to fit the free-tier memory budget. Documents are embedded once at build time; queries are embedded per request with the model's asymmetric query prefix.
- **HNSW.** A multi-layer proximity graph. Nodes receive a random level (`P(level ≥ k) = 0.5^k`); insertion searches each layer for candidate neighbours and applies the diversity heuristic from Malkov and Yashunin to maintain connectivity; search performs greedy descent through the upper layers followed by a best-first beam search (`efSearch`) at layer 0. The index supports persistence to disk.
- **BM25.** An inverted index with document-length normalization and the standard `k1` and `b` parameters. Term frequencies are stored at build time, and postings are packed into compact C-int arrays rather than tuples.
- **Reciprocal Rank Fusion.** Combines vector and keyword rankings without requiring normalization across incompatible score scales.
- **Int8 quantization.** Symmetric per-dimension scalar quantization with stored scales; reduces index memory by 4× (24.6 MB → 6.1 MB at 16,770 chunks) while preserving ranking: 0.9999 average cosine similarity and 0.99 brute-force recall@10 overlap. Quantization is a benchmarked capability rather than the serving path — the API serves float32 vectors.


## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Index version, vector count, chunk count. |
| POST | `/search` | Assign a variant, retrieve ranked passages, log an exposure. |
| POST | `/answer` | Retrieve and generate a cited answer (non-streaming). |
| GET | `/answer/stream` | Server-Sent Events stream of retrieved chunks, answer tokens, and citations. |
| POST | `/events` | Batched, idempotent outcome events. |
| GET | `/experiments` | List experiments and the active experiment. |
| GET | `/experiments/{id}` | Retrieve a single experiment. |
| POST | `/experiments` | Create an experiment (validated). |
| PATCH | `/experiments/{id}` | Update name, status, primary metric, or configuration. |
| DELETE | `/experiments/{id}` | Delete an experiment. |
| POST | `/experiments/{id}/activate` | Make an experiment the one serving traffic. |
| POST | `/experiments/{id}/simulate` | Generate synthetic exposures and outcomes. |
| GET | `/experiments/{id}/results` | Aggregate statistics, hypothesis test, SRM status, and verdict. |
| GET/POST | `/eval/sets` | List or create labelled evaluation sets. |
| POST | `/eval/sets/sample` | Create a corpus-derived sample set. |
| POST | `/eval/run` | Score one or more retrieval configurations. |
| GET | `/eval/runs` | Recent evaluation runs. |


## Experimentation framework

Assignment is deterministic: `SHA-256(experiment_id + ":" + user_id) mod 10000`, mapped to variants by cumulative weight. A stable hash function is used deliberately — Python's built-in `hash()` is not stable across processes.

Design properties:

- **Active experiment.** Exactly one experiment serves traffic at a time. `POST /experiments/{id}/activate` switches it and pauses the previous one; newly created experiments remain idle until activated.
- **User-scoped outcomes.** The unit of analysis is the user: at most one exposure and one vote per user per variant. This guarantees `outcomes ≤ exposures` and makes the thumbs-up rate a per-user proportion.
- **Idempotent logging.** Events are keyed by a unique `event_id`; retries are no-ops.
- **SRM detection.** A chi-square test compares observed variant counts with those expected from the configured weights. A flagged experiment is reported as untrustworthy.
- **Traffic simulation.** `POST /experiments/{id}/simulate` generates synthetic users and routes them through the same bucketing, exposure, and outcome functions as live traffic, allowing experiments to be evaluated before sufficient real traffic exists. See `docs/simulation.md`.

The framework can be exercised without the API:

```bash
PYTHONPATH=. python experiments/simulation.py
```

This reports bucketing validation, the A/A false-positive rate, A/B detection power, SRM cases, and a complete experiment.


## Offline evaluation

The evaluation harness scores a retrieval configuration against a labelled set of `query → relevant chunk ids` pairs and reports recall@k, Mean Reciprocal Rank, and nDCG@k. A corpus-derived sample set can be generated from the UI; sets may also be supplied as JSON.

```
POST /eval/run  { "set_id": "...", "top_k": 10, "compare": ["vector", "bm25", "hybrid"] }
```

On the bundled 100-query corpus-derived set, vector retrieval scores recall@10 0.970 / MRR 0.884 / nDCG@10 0.905, hybrid fusion scores 1.000 / 0.948 / 0.961, and BM25 alone scores 1.000 / 0.985 / 0.989. BM25 is strongest on this particular set because each query reuses rare terms from its source chunk — a property of corpus-derived queries rather than a general result. Supported configs are `vector`, `bm25`, and `hybrid`.


## Answer generation

`POST /answer` and `GET /answer/stream` retrieve the top passages, build a citation-constrained RAG prompt, and produce an answer that cites them. The prose itself comes from a **deterministic stub** (`StubProvider`): retrieval, ranking, streaming and citation highlighting are all real, but no language model is called.

That is a deliberate trade-off — it keeps the demo and the test suite offline, reproducible, and free of API keys and rate limits. The `LLMProvider` interface (`generate` / `generate_stream`) exists precisely so a hosted model is a drop-in swap: implement it and return it from `api/main.py`, and nothing else changes.

The Answer view labels this in the UI itself with a **"Demo mode · templated answer from retrieved passages"** badge, so the stub can't be mistaken for a real model's output.

| Real | Stubbed |
|---|---|
| Hybrid retrieval, ranking, scores | The answer prose |
| Citation-constrained prompt (`build_rag_prompt`) | |
| Token-by-token SSE streaming | |
| Click-to-highlight citations | |


## Benchmarks

| Script | Measures |
|---|---|
| `bench/benchmark_engine.py` | Brute-force, BM25, and RRF correctness; int8 memory footprint. |
| `bench/benchmark_hnsw.py` | HNSW recall@10, latency, and brute-force speedup across `efSearch`. |
| `bench/benchmark_quantization.py` | float32 versus int8 recall and memory. |
| `bench/benchmark_day9.py` | MRR@10 and nDCG@10 for vector, BM25, and hybrid retrieval. |
| `bench/run_all_benchmarks.py` | Executes all of the above. |


## Testing

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest -q       # 36 passing
```

Coverage spans the retrieval engine (brute force, BM25, RRF, quantization, HNSW, persistence), the evaluation harness (recall@k, MRR, nDCG for vector / BM25 / hybrid), the experimentation primitives (bucketing, weight distribution, hypothesis tests, SRM), and the HTTP API (search, streaming, idempotent events, experiment lifecycle, activation routing, evaluation, simulation).


## Local development

### Backend

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # runtime + datasets + pytest

bash build.sh                           # (re)build artifacts; ~16.8k chunks
PYTHONPATH=. uvicorn api.main:app --port 8000
```

The committed corpus artifacts mean `pip install -r requirements.txt` (runtime only) is enough to serve; `requirements-dev.txt` is only needed to rebuild the corpus or run the tests/benchmarks.

Interactive API documentation is available at `http://localhost:8000/docs`.

### Frontend

```bash
cd web
npm install
npm run dev                         # http://localhost:5173
```

The Vite development server proxies `/search`, `/answer`, `/events`, `/health`, `/experiments`, and `/eval` to the backend on port 8000.


## Deployment

Sized to run within a free tier — Render **512 MB / 0.1 CPU** for the API plus Vercel/Cloudflare static hosting for the frontend:

- `render.yaml` — Render free web service. The build only runs `pip install` plus a one-time prefetch of the ONNX embedding model; the corpus artifacts (`data/chunks.jsonl`, `data/vectors.npy`) are committed, so **no corpus embedding happens at deploy time**. A `/health` check is configured, and `CORS_ORIGINS` sets the allowed frontend origin.
- `vercel.json` — static frontend hosting with SPA fallback. Set **`VITE_API_URL`** to the deployed API URL at build time; the frontend then calls the API cross-origin and the backend sends CORS headers.
- `web/.env.example` documents the one frontend build-time variable (`VITE_API_URL`). Backend variables (`CORS_ORIGINS`, `DEMO_SEED_USERS`) are set on the host.
- No database is required — experiment state is in memory (see Limitations).
- **Answers** are produced by the deterministic stub, so no LLM key or external service is needed — see [Answer generation](#answer-generation).
- **Cold starts.** Render's free tier sleeps after inactivity, so the first request can take up to a minute while the instance wakes and reloads the index. The landing page warns visitors about this.
- **Demo data.** Because state is not persisted, the default experiment is seeded with synthetic traffic at startup so the dashboard is not empty; it is flagged as simulated in the UI.

> **Why the runtime fits in 512 MB.** Embeddings run on **ONNX Runtime** (`fastembed`), not PyTorch — importing torch + transformers alone costs ~600 MB. Measured runtime peak is **~371 MB** (index ~126 MB + ONNX embedder ~245 MB); the corpus size is not the constraint, since the index is only ~100 MB thanks to compact BM25 postings.


## Repository structure

```
engine/        retrieval engine (HNSW, BM25, RRF, quantization); no web dependencies
pipeline/      chunking, embedding, and artifact generation
experiments/   bucketing, hypothesis tests, SRM, simulation
eval/          retrieval metrics and the labelled-set harness
api/           FastAPI application and language-model provider
web/           React + TypeScript + Tailwind frontend
bench/         reproducible benchmark scripts
scripts/       PostgreSQL schema (future persistence; not required to run)
tests/         pytest suite
docs/          design notes and simulation guide
data/          corpus artifacts — chunks.jsonl + vectors.npy (committed for deploy)
```

## Engineering notes

Two results that first looked like ordinary ANN / quantization trade-offs were actually implementation bugs. Both are reproducible from the benchmarks and are now fixed.

**int8 quantization was using 3 of 256 levels.** `quantize_int8` scaled each dimension by its maximum magnitude, so `vectors / scale` landed in `[-1, 1]` and rounding produced only `{-1, 0, 1}`. The scale must map the maximum to 127 — the largest representable int8 value — and clipping must happen *before* the cast (casting an out-of-range float to `int8` wraps rather than saturates). Fix: `scale = max_abs / 127`. Effect: average cosine similarity **0.711 → 0.9999**, brute-force recall@10 overlap **26% → 99%**.

**HNSW capped layer-0 links at M instead of 2·M.** HNSW admits `Mmax0 = 2·M` links at the base layer; capping at `M` pruned every newly inserted node back out of its neighbours' adjacency lists. The graph stayed *weakly* connected (a single component) while **forward** reachability from the entry point collapsed to **41 / 5000** nodes — which capped recall and made `efSearch` a no-op. The tell was identical recall at every `efSearch` from 10 to 200, when recall should rise with the beam width. Fix: `_max_links(level) = 2*M` at level 0. Effect: forward reachability **41/5000 → 5000/5000**, and recall@10 now tracks `efSearch` (0.68 / 0.96 / 1.00 at 10 / 40 / 80).


## Limitations and future work

- **Answer generation.** `/answer` returns a deterministic stub answer: retrieval, streaming and citations are real, but the prose is templated. The `LLMProvider` interface is in place for a drop-in hosted model, and answer quality is not evaluated (only retrieval is).
- **State.** Experiment definitions and events are held in memory, so they reset on restart. To keep the dashboard meaningful on a non-persistent free tier, the default experiment is seeded with synthetic traffic at startup (`DEMO_SEED_USERS`, flagged `demo_seeded`). The PostgreSQL schema (`scripts/init_db.sql`) and the statistics functions are complete, but the persistence layer is not yet wired, and the scheduled statistics job is not implemented.
- **HNSW performance.** Recall@10 reaches 0.96–1.00 at `efSearch` 40–80, but the index is Python-level (a NumPy dot product per visited node): the graph build takes minutes at ~17k vectors and, at correct recall, is around parity with NumPy brute force at this scale. It is a benchmarked component, not the serving path — the served search is exact brute force.
- **Evaluation set.** The bundled evaluation set is corpus-derived and intended to demonstrate the harness; a hand-verified question set is required for publication-quality retrieval metrics.


## License

MIT
