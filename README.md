# Cairn

A retrieval-augmented search engine with a from-scratch vector index and an integrated A/B experimentation platform.

Cairn implements its own approximate-nearest-neighbour index (HNSW), keyword index (BM25), hybrid rank fusion, and scalar quantization in pure Python, then layers a statistically grounded experimentation framework on top. The system is designed to be deployable on free-tier infrastructure without a managed vector database.

---

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key results](#key-results)
- [Components](#components)
- [API reference](#api-reference)
- [Experimentation framework](#experimentation-framework)
- [Offline evaluation](#offline-evaluation)
- [Benchmarks](#benchmarks)
- [Testing](#testing)
- [Local development](#local-development)
- [Deployment](#deployment)
- [Repository structure](#repository-structure)
- [Limitations and future work](#limitations-and-future-work)
- [License](#license)

---

## Overview

The project addresses two problems:

1. **Retrieval.** Given a query, identify the most relevant passages from a corpus. Cairn implements the retrieval stack directly — HNSW graph search, BM25 ranking, Reciprocal Rank Fusion, and int8 quantization — rather than depending on an external vector database.

2. **Measurement.** Determine, with statistical confidence, which retrieval configuration performs better. Cairn provides deterministic user bucketing, two-proportion and Welch's t-tests, Sample Ratio Mismatch detection, idempotent event logging, and an offline evaluation harness.

The application exposes four views:

| View | Function |
|---|---|
| Search | Ranked corpus passages by cosine similarity, with per-result score and confidence. No language model involved. |
| Answer | A streamed, citation-grounded answer; selecting a citation highlights the corresponding source passage. |
| Experiments | A/B dashboard and administration console: create, activate, pause, and delete experiments; edit variants and weights; generate traffic; inspect hypothesis tests and SRM status. |
| Eval | Offline evaluation of retrieval configurations against a labelled set (recall@k, MRR, nDCG@k). |

A public landing page is served at `/` and a documentation page at `/docs`.

---

## Architecture

The system separates an **offline indexing path** from an **online serving path**.

**Offline path** (executed once, on a workstation or in CI):

```
documents → chunking → embedding (384-dim) → index artifacts (JSONL + NPY)
```

**Online path** (per request):

```
query → variant assignment → retrieval (vector and/or BM25) → rank fusion
      → results → outcome logging → aggregate statistics
```

Retrieval and the experimentation framework reside in separate, dependency-free Python packages (`engine/` and `experiments/`) so that both are independently unit-testable and free of web-framework dependencies. The FastAPI application (`api/`) composes them, and the React frontend (`web/`) consumes the HTTP API.

---

## Key results

All figures are reproducible from the scripts referenced in the final column. Corpus: 610 chunks, 384 dimensions.

| Metric | Value | Source |
|---|---|---|
| int8 quantization, memory | 0.89 MB → 0.22 MB (4×) | `bench/benchmark_quantization.py` |
| int8 vs float32, brute-force recall@10 | 60–80% overlap | same |
| int8 average cosine similarity (after dequantization) | 0.684 | same |
| HNSW recall@10 (M=32, efConstruction=200) | ~0.98 | `bench/benchmark_hnsw.py` |
| HNSW query latency (610 vectors) | ~0.5 ms | same |
| Retrieval — vector: MRR / nDCG@10 | 0.855 / 0.890 | `/eval/run` |
| Retrieval — hybrid: MRR / nDCG@10 | 0.975 / 0.982 | same |
| Bucketing split (20,000 users, 50/50) | 49.41% / 50.59% | `experiments/simulation.py` |
| Cross-experiment bucket correlation | +0.007 | same |
| A/A false-positive rate (α = 0.05) | 4.3% | same |
| A/B detection power (+2 pp, 3,000/arm) | 0.73 | same |
| End-to-end experiment | control 9.95% vs treatment 12.73% thumbs-up; +27.9% relative; z = −6.20 | same |
| SRM detection | 50/50 not flagged; 52/48, 60/40, 90/10 flagged | same |

At the current corpus size, HNSW latency is comparable to brute force, because pure-Python graph traversal overhead dominates; the asymptotic advantage is realised at corpus sizes above roughly 10,000 vectors.

---

## Components

| Path | Responsibility |
|---|---|
| `pipeline/` | Corpus loading, chunking, embedding, and artifact generation. |
| `engine/` | Brute-force and HNSW nearest-neighbour search, BM25, Reciprocal Rank Fusion, int8 quantization. Pure Python, no web dependencies. |
| `experiments/` | Deterministic bucketing, hypothesis tests, SRM detection, idempotent logging, and simulation. Pure Python. |
| `eval/` | Retrieval metrics (recall@k, MRR, nDCG) and the labelled-set harness. |
| `api/` | FastAPI application, request/response models, and the language-model provider interface. |
| `web/` | React, TypeScript, and Tailwind frontend built with Vite. |
| `bench/` | Reproducible benchmark scripts. |
| `tests/` | pytest suite (36 tests). |
| `scripts/` | SQL schema for the production database. |
| `docs/` | Design notes and the simulation guide. |

### Retrieval engine

- **HNSW.** A multi-layer proximity graph. Nodes receive a random level (`P(level ≥ k) = 0.5^k`); insertion searches each layer for candidate neighbours and applies the diversity heuristic from Malkov and Yashunin to maintain connectivity; search performs greedy descent through the upper layers followed by a best-first beam search (`efSearch`) at layer 0. The index supports persistence to disk.
- **BM25.** An inverted index with document-length normalization and the standard `k1` and `b` parameters.
- **Reciprocal Rank Fusion.** Combines vector and keyword rankings without requiring normalization across incompatible score scales.
- **Int8 quantization.** Per-dimension scalar quantization with stored scales; reduces index memory by 4× at a measured cost of approximately 0.68 cosine similarity and 60–80% brute-force recall overlap.

---

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

---

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

---

## Offline evaluation

The evaluation harness scores a retrieval configuration against a labelled set of `query → relevant chunk ids` pairs and reports recall@k, Mean Reciprocal Rank, and nDCG@k. A corpus-derived sample set can be generated from the UI; sets may also be supplied as JSON.

```
POST /eval/run  { "set_id": "...", "top_k": 10, "compare": ["vector", "hybrid"] }
```

On the sample set, hybrid retrieval improves MRR from 0.855 to 0.975 and nDCG@10 from 0.890 to 0.982 relative to vector-only retrieval.

---

## Benchmarks

| Script | Measures |
|---|---|
| `bench/benchmark_engine.py` | Brute-force, BM25, and RRF correctness; int8 memory footprint. |
| `bench/benchmark_hnsw.py` | HNSW recall@10 and latency across `efSearch`. |
| `bench/benchmark_quantization.py` | float32 versus int8 recall and memory. |
| `bench/benchmark_day9.py` | MRR@10 and nDCG@10 for vector, BM25, and hybrid retrieval. |
| `bench/run_all_benchmarks.py` | Executes all of the above. |

---

## Testing

```bash
PYTHONPATH=. pytest -q       # 36 passing
```

Coverage spans the retrieval engine (brute force, BM25, RRF, quantization, HNSW, persistence), the experimentation primitives (bucketing, weight distribution, hypothesis tests, SRM), and the HTTP API (search, streaming, idempotent events, experiment lifecycle, activation routing, evaluation, simulation).

---

## Local development

### Backend

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash build.sh                       # generates data/chunks.jsonl and data/vectors.npy
PYTHONPATH=. uvicorn api.main:app --port 8000
```

Interactive API documentation is available at `http://localhost:8000/docs`.

### Frontend

```bash
cd web
npm install
npm run dev                         # http://localhost:5173
```

The Vite development server proxies `/search`, `/answer`, `/events`, `/health`, `/experiments`, and `/eval` to the backend on port 8000.

---

## Deployment

The system targets free-tier infrastructure:

- `render.yaml` — Render free web service and free PostgreSQL; the build step installs dependencies and runs `build.sh`.
- `vercel.json` — static frontend hosting (Vercel or Cloudflare Pages) with SPA fallback and API rewrites.
- `scripts/init_db.sql` — PostgreSQL schema for experiments, variants, exposures, outcomes, and results.

Set `VITE_API_URL` at frontend build time to target the deployed API. Replacing `StubProvider` in `api/llm_provider.py` with a free-tier language-model provider enables real answer generation.

---

## Repository structure

```
engine/        retrieval engine (HNSW, BM25, RRF, quantization); no web dependencies
pipeline/      chunking, embedding, and artifact generation
experiments/   bucketing, hypothesis tests, SRM, simulation
eval/          retrieval metrics and the labelled-set harness
api/           FastAPI application and language-model provider
web/           React + TypeScript + Tailwind frontend
bench/         reproducible benchmark scripts
scripts/       database schema
tests/         pytest suite
docs/          design notes and simulation guide
data/          generated artifacts (git-ignored)
```

---

## Limitations and future work

- **Answer generation.** The `/answer` endpoint currently uses a deterministic stub provider. The provider interface is in place; a real model can be substituted without API changes.
- **State.** Experiment definitions and events are held in memory. The PostgreSQL schema (`scripts/init_db.sql`) and the statistics functions are complete, but the persistence layer is not yet wired, and the scheduled statistics job is not implemented.
- **HNSW performance.** At the current corpus size, latency is on par with brute force. Larger corpora are required to demonstrate the asymptotic advantage.
- **Evaluation set.** The bundled evaluation set is corpus-derived and intended to demonstrate the harness; a hand-verified question set is required for publication-quality retrieval metrics.

---

## License

MIT
