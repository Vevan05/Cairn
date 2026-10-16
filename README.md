# Cairn

A from-scratch RAG search engine with a built-in A/B experimentation platform.
Pure-Python retrieval (no vector-DB dependency), a React UI, and a free-tier deploy path.

**Status:** feature-complete scaffold — every phase of the build plan implemented and tested (26 tests passing).

---

## What it does

Cairn answers two questions:

1. **Retrieval** — given a query, which stored passages are most relevant? (HNSW + BM25 + RRF fusion, no external vector DB)
2. **Experimentation** — which retrieval configuration is *statistically* better? (deterministic bucketing, z-tests, Sample Ratio Mismatch detection)

The UI has three tabs:

| Tab | What it does |
|---|---|
| **Search** | Ranks the closest corpus passages by cosine similarity. No LLM. |
| **Answer** | Retrieves passages, then generates a cited answer via an LLM provider. |
| **Experiments** | Live A/B dashboard: per-variant rates, z-test, lift, SRM flag, verdict. |

Plus a **landing page** at `/`.

---

## Components

| Path | Role |
|---|---|
| `pipeline/` | chunking, sentence-transformers embeddings, artifact builder |
| `engine/` | brute-force & HNSW NN, BM25, RRF fusion, int8 quantization (pure Python) |
| `experiments/` | SHA-256 bucketing, z-test / Welch's t / χ² SRM, idempotent logging, simulations |
| `api/` | FastAPI: `/health`, `/search`, `/answer`, `/events`, `/experiments`, `/experiments/{id}/results` |
| `web/` | Vite + React + TypeScript + Tailwind UI |
| `bench/` | reproducible benchmarks (HNSW, quantization, MRR/nDCG) |
| `tests/` | pytest suite (engine, experiments, API integration) |
| `scripts/` | `init_db.sql` for the Postgres schema |

---

## Quick start

### Backend

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# build the offline index (first run downloads a small corpus + embedding model)
bash build.sh                       # writes data/chunks.jsonl + data/vectors.npy

# run API
PYTHONPATH=. uvicorn api.main:app --port 8000
# interactive docs: http://localhost:8000/docs
```

### Frontend

```bash
cd web
npm install
npm run dev                         # http://localhost:5173
```

Vite proxies `/search`, `/answer`, `/events`, `/health`, `/experiments` to the backend on `:8000`.

### Tests

```bash
PYTHONPATH=. pytest -q              # 26 passing
```

---

## Try it

The corpus is a Wikipedia sample, so ask about what's actually in it:

```bash
curl -s -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Zrínyi battleship","user_id":"u1"}'
```

Questions about the corpus (Valkyria Chronicles III, Egyptian deities, Zrínyi, Zapata, gold dollar, aircraft) retrieve well. Off-topic questions return low similarity scores (~0.2) — which the UI surfaces as low confidence rather than a confident wrong answer.

---

## Experimentation

Each `/search` call assigns a variant via `hash(experiment_id + ":" + user_id) % 10000` and logs an exposure. Thumbs-up/down post to `/events` as outcomes. `GET /experiments/{id}/results` computes the stats.

See the layer run end-to-end:

```bash
PYTHONPATH=. python experiments/simulation.py
```

Prints bucketing validation, an A/A false-positive check (≈ α), A/B detection power, SRM cases, and a full experiment table.

---

## Benchmarks

| Script | Measures |
|---|---|
| `bench/benchmark_engine.py` | int8 quantization: **4× memory reduction** (0.89 → 0.22 MB) |
| `bench/benchmark_hnsw.py` | HNSW recall vs latency across `efSearch` |
| `bench/benchmark_quantization.py` | float32 vs int8 recall + memory trade-off |
| `bench/benchmark_day9.py` | MRR@10, nDCG@10 — vector vs BM25 vs RRF hybrid |
| `bench/run_all_benchmarks.py` | runs them all |

---

## Deployment (free tier)

- `render.yaml` — Render free web service + free Postgres; build runs `pip install` + `bash build.sh`.
- `vercel.json` — static frontend (Vercel/Cloudflare Pages) with SPA fallback + API rewrites.
- `scripts/init_db.sql` — Postgres schema for experiments, variants, exposures, outcomes, results.
- `.github/workflows/stats.yml` — nightly stats job skeleton.

Set `VITE_API_URL` on the frontend build to point at the deployed API. Swap `StubProvider` in `api/llm_provider.py` for any free-tier LLM to make `/answer` real.

---

## Layout

```
/engine        pure-Python retrieval (no web deps)
/pipeline      chunking, embedding, artifact builder
/experiments   bucketing, stats, SRM, simulations
/api           FastAPI wiring + LLM provider
/web           Vite + React + TS + Tailwind frontend
/bench         reproducible benchmark scripts
/scripts       DB schema
/tests         pytest suite
/data          gitignored, produced by build.sh
```

---

## License

MIT
