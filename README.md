# Cairn

A from-scratch RAG search engine with a built-in A/B experimentation platform.
Pure-Python retrieval (no vector-DB dependency), a React UI, and a free-tier deploy path.

**Status:** feature-complete — every phase of the build plan implemented and tested (**36 tests passing**).

---

## What it does

Cairn answers two questions:

1. **Retrieval** — given a query, which stored passages are most relevant? (HNSW + BM25 + RRF fusion, no external vector DB)
2. **Experimentation** — which retrieval configuration is *statistically* better? (deterministic bucketing, z-tests, Sample Ratio Mismatch detection)

The UI has four tabs:

| Tab | What it does |
|---|---|
| **Search** | Ranks the closest corpus passages by cosine similarity, with a per-result score + confidence bar. No LLM. |
| **Answer** | Streams a cited answer token-by-token; clicking a citation highlights and scrolls to its source chunk. |
| **Experiments** | A/B dashboard **and admin console** — create/activate/pause/delete experiments, edit variants and auto-balanced weights, run a traffic simulator, read z-tests + SRM. |
| **Eval** | Offline eval harness — score vector vs hybrid on a labeled set with recall@k, MRR, nDCG@k. |

Plus a **landing page** at `/` and a **docs page** at `/docs` (how to simulate experiments).

---

## Results

| Metric | Value | Source |
|---|---|---|
| int8 quantization memory | **0.89 MB → 0.22 MB** (**4×**) | `bench/benchmark_quantization.py` |
| int8 vs float32, brute-force recall@10 | **60–80%** overlap | same |
| int8 avg cosine similarity (dequantized) | **0.684** | same |
| HNSW recall@10 (M=32, ef₍construction₎=200) | **~0.98** on corpus queries | `bench/benchmark_hnsw.py` |
| HNSW query latency (610 vectors) | **~0.5 ms** (≈ brute force at this scale) | same |
| Retrieval — vector: MRR / nDCG@10 | **0.855 / 0.890** | Eval tab / `/eval/run` |
| Retrieval — hybrid: MRR / nDCG@10 | **0.975 / 0.982** | same |
| Bucketing split (20k users, 50/50 config) | **49.41% / 50.59%** | `experiments/simulation.py` |
| Cross-experiment bucket correlation | **+0.007** (want ~0) | same |
| A/A false-positive rate (α = 0.05) | **4.3%** | same |
| A/B detection power (+2pp, 3k/arm) | **0.73** (slightly underpowered) | same |
| Full-experiment result | control **9.95%** vs treatment **12.73%** thumbs-up, **+27.9%** relative, z = −6.20 → *treatment better* | same |
| SRM detection | 50/50 ok; 52/48, 60/40, 90/10 all flagged | same |

> At ~600 vectors, HNSW latency is comparable to brute force — pure-Python graph traversal overhead dominates. The win appears at 10k+ vectors, where brute force is O(N) per query (see Plan.md).

---

## Components

| Path | Role |
|---|---|
| `pipeline/` | chunking, sentence-transformers embeddings, artifact builder |
| `engine/` | brute-force & HNSW NN, BM25, RRF fusion, int8 quantization (pure Python) |
| `experiments/` | SHA-256 bucketing, z-test / Welch's t / χ² SRM, idempotent logging, simulations |
| `eval/` | offline metrics (recall@k, MRR, nDCG) + labeled-set harness |
| `api/` | FastAPI: `/health`, `/search`, `/answer(+/stream)`, `/events`, `/experiments` (+CRUD/+activate/+simulate), `/eval` |
| `web/` | Vite + React + TypeScript + Tailwind UI |
| `bench/` | reproducible benchmarks (HNSW, quantization, MRR/nDCG) |
| `tests/` | pytest suite (36 tests) |
| `scripts/` | `init_db.sql` for the Postgres schema |
| `docs/` | `simulation.md` guide |

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

Vite proxies `/search`, `/answer`, `/events`, `/health`, `/experiments`, `/eval` to the backend on `:8000`.

### Tests

```bash
PYTHONPATH=. pytest -q              # 36 passing
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

Each `/search` call assigns a variant via `SHA-256(experiment_id + ":" + user_id) % 10000` and logs an exposure. Thumbs up/down post to `/events` as outcomes. `GET /experiments/{id}/results` computes the stats.

- **Active experiment** — only one experiment serves traffic at a time; `POST /experiments/{id}/activate` switches it (and pauses the previous).
- **User-scoped outcomes** — the unit of analysis is the user: one exposure and at most one vote per user per variant, so `outcomes ≤ exposures` always holds.
- **Admin console** — create/pause/delete experiments with server-side validation (≥ 2 variants, unique names, positive weights). Splits auto-balance to 100 (typing 70 sets the other to 30).
- **Traffic simulator** — `POST /experiments/{id}/simulate` fills an experiment with synthetic users so the z-test and SRM checks actually run (see below).

See the layer run end-to-end:

```bash
PYTHONPATH=. python experiments/simulation.py
```

Prints bucketing validation, an A/A false-positive check (≈ α), A/B detection power, SRM cases, and a full experiment table.

### Simulating traffic

The admin console has a **Simulate traffic** row (users + lift). It generates distinct users, buckets each with the real hashing function, and logs exposures + thumbs with a configurable lift — through the same helpers as live traffic. Full guide: [`docs/simulation.md`](docs/simulation.md) or `/docs` in the app.

```bash
curl -s -X POST localhost:8000/experiments/demo/simulate \
  -H 'Content-Type: application/json' \
  -d '{"n_users":1000,"treatment_lift":0.10,"seed":1}'
```

---

## Offline evaluation

The Eval tab scores any config on a labeled set (`question → relevant chunk ids`) and returns a recall@k / MRR / nDCG@k table for **vector vs hybrid**. Seed a corpus-derived sample set from the UI, or paste your own JSON.

API: `GET /eval/sets`, `POST /eval/sets`, `POST /eval/sets/sample`, `POST /eval/run`, `GET /eval/runs`.

---

## Benchmarks

| Script | Measures |
|---|---|
| `bench/benchmark_engine.py` | brute-force / BM25 / RRF sanity + int8 memory |
| `bench/benchmark_hnsw.py` | HNSW recall@10 and latency across `efSearch` |
| `bench/benchmark_quantization.py` | float32 vs int8 recall + memory trade-off |
| `bench/benchmark_day9.py` | MRR@10, nDCG@10 — vector vs BM25 vs RRF hybrid |
| `bench/run_all_benchmarks.py` | runs them all |

---

## Deployment (free tier)

- `render.yaml` — Render free web service + free Postgres; build runs `pip install` + `bash build.sh`.
- `vercel.json` — static frontend (Vercel/Cloudflare Pages) with SPA fallback + API rewrites.
- `scripts/init_db.sql` — Postgres schema for experiments, variants, exposures, outcomes, results.

Set `VITE_API_URL` on the frontend build to point at the deployed API. Swap `StubProvider` in `api/llm_provider.py` for any free-tier LLM to make `/answer` real.

> A scheduled stats job (GitHub Actions cron reading exposures and writing the `results` table) is a planned addition — the schema and stats functions are ready, but no workflow is committed yet.

---

## Layout

```
/engine        pure-Python retrieval (no web deps)
/pipeline      chunking, embedding, artifact builder
/experiments   bucketing, stats, SRM, simulations
/eval          offline metrics + labeled-set harness
/api           FastAPI wiring + LLM provider
/web           Vite + React + TS + Tailwind frontend
/bench         reproducible benchmark scripts
/scripts       DB schema
/tests         pytest suite
/docs          design notes + simulation guide
/data          gitignored, produced by build.sh
```

---

## License

MIT
