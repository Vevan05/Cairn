# Cairn

A from-scratch RAG search engine + built-in A/B experimentation platform.
Pure Python engine, no vector-DB dependency, free-tier deployable.

## Components

- **Pipeline** `pipeline/` — chunker + sentence-transformers embedder → artifacts.
- **Engine** `engine/` — brute-force & HNSW nearest-neighbor, BM25, RRF fusion, int8 quantization.
- **Experiments** `experiments/` — deterministic bucketing, z-test/Welch's/SRM, idempotent event log.
- **API** `api/` — FastAPI: `/health`, `/search`, `/answer`, `/events`.
- **Bench** `bench/` — HNSW, quantization, MRR/nDCG reproducible benchmarks.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# build index
PYTHONPATH=. python pipeline/build.py --max-docs 500

# run
pytest
PYTHONPATH=. uvicorn api.main:app --port 8000
```

Sample API:
```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"what is the capital","user_id":"u1"}'
```

## Deployment (free)

- `render.yaml` — Render free web service + free Postgres.
- `vercel.json` — static frontend (Vercel or Cloudflare Pages).
- `docs/` — design notes.
- Replace `StubProvider` in `api/llm_provider.py` with any free-tier LLM for `/answer`.

## Benchmarks

- `bench/benchmark_engine.py` — approves int8 quantization 4x memory reduction (0.89 → 0.22 MB).
- `bench/benchmark_hnsw.py` — HNSW recall/latency trade-off vs efSearch.
- `bench/benchmark_quantization.py` — dequantized search 80-90% cosine retention.
- `bench/benchmark_day9.py` — MRR@10 + nDCG@10, vector vs BM25 vs RRF hybrid.
- `experiments/idempotent_logging.py` — idempotent logging + A/A false-positive (5%).

## Layout

```
/engine        pure-Python retrieval (no web deps)
/pipeline      chunking, embedding, artifact builder
/experiments   bucketing, stats, SRM
/api           FastAPI wiring + LLM provider
/bench         reproducible benchmark scripts & outputs
/data          (gitignored, produced by pipeline)
```

## License
MIT — see LICENSE (add one before publish).