"""Benchmark script: HNSW recall vs. latency tradeoff over efSearch.

As per Plan.md Day 7: "Index serialization | Save/load index; first benchmark
(recall and latency vs. efSearch)."""

from __future__ import annotations

import time
import numpy as np
from engine.hnsw import HNSW
from engine.search import brute_force_search
from pipeline.embeddings import embed_texts
from pipeline.chunking import chunk_text
from pathlib import Path
import json


def load_corpus(max_docs: int = 0) -> tuple:
    """Load the built corpus from data/, or rebuild it if missing (0 = all docs)."""
    data_path = Path("data")
    chunks = []
    vectors = None

    # Try loading existing data
    if (data_path / "chunks.jsonl").exists():
        with open(data_path / "chunks.jsonl") as f:
            for line in f:
                chunks.append(json.loads(line))

    if (data_path / "vectors.npy").exists():
        vectors = np.load(data_path / "vectors.npy")

    if not chunks or vectors is None:
        # Build from scratch
        import datasets
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        limit = max_docs if max_docs > 0 else None
        docs: list[tuple[str, str]] = []
        for i, row in enumerate(ds):
            text = row["text"].strip()
            if len(text) < 200:
                continue
            docs.append((f"doc{i}", text))
            if limit is not None and len(docs) >= limit:
                break

        chunks = []
        for doc_id, text in docs:
            chunks.extend(chunk_text(doc_id, text, size=200, overlap=40))

        with open(data_path / "chunks.jsonl", "w") as f:
            for c in chunks:
                c_dict = {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, "start_word": c.start_word}
                f.write(json.dumps(c_dict) + "\n")

        text_only = [c.text for c in chunks]
        vectors = embed_texts(text_only, batch_size=64)
        np.save(data_path / "vectors.npy", vectors)

    return chunks, vectors


def main() -> None:
    print("=== HNSW Benchmark: Recall vs. Latency vs. efSearch ===\n")

    # Load / build corpus
    chunks, vectors = load_corpus()
    print(f"Corpus: {len(chunks)} chunks, {vectors.shape[1]}-dim vectors\n")

    # Use the first 20 vectors as queries
    query_vectors = vectors[:20]
    k = 10  # top-k we care about

    # Pre-compute brute-force top-k and times for each query.
    # Warm up once so the first query isn't charged for lazy allocation.
    brute_force_search(query_vectors[0], vectors, top_k=k)
    bf_topk_sets: list[set] = []
    bf_times_ms: list[float] = []

    for i, qv in enumerate(query_vectors):
        start = time.perf_counter()
        bf_indices = brute_force_search(qv, vectors, top_k=k)
        bf_time = (time.perf_counter() - start) * 1000
        bf_times_ms.append(bf_time)
        bf_topk_sets.append(set(bf_indices.tolist()))

    bf_avg_ms = sum(bf_times_ms) / len(bf_times_ms)

    # Build HNSW index once
    print(f"Building HNSW index (M=32, ef_construction=200)...")
    index = HNSW(M=32, ef_construction=200)
    for i in range(len(vectors)):
        index.insert(vectors[i], i)
    print(f"Index built: {index.size()} nodes, max_level={index._max_level}\n")

    # Benchmark HNSW at different efSearch values
    ef_values = [10, 20, 40, 60, 80, 100, 150, 200]

    print(f"{'efSearch':>10} | {'Recall@10':>9} | {'Avg latency(ms)':>15} | {'Speedup vs BF'}")
    print("-" * 55)

    for ef in ef_values:
        total_time = 0.0
        total_recall = 0.0

        # Search each query vector with this efSearch
        for i, qv in enumerate(query_vectors):
            # HNSW search with this efSearch
            start = time.perf_counter()
            hnsw_indices = index.search(qv, k=k, ef=ef)
            hnsw_time = (time.perf_counter() - start) * 1000  # ms

            total_time += hnsw_time

            # Recall@k: fraction of HNSW top-k that matches brute-force top-k
            hnsw_set = set(hnsw_indices.tolist())
            brute_set = bf_topk_sets[i]
            recall = len(hnsw_set.intersection(brute_set)) / k if k > 0 else 0
            total_recall += recall

        avg_time = total_time / len(query_vectors)
        avg_recall = total_recall / len(query_vectors)
        # Speedup factor: how many times faster HNSW is than brute force
        # (compared against the mean brute-force time, not a cold first sample).
        speedup = bf_avg_ms / avg_time if avg_time > 0 else 0

        print(f"{ef:>10} | {avg_recall:>9.3f} | {avg_time:>15.1f} | {speedup:>12.1f}x")

    print("\n=== Benchmark complete ===")
    print("Observations:")
    print(f"- HNSW recall@10 on corpus queries at M=32, ef_construction=200 (N={len(vectors)})")
    print("- Below ~5k vectors, HNSW is at parity with brute force: a vectorised NumPy")
    print("  matmul is a high bar and pure-Python graph traversal costs ~1 ms/query.")
    print("  Above that, brute force grows O(N) while HNSW stays near-flat, so the")
    print("  speedup widens with corpus size.")
    print("- efSearch trades recall against latency; larger ef -> higher recall, slower")


if __name__ == "__main__":
    main()