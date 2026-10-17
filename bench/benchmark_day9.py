"""Benchmark: MRR@10 and nDCG@10 for vector, BM25, and hybrid retrieval.

Uses the shared eval harness (`eval/harness.py`) so these numbers are computed
by exactly the same code that backs the `/eval` API.

The labeled set is corpus-derived (each query is the opening of a real chunk and
that chunk is the relevant document), so it exercises the harness end to end
rather than constituting a hand-verified benchmark set.
"""

from __future__ import annotations

import json
import time

import numpy as np

from engine.search import BM25
from pipeline.embeddings import embed_query
from eval import EvalConfig, run_eval


def load_index() -> tuple[list[dict], np.ndarray]:
    with open("data/chunks.jsonl") as f:
        chunks = [json.loads(line) for line in f]
    vectors = np.load("data/vectors.npy")
    return chunks, vectors


def build_labeled_set(chunks: list[dict], n: int = 100, prefix: int = 80) -> list[dict]:
    """Sample n chunks; the query is its opening text, the relevant id is itself."""
    step = max(1, len(chunks) // n)
    labeled: list[dict] = []
    for c in chunks[::step][:n]:
        text = c.get("text", "")
        if len(text) < prefix:
            continue
        labeled.append({"query": text[:prefix], "relevant_chunk_ids": [c["chunk_id"]]})
    return labeled


def main() -> None:
    print("=== Day 9: MRR, nDCG + Hybrid Comparison ===\n")

    chunks, vectors = load_index()
    bm25 = BM25([c["text"] for c in chunks])
    labeled = build_labeled_set(chunks, n=100)
    print(f"Corpus: {len(chunks)} chunks, {vectors.shape[1]}-dim vectors")
    print(f"Labeled evaluation set: {len(labeled)} queries (corpus-derived)\n")

    rows = []
    for retrieval in ("vector", "bm25", "hybrid"):
        t0 = time.perf_counter()
        rep = run_eval(
            labeled, chunks, vectors, bm25,
            EvalConfig(retrieval=retrieval, top_k=10), embed_query,
        )
        rows.append((retrieval, rep, time.perf_counter() - t0))

    print(f"{'Config':>8} | {'Recall@10':>9} | {'MRR@10':>8} | {'nDCG@10':>8} | {'secs':>6}")
    print("-" * 54)
    for name, rep, secs in rows:
        print(f"{name:>8} | {rep.recall_at_k:>9.4f} | {rep.mrr:>8.4f} | {rep.ndcg_at_k:>8.4f} | {secs:>6.1f}")

    results = {
        "corpus_size": len(chunks),
        "labeled_set_size": len(labeled),
        "recall_at_10": {name: rep.recall_at_k for name, rep, _ in rows},
        "mrr": {name: rep.mrr for name, rep, _ in rows},
        "ndcg_at_10": {name: rep.ndcg_at_k for name, rep, _ in rows},
    }
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nEvaluation results saved to eval_results.json")
    print("=== Day 9 Complete ===")


if __name__ == "__main__":
    main()
