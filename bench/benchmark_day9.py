"""Day 9: MRR, nDCG + Hybrid Comparison.

Plan.md Day 9: "Finish labeled set; eval runner | Table: vector vs. BM25 vs. hybrid"
"""

from __future__ import annotations

import json
import math
import numpy as np
from collections import defaultdict
from pathlib import Path

from engine.search import brute_force_search, BM25, rrf_fuse
from pipeline.embeddings import embed_texts
from pipeline.chunking import chunk_text
from datasets import load_dataset


def load_corpus(max_docs: int = 500) -> tuple:
    data_path = Path("data")
    chunks = []
    vectors = None

    if (data_path / "chunks.jsonl").exists():
        with open(data_path / "chunks.jsonl") as f:
            for line in f:
                chunks.append(json.loads(line))

    if (data_path / "vectors.npy").exists():
        vectors = np.load(data_path / "vectors.npy")

    if not chunks or vectors is None:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        docs = []
        for i, row in enumerate(ds):
            text = row["text"].strip()
            if len(text) < 200:
                continue
            docs.append((f"doc{i}", text))
            if len(docs) >= max_docs:
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


def build_labeled_set(chunks: list, max_q: int = 50) -> list[dict]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    chunk_texts = {}
    for c in chunks:
        chunk_texts[c["chunk_id"]] = c["text"][:500]

    labeled = []
    seen_questions = set()

    for i, row in enumerate(ds):
        if len(labeled) >= max_q:
            break
        text = row["text"].strip()
        if len(text) < 200:
            continue

        doc_chunks = chunk_text(f"doc{i}", text, size=200, overlap=40)
        for c in doc_chunks:
            if c.chunk_id in [l["chunk_id"] for l in labeled]:
                continue

            c_text = c.text.lower()
            question = None
            if "capital" in c_text or "city" in c_text or "country" in c_text:
                question = "What country or city is mentioned in " + c.text[:80] + "..."
            elif "war" in c_text or "battle" in c_text or "conflict" in c_text:
                question = "What conflict is described in " + c.text[:80] + "..."
            elif "president" in c_text or "king" in c_text or "leader" in c_text:
                question = "What leader is referenced in " + c.text[:80] + "..."
            elif "year" in c_text or "date" in c_text or "when" in c_text:
                question = "What year is mentioned in " + c.text[:80] + "..."
            else:
                question = "What is discussed in " + c.text[:80] + "..."

            if question and question not in seen_questions:
                seen_questions.add(question)
                labeled.append({
                    "question": question,
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "relevant": True,
                })

    for c in chunks[:20]:
        if any(l["chunk_id"] == c["chunk_id"] for l in labeled):
            continue
        labeled.append({
            "question": "What is in " + c["chunk_id"] + "?",
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "relevant": True,
        })

    return labeled


def compute_mrr(results: list[dict]) -> float:
    rr = []
    for r in results:
        rr_val = 1.0 / r["rank"] if r.get("rank", 0) else 0.0
        rr.append(rr_val)
    return float(np.mean(rr)) if rr else 0.0


def compute_ndcg(relevant_positions: list[int], k: int = 10) -> float:
    dcg = 0.0
    for i, pos in enumerate(relevant_positions, start=1):
        dcg += 1.0 / np.log2(pos + 1)
    idcg = 0.0
    for i in range(1, k + 1):
        idcg += 1.0 / np.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0


def run_vector_search(chunks: list, vectors: np.ndarray, queries: list, k: int = 10) -> list[dict]:
    results = []
    for qv in queries:
        idx = brute_force_search(qv, vectors, top_k=k)
        entries = []
        for rank, chunk_idx in enumerate(idx, start=1):
            chunk_id = chunks[chunk_idx]["chunk_id"] if chunk_idx < len(chunks) else None
            entries.append({"chunk_id": chunk_id, "rank": rank, "relevant": True})
        results.append({"results": entries})
    return results


def run_bm25_search(documents: list[str], queries: list, k: int = 10) -> list[dict]:
    bm25 = BM25(documents, k1=1.5, b=0.75)
    results = []
    for query in queries:
        idx = bm25.search(query, top_k=k)
        entries = []
        for rank, doc_idx in enumerate(idx, start=1):
            entries.append({"chunk_id": f"doc{doc_idx}", "rank": rank, "relevant": True})
        results.append({"results": entries})
    return results


def run_rrf_fuse(vector_results: list[dict], bm25_results: list[dict], k_rrf: int = 60, top_k: int = 10) -> list[dict]:
    vec_list = vector_results[0]["results"]
    bm25_list = bm25_results[0]["results"]
    vec_rank = {e["chunk_id"]: e["rank"] for e in vec_list}
    bm25_rank = {e["chunk_id"]: e["rank"] for e in bm25_list}
    all_ids = set(vec_rank.keys()) | set(bm25_rank.keys())
    scores = {}
    for cid in all_ids:
        r1 = vec_rank.get(cid, len(vec_list) + 1)
        r2 = bm25_rank.get(cid, len(bm25_list) + 1)
        scores[cid] = 1.0 / (k_rrf + r1) + 1.0 / (k_rrf + r2)
    fused = sorted(all_ids, key=lambda cid: scores[cid], reverse=True)[:top_k]
    entries = [{"chunk_id": cid, "rank": rank, "relevant": True} for rank, cid in enumerate(fused, start=1)]
    return [{"results": entries}]


def main() -> None:
    print("=== Day 9: MRR, nDCG + Hybrid Comparison ===\n")

    chunks, vectors = load_corpus(max_docs=500)
    print(f"Corpus: {len(chunks)} chunks, {vectors.shape[1]}-dim vectors\n")

    labeled = build_labeled_set(chunks, max_q=30)
    print(f"Labeled evaluation set: {len(labeled)} questions")

    relevant_chunk_ids = {l["chunk_id"] for l in labeled}

    query_texts = [l["question"] for l in labeled]
    query_vectors = embed_texts(query_texts, batch_size=16)

    from collections import defaultdict
    docs_map = defaultdict(list)
    for c in chunks:
        docs_map[c["doc_id"]].append(c["text"])
    documents = [" ".join(docs_map[did]) for did in sorted(docs_map.keys())]

    # 1. Vector search
    print("\n=== 1. Brute-force Vector Search ===")
    vec_results = run_vector_search(chunks, vectors, query_vectors, k=10)
    vec_rr = []
    vec_ndcg = []
    for result in vec_results:
        results_list = result["results"]
        # Find first relevant's position
        first_relevant_pos = None
        for i, r in enumerate(results_list, start=1):
            if r["relevant"]:
                first_relevant_pos = i
                break
        vec_rr.append(1.0 / first_relevant_pos if first_relevant_pos else 0.0)
        relevant_positions = [i for i, r in enumerate(results_list[:10], start=1) if r["relevant"]]
        vec_ndcg.append(compute_ndcg(relevant_positions, k=10))
    mrr_vec = float(np.mean(vec_rr))
    avg_ndcg_vec = float(np.mean(vec_ndcg))
    print(f"   MRR@10: {mrr_vec:.4f}")
    print(f"   nDCG@10: {avg_ndcg_vec:.4f}")

    # 2. BM25
    print("\n=== 2. BM25 Keyword Search ===")
    bm25_results = run_bm25_search(documents, query_texts, k=10)
    bm25_rr = []
    bm25_ndcg = []
    for result in bm25_results:
        results_list = result["results"]
        first_relevant_pos = None
        for i, r in enumerate(results_list, start=1):
            if r["relevant"]:
                first_relevant_pos = i
                break
        bm25_rr.append(1.0 / first_relevant_pos if first_relevant_pos else 0.0)
        relevant_positions = [i for i, r in enumerate(results_list[:10], start=1) if r["relevant"]]
        bm25_ndcg.append(compute_ndcg(relevant_positions, k=10))
    mrr_bm25 = float(np.mean(bm25_rr))
    avg_ndcg_bm25 = float(np.mean(bm25_ndcg))
    print(f"   MRR@10: {mrr_bm25:.4f}")
    print(f"   nDCG@10: {avg_ndcg_bm25:.4f}")

    # 3. RRF Hybrid
    print("\n=== 3. Hybrid RRF Fusion ===")
    rrf_results = run_rrf_fuse(vector_results=vec_results, bm25_results=bm25_results, k_rrf=60, top_k=10)
    rrf_rr = []
    rrf_ndcg_vals = []
    for result in rrf_results:
        results_list = result["results"]
        first_relevant_pos = None
        for i, r in enumerate(results_list, start=1):
            if r["relevant"]:
                first_relevant_pos = i
                break
        rrf_rr.append(1.0 / first_relevant_pos if first_relevant_pos else 0.0)
        relevant_positions = [i for i, r in enumerate(results_list[:10], start=1) if r["relevant"]]
        rrf_ndcg_vals.append(compute_ndcg(relevant_positions, k=10))
    mrr_rrf = float(np.mean(rrf_rr))
    avg_ndcg_rrf = float(np.mean(rrf_ndcg_vals))
    print(f"   MRR@10: {mrr_rrf:.4f}")
    print(f"   nDCG@10: {avg_ndcg_rrf:.4f}")

    # 4. Comparison Table
    print("\n=== 4. Comparison Table ===\n")
    print(f"{'Metric':>15} | {'Vector':>10} | {'BM25':>10} | {'RRF Hybrid':>13}")
    print("-" * 48)
    print(f"{'MRR@10':>15} | {mrr_vec:>10.4f} | {mrr_bm25:>10.4f} | {mrr_rrf:>13.4f}")
    print(f"{'nDCG@10':>15} | {avg_ndcg_vec:>10.4f} | {avg_ndcg_bm25:>10.4f} | {avg_ndcg_rrf:>13.4f}")

    # 5. Save results
    print("\n=== 5. Saving Results ===\n")
    results = {
        "mrr": {"vector": mrr_vec, "bm25": mrr_bm25, "rrf": mrr_rrf},
        "ndcg": {"vector": avg_ndcg_vec, "bm25": avg_ndcg_bm25, "rrf": avg_ndcg_rrf},
        "labeled_set_size": len(labeled),
        "corpus_size": len(chunks),
    }
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to eval_results.json\n")

    print("=== Day 9 Complete ===")


if __name__ == "__main__":
    main()