"""Benchmark: float32 vs int8 quantization recall + memory tradeoff.

Day 8 phase: scalar quantization memory/recall comparison.
"""

from __future__ import annotations

import numpy as np
import time
from engine.hnsw import HNSW
from engine.search import brute_force_search
from engine.quantization import quantize_int8, dequantize_int8
from pipeline.embeddings import embed_texts
from pipeline.chunking import chunk_text
from datasets import load_dataset
import json
from pathlib import Path


def load_corpus(max_docs: int = 500) -> tuple:
    """Load or build the corpus."""
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
        docs: list[tuple[str, str]] = []
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


def main() -> None:
    print("=== Day 8: Quantization Memory/Recall Benchmark ===\n")

    # Load corpus
    chunks, vectors = load_corpus(max_docs=500)
    print(f"Corpus: {len(chunks)} chunks, {vectors.shape[1]}-dim vectors\n")

    # ============================================================
    # 1. Quantization metrics (the core Day 8 deliverable)
    # ============================================================
    print("=== 1. Quantization Metrics ===")

    q_vectors, scales = quantize_int8(vectors)
    dq_vectors = dequantize_int8(q_vectors, scales)

    original_mem = vectors.nbytes / (1024 * 1024)
    quantized_mem = q_vectors.nbytes / (1024 * 1024)
    mem_reduction = original_mem / quantized_mem

    # Cosine similarity preservation at brute force level
    sims_orig = np.sum(vectors * dq_vectors, axis=1) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(dq_vectors, axis=1)
    )
    avg_cos_sim = float(np.mean(sims_orig))

    print(f"Original vectors: float32, {original_mem:.2f} MB")
    print(f"Quantized vectors: int8, {quantized_mem:.2f} MB")
    print(f"Memory reduction: {mem_reduction:.1f}x")
    print(f"Avg cosine similarity (brute force): {avg_cos_sim:.4f}\n")

    # ============================================================
    # 2. Recall benchmark: brute force on original vs dequantized
    # ============================================================
    print("=== 2. Recall@k: Original vs Dequantized (Brute Force) ===\n")

    k = 10
    query_vectors = vectors[:3]  # 3 query vectors

    # Brute force on original float32 (ground truth)
    bf_original_sets: list[set] = []
    for qv in query_vectors:
        idx = brute_force_search(qv, vectors, top_k=k)
        bf_original_sets.append(set(idx.tolist()))

    # Brute force on dequantized int8 vectors
    bf_dequant_sets: list[set] = []
    for i, qv in enumerate(query_vectors):
        dq_qv = dq_vectors[i]
        idx = brute_force_search(dq_qv, dq_vectors, top_k=k)
        bf_dequant_sets.append(set(idx.tolist()))

    print("Query | Orig BF Recall@10 | Dequant BF Recall@10 | Overlap")
    print("-" * 45)
    overlaps = []
    for i in range(len(query_vectors)):
        orig_set = bf_original_sets[i]
        deq_set = bf_dequant_sets[i]
        overlap = len(orig_set.intersection(deq_set))
        recall_orig = 1.0  # by definition, since it's the ground truth
        recall_deq = overlap / k
        overlaps.append(overlap)
        print(f"  {i+1:3d}       | {recall_orig:15.3f} | {recall_deq:19.3f} | {overlap}/10 ({100*overlap/k:.0f}%)")

    # ============================================================
    # 3. HNSW on dequantized: recall + latency
    # ============================================================
    print("\n=== 3. HNSW on Dequantized Vectors ===\n")

    index = HNSW(M=16, ef_construction=40, ef_search=40, random_seed=42)
    for i in range(len(dq_vectors)):
        index.insert(dq_vectors[i], i)

    print(f"HNSW index built: {index.size()} nodes, max_level={index._max_level}\n")

    # Test at different efSearch values
    ef_values = [40, 100, 200]
    hnsw_recall: list[dict] = []

    for ef in ef_values:
        index.ef_search = ef
        total_recall = 0.0
        total_latency = 0.0

        for i in range(len(query_vectors)):
            qv_dq = dq_vectors[i]
            start = time.perf_counter()
            hnsw_idx = index.search(qv_dq, k=k)
            latency = (time.perf_counter() - start) * 1000

            # Recall against brute force dequantized ground truth
            deq_set = bf_dequant_sets[i]
            hnsw_set = set(hnsw_idx.tolist())
            overlap = len(hnsw_set.intersection(deq_set))
            recall = overlap / k

            total_recall += recall
            total_latency += latency

        avg_recall = total_recall / len(query_vectors)
        avg_latency = total_latency / len(query_vectors)
        hnsw_recall.append({"ef": ef, "recall": avg_recall, "latency": avg_latency})
        print(f"efSearch={ef}: Recall@10 = {avg_recall:.3f} ({100*avg_recall:.0f}%), "
              f"latency = {avg_latency:.1f} ms")

    # ============================================================
    # 4. Summary table
    # ============================================================
    print("\n=== 4. Summary: float32 vs int8 Tradeoff ===\n")
    print(f"{'Metric':>20} | {'float32':>12} | {'int8':>12} | {'Change':>10}")
    print("-" * 55)
    print(f"{'Memory (MB)':>20} | {original_mem:>12.2f} | {quantized_mem:>12.2f} | {-100/mem_reduction:>9.1f}%")
    print(f"{'Avg cosine sim':>20} | {avg_cos_sim:>12.4f} | {avg_cos_sim:>12.4f} | {0:>9.0f}%")
    # HNSW recall summary
    best_hnsw_recall = hnsw_recall[1]["recall"] if len(hnsw_recall) > 1 else hnsw_recall[0]["recall"]
    print(f"{'HNSW Recall@10':>20} | {'-':>12} | {'-':>12} | {best_hnsw_recall*100:>9.1f}%")

    # ============================================================
    # 5. Save quantization metadata
    # ============================================================
    print("\n=== 5. Saving Metadata ===\n")
    meta = {
        "corpus_size": len(chunks),
        "vector_dim": vectors.shape[1],
        "original_dtype": str(vectors.dtype),
        "quantized_dtype": "int8",
        "memory_reduction_x": float(mem_reduction),
        "avg_cosine_sim_brute_force": float(avg_cos_sim),
        "hnsw_recall_at_ef100": best_hnsw_recall,
        "ef_search_tested": ef_values,
    }
    with open("quantization_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Quantization metadata saved to quantization_meta.json\n")

    # ============================================================
    # 6. Draft evaluation questions
    # ============================================================
    print("=== 6. Draft Evaluation Questions ===\n")
    print("These questions can be expanded from 5 to 50-100 pairs for the full eval set:")
    print("1. What is the capital of the main country discussed in the first document?")
    print("2. Describe the main conflict in the second document's topic area.")
    print("3. Which year is mentioned most frequently across the corpus?")
    print("4. What are the key terms associated with machine learning in the documents?")
    print("5. Compare the approaches taken in documents 1 and 2 on the given topic.")
    print("6. Identify the geographic region referenced in document 3.")
    print("7. What time period is most commonly referenced in the corpus?")
    print("8. Name a key figure or entity mentioned in document 4.")
    print("9. What contradiction exists between documents 2 and 5?")
    print("10. Synthesize the main argument across documents 3-7.\n")

    print("=== Day 8 Complete ===")


if __name__ == "__main__":
    main()