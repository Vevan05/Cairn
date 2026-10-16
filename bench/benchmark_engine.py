"""Benchmark script for the Cairn engine: brute-force search, BM25, RRF fusion, and quantization."""

from __future__ import annotations

import numpy as np
import time

from pipeline.chunking import chunk_text
from pipeline.embeddings import embed_texts, cosine_similarity
from engine.search import brute_force_search, BM25, rrf_fuse
from engine.quantization import quantize_int8, dequantize_int8


def main() -> None:
    # --- Load the existing data ---
    from pathlib import Path
    data_path = Path("data")
    chunks = []
    with open(data_path / "chunks.jsonl") as f:
        for line in f:
            chunks.append(__import__("json").loads(line))

    vectors = np.load(data_path / "vectors.npy")
    print(f"Loaded {len(chunks)} chunks and vectors shape {vectors.shape}")

    # --- Day 2 style: Brute-force vector search ---
    # Pick a random query from the corpus
    query_idx = 0
    query_vector = vectors[query_idx]
    # Ensure query is also normalized (the embeddings model normalizes, but double-check)
    # cosine_similarity expects L2-normalized vectors; our embeddings.py normalizes.

    start = time.perf_counter()
    brute_indices = brute_force_search(query_vector, vectors, top_k=10)
    brute_time = time.perf_counter() - start

    print("\n=== Brute-force vector search ===")
    print(f"Top 10 indices: {brute_indices.tolist()}")
    print(f"Top 10 similarities: {np.max(vectors @ query_vector):.4f} max, "
          f"{np.sort(vectors @ query_vector)[-10:][::-1]}")
    print(f"Brute-force search time: {brute_time*1000:.1f} ms")

    # Show top result chunk text
    top_idx = brute_indices[0]
    print(f"\nTop result chunk text: {chunks[top_idx]['text'][:120]}...")

    # --- Day 3 style: BM25 keyword search ---
    # Rebuild documents from chunks for BM25
    # Join chunk text per doc_id
    from collections import defaultdict
    docs_map = defaultdict(list)
    for c in chunks:
        docs_map[c["doc_id"]].append(c["text"])

    documents = []
    doc_ids = []
    for doc_id in sorted(docs_map.keys()):
        full_text = " ".join(docs_map[doc_id])
        documents.append(full_text)
        doc_ids.append(doc_id)

    print(f"\n=== BM25 index built from {len(documents)} documents ===")

    bm25 = BM25(documents, k1=1.5, b=0.75)

    query = "what is the capital of france"  # example query
    # Actually, let's use a real chunk's text as query for a fair comparison
    query_text = chunks[0]["text"]
    bm25_indices = bm25.search(query_text, top_k=10)

    print("\n=== BM25 keyword search ===")
    print(f"Top 10 doc indices: {bm25_indices.tolist()}")
    for i, idx in enumerate(bm25_indices):
        print(f"  {i+1}. doc {idx}: {documents[idx][:80]}...")

    # --- RRF Fusion ---
    print("\n=== RRF Fusion ===")
    fused = rrf_fuse(brute_indices, bm25_results=bm25_indices, k=60)
    print(f"Fused top 10 indices: {fused.tolist()}")
    for i, idx in enumerate(fused):
        # Map doc index back to chunk
        doc_id = doc_ids[idx] if idx < len(doc_ids) else "?"
        chunk_text_sample = chunks[0]["text"][:40]  # fallback
        print(f"  {i+1}. doc {idx} ({doc_id}): mapping needed")

    # --- Quantization ---
    print("\n=== Quantization ===")
    original_norm = np.linalg.norm(vectors[0])
    print(f"Original vector norm: {original_norm:.4f}")

    q_vectors, scales = quantize_int8(vectors)
    print(f"Quantized vector dtype: {q_vectors.dtype}, min={q_vectors.min()}, max={q_vectors.max()}")

    dequanted = dequantize_int8(q_vectors, scales)
    dequant_norm = np.linalg.norm(dequanted[0])
    print(f"Dequantized vector norm: {dequant_norm:.4f}")

    # Recall after dequantization
    # Recompute similarity with dequantized vectors (both 1D, dot product = cosine if normalized)
    dequanted_norm = np.linalg.norm(dequanted[0])
    query_norm = np.linalg.norm(query_vector)
    cos_sim = float(np.dot(query_vector, dequanted[0]) / (query_norm * dequanted_norm))
    print(f"Cosine similarity with dequantized: {cos_sim:.4f}")

    # Memory savings
    original_mem = vectors.nbytes / (1024 * 1024)
    quantized_mem = q_vectors.nbytes / (1024 * 1024)
    print(f"Memory: original {original_mem:.1f} MB, quantized {quantized_mem:.1f} MB, "
          f"{original_mem/quantized_mem:.1f}x reduction")

    print("\n=== All engine benchmarks completed ===")


if __name__ == "__main__":
    main()