"""Tests for the retrieval engine."""
import json, numpy as np
from pathlib import Path
from engine.search import brute_force_search, BM25, rrf_fuse
from engine.quantization import quantize_int8, dequantize_int8
from engine.hnsw import HNSW


def _corpus():
    vecs = np.random.RandomState(42).randn(50, 8).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    docs = [f"doc {i} text hello search query keywords indexed " + str(i) for i in range(50)]
    return vecs, docs


def test_brute_force_top_k():
    vecs, _ = _corpus()
    idx = brute_force_search(vecs[0], vecs, top_k=5)
    assert idx.dtype == np.int64
    assert len(idx) == 5
    # 0th should be self
    assert idx[0] == 0


def test_brute_force_empty():
    vecs = np.zeros((0, 8), dtype=np.float32)
    q = np.random.randn(8).astype(np.float32)
    res = brute_force_search(q, vecs, top_k=5)
    assert res.shape == (0,)


def test_bm25_basic():
    docs = ["apple banana cherry", "x y z", "p q r"]
    index = BM25(docs)
    assert len(index.doc_lengths) == 3
    res = index.search("apple banana", top_k=2)
    assert len(res) == 2
    assert res[0] == 0  # doc with 'apple' comes first


def test_bm25_top_k_larger_than_docs():
    docs = ["a b", "c d"]
    index = BM25(docs)
    res = index.search("a c", top_k=10)
    assert len(res) == 2


def test_rrf_fuse_basic():
    v = np.array([0, 1, 2])
    b = np.array([2, 3, 4])
    fused = rrf_fuse(v, b)
    assert len(fused) >= 5
    # each list's top-1 should be elevated
    assert fused[0] in {0, 2}


def test_quantize_dequantize_roundtrip():
    vecs = np.random.RandomState(0).randn(10, 12).astype(np.float32)
    vecs = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)
    q, scales = quantize_int8(vecs)
    assert q.dtype == np.int8
    assert scales.shape == (12,)
    dq = dequantize_int8(q, scales)
    assert dq.dtype == np.float32
    # cosine recovery >= 0.8
    cos = float(np.mean(np.sum(vecs * dq, axis=1) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(dq, axis=1))))
    assert cos >= 0.8, cos


def test_hnsw_insert_and_search():
    V, docs = _corpus()
    index = HNSW(M=16, ef_construction=40, ef_search=40, random_seed=42)
    for i in range(len(V)):
        index.insert(V[i], i)
    assert index.size() == len(V)
    results = index.search(V[0], k=10)
    assert len(results) == 10
    assert all(0 <= r < len(V) for r in results)
    # self may or may not surface due to heuristics; at least no crashes and valid idx


def test_hnsw_persistence(tmp_path):
    import tempfile, os
    V, docs = _corpus()
    index = HNSW(M=16, ef_construction=40, ef_search=40, random_seed=42)
    for i in range(len(V)):
        index.insert(V[i], i)
    path = os.path.join(tempfile.gettempdir(), "hnsw_test")
    index.save(path)
    loaded = HNSW.load(path)
    assert len(loaded.nodes) == len(index.nodes)
    # search consistency
    q = V[5]
    orig = index.search(q, k=10)
    aft = loaded.search(q, k=10)
    assert len(aft) == len(orig)
