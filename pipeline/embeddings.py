"""Embed chunks and queries with a small ONNX model; L2-normalize for cosine.

Uses `fastembed` (ONNX Runtime + Rust tokenizers) rather than
sentence-transformers. The deploy target is a 512 MB free tier, and importing
torch + transformers alone costs ~600 MB; the ONNX path keeps the whole
process (index included) under ~400 MB.

The model is loaded once and cached at module level — loading it per call is a
100x latency bug (re-reads weights + tokenizer each request).
"""

from __future__ import annotations

import numpy as np

# 384-dim, fast on CPU, and a standard retrieval baseline. Queries use the
# model's asymmetric query prefix (via `query_embed`); documents do not.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384

_MODEL_CACHE: dict = {}


def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        from fastembed import TextEmbedding
        _MODEL_CACHE[model_name] = TextEmbedding(model_name)
    return _MODEL_CACHE[model_name]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> np.ndarray:
    """Embed documents; returns an L2-normalized (N, D) float32 array."""
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    model = _get_model(model_name)
    vectors = np.asarray(list(model.embed(texts, batch_size=batch_size)), dtype=np.float32)
    return _l2_normalize(vectors)


def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed a single query; returns an L2-normalized (D,) float32 vector."""
    model = _get_model(model_name)
    vector = np.asarray(list(model.query_embed([query]))[0], dtype=np.float32)
    return _l2_normalize(vector)


def cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Assumes both are L2-normalized rows; returns dot products."""
    return corpus @ query
