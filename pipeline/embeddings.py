"""Embed chunks with a small sentence-transformers model; L2-normalize for cosine.

The model is loaded once and cached at module level. Loading it per call is a
100x latency bug (re-reads weights + tokenizer each request).
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_MODEL_CACHE: dict = {}


def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> np.ndarray:
    model = _get_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """Embed a single query string; returns a normalized (D,) float32 vector."""
    return embed_texts([query], model_name=model_name, batch_size=1)[0]


def cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Assumes both are L2-normalized rows; returns dot products."""
    return corpus @ query