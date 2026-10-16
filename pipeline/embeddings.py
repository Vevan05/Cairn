"""Embed chunks with a small sentence-transformers model; L2-normalize for cosine."""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


def cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Assumes both are L2-normalized rows; returns dot products."""
    return corpus @ query
