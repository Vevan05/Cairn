"""Brute-force vector search and BM25 keyword search."""

from __future__ import annotations

import re
from array import array
from collections import defaultdict
from math import log

import numpy as np


def brute_force_search(
    query_vector: np.ndarray,
    vectors: np.ndarray,
    top_k: int = 10,
) -> np.ndarray:
    """Return indices of the top-k most similar vectors using cosine similarity.

    Assumes query_vector and all rows of vectors are L2-normalized.
    """
    if vectors.size == 0:
        return np.array([], dtype=np.int64)

    # Dot product gives cosine similarity when both are L2-normalized.
    sims = vectors @ query_vector  # shape: (N,)

    # Sort descending, get top-k indices
    k = min(top_k, len(sims))
    # Use argpartition for efficiency, then sort the top-k
    if k >= len(sims):
        indices = np.argsort(-sims)  # descending
    else:
        # Partition to find top-k, then sort those k
        indices = np.argpartition(sims, -k)[-k:]
        # Sort the selected k by similarity descending
        top_scores = sims[indices]
        order = np.argsort(-top_scores)
        indices = indices[order]

    return indices.astype(np.int64)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    # keep alphanumerics and apostrophes, replace everything else with space
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    tokens = text.split()
    # simple stopword filter to improve BM25 quality on general text
    stopwords = {
    "the", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "of", "in", "to", "for", "on", "at",
    "by", "from", "as", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "up", "down",
    "out", "off", "over", "under", "again", "further", "then", "once",
    }
    tokens = [t for t in tokens if t not in stopwords]
    return tokens


class BM25:
    """BM25 ranking model with an inverted index."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.N = len(documents)
        self.doc_lengths: np.ndarray = np.zeros(self.N, dtype=np.int64)
        self.avg_doc_len: float = 0.0
        self._term_doc_freq: dict[str, int] = defaultdict(int)  # df per term
        # term -> array('i') of interleaved [doc_idx, freq, doc_idx, freq, ...].
        # Compact C ints instead of (int, int) tuples: ~4 bytes vs ~80 bytes per
        # posting, which matters at scale (1M postings at 16.8k chunks).
        # Frequencies are computed at build time so search never re-tokenizes.
        self._postings: dict[str, array] = {}
        self._build()

    def _build(self):
        for di, doc in enumerate(self.documents):
            tokens = _tokenize(doc)
            self.doc_lengths[di] = len(tokens)
            # Compute per-document term frequencies once.
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            for tok, freq in tf.items():
                self._term_doc_freq[tok] += 1
                arr = self._postings.get(tok)
                if arr is None:
                    arr = array("i")
                    self._postings[tok] = arr
                arr.append(di)
                arr.append(freq)

        # Average document length
        if self.N > 0:
            self.avg_doc_len = float(self.doc_lengths.sum()) / self.N

    def search(self, query: str, top_k: int = 10) -> np.ndarray:
        """Return doc indices ranked by BM25 score for the given query string."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return np.array([], dtype=np.int64)

        scores = np.zeros(self.N, dtype=np.float64)

        for tok in query_tokens:
            df = self._term_doc_freq.get(tok, 0)
            idf = log((self.N - df + 0.5) / (df + 0.5) + 1.0)  # +1 smoothing

            postings = self._postings.get(tok)
            if postings is None:
                continue
            for i in range(0, len(postings), 2):
                di = postings[i]
                freq = postings[i + 1]
                doc_len = self.doc_lengths[di]
                # BM25 component for this term in this doc
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len) if self.avg_doc_len > 0 else 1.0
                score_contrib = idf * numerator / denominator
                scores[di] += score_contrib

        # Sort descending
        rank = np.argsort(-scores)
        k = min(top_k, self.N)
        return rank[:k].astype(np.int64)


def rrf_fuse(vector_results: np.ndarray, bm25_results: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion of two result lists.

    vector_results: indices ranked by vector search (best first)
    bm25_results: indices ranked by BM25 search (best first)
    Returns fused indices sorted by combined score.
    """
    # Build rank lookup for each list
    rank_vector = {idx: rank for rank, idx in enumerate(vector_results)}
    rank_bm25 = {idx: rank for rank, idx in enumerate(bm25_results)}

    # Collect all unique doc indices
    all_indices = set(vector_results) | set(bm25_results)

    # Compute RRF score per document
    scores: dict[int, float] = {}
    for idx in all_indices:
        r1 = rank_vector.get(idx, len(vector_results))
        r2 = rank_bm25.get(idx, len(bm25_results))
        scores[idx] = 1.0 / (k + r1) + 1.0 / (k + r2)

    # Sort descending by score
    fused = sorted(all_indices, key=lambda idx: scores[idx], reverse=True)
    return np.array(fused, dtype=np.int64)