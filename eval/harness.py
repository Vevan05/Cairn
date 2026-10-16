"""Offline evaluation metrics and harness.

Fills the `/eval` layer: given a labeled set of (query -> relevant chunk ids),
score any retrieval config with recall@k, MRR and nDCG@k.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k ranked results."""
    if not relevant:
        return 0.0
    top = ranked_ids[:k]
    return len(set(top) & relevant) / len(relevant)


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant result (0 if none)."""
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """Normalized discounted cumulative gain for binary relevance."""
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal: all relevant docs at the top
    ideal = sorted((1.0 for _ in range(min(len(relevant), k))), reverse=True)
    idcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    retrieval: str = "vector"   # "vector" | "hybrid"
    top_k: int = 10
    alpha: float = 0.6          # hybrid weight for vector side


@dataclass
class EvalReport:
    config: dict
    n_queries: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    per_query: list[dict] = field(default_factory=list)


def _rank_vector(query_vector, vectors, chunk_ids, k):
    """Rank chunk ids by cosine similarity (vectors L2-normalized)."""
    import numpy as np
    sims = vectors @ query_vector
    order = np.argsort(-sims)[:k]
    return [chunk_ids[i] for i in order], [float(sims[i]) for i in order]


def _rank_hybrid(query, query_vector, vectors, chunk_ids, bm25, k, alpha):
    """RRF fusion of vector + BM25 rankings."""
    import numpy as np
    from engine.search import rrf_fuse

    n = len(chunk_ids)
    # vector side
    sims = vectors @ query_vector
    vec_order = np.argsort(-sims)[: max(k * 3, k)]
    # bm25 side
    bm_order = bm25.search(query, top_k=max(k * 3, k))

    fused = rrf_fuse(vec_order, bm_order, k=60)
    return [chunk_ids[i] for i in fused[:k]], [0.0] * min(k, len(fused))


def run_eval(labeled: list[dict], chunks: list[dict], vectors, bm25, config: EvalConfig,
             embed_query, k: int | None = None) -> EvalReport:
    """Run a labeled set against one retrieval config.

    labeled: list of {"query": str, "relevant_chunk_ids": [str, ...]}
    """
    k = k or config.top_k
    chunk_ids = [c["chunk_id"] for c in chunks]

    recalls, rrs, ndcgs = [], [], []
    per_query = []

    for item in labeled:
        q = item["query"]
        relevant = set(item.get("relevant_chunk_ids", []))
        if not relevant:
            continue

        qv = embed_query(q)
        if config.retrieval == "hybrid" and bm25 is not None:
            ranked, _ = _rank_hybrid(q, qv, vectors, chunk_ids, bm25, k, config.alpha)
        else:
            ranked, _ = _rank_vector(qv, vectors, chunk_ids, k)

        r = recall_at_k(ranked, relevant, k)
        rr = reciprocal_rank(ranked, relevant)
        nd = ndcg_at_k(ranked, relevant, k)
        recalls.append(r)
        rrs.append(rr)
        ndcgs.append(nd)
        per_query.append({
            "query": q,
            "recall_at_k": round(r, 4),
            "reciprocal_rank": round(rr, 4),
            "ndcg_at_k": round(nd, 4),
            "hits": [cid for cid in ranked if cid in relevant],
        })

    n = len(recalls) or 1
    return EvalReport(
        config={"retrieval": config.retrieval, "top_k": k, "alpha": config.alpha},
        n_queries=len(recalls),
        recall_at_k=round(sum(recalls) / n, 4),
        mrr=round(sum(rrs) / n, 4),
        ndcg_at_k=round(sum(ndcgs) / n, 4),
        per_query=per_query,
    )
