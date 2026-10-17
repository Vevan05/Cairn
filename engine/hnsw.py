"""HNSW (Hierarchical Navigable Small World) graph index from scratch.

HNSW is an approximate nearest-neighbor graph structure that organizes vectors
into a multi-layer skip-list. Higher layers have fewer nodes and provide
long-range connections; the bottom layer (level 0) contains all nodes.

Key parameters:
- M: max connections per layer (typical 16-64)
- ef_construction: size of candidate list during graph build (affects build time/quality)
- ef_search: size of candidate list during search (affects query time/recall tradeoff)
- m_0: number of levels for the first node (usually 1)

Day 5-6 plan: multi-layer HNSW with proper neighbor-selection heuristic.
"""

from __future__ import annotations

import numpy as np
import random


class HNSWNode:
    """Node in the HNSW graph.

    Stores the vector, its level, and adjacency lists per layer.
    Adjacency[level] = list of neighbor node IDs at that layer.
    """

    __slots__ = ["vector", "id", "level", "adj"]

    def __init__(self, vector: np.ndarray, node_id: int, level: int):
        self.vector = vector  # L2-normalized float32 row (D,)
        self.id = node_id
        self.level = level  # this node's highest level (0 = only bottom layer)
        self.adj: list[list[int] | None] = [None] * (level + 1)


class HNSW:
    """HNSW index for approximate nearest-neighbor search.

    Full multi-layer skip-list implementation:
    - Each new node gets a random level P(level >= k) = 0.5^(k - m_0 + 1) for k >= m_0
    - Higher layers have fewer nodes; top layer typically has O(log N) nodes
    - Insertion: random level, greedy descent from entry_point, connect to M nearest neighbors
    - Search: descent from entry_point through upper layers (top-down), then best-first at level 0
    - The ef parameter during search controls the candidate list size, trading off recall vs. latency

    Distance: cosine similarity via L2-normalized dot products, vectorized with NumPy.
    """

    def __init__(
        self,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 100,
        m_0: int = 1,
        random_seed: int | None = 42,
    ):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m_0 = m_0

        if random_seed is not None:
            random.seed(random_seed)

        self._max_level: int = 0  # current max level in the entire graph
        self.entry_point: int | None = None  # node ID acting as entry point at top level
        self.nodes: dict[int, HNSWNode] = {}  # node_id -> HNSWNode
        self.vectors: np.ndarray | None = None  # (N, D) float32, indexed by node_id
        self._dim: int | None = None  # vector dimension, fixed on first insert

    # ---- Level randomisation ----

    @staticmethod
    def _random_level(m_0: int = 1, max_extra: int = 63) -> int:
        """Generate a random level for a new node.

        Probabilistic rule: P(level >= k) = 0.5^(k - m_0 + 1) for k >= m_0.
        We flip coins starting from level m_0 - 1 upward.
        """
        level = m_0 - 1
        while level < max_extra and random.random() < 0.5:
            level += 1
        return level

    # ---- Distance ----

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Dot product of L2-normalized vectors = cosine similarity."""
        return float(np.dot(a, b))

    # ---- Vectors helpers ----

    def _get_vector(self, node_id: int) -> np.ndarray:
        """Return the vector for a node_id."""
        if self.vectors is not None and 0 <= node_id < self.vectors.shape[0]:
            return self.vectors[node_id]
        # Node not materialised yet (e.g. the node currently being inserted):
        # fall back to a correctly-sized zero vector so similarity calls don't
        # raise on a dimension mismatch.
        d = self._dim or (self.vectors.shape[1] if self.vectors is not None and self.vectors.ndim >= 2 else 8)
        return np.zeros(d, dtype=np.float32)

    # ---- Insertion (multi-layer) ----

    def insert(self, vector: np.ndarray, node_id: int) -> None:
        """Insert a vector into the multi-layer HNSW graph.

        Standard insertion (Malkov & Yashunin):
        1. Normalise the vector and draw a random level.
        2. Greedily descend from the entry point to the new node's top level.
        3. For each level from min(new_level, max_level) down to 0:
             - run a best-first search (SEARCH-LAYER) to find the ef_construction
               nearest candidates,
             - pick the M nearest as neighbours, link bidirectionally,
             - prune neighbours that exceed M links.
        4. If the new node is taller than the entry point, it becomes the entry point.

        Linking to a *searched* candidate set (not just the entry point's
        immediate neighbours) is what keeps the graph connected.
        """
        vec = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        if self._dim is None:
            self._dim = int(vec.shape[0])

        new_level = self._random_level()
        new_node = HNSWNode(vec, node_id, new_level)

        # First node becomes the entry point.
        if self.entry_point is None:
            self.entry_point = node_id
            self._max_level = new_level
            self.nodes[node_id] = new_node
            self.vectors = vec.reshape(1, -1)
            return

        # Greedy descent from the current entry point down to the new node's level.
        ep = self.entry_point
        for level in range(self._max_level, new_level, -1):
            ep = self._greedy_descend(vec, ep, level)

        # Connect at each level the new node participates in.
        for level in range(min(new_level, self._max_level), -1, -1):
            candidates = self._search_layer(vec, ep, self.ef_construction, level)
            neighbors = self._select_neighbors(vec, candidates, self.M)
            if neighbors:
                new_node.adj[level] = list(neighbors)
                for nid in neighbors:
                    self._link(nid, node_id, level)
                ep = candidates[0][1]

        self.nodes[node_id] = new_node
        self.vectors = vec.reshape(1, -1) if self.vectors is None else np.vstack([self.vectors, vec])

        if new_level > self._max_level:
            self._max_level = new_level
            self.entry_point = node_id

    def _max_links(self, level: int) -> int:
        """Admission cap for a node's adjacency at `level`.

        HNSW allows Mmax0 = 2*M links at the base layer and M above it
        (Malkov & Yashunin, Algorithm 1). Capping level 0 at M as well prunes
        far more aggressively: every newly inserted node is evicted from its
        neighbours' lists, so the graph stays *weakly* connected while
        forward-reachable sets stay tiny — which caps recall and makes
        efSearch ineffective.
        """
        return self.M * 2 if level == 0 else self.M

    def _link(self, src: int, dst: int, level: int) -> None:
        """Add a link src->dst at `level`, pruning src back to its cap."""
        src_node = self.nodes[src]
        if src_node.adj[level] is None:
            src_node.adj[level] = []
        if dst not in src_node.adj[level]:
            src_node.adj[level].append(dst)
        cap = self._max_links(level)
        if len(src_node.adj[level]) > cap:
            vec = self._get_vector(src)
            candidates = [(self._cosine_similarity(vec, self._get_vector(n)), n) for n in src_node.adj[level]]
            src_node.adj[level] = self._select_neighbors(vec, candidates, cap)

    def _select_neighbors(self, base_vec: np.ndarray, candidates: list[tuple[float, int]], m: int) -> list[int]:
        """Select up to `m` neighbours using the HNSW diversity heuristic.

        A candidate is kept only if it is closer to `base_vec` than to any
        already-selected neighbour. This avoids the "everything connects to the
        same few hubs" clustering that hurts reachability; a simple M-nearest
        rule would leave the graph fragmented.
        """
        ordered = sorted(candidates, key=lambda x: -x[0])
        selected: list[int] = []
        for sim, nid in ordered:
            if len(selected) >= m:
                break
            nvec = self._get_vector(nid)
            if all(self._cosine_similarity(nvec, self._get_vector(s)) < sim for s in selected):
                selected.append(nid)
        # Backfill with the closest remaining if the heuristic was too strict.
        if len(selected) < m:
            for _, nid in ordered:
                if nid not in selected:
                    selected.append(nid)
                    if len(selected) >= m:
                        break
        return selected

    # ---- Search (multi-layer with ef parameter) ----

    def search(self, query_vector: np.ndarray, k: int = 10, ef: int | None = None) -> np.ndarray:
        """Search for k nearest neighbours using the multi-layer HNSW graph.

        Standard HNSW search:
        1. Greedy descent from the entry point through the upper layers
           (level = max_level .. 1), moving to the closest neighbour at each step.
        2. Best-first (beam) search confined to layer 0, bounded by `ef`.
        3. Return the top-k from the beam.

        `ef` is the beam width at layer 0: larger ef -> higher recall, slower.
        """
        if len(self.nodes) == 0:
            return np.array([], dtype=np.int64)

        q = np.asarray(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        search_ef = max(k, ef if ef is not None else self.ef_search)

        # 1. greedy descent through the upper layers
        current_id = self.entry_point
        for level in range(self._max_level, 0, -1):
            current_id = self._greedy_descend(q, current_id, level)

        # 2. best-first beam search at layer 0
        beam = self._search_layer(q, current_id, search_ef, level=0)

        # 3. top-k
        result = [nid for _, nid in beam[:k]]
        return np.array(result, dtype=np.int64)

    def _greedy_descend(self, q: np.ndarray, entry_id: int, level: int) -> int:
        """Repeatedly move to the closest neighbour at `level` until no improvement."""
        current = entry_id
        current_sim = self._cosine_similarity(q, self._get_vector(current))
        improved = True
        while improved:
            improved = False
            for nid in self._get_neighbors_at_level(current, level):
                sim = self._cosine_similarity(q, self._get_vector(nid))
                if sim > current_sim:
                    current, current_sim = nid, sim
                    improved = True
        return current

    def _search_layer(self, q: np.ndarray, entry_id: int, ef: int, level: int) -> list[tuple[float, int]]:
        """Best-first search within one layer, keeping the `ef` best candidates.

        Returns a list of (similarity, node_id) sorted by descending similarity.
        """
        import heapq

        visited = {entry_id}
        entry_sim = self._cosine_similarity(q, self._get_vector(entry_id))
        # candidates: max-heap by similarity (explore the closest first)
        candidates: list[tuple[float, int]] = [(-entry_sim, entry_id)]
        # results: min-heap by similarity (so the worst is at the top for eviction)
        results: list[tuple[float, int]] = [(entry_sim, entry_id)]

        while candidates:
            neg_sim, cid = heapq.heappop(candidates)
            sim = -neg_sim
            # stop if the closest unexpanded candidate is worse than our worst result
            if len(results) >= ef and sim < results[0][0]:
                break
            for nid in self._get_neighbors_at_level(cid, level):
                if nid in visited:
                    continue
                visited.add(nid)
                nsim = self._cosine_similarity(q, self._get_vector(nid))
                if len(results) < ef or nsim > results[0][0]:
                    heapq.heappush(candidates, (-nsim, nid))
                    heapq.heappush(results, (nsim, nid))
                    if len(results) > ef:
                        heapq.heappop(results)  # evict the worst

        return sorted(results, key=lambda x: -x[0])

    # ---- Neighbor access helpers ----

    def _get_neighbors_at_level(self, node_id: int, level: int) -> list[int]:
        """Return neighbor node IDs at the given level for a node.

        Returns empty list if node doesn't exist or has no neighbors at that level.
        """
        if node_id not in self.nodes:
            return []
        node = self.nodes[node_id]
        if level < 0 or level > node.level:
            return []
        adj = node.adj[level]
        return adj if adj is not None else []

    def size(self) -> int:
        """Return the number of vectors in the index."""
        return len(self.nodes)

    # ---- Persistence ----

    def save(self, path: str) -> None:
        """Serialize the HNSW index to disk.

        Saves:
        - vectors as .npy (N, D) float32
        - metadata JSON: M, ef_construction, ef_search, m_0, max_level, entry_point
        - adjacency: dict level -> dict node_id -> [neighbor_ids]
        """
        import json

        # Save vectors
        if self.vectors is not None:
            np.save(f"{path}_vectors.npy", self.vectors)

        # Save metadata
        metadata = {
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "m_0": self.m_0,
            "max_level": self._max_level,
            "entry_point": self.entry_point,
        }
        with open(f"{path}_metadata.json", "w") as f:
            json.dump(metadata, f)

        # Save adjacency: dict level -> dict node_id -> [neighbor_ids]
        adj_data: dict[int, dict[int, list[int]]] = {}
        for nid, node in self.nodes.items():
            for lvl in range(node.level + 1):
                if node.adj[lvl] is not None:
                    adj_data.setdefault(lvl, {})[nid] = node.adj[lvl]

        with open(f"{path}_adjacency.json", "w") as f:
            json.dump(adj_data, f)

        print(f"HNSW index saved to {path}")

    @classmethod
    def load(cls, path: str) -> HNSW:
        """Load an HNSW index from disk."""
        import json

        with open(f"{path}_metadata.json") as f:
            metadata = json.load(f)

        index = cls(
            M=metadata["M"],
            ef_construction=metadata.get("ef_construction", 40),
            ef_search=metadata["ef_search"],
            m_0=metadata.get("m_0", 1),
            random_seed=42,
        )
        index._max_level = metadata["max_level"]
        index.entry_point = metadata["entry_point"]
        index.vectors = np.load(f"{path}_vectors.npy")

        # Load adjacency
        with open(f"{path}_adjacency.json") as f:
            adj_data = json.load(f)

        # Rebuild nodes: first pass, determine each node's max level across all levels
        node_max_level: dict[int, int] = {}
        for lvl_str, nodes_at_level in adj_data.items():
            lvl = int(lvl_str)
            for nid_str in nodes_at_level:
                nid = int(nid_str)
                node_max_level[nid] = max(node_max_level.get(nid, lvl), lvl)

        # Second pass: create HNSWNode objects
        for nid, max_node_level in node_max_level.items():
            vec = index.vectors[nid] if index.vectors is not None and nid < len(index.vectors) else np.zeros(384, dtype=np.float32)
            index.nodes[nid] = HNSWNode(vec, nid, max_node_level)

        # Third pass: set adjacency at each level
        for lvl_str, nodes_at_level in adj_data.items():
            lvl = int(lvl_str)
            for nid_str, neighbors in nodes_at_level.items():
                nid = int(nid_str)
                if index.nodes[nid].adj[lvl] is None:
                    index.nodes[nid].adj[lvl] = []
                index.nodes[nid].adj[lvl].extend(neighbors)

        return index


# ---- Quick test / benchmark ----

if __name__ == "__main__":
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    import time

    np.random.seed(42)
    n = 500  # 500 vectors (larger than before to see multi-layer benefits)
    d = 384  # 384 dimensions

    # Create random vectors and L2-normalize
    vectors = np.random.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    vectors = vectors / norms

    # Build HNSW index
    print(f"Building HNSW index with {n} vectors, M={16}, ef_construction={40}, ef_search={40}")
    index = HNSW(M=16, ef_construction=40, ef_search=40, random_seed=42)

    t0 = time.time()
    for i in range(n):
        index.insert(vectors[i], i)
    t_build = time.time() - t0
    print(f"Build time: {t_build:.2f}s, index size: {index.size()} nodes, max_level={index._max_level}")

    # Test search: query with first vector
    query = vectors[0]
    t0 = time.time()
    indices = index.search(query, k=10)
    t_search = time.time() - t0
    print(f"Search time: {t_search*1000:.1f}ms for k=10")

    # Brute-force ground truth
    sims = cosine_similarity(query.reshape(1, -1), vectors.reshape(n, -1)).flatten()
    brute_top10 = np.argsort(-sims)[:10]
    print(f"Brute-force top-10: {brute_top10.tolist()}")

    # Recall@10
    recall = len(set(indices.tolist()).intersection(set(brute_top10.tolist())))
    print(f"Recall@10: {recall}/10 ({100 * recall / 10:.0f}%)")

    # Test with different efSearch values
    print("\n--- efSearch sensitivity ---")
    for ef in [10, 40, 100, 200]:
        index.ef_search = ef
        indices_ef = index.search(query, k=10)
        recall_ef = len(set(indices_ef.tolist()).intersection(set(brute_top10.tolist())))
        print(f"efSearch={ef}: Recall@10 = {recall_ef}/10 ({100*recall_ef/10:.0f}%)")

    # Save and reload
    index.save("/tmp/hnw_multi")
    index_loaded = HNSW.load("/tmp/hnw_multi")
    indices_rl = index_loaded.search(query, k=10)
    recall_rl = len(set(indices_rl.tolist()).intersection(set(brute_top10.tolist())))
    print(f"\nAfter reload - Recall@10: {recall_rl}/10")

    # Test a few different queries
    print("\n--- Multi-query recall ---")
    for q_idx in [0, 100, 250, 499]:
        qv = vectors[q_idx]
        hnsw_idx = index.search(qv, k=10)
        bf_idx = np.argsort(-cosine_similarity(qv.reshape(1, -1), vectors.reshape(n, -1)).flatten())[:10]
        overlap = len(set(hnsw_idx.tolist()).intersection(set(bf_idx.tolist())))
        print(f"Query {q_idx}: Recall@10 = {overlap}/10 ({100*overlap/10:.0f}%)")