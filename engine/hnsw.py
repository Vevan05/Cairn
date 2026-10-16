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
        M: int = 16,
        ef_construction: int = 40,
        ef_search: int = 40,
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
        d = self.vectors.shape[1] if self.vectors is not None and self.vectors.ndim >= 2 else 8
        return np.zeros(d, dtype=np.float32)

    # ---- Insertion (multi-layer) ----

    def insert(self, vector: np.ndarray, node_id: int) -> None:
        """Insert a new vector into the HNSW multi-layer graph.

        Steps (following the standard HNSW insertion algorithm from Malkov & Yashunin 2018):
        1. Normalize the vector to L2 length 1
        2. Draw a random level for the new node
        3. If new level > current max_level, update max_level and set the new node
           as the entry point at the new top level
        4. Greedy descent from entry_point through layers (highest -> 1):
           at each level, consider the current node + its neighbors at that level,
           pick the top 'ef_construction' candidates by cosine similarity,
           move to the best one
        5. After descent to level 0, the "current node" is the entry point at bottom
        6. At each level l = 1..new_level:
           - Gather candidates: the entry node + its neighbors at this level
           - Pick the M nearest by cosine similarity to the new vector
           - Set the new node's adjacency at this level to these M neighbors
           - Rewire: add the new node to each of the M neighbors' adjacency lists
        7. At level 0:
           - Gather candidates: the descent entry node + its level-0 neighbors
           - Pick M nearest by cosine similarity
           - Set the new node's level-0 adjacency
           - Rewire level-0 neighbors

        The key heuristic: at each level, we consider a larger candidate pool
        (ef_construction) than the M we keep, which creates a more well-connected
        graph than just picking M from the immediate neighbors.
        """
        # Step 1: normalize
        vec = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # Step 2: random level
        new_level = self._random_level()

        # Create the new node
        new_node = HNSWNode(vec, node_id, new_level)

        # Step 3: grow max_level if needed
        if new_level > self._max_level:
            self._max_level = new_level
            self.entry_point = node_id
            # New node becomes entry point at the new top level
            if new_level > 0:
                new_node.adj[new_level] = []  # no neighbors yet at this level

        # Ensure entry point is set even on the first insert (new_level == 0)
        if self.entry_point is None:
            self.entry_point = node_id

        # Step 4: greedy descent from entry_point through layers
        # We descend from _max_level down to 1
        # At each level, we consider ef_construction candidates and pick the best
        current_id = self.entry_point

        for level in range(self._max_level, 0, -1):
            # Get neighbors of current_id at this level
            neighbors = self._get_neighbors_at_level(current_id, level)
            # Candidates: current node + its neighbors
            candidates = [current_id] + (neighbors if neighbors else [])

            # Compute cosine similarity to vec for each candidate
            scored: list[tuple[float, int]] = []
            for cid in candidates:
                sim = self._cosine_similarity(vec, self._get_vector(cid))
                scored.append((sim, cid))

            # Sort descending and keep top 'ef_construction'
            scored.sort(key=lambda x: x[0], reverse=True)
            top_candidates = [cid for _, cid in scored[: self.ef_construction]]

            # Move to the best candidate among the top ef_construction
            if top_candidates:
                current_id = top_candidates[0]

        # After the loop, current_id is the node at level 0 reached during descent
        # Step 5: connect at each level from 1 to new_level
        ep = self.entry_point
        # Ensure entry point has adj entries for all needed levels
        for lvl in range(1, max(new_level, self._max_level) + 1):
            if ep is not None and ep in self.nodes:
                if self.nodes[ep].adj[lvl] is None:
                    self.nodes[ep].adj[lvl] = []

        for lvl in range(1, new_level + 1):
            # Gather candidates: entry point + its neighbors at this level
            if ep is not None and ep in self.nodes:
                neighbors_at_lvl = self._get_neighbors_at_level(ep, lvl)
            else:
                neighbors_at_lvl = []

            candidates = [ep] + (neighbors_at_lvl if neighbors_at_lvl else [])

            # Compute similarity to new vector
            scored: list[tuple[float, int]] = []
            for cid in candidates:
                sim = self._cosine_similarity(vec, self._get_vector(cid))
                scored.append((sim, cid))

            # Sort descending, keep top M
            scored.sort(key=lambda x: x[0], reverse=True)
            top_M_ids = [cid for _, cid in scored[: self.M]]

            # Set new node's adjacency at this level
            new_node.adj[lvl] = top_M_ids

            # Rewire: add new node to each of the top-M neighbors' adjacency
            for nid in top_M_ids:
                if nid in self.nodes:
                    neighbor_node = self.nodes[nid]
                    if neighbor_node.adj[lvl] is None:
                        neighbor_node.adj[lvl] = []
                    if node_id not in neighbor_node.adj[lvl]:
                        neighbor_node.adj[lvl].append(node_id)

        # Also connect at level 0
        level_0_neighbors = self._get_neighbors_at_level(current_id, 0)
        level0_candidates = [current_id] + (level_0_neighbors if level_0_neighbors else [])

        scored0: list[tuple[float, int]] = []
        for cid in level0_candidates:
            sim = self._cosine_similarity(vec, self._get_vector(cid))
            scored0.append((sim, cid))

        scored0.sort(key=lambda x: x[0], reverse=True)
        top_M_ids_0 = [cid for _, cid in scored0[: self.M]]

        new_node.adj[0] = top_M_ids_0

        # Rewire level-0 neighbors
        for nid in top_M_ids_0:
            if nid in self.nodes:
                neighbor_node = self.nodes[nid]
                if neighbor_node.adj[0] is None:
                    neighbor_node.adj[0] = []
                if node_id not in neighbor_node.adj[0]:
                    neighbor_node.adj[0].append(node_id)

        # Step 6: add the new node to the index
        self.nodes[node_id] = new_node

        # Update vectors array
        if self.vectors is None:
            self.vectors = vec.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vec])

        # Update entry_point if this new node has a higher level
        if new_level > self._max_level:
            # This shouldn't happen since we already updated _max_level, but be safe
            self._max_level = new_level
            self.entry_point = node_id

    # ---- Search (multi-layer with ef parameter) ----

    def search(self, query_vector: np.ndarray, k: int = 10, ef: int | None = None) -> np.ndarray:
        """Search for k nearest neighbors using the multi-layer HNSW graph.

        Algorithm:
        1. Start at entry_point at the highest level (_max_level)
        2. For each level from _max_level down to 1:
           - Among the current node + its neighbors at this level,
             pick the single best neighbor by cosine similarity to the query
           - Move to that neighbor
           This descent uses the graph structure to quickly home in on
           the right region of the space. At each higher level, we get
           long-range jumps; at level 0, we have fine-grained locality.
        3. At level 0, after descent, collect candidates from the vicinity:
           the final node + its level-0 neighbors.
           Consider up to 'ef' candidates total.
        4. Return the top-k from the collected candidates.

        The `ef` parameter controls the candidate list size at level 0:
        - Larger ef -> considers more candidates -> better recall but slower
        - Smaller ef -> faster search but potentially lower recall
        Typical values: ef=40 for demo, ef=100+ for production-quality recall.

        Important: we pick the single best neighbor at each level during descent,
        and only use the ef parameter at level 0 for the final candidate pool.
        This is the standard HNSW search algorithm.
        """
        if len(self.nodes) == 0:
            return np.array([], dtype=np.int64)

        # Normalize query vector
        q = np.asarray(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        # Step 1: descent from entry_point through upper layers
        # At each level, pick the single best neighbor (current node + neighbors)
        # by cosine similarity to the query, and move to it.
        # This is the core of HNSW's efficiency - we use the graph structure
        # to skip large portions of the search space.
        current_id = self.entry_point

        for level in range(self._max_level, 0, -1):
            neighbors = self._get_neighbors_at_level(current_id, level)
            if not neighbors:
                # No neighbors at this level; stay at current node
                continue

            # Find the best neighbor by cosine similarity to query q
            best_id = current_id
            best_sim = self._cosine_similarity(q, self._get_vector(current_id))

            for nid in neighbors:
                sim = self._cosine_similarity(q, self._get_vector(nid))
                if sim > best_sim:
                    best_sim = sim
                    best_id = nid

            # Move to the best neighbor (or stay if none is better)
            if best_id != current_id:
                current_id = best_id
            # If no neighbor is better, we stay and descend to the next level

        # Step 2: at level 0, collect candidates from the vicinity
        # of the final current_id node
        # Gather: the node itself + its level-0 neighbors
        initial_ids: list[int] = [current_id]
        neighbors_0 = self._get_neighbors_at_level(current_id, 0)
        initial_ids.extend(neighbors_0 if neighbors_0 else [])

        # Compute similarities for all initial candidates
        scored: list[tuple[float, int]] = []
        for cid in initial_ids:
            sim = self._cosine_similarity(q, self._get_vector(cid))
            scored.append((sim, cid))

        # Sort descending by similarity
        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top 'ef' candidates (use provided ef or self.ef_search)
        search_ef = ef if ef is not None else self.ef_search
        candidates = scored[:search_ef]

        # Step 3: return top-k
        top_k = min(k, len(candidates))
        result_indices = np.array([cid for _, cid in candidates[:top_k]], dtype=np.int64)
        return result_indices

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