from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np


class ButinaClusterer:
    """
    Butina (greedy) clustering for small-to-moderate datasets.

    The algorithm:
      - build neighbor sets: for each item i, neighbors = { j | sim[i,j] >= cutoff }
      - greedily pick the item with the largest number of neighbors among remaining items,
        form a cluster as that neighbor set intersect remaining, and remove those members.
      - repeat until no items remain.

    :param cutoff: similarity threshold (0..1). Items i,j are neighbors if sim[i,j] >= cutoff.
    :type cutoff: float
    :param include_self: If True (default) each item's neighbor set includes itself.
    :type include_self: bool
    :param sort_clusters: If True (default), returned clusters are sorted by size (desc).
    :type sort_clusters: bool
    :param min_cluster_size: If provided (>0), clusters smaller than this are merged into singletons
                             (i.e. filtered out) — using None keeps all clusters.
    :type min_cluster_size: Optional[int]
    :param metric: When providing raw feature vectors, choose 'tanimoto' (binary) or 'cosine' (float).
                   If sim matrix provided, metric is ignored.
    :type metric: str

    Example
    -------
    >>> import numpy as np
    >>> from synrxn.cluster.butina import ButinaClusterer
    >>> # simple toy - two tight groups
    >>> features = np.array([
    ...   [1,1,0,0],
    ...   [1,1,0,0],
    ...   [0,0,1,1],
    ...   [0,0,1,1],
    ... ])
    >>> clusterer = ButinaClusterer(cutoff=0.9, metric="tanimoto")
    >>> clusterer.fit(features=features)
    >>> print(clusterer.clusters_)
    [[0, 1], [2, 3]]
    >>> print(clusterer.labels_)
    [0,0,1,1]
    """

    def __init__(
        self,
        cutoff: float = 0.7,
        include_self: bool = True,
        sort_clusters: bool = True,
        min_cluster_size: Optional[int] = None,
        metric: str = "tanimoto",
    ):
        if not (0.0 <= cutoff <= 1.0):
            raise ValueError("cutoff must be between 0 and 1")
        if metric not in ("tanimoto", "cosine"):
            raise ValueError("metric must be 'tanimoto' or 'cosine'")
        if min_cluster_size is not None and min_cluster_size <= 0:
            raise ValueError("min_cluster_size must be > 0 or None")

        self.cutoff = float(cutoff)
        self.include_self = bool(include_self)
        self.sort_clusters = bool(sort_clusters)
        self.min_cluster_size = (
            None if min_cluster_size is None else int(min_cluster_size)
        )
        self.metric = metric

        # populated after fit()
        self.sim_matrix_: Optional[np.ndarray] = None
        self.neighbor_sets_: Optional[List[set]] = None
        self.clusters_: Optional[List[List[int]]] = None
        self.labels_: Optional[np.ndarray] = None

    # ------------- Public API -------------
    def fit(
        self,
        sim: Optional[np.ndarray] = None,
        features: Optional[Sequence[Sequence]] = None,
    ) -> "ButinaClusterer":
        """
        Fit the clusterer using either a precomputed similarity matrix `sim` (square numpy array)
        or `features` (list/array of vectors). If both provided, `sim` is used.

        :param sim: Precomputed full symmetric similarity matrix (n x n). Optional.
        :type sim: Optional[np.ndarray]
        :param features: Raw feature vectors. If provided and metric == 'tanimoto', input must be binary/integers.
        :type features: Optional[Sequence[Sequence]]
        :returns: self
        """
        if sim is None and features is None:
            raise ValueError("Provide either 'sim' or 'features'")
        if sim is not None:
            sim = np.asarray(sim, dtype=float)
            if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
                raise ValueError("sim must be a square (n,n) matrix")
            if np.isnan(sim).any():
                raise ValueError("sim contains NaN values")
            self.sim_matrix_ = sim.copy()
        else:
            X = np.asarray(features)
            if X.ndim != 2:
                raise ValueError(
                    "features must be a 2D array-like (n_samples, n_features)"
                )
            if self.metric == "tanimoto":
                self.sim_matrix_ = self._tanimoto_matrix(X)
            else:
                self.sim_matrix_ = self._cosine_matrix(X)

        # build neighbor sets
        self.neighbor_sets_ = self._build_neighbor_sets(
            self.sim_matrix_, self.cutoff, include_self=self.include_self
        )

        # greedy clustering
        self.clusters_ = self._greedy_butina(self.neighbor_sets_)

        # optionally filter by min_cluster_size (if provided) — keep clusters >= threshold
        if self.min_cluster_size is not None:
            filtered = [c for c in self.clusters_ if len(c) >= self.min_cluster_size]
            # clusters smaller than min_cluster_size become singletons appended (common alternative)
            # Here we discard small clusters; if you want them as singletons, change logic.
            self.clusters_ = filtered

        # sort clusters by size desc if requested
        if self.sort_clusters:
            self.clusters_.sort(key=lambda c: -len(c))

        # compute labels array mapping each index -> cluster id (or -1 if missing)
        n = self.sim_matrix_.shape[0]
        labels = -1 * np.ones(n, dtype=int)
        for cid, cluster in enumerate(self.clusters_):
            for i in cluster:
                labels[i] = cid
        self.labels_ = labels

        return self

    def get_clusters(self) -> List[List[int]]:
        """Return clusters (list of lists of integer indices)."""
        if self.clusters_ is None:
            raise RuntimeError("Call fit(...) before get_clusters()")
        return self.clusters_

    def get_labels(self) -> np.ndarray:
        """Return labels array of length n_samples with cluster ids (or -1)."""
        if self.labels_ is None:
            raise RuntimeError("Call fit(...) before get_labels()")
        return self.labels_

    # ------------- Internal helpers -------------
    @staticmethod
    def _tanimoto_matrix(X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Tanimoto (Jaccard for binary) similarity matrix for binary matrix X.
        Vectorized: intersection = X @ X.T, union = a_sum[:,None] + a_sum[None,:] - intersection.
        """
        Xb = X.astype(bool).astype(np.uint8)
        # intersection counts
        inter = Xb.dot(Xb.T).astype(float)
        a_sum = Xb.sum(axis=1).astype(float)
        union = a_sum[:, None] + a_sum[None, :] - inter
        # avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0.0, inter / union, 0.0)
        # clip small numerical noise
        sim = np.clip(sim, 0.0, 1.0)
        return sim

    @staticmethod
    def _cosine_matrix(X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix (dense).
        """
        Xf = X.astype(float)
        norms = np.linalg.norm(Xf, axis=1)
        # avoid div-by-zero: if zero norm, keep as zero vector (cosine=0)
        # compute dot product matrix
        dot = Xf.dot(Xf.T)
        denom = norms[:, None] * norms[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(denom > 0.0, dot / denom, 0.0)
        sim = np.clip(sim, -1.0, 1.0)
        return sim

    @staticmethod
    def _build_neighbor_sets(
        sim: np.ndarray, cutoff: float, include_self: bool = True
    ) -> List[set]:
        """
        Build list of neighbor sets: neighbor_sets[i] = set(j | sim[i,j] >= cutoff).
        """
        n = sim.shape[0]
        neighbor_sets: List[set] = []
        # vectorized thresholding per row
        for i in range(n):
            neigh = set(np.nonzero(sim[i] >= cutoff)[0].tolist())
            if include_self:
                neigh.add(i)
            neighbor_sets.append(neigh)
        return neighbor_sets

    @staticmethod
    def _greedy_butina(neighbor_sets: List[set]) -> List[List[int]]:
        """
        Greedy cluster formation:
          - remaining = set(all indices)
          - choose best = argmax_i |neighbor_sets[i] & remaining|
          - cluster = sorted(neighbor_sets[best] & remaining)
          - remove them and repeat
        """
        n = len(neighbor_sets)
        remaining = set(range(n))
        clusters: List[List[int]] = []
        while remaining:
            best = max(remaining, key=lambda x: len(neighbor_sets[x] & remaining))
            members = sorted(neighbor_sets[best] & remaining)
            clusters.append(members)
            for m in members:
                remaining.discard(m)
        return clusters
