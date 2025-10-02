from __future__ import annotations
import hashlib
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import networkx as nx


def _norm_value(v: Any) -> Any:
    """
    Normalize values to deterministic, hashable forms.
    """
    if v is None:
        return None
    if isinstance(v, (str, int, bool, float)):
        return v
    if isinstance(v, (list, tuple)):
        return tuple(_norm_value(x) for x in v)
    if isinstance(v, set):
        return tuple(sorted(_norm_value(x) for x in v))
    if isinstance(v, dict):
        return tuple(sorted((k, _norm_value(v[k])) for k in sorted(v.keys())))
    try:
        import numpy as _np  # type: ignore

        if isinstance(v, _np.generic):
            return v.item()
    except Exception:
        pass
    return repr(v)


class WLFP:
    """
    Weisfeiler–Lehman Fingerprints (WLFP) for NetworkX graphs.

    This class computes WL-style node-label refinement and collects counts of the
    refined labels across iterations to form a fingerprint.

    :param iterations: Number of WL iterations (h). Typical small values: 1..3.
    :param n_bits: Size of folded fingerprint vector (number of bits).
    :param use_counts: If True fingerprint values are counts per folded bit; otherwise 0/1 presence.
    :param node_attrs: List of node attribute names to form the initial label. If None,
                       defaults to ['element', 'aromatic', 'hcount', 'charge'].
    :param edge_attrs: List of edge attribute names to include when forming neighbor descriptors.
                       If None, defaults to ['standard_order', 'order'].
    :param use_edge_attrs: If True, neighbor descriptors include the configured edge_attrs.
    :param hash_bits: Number of bits used when truncating the SHA-1 digest (32,64,128,160,256).
                      Default 64 (good trade-off).
    :param stable_sort_neighbors: If True, neighbor multisets are assembled in sorted neighbor-id
                                  order before canonicalizing; this improves determinism across
                                  NetworkX node ordering differences.
    :raises ValueError: for invalid parameters.

    :returns: See `fingerprint` return descriptions.

    Example
    -------
    .. code-block:: python

       import networkx as nx
       from synrxn.fps.wlfp import WLFP
       G = nx.Graph()
       G.add_node(1, element='C')
       G.add_node(2, element='O')
       G.add_node(3, element='N')
       G.add_edge(1, 2, standard_order=1.0)
       G.add_edge(1, 3, standard_order=1.0)

       wl = WLFP(iterations=2, n_bits=1024,
                 node_attrs=['element'],
                 edge_attrs=['standard_order'],
                 use_edge_attrs=True,
                 use_counts=False)

       vec = wl.fingerprint(G)               # dense numpy vector (uint8)
       idx, vals = wl.fingerprint(G, return_sparse=True)
       dense, fmap = wl.fingerprint(G, return_feature_map=True)
       print("bits on:", int(vec.sum()))
       print("feature map sample:", list(fmap.items())[:5])

    """

    DEFAULT_NODE_ATTRS = ["element", "aromatic", "hcount", "charge"]
    DEFAULT_EDGE_ATTRS = ["standard_order", "order"]

    def __init__(
        self,
        iterations: int = 2,
        n_bits: int = 2048,
        use_counts: bool = False,
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None,
        use_edge_attrs: bool = True,
        hash_bits: int = 64,
        stable_sort_neighbors: bool = True,
    ):
        if iterations < 0:
            raise ValueError("iterations must be >= 0")
        if hash_bits not in (32, 64, 128, 160, 256):
            raise ValueError("hash_bits must be one of (32,64,128,160,256)")
        self.iterations = int(iterations)
        self.n_bits = int(n_bits)
        self.use_counts = bool(use_counts)
        self.node_attrs = (
            list(node_attrs)
            if node_attrs is not None
            else list(self.DEFAULT_NODE_ATTRS)
        )
        self.edge_attrs = (
            list(edge_attrs)
            if edge_attrs is not None
            else list(self.DEFAULT_EDGE_ATTRS)
        )
        self.use_edge_attrs = bool(use_edge_attrs)
        self.hash_bits = int(hash_bits)
        self.stable_sort_neighbors = bool(stable_sort_neighbors)

    # ---------------- Public API ----------------
    def fingerprint(
        self,
        G: nx.Graph,
        return_sparse: bool = False,
        return_feature_map: bool = False,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, Dict[int, Tuple[int, int]]],
    ]:
        """
        Compute WL fingerprint.

        :param G: networkx.Graph or networkx.MultiGraph with node/edge attributes.
        :param return_sparse: If True returns (indices, values) where indices are bit positions (np.int32)
                              and values are counts (if use_counts) or 1s (if not).
        :param return_feature_map: If True returns (dense_vector, feature_map) where feature_map maps
                                   feature_hash -> (iteration_level, count).
        :returns:
            - dense numpy vector (np.uint8 if use_counts=False else np.int32) when both flags False,
            - (indices, values) when return_sparse True,
            - (dense_vector, feature_map) when return_feature_map True.
        :raises TypeError: if G is not a networkx Graph/MultiGraph.
        """
        if not isinstance(G, (nx.Graph, nx.MultiGraph)):
            raise TypeError("G must be a networkx.Graph or networkx.MultiGraph")

        # 1) initial node labels
        labels: Dict[Any, int] = {}
        label_str_map: Dict[Any, str] = {}  # store string repr for convenience
        for n in G.nodes():
            s = self._initial_label_str(G, n)
            hid = self._hash_to_int(s)
            labels[n] = hid
            label_str_map[n] = s

        # feature counter: feature_hash -> count
        feat_counter: Counter = Counter()
        feat_iter: Dict[int, int] = {}

        # count initial labels (iteration 0)
        for n, hid in labels.items():
            feat_counter[hid] += 1
            feat_iter[hid] = 0

        # 2) WL iterations
        cur_labels = dict(labels)
        for it in range(1, self.iterations + 1):
            next_labels: Dict[Any, int] = {}
            for n in G.nodes():
                # neighbor label multiset (optionally include edge attrs)
                neighbor_descriptors = []
                neighs = list(G.neighbors(n))
                if self.stable_sort_neighbors:
                    neighs = sorted(neighs)
                for nbr in neighs:
                    nbr_label = cur_labels[nbr]
                    if self.use_edge_attrs:
                        edge_desc = self._edge_attr_tuple(G, n, nbr)
                        # Represent neighbor as (edge_desc, neighbor_label)
                        neighbor_descriptors.append((edge_desc, nbr_label))
                    else:
                        neighbor_descriptors.append((None, nbr_label))
                neighbor_descriptors.sort()
                center_label = cur_labels[n]
                rep = ("WL", it, center_label, tuple(neighbor_descriptors))
                hid = self._hash_to_int(repr(rep))
                next_labels[n] = hid
                feat_counter[hid] += 1
                if hid not in feat_iter or feat_iter[hid] > it:
                    feat_iter[hid] = it
            cur_labels = next_labels

        # 3) produce outputs
        if return_feature_map:
            dense = self._fold_counter_to_vector(feat_counter)
            return dense, {k: (feat_iter[k], int(v)) for k, v in feat_counter.items()}

        if return_sparse:
            idx_vals: Dict[int, int] = defaultdict(int)
            for hid, cnt in feat_counter.items():
                idx = int(hid % self.n_bits)
                if self.use_counts:
                    idx_vals[idx] += int(cnt)
                else:
                    idx_vals[idx] = 1
            if not idx_vals:
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
            indices = np.fromiter(idx_vals.keys(), dtype=np.int32)
            values = np.fromiter(idx_vals.values(), dtype=np.int32)
            order = np.argsort(indices)
            return indices[order], values[order]

        dense = self._fold_counter_to_vector(feat_counter)
        return dense

    # ---------------- Internal helpers ----------------
    def _initial_label_str(self, G: nx.Graph, node: Any) -> str:
        """
        Build a string representation for initial node label from node_attrs.
        """
        na = G.nodes[node]
        vals = tuple(_norm_value(na.get(k, None)) for k in self.node_attrs)
        return repr(("INIT", vals))

    def _edge_attr_tuple(self, G: nx.Graph, a: Any, b: Any) -> Tuple[Any, ...]:
        """
        Extract edge attributes as tuple (for MultiGraph uses first key deterministically).
        """
        data: Dict[str, Any] = {}
        if G.is_multigraph():
            keys = sorted(G[a][b].keys())
            if keys:
                data = G[a][b][keys[0]] or {}
            else:
                data = {}
        else:
            data = G.get_edge_data(a, b, {}) or {}
        return tuple(_norm_value(data.get(k, None)) for k in self.edge_attrs)

    def _hash_to_int(self, s: Union[str, bytes]) -> int:
        """
        Deterministic hashing: sha1 truncated to `hash_bits` -> integer.
        """
        if isinstance(s, str):
            s = s.encode("utf8", errors="surrogateescape")
        digest = hashlib.sha1(s).digest()
        nbytes = self.hash_bits // 8
        truncated = digest[:nbytes]
        return int.from_bytes(truncated, "big", signed=False)

    def _fold_counter_to_vector(self, ctr: Counter) -> np.ndarray:
        if self.use_counts:
            vec = np.zeros(self.n_bits, dtype=np.int32)
            for hid, cnt in ctr.items():
                idx = int(hid % self.n_bits)
                vec[idx] += int(cnt)
            return vec
        else:
            vec = np.zeros(self.n_bits, dtype=np.uint8)
            for hid in ctr.keys():
                idx = int(hid % self.n_bits)
                vec[idx] = 1
            return vec

    def to_bitstring(self, vec: np.ndarray) -> str:
        """Return '0'/'1' bitstring for a dense vector."""
        bv = (vec != 0).astype(np.uint8)
        return "".join("1" if b else "0" for b in bv.tolist())
