from __future__ import annotations
import hashlib
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import networkx as nx


def _norm_value(v: Any) -> Any:
    """
    Normalize attribute values to deterministic, hashable forms:
      - lists/tuples -> tuples
      - dict -> tuple(sorted((k, norm(v))...))
      - set -> tuple(sorted(...))
      - numpy scalars -> Python scalars
      - None -> None
      - fallback -> repr(v)
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
    # numpy scalar?
    try:
        import numpy as _np  # type: ignore

        if isinstance(v, _np.generic):
            return v.item()
    except Exception:
        pass
    return repr(v)


class ECFP:
    """
    ECFP (Morgan-style) fingerprint generator for NetworkX graphs (suitable for ITS graphs).

    This implementation is deterministic, configurable via node_attrs and edge_attrs,
    and returns folded bit-vectors (dense) or sparse/indexed outputs. It uses truncated
    SHA-1 for stable hashing (configurable via `hash_bits`).

    :param radius: Morgan radius (ECFP2 -> radius=1, ECFP4 -> radius=2).
    :param n_bits: Length of folded fingerprint vector (number of bits).
    :param use_counts: If True, outputs counts per folded bit; else presence/absence.
    :param node_attrs: List of node attribute names to include in the initial atom invariant.
                       If None, defaults to ["element", "aromatic", "hcount", "charge"].
    :param edge_attrs: List of edge attribute names to include in bond descriptor tuples.
                       If None, defaults to ["standard_order", "order"].
    :param include_neighbor_attrs: If True, include neighbor node_attrs summary into the
                                   radius-0 atom invariant (default: False).
    :param hash_bits: Number of bits to keep from SHA-1 truncation (32,64,128,160,256).
                      Recommended: 64.
    :raises ValueError: for invalid parameters (negative radius or invalid hash_bits).
    :returns: See `fingerprint` method for concrete return types.

    Example
    -------
    .. code-block:: python

       import networkx as nx
       from synrxn.fps.ecfp import ECFP

       # Build a tiny ITS-like graph
       G = nx.Graph()
       G.add_node(1, element="Cl", aromatic=False, hcount=0, charge=0,
                  typesGH=(("Cl", False, 0, 0, ["C"]),))
       G.add_node(5, element="C", aromatic=False, hcount=0, charge=0,
                  typesGH=(("C", False, 0, 0, ["C","Cl","N","N"]),))
       G.add_node(4, element="C", aromatic=False, hcount=0, charge=0)
       G.add_edge(1, 5, standard_order=1.0)
       G.add_edge(5, 4, order=(1.0, 2.0))

       fp = ECFP(radius=2, n_bits=1024,
                 node_attrs=["element","aromatic","hcount","charge","typesGH"],
                 edge_attrs=["standard_order","order"],
                 include_neighbor_attrs=False,
                 use_counts=False)
       vec = fp.fingerprint(G)               # dense numpy vector (uint8)
       indices, values = fp.fingerprint(G, return_sparse=True)  # sparse indices + values
       dense, fmap = fp.fingerprint(G, return_feature_map=True) # dense + feature map

       print("bits on:", int(vec.sum()))
       print("sparse:", indices.tolist(), values.tolist())
       print("feature map items (sample):", list(fmap.items())[:3])

    """

    DEFAULT_NODE_ATTRS = ["element", "aromatic", "hcount", "charge"]
    DEFAULT_EDGE_ATTRS = ["standard_order", "order"]

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        use_counts: bool = False,
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None,
        include_neighbor_attrs: bool = False,
        hash_bits: int = 64,
    ):
        if radius < 0:
            raise ValueError("radius must be >= 0")
        if hash_bits not in (32, 64, 128, 160, 256):
            raise ValueError("hash_bits should be one of (32,64,128,160,256)")
        self.radius = int(radius)
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
        self.include_neighbor_attrs = bool(include_neighbor_attrs)
        self.hash_bits = int(hash_bits)

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
        Compute the ECFP fingerprint from NetworkX graph `G`.

        :param G: networkx.Graph or networkx.MultiGraph with node/edge attributes.
        :param return_sparse: If True, return tuple (indices, values) where indices are bit positions
                              (np.int32) and values are counts (if use_counts) or 1s (if not).
        :param return_feature_map: If True, return (dense_vector, feature_map) where feature_map is a
                                   dict mapping feature_hash -> (radius_layer, count).
        :returns:
            - If return_sparse=False and return_feature_map=False:
                dense folded vector (numpy.ndarray). dtype: np.uint8 if use_counts=False else np.int32.
            - If return_sparse=True:
                (indices: np.ndarray[int32], values: np.ndarray[int32])
            - If return_feature_map=True:
                (dense_vector: np.ndarray, feature_map: Dict[int, Tuple[int, int]])
        :raises TypeError: if G is not a NetworkX Graph/MultiGraph.
        """
        if not isinstance(G, (nx.Graph, nx.MultiGraph)):
            raise TypeError("G must be a networkx.Graph or networkx.MultiGraph")

        # Initial atom invariants (radius 0)
        atom_id: Dict[Any, int] = {}
        for n in G.nodes():
            atom_id[n] = self._initial_atom_invariant(G, n)

        feat_counter: Counter = Counter()
        feat_radius: Dict[int, int] = {}

        # radius 0 features
        for n, hid in atom_id.items():
            feat_counter[hid] += 1
            feat_radius[hid] = 0

        cur_ids = dict(atom_id)

        # Morgan iterations
        for r in range(1, self.radius + 1):
            next_ids: Dict[Any, int] = {}
            for n in G.nodes():
                # deterministic neighbor order
                neigh_tuples = []
                for nbr in sorted(G.neighbors(n)):
                    nbr_id = cur_ids[nbr]
                    bond_desc = self._edge_descriptor_tuple(G, n, nbr)
                    neigh_tuples.append((bond_desc, nbr_id))
                neigh_tuples.sort()
                center = cur_ids[n]
                rep = ("MORGAN", r, center, tuple(neigh_tuples))
                hid = self._hash_to_int(repr(rep))
                next_ids[n] = hid
                feat_counter[hid] += 1
                if hid not in feat_radius or feat_radius[hid] > r:
                    feat_radius[hid] = r
            cur_ids = next_ids

        # Return feature map + dense if requested
        if return_feature_map:
            dense = self._fold_counter_to_vector(feat_counter)
            return dense, {k: (feat_radius[k], int(v)) for k, v in feat_counter.items()}

        # Sparse output
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

        # Dense folded vector
        vec = self._fold_counter_to_vector(feat_counter)
        return vec

    # ---------------- internal helpers ----------------
    def _initial_atom_invariant(self, G: nx.Graph, node: Any) -> int:
        """
        Build initial atom invariant from node_attrs and, optionally, neighbor summaries.
        """
        na = G.nodes[node]
        node_vals = tuple(_norm_value(na.get(k, None)) for k in self.node_attrs)

        neigh_summary: Tuple[Any, ...] = ()
        if self.include_neighbor_attrs:
            s = []
            for nbr in sorted(G.neighbors(node)):
                nna = G.nodes[nbr]
                tup = tuple(_norm_value(nna.get(k, None)) for k in self.node_attrs)
                s.append(tup)
            neigh_summary = tuple(sorted(s))

        rep = ("ATOM0", node_vals, neigh_summary)
        return self._hash_to_int(repr(rep))

    def _edge_descriptor_tuple(self, G: nx.Graph, a: Any, b: Any) -> Tuple[Any, ...]:
        """
        Extract edge attributes (in configured order) as a tuple. For MultiGraph uses the
        first edge's data deterministically (sorted keys).
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
        Deterministic truncated hash -> int. Uses sha1 and truncates to `hash_bits`.
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
        """
        Return compact bitstring representation ('0'/'1') of a dense vector.
        """
        bv = (vec != 0).astype(np.uint8)
        return "".join("1" if b else "0" for b in bv.tolist())
