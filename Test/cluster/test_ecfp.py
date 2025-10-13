import unittest
import numpy as np
import networkx as nx
from numpy.testing import assert_array_equal

from synrxn.cluster.ecfp import ECFP, _norm_value


class TestNormValue(unittest.TestCase):
    def test_norm_primitives(self):
        self.assertEqual(_norm_value(None), None)
        self.assertEqual(_norm_value("abc"), "abc")
        self.assertEqual(_norm_value(5), 5)
        self.assertEqual(_norm_value(3.14), 3.14)
        self.assertEqual(_norm_value(True), True)

    def test_norm_sequences_and_sets(self):
        self.assertEqual(_norm_value([1, 2, 3]), (1, 2, 3))
        self.assertEqual(_norm_value((1, 2)), (1, 2))
        # set becomes sorted tuple
        self.assertEqual(_norm_value({3, 1, 2}), (1, 2, 3))

    def test_norm_dict(self):
        d = {"b": 2, "a": 1}
        # dict -> tuple of (k, norm(v)) sorted by key
        self.assertEqual(_norm_value(d), (("a", 1), ("b", 2)))

    def test_norm_numpy_scalar(self):
        self.assertEqual(_norm_value(np.int32(7)), 7)
        self.assertEqual(_norm_value(np.float64(2.5)), 2.5)


class TestECFPBasic(unittest.TestCase):
    def make_simple_graph(self):
        """
        Graph layout:
          1 -- 2 -- 3

        Node attributes: element, aromatic, hcount, charge
        Edge attributes: standard_order
        """
        G = nx.Graph()
        for i, elem in enumerate(["C", "C", "O"], start=1):
            G.add_node(i, element=elem, aromatic=False, hcount=0, charge=0)
        G.add_edge(1, 2, standard_order=1.0)
        G.add_edge(2, 3, standard_order=1.0)
        return G

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            ECFP(radius=-1)
        with self.assertRaises(ValueError):
            ECFP(hash_bits=7)  # invalid hash_bits

    def test_type_error_for_non_graph(self):
        fp = ECFP()
        with self.assertRaises(TypeError):
            fp.fingerprint("not a graph")  # must be networkx graph

    def test_dense_and_sparse_equivalence_presence(self):
        G = self.make_simple_graph()
        fp = ECFP(radius=1, n_bits=128, use_counts=False, hash_bits=64)
        vec = fp.fingerprint(G)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.dtype, np.uint8)
        self.assertEqual(vec.shape[0], 128)

        # sparse output should reconstruct to the same dense vector
        idxs, vals = fp.fingerprint(G, return_sparse=True)
        self.assertEqual(idxs.dtype, np.int32)
        self.assertEqual(vals.dtype, np.int32)
        # reconstructed
        recon = np.zeros_like(vec, dtype=np.uint8)
        if idxs.size > 0:
            recon[idxs] = (vals != 0).astype(np.uint8)
        assert_array_equal(recon, vec)

    def test_dense_and_sparse_equivalence_counts(self):
        G = self.make_simple_graph()
        fp = ECFP(radius=1, n_bits=128, use_counts=True, hash_bits=64)
        vec = fp.fingerprint(G)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.dtype, np.int32)
        self.assertEqual(vec.shape[0], 128)

        idxs, vals = fp.fingerprint(G, return_sparse=True)
        # reconstruct counts
        recon = np.zeros_like(vec, dtype=np.int32)
        if idxs.size > 0:
            recon[idxs] = vals
        assert_array_equal(recon, vec)

    def test_return_feature_map_structure(self):
        G = self.make_simple_graph()
        fp = ECFP(radius=2, n_bits=256, use_counts=True, hash_bits=64)
        dense, fmap = fp.fingerprint(G, return_feature_map=True)
        # dense vector shape and types
        self.assertIsInstance(dense, np.ndarray)
        self.assertEqual(dense.shape[0], 256)
        # fmap should be a dict mapping int -> (radius:int, count:int)
        self.assertIsInstance(fmap, dict)
        for k, v in fmap.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, tuple)
            self.assertGreaterEqual(len(v), 2)
            radius_val, count_val = v[0], v[1]
            self.assertIsInstance(radius_val, int)
            self.assertIsInstance(count_val, int)
            self.assertGreaterEqual(radius_val, 0)
            self.assertGreaterEqual(count_val, 1)

    def test_include_neighbor_attrs_changes_fingerprint(self):
        # Two nodes in line: include_neighbor_attrs toggles representation
        G = self.make_simple_graph()
        fp_no = ECFP(radius=1, n_bits=256, include_neighbor_attrs=False, hash_bits=64)
        fp_yes = ECFP(radius=1, n_bits=256, include_neighbor_attrs=True, hash_bits=64)
        vec_no = fp_no.fingerprint(G)
        vec_yes = fp_yes.fingerprint(G)
        # They should differ in general
        self.assertFalse(np.array_equal(vec_no, vec_yes))

    def test_multigraph_edge_handling(self):
        MG = nx.MultiGraph()
        MG.add_node(1, element="C", aromatic=False, hcount=0, charge=0)
        MG.add_node(2, element="O", aromatic=False, hcount=0, charge=0)
        # add two parallel edges with different data; implementation picks first deterministically
        MG.add_edge(1, 2, key="a", standard_order=1.0)
        MG.add_edge(1, 2, key="b", standard_order=2.0)
        fp = ECFP(radius=1, n_bits=128, hash_bits=64)
        vec = fp.fingerprint(MG)
        # vector produced and is deterministic
        self.assertIsInstance(vec, np.ndarray)
        # call again to ensure deterministic (same)
        vec2 = fp.fingerprint(MG)
        assert_array_equal(vec, vec2)

    def test_determinism_across_instances(self):
        G = self.make_simple_graph()
        a = ECFP(radius=1, n_bits=128, hash_bits=64)
        b = ECFP(radius=1, n_bits=128, hash_bits=64)
        v1 = a.fingerprint(G)
        v2 = b.fingerprint(G)
        assert_array_equal(v1, v2)

    def test_empty_graph_zero_vector_and_sparse_empty(self):
        G = nx.Graph()
        fp = ECFP(radius=1, n_bits=64, hash_bits=64)
        vec = fp.fingerprint(G)
        # empty graph -> zero vector
        assert_array_equal(vec, np.zeros(64, dtype=vec.dtype))
        idxs, vals = fp.fingerprint(G, return_sparse=True)
        self.assertEqual(idxs.size, 0)
        self.assertEqual(vals.size, 0)

    def test_to_bitstring(self):
        G = self.make_simple_graph()
        fp = ECFP(radius=1, n_bits=32, hash_bits=64)
        vec = fp.fingerprint(G)
        bs = fp.to_bitstring(vec)
        self.assertIsInstance(bs, str)
        self.assertEqual(len(bs), 32)
        # number of '1' in bitstring equals sum of nonzero entries
        self.assertEqual(bs.count("1"), int(vec.astype(bool).sum()))


if __name__ == "__main__":
    unittest.main()
