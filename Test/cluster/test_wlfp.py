import unittest
import numpy as np
import networkx as nx
from numpy.testing import assert_array_equal

from synrxn.cluster.wlfp import WLFP, _norm_value


class TestNormValue(unittest.TestCase):
    def test_norm_primitives(self):
        self.assertIsNone(_norm_value(None))
        self.assertEqual(_norm_value("abc"), "abc")
        self.assertEqual(_norm_value(42), 42)
        self.assertEqual(_norm_value(3.14), 3.14)
        self.assertEqual(_norm_value(True), True)

    def test_norm_sequences_and_sets(self):
        self.assertEqual(_norm_value([1, 2, 3]), (1, 2, 3))
        self.assertEqual(_norm_value((1, 2)), (1, 2))
        self.assertEqual(_norm_value({3, 1, 2}), (1, 2, 3))

    def test_norm_dict_and_numpy_scalar(self):
        d = {"b": 2, "a": 1}
        self.assertEqual(_norm_value(d), (("a", 1), ("b", 2)))
        self.assertEqual(_norm_value(np.int32(7)), 7)
        self.assertEqual(_norm_value(np.float64(2.5)), 2.5)


class TestWLFPBasics(unittest.TestCase):
    def make_simple_graph(self):
        """
        Graph:
            1
           / \
          2   3
        Node attrs: element only (kept simple)
        """
        G = nx.Graph()
        G.add_node(1, element="C")
        G.add_node(2, element="O")
        G.add_node(3, element="N")
        G.add_edge(1, 2, standard_order=1.0)
        G.add_edge(1, 3, standard_order=1.0)
        return G

    def make_reordered_graph(self):
        # same graph but add nodes in a different order (to test stable_sort_neighbors)
        G = nx.Graph()
        G.add_node(3, element="N")
        G.add_node(1, element="C")
        G.add_node(2, element="O")
        G.add_edge(1, 2, standard_order=1.0)
        G.add_edge(1, 3, standard_order=1.0)
        return G

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            WLFP(iterations=-1)
        with self.assertRaises(ValueError):
            WLFP(hash_bits=7)

    def test_type_error_for_non_graph(self):
        wl = WLFP()
        with self.assertRaises(TypeError):
            wl.fingerprint("not a graph")

    def test_dense_and_sparse_equivalence_presence(self):
        G = self.make_simple_graph()
        wl = WLFP(iterations=1, n_bits=128, use_counts=False, hash_bits=64)
        vec = wl.fingerprint(G)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.dtype, np.uint8)
        self.assertEqual(vec.shape[0], 128)

        idxs, vals = wl.fingerprint(G, return_sparse=True)
        self.assertEqual(idxs.dtype, np.int32)
        self.assertEqual(vals.dtype, np.int32)

        recon = np.zeros_like(vec, dtype=np.uint8)
        if idxs.size > 0:
            recon[idxs] = (vals != 0).astype(np.uint8)
        assert_array_equal(recon, vec)

    def test_dense_and_sparse_equivalence_counts(self):
        G = self.make_simple_graph()
        wl = WLFP(iterations=2, n_bits=128, use_counts=True, hash_bits=64)
        vec = wl.fingerprint(G)
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.dtype, np.int32)
        self.assertEqual(vec.shape[0], 128)

        idxs, vals = wl.fingerprint(G, return_sparse=True)
        recon = np.zeros_like(vec, dtype=np.int32)
        if idxs.size > 0:
            recon[idxs] = vals
        assert_array_equal(recon, vec)

    def test_return_feature_map_structure(self):
        G = self.make_simple_graph()
        wl = WLFP(iterations=2, n_bits=256, use_counts=True, hash_bits=64)
        dense, fmap = wl.fingerprint(G, return_feature_map=True)
        self.assertIsInstance(dense, np.ndarray)
        self.assertEqual(dense.shape[0], 256)
        self.assertIsInstance(fmap, dict)
        # feature map entries: key=int, value=(iteration:int, count:int)
        for k, v in fmap.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, tuple)
            self.assertGreaterEqual(len(v), 2)
            iter_level, count_val = v[0], v[1]
            self.assertIsInstance(iter_level, int)
            self.assertIsInstance(count_val, int)
            self.assertGreaterEqual(iter_level, 0)
            self.assertGreaterEqual(count_val, 1)

    def test_use_edge_attrs_changes_output(self):
        G = self.make_simple_graph()
        wl_yes = WLFP(iterations=1, n_bits=256, use_edge_attrs=True, hash_bits=64)
        wl_no = WLFP(iterations=1, n_bits=256, use_edge_attrs=False, hash_bits=64)
        vec_yes = wl_yes.fingerprint(G)
        vec_no = wl_no.fingerprint(G)
        # enabling edge attrs should generally change the fingerprint
        self.assertFalse(np.array_equal(vec_yes, vec_no))

    def test_stable_sort_neighbors_determinism(self):
        # two graphs with different insertion order should give same result when stable_sort_neighbors=True
        G1 = self.make_simple_graph()
        G2 = self.make_reordered_graph()
        wl_stable = WLFP(
            iterations=2, n_bits=256, stable_sort_neighbors=True, hash_bits=64
        )
        v1 = wl_stable.fingerprint(G1)
        v2 = wl_stable.fingerprint(G2)
        assert_array_equal(v1, v2)

        # We only ensure determinism with stable_sort_neighbors True.
        # When False, behaviour may depend on insertion order; we do not assert inequality (could be equal by chance).

    def test_multigraph_edge_handling(self):
        MG = nx.MultiGraph()
        MG.add_node(1, element="C")
        MG.add_node(2, element="O")
        MG.add_edge(1, 2, key="a", standard_order=1.0)
        MG.add_edge(1, 2, key="b", standard_order=2.0)
        wl = WLFP(iterations=1, n_bits=128, hash_bits=64)
        v1 = wl.fingerprint(MG)
        v2 = wl.fingerprint(MG)
        assert_array_equal(v1, v2)

    def test_determinism_across_instances(self):
        G = self.make_simple_graph()
        a = WLFP(iterations=2, n_bits=128, hash_bits=64)
        b = WLFP(iterations=2, n_bits=128, hash_bits=64)
        v1 = a.fingerprint(G)
        v2 = b.fingerprint(G)
        assert_array_equal(v1, v2)

    def test_empty_graph_zero_vector_and_sparse_empty(self):
        G = nx.Graph()
        wl = WLFP(iterations=1, n_bits=64, hash_bits=64)
        vec = wl.fingerprint(G)
        assert_array_equal(vec, np.zeros(64, dtype=vec.dtype))
        idxs, vals = wl.fingerprint(G, return_sparse=True)
        self.assertEqual(idxs.size, 0)
        self.assertEqual(vals.size, 0)

    def test_to_bitstring(self):
        G = self.make_simple_graph()
        wl = WLFP(iterations=1, n_bits=32, hash_bits=64)
        vec = wl.fingerprint(G)
        bs = wl.to_bitstring(vec)
        self.assertIsInstance(bs, str)
        self.assertEqual(len(bs), 32)
        self.assertEqual(bs.count("1"), int(vec.astype(bool).sum()))


if __name__ == "__main__":
    unittest.main()
