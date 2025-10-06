import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from synrxn.cluster.butina import ButinaClusterer


class TestButinaClusterer(unittest.TestCase):
    def test_init_validation(self):
        with self.assertRaises(ValueError):
            ButinaClusterer(cutoff=-0.1)
        with self.assertRaises(ValueError):
            ButinaClusterer(cutoff=1.1)
        with self.assertRaises(ValueError):
            ButinaClusterer(metric="euclid")
        with self.assertRaises(ValueError):
            ButinaClusterer(min_cluster_size=0)

    def test_fit_requires_input(self):
        c = ButinaClusterer()
        with self.assertRaises(ValueError):
            c.fit()  # neither sim nor features

    def test_fit_sim_validation(self):
        c = ButinaClusterer()
        # non-square
        with self.assertRaises(ValueError):
            c.fit(sim=np.ones((2, 3)))
        # contains NaN
        sim = np.eye(3)
        sim[0, 1] = np.nan
        with self.assertRaises(ValueError):
            c.fit(sim=sim)

    def test_get_before_fit_raises(self):
        c = ButinaClusterer()
        with self.assertRaises(RuntimeError):
            c.get_clusters()
        with self.assertRaises(RuntimeError):
            c.get_labels()

    def test_tanimoto_on_binary_features_two_clusters(self):
        # from the docstring example: two tight groups [0,1] and [2,3]
        features = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ]
        )
        c = ButinaClusterer(cutoff=0.9, metric="tanimoto")
        c.fit(features=features)
        clusters = c.get_clusters()
        labels = c.get_labels()
        # clusters may be in either order if sizes equal; sort for assertion
        sorted_clusters = sorted([sorted(cl) for cl in clusters])
        self.assertEqual(sorted_clusters, [[0, 1], [2, 3]])
        # labels mapping: two clusters, cluster ids 0 or 1 — check cluster membership consistency
        self.assertIn(labels[0], (0, 1))
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        # ensure different cluster ids for the two groups
        self.assertNotEqual(labels[0], labels[2])

    def test_cosine_on_float_features(self):
        # two orthogonal vectors plus a duplicate -> cluster of size 2 and singleton
        features = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        c = ButinaClusterer(cutoff=0.9, metric="cosine")
        c.fit(features=features)
        clusters = c.get_clusters()
        # Expect cluster of [0,1] and [2]
        sorted_clusters = sorted([sorted(cl) for cl in clusters])
        self.assertEqual(sorted_clusters, [[0, 1], [2]])
        labels = c.get_labels()
        # check labels reflect clusters
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_zero_vectors_tanimoto(self):
        # rows of zeros -> union = 0, tanimoto should produce 0 similarity
        features = np.zeros((3, 4), dtype=int)
        c = ButinaClusterer(cutoff=0.1, metric="tanimoto")
        c.fit(features=features)
        clusters = c.get_clusters()
        # With include_self True default, each index forms singleton cluster
        sorted_clusters = sorted([sorted(cl) for cl in clusters])
        self.assertEqual(sorted_clusters, [[0], [1], [2]])
        labels = c.get_labels()
        assert_array_equal(labels, np.array([0, 1, 2], dtype=int))

    def test_build_neighbor_sets_include_self_false(self):
        sim = np.array([[1.0, 0.8], [0.8, 1.0]])
        c = ButinaClusterer(cutoff=0.9, include_self=False)
        # call fit with sim so neighbor sets created
        c.fit(sim=sim)
        clusters = c.get_clusters()
        sorted_clusters = sorted([sorted(cl) for cl in clusters])
        self.assertEqual(sorted_clusters, [[0], [1]])

    def test_min_cluster_size_filtering(self):
        # create sim that makes one cluster of size 3 and one singleton
        sim = np.array(
            [
                [1.0, 0.95, 0.95, 0.0],
                [0.95, 1.0, 0.95, 0.0],
                [0.95, 0.95, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        c = ButinaClusterer(cutoff=0.9, min_cluster_size=2)
        c.fit(sim=sim)
        clusters = c.get_clusters()
        # cluster of size 3 meets threshold, singleton should be filtered out
        self.assertEqual(clusters, [[0, 1, 2]])
        labels = c.get_labels()
        # indices 0,1,2 -> cluster 0, index 3 -> -1
        expected = np.array([0, 0, 0, -1], dtype=int)
        assert_array_equal(labels, expected)

    def test_sort_clusters_flag(self):
        # setup: cluster of size 1 and cluster of size 3; verify sort_clusters controls order
        sim = np.array(
            [
                [1.0, 0.99, 0.99, 0.0],
                [0.99, 1.0, 0.99, 0.0],
                [0.99, 0.99, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        c_sorted = ButinaClusterer(cutoff=0.9, sort_clusters=True)
        c_sorted.fit(sim=sim)
        self.assertEqual(c_sorted.get_clusters()[0], [0, 1, 2])  # largest first

        c_unsorted = ButinaClusterer(cutoff=0.9, sort_clusters=False)
        c_unsorted.fit(sim=sim)
        # greedy algorithm picks the best neighbor-count first; best may be the big cluster
        # but order may differ; assert that when not sorting the clusters are a permutation of the sorted ones
        clusters_unsorted_sorted = sorted(
            [sorted(c) for c in c_unsorted.get_clusters()]
        )
        clusters_sorted_sorted = sorted([sorted(c) for c in c_sorted.get_clusters()])
        self.assertEqual(clusters_unsorted_sorted, clusters_sorted_sorted)

    def test_features_must_be_2d(self):
        c = ButinaClusterer()
        with self.assertRaises(ValueError):
            c.fit(features=[1, 2, 3])  # 1D-like

    def test_internal_tanimoto_matrix_values(self):
        X = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 0]])
        sim = ButinaClusterer._tanimoto_matrix(X)
        # compute expected pairwise manually: rows 0 and 1 intersection 1, union 3 -> 1/3
        self.assertAlmostEqual(sim[0, 1], 1.0 / 3.0)
        # zero row vs others -> 0
        self.assertEqual(sim[2, 0], 0.0)
        self.assertEqual(sim[2, 1], 0.0)
        # diagonal: current impl yields 1.0 for non-zero rows, 0.0 for zero rows
        row_nonzero = X.astype(bool).sum(axis=1) > 0
        expected_diag = np.where(row_nonzero, 1.0, 0.0)
        assert_allclose(np.diag(sim), expected_diag)

    def test_internal_cosine_matrix_values(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        sim = ButinaClusterer._cosine_matrix(X)
        # orthogonal vectors -> 0
        assert_allclose(sim[0, 1], 0.0)
        # vector with itself -> 1
        assert_allclose(sim[2, 2], 1.0)
        # cosine between [1,0] and [1,1] -> 1/sqrt(2)
        assert_allclose(sim[0, 2], 1.0 / np.sqrt(2.0), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
