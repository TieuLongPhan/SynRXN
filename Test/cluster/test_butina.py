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
            c.fit()  

    def test_fit_sim_validation(self):
        c = ButinaClusterer()
        with self.assertRaises(ValueError):
            c.fit(sim=np.ones((2, 3)))
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
        sorted_clusters = sorted([sorted(cl) for cl in clusters])
        self.assertEqual(sorted_clusters, [[0, 1], [2, 3]])
        self.assertIn(labels[0], (0, 1))
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_cosine_on_float_features(self):
        features = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        c = ButinaClusterer(cutoff=0.9, metric="cosine")
        c.fit(features=features)
        clusters = c.get_clusters()
        sorted_clusters = sorted([sorted(cl) for cl in clusters])
        self.assertEqual(sorted_clusters, [[0, 1], [2]])
        labels = c.get_labels()
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])

    def test_zero_vectors_tanimoto(self):
        features = np.zeros((3, 4), dtype=int)
        c = ButinaClusterer(cutoff=0.1, metric="tanimoto")
        c.fit(features=features)
        clusters = c.get_clusters()
        sorted_clusters = sorted([sorted(cl) for cl in clusters])
        self.assertEqual(sorted_clusters, [[0], [1], [2]])
        labels = c.get_labels()
        assert_array_equal(labels, np.array([0, 1, 2], dtype=int))

    def test_build_neighbor_sets_include_self_false(self):
        sim = np.array([[1.0, 0.8], [0.8, 1.0]])
        c = ButinaClusterer(cutoff=0.9, include_self=False)
        c.fit(sim=sim)
        clusters = c.get_clusters()
        sorted_clusters = sorted([sorted(cl) for cl in clusters])
        self.assertEqual(sorted_clusters, [[0], [1]])

    def test_min_cluster_size_filtering(self):
        
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
        self.assertEqual(clusters, [[0, 1, 2]])
        labels = c.get_labels()
        expected = np.array([0, 0, 0, -1], dtype=int)
        assert_array_equal(labels, expected)

    def test_sort_clusters_flag(self):
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
        self.assertEqual(c_sorted.get_clusters()[0], [0, 1, 2])  

        c_unsorted = ButinaClusterer(cutoff=0.9, sort_clusters=False)
        c_unsorted.fit(sim=sim)
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

        self.assertAlmostEqual(sim[0, 1], 1.0 / 3.0)
        
        self.assertEqual(sim[2, 0], 0.0)
        self.assertEqual(sim[2, 1], 0.0)
       
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
