"""Shared graph-fingerprint contracts and algorithm-specific behavior."""

import networkx as nx
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from synrxn.cluster.ecfp import ECFP, _norm_value as ecfp_norm
from synrxn.cluster.wlfp import WLFP, _norm_value as wlfp_norm


@pytest.fixture(params=[ECFP, WLFP], ids=["ecfp", "wlfp"])
def fingerprint_type(request):
    return request.param


def make_graph(fingerprint_type):
    graph = nx.Graph()
    if fingerprint_type is ECFP:
        for node, element in enumerate(["C", "C", "O"], start=1):
            graph.add_node(node, element=element, aromatic=False, hcount=0, charge=0)
        graph.add_edges_from([(1, 2), (2, 3)], standard_order=1.0)
    else:
        for node, element in enumerate(["C", "O", "N"], start=1):
            graph.add_node(node, element=element)
        graph.add_edges_from([(1, 2), (1, 3)], standard_order=1.0)
    return graph


def make_fingerprint(fingerprint_type, level=1, **options):
    parameter = "radius" if fingerprint_type is ECFP else "iterations"
    return fingerprint_type(**{parameter: level}, **options)


@pytest.mark.parametrize("normalize,pairs", [
    pytest.param(ecfp_norm, [(None, None), ("abc", "abc"), (5, 5),
                            (3.14, 3.14), (True, True)], id="ecfp-primitives"),
    pytest.param(ecfp_norm, [([1, 2, 3], (1, 2, 3)), ((1, 2), (1, 2)),
                            ({3, 1, 2}, (1, 2, 3))], id="ecfp-sequences"),
    pytest.param(ecfp_norm, [({"b": 2, "a": 1}, (("a", 1), ("b", 2)))],
                 id="ecfp-dictionary"),
    pytest.param(ecfp_norm, [(np.int32(7), 7), (np.float64(2.5), 2.5)],
                 id="ecfp-numpy-scalars"),
    pytest.param(wlfp_norm, [(None, None), ("abc", "abc"), (42, 42),
                            (3.14, 3.14), (True, True)], id="wlfp-primitives"),
    pytest.param(wlfp_norm, [([1, 2, 3], (1, 2, 3)), ((1, 2), (1, 2)),
                            ({3, 1, 2}, (1, 2, 3))], id="wlfp-sequences"),
    pytest.param(wlfp_norm, [({"b": 2, "a": 1}, (("a", 1), ("b", 2))),
                            (np.int32(7), 7), (np.float64(2.5), 2.5)],
                 id="wlfp-dictionary-and-scalars"),
])
def test_normalization(normalize, pairs):
    for value, expected in pairs:
        assert normalize(value) == expected


def test_init_validation(fingerprint_type):
    with pytest.raises(ValueError):
        make_fingerprint(fingerprint_type, level=-1)
    with pytest.raises(ValueError):
        fingerprint_type(hash_bits=7)


def test_type_error_for_non_graph(fingerprint_type):
    with pytest.raises(TypeError):
        fingerprint_type().fingerprint("not a graph")


def test_dense_and_sparse_equivalence_presence(fingerprint_type):
    fp = make_fingerprint(fingerprint_type, n_bits=128, use_counts=False, hash_bits=64)
    graph = make_graph(fingerprint_type)
    vector = fp.fingerprint(graph)
    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.uint8
    assert vector.shape == (128,)
    indices, values = fp.fingerprint(graph, return_sparse=True)
    assert indices.dtype == np.int32
    assert values.dtype == np.int32
    reconstructed = np.zeros_like(vector, dtype=np.uint8)
    reconstructed[indices] = (values != 0).astype(np.uint8)
    assert_array_equal(reconstructed, vector)


def test_dense_and_sparse_equivalence_counts(fingerprint_type):
    level = 1 if fingerprint_type is ECFP else 2
    fp = make_fingerprint(fingerprint_type, level, n_bits=128, use_counts=True, hash_bits=64)
    graph = make_graph(fingerprint_type)
    vector = fp.fingerprint(graph)
    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.int32
    assert vector.shape == (128,)
    indices, values = fp.fingerprint(graph, return_sparse=True)
    reconstructed = np.zeros_like(vector, dtype=np.int32)
    reconstructed[indices] = values
    assert_array_equal(reconstructed, vector)


def test_return_feature_map_structure(fingerprint_type):
    fp = make_fingerprint(fingerprint_type, 2, n_bits=256, use_counts=True, hash_bits=64)
    vector, features = fp.fingerprint(make_graph(fingerprint_type), return_feature_map=True)
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (256,)
    assert isinstance(features, dict)
    for key, feature in features.items():
        assert isinstance(key, int)
        assert isinstance(feature, tuple)
        assert len(feature) >= 2
        level, count = feature[:2]
        assert isinstance(level, int)
        assert isinstance(count, int)
        assert level >= 0
        assert count >= 1


def test_multigraph_edge_handling(fingerprint_type):
    graph = nx.MultiGraph()
    for node, element in [(1, "C"), (2, "O")]:
        attributes = {"aromatic": False, "hcount": 0, "charge": 0} if fingerprint_type is ECFP else {}
        graph.add_node(node, element=element, **attributes)
    graph.add_edge(1, 2, key="a", standard_order=1.0)
    graph.add_edge(1, 2, key="b", standard_order=2.0)
    fp = make_fingerprint(fingerprint_type, n_bits=128, hash_bits=64)
    vector = fp.fingerprint(graph)
    assert isinstance(vector, np.ndarray)
    assert_array_equal(vector, fp.fingerprint(graph))


def test_determinism_across_instances(fingerprint_type):
    graph = make_graph(fingerprint_type)
    level = 1 if fingerprint_type is ECFP else 2
    first = make_fingerprint(fingerprint_type, level, n_bits=128, hash_bits=64)
    second = make_fingerprint(fingerprint_type, level, n_bits=128, hash_bits=64)
    assert_array_equal(first.fingerprint(graph), second.fingerprint(graph))


def test_empty_graph_zero_vector_and_sparse_empty(fingerprint_type):
    fp = make_fingerprint(fingerprint_type, n_bits=64, hash_bits=64)
    graph = nx.Graph()
    vector = fp.fingerprint(graph)
    assert_array_equal(vector, np.zeros(64, dtype=vector.dtype))
    indices, values = fp.fingerprint(graph, return_sparse=True)
    assert indices.size == values.size == 0


def test_to_bitstring(fingerprint_type):
    fp = make_fingerprint(fingerprint_type, n_bits=32, hash_bits=64)
    vector = fp.fingerprint(make_graph(fingerprint_type))
    bitstring = fp.to_bitstring(vector)
    assert isinstance(bitstring, str)
    assert len(bitstring) == 32
    assert bitstring.count("1") == int(vector.astype(bool).sum())


def test_ecfp_neighbor_attributes_change_fingerprint():
    graph = make_graph(ECFP)
    without = ECFP(radius=1, n_bits=256, include_neighbor_attrs=False, hash_bits=64)
    with_neighbors = ECFP(radius=1, n_bits=256, include_neighbor_attrs=True, hash_bits=64)
    assert not np.array_equal(without.fingerprint(graph), with_neighbors.fingerprint(graph))


def test_wlfp_edge_attributes_change_fingerprint():
    graph = make_graph(WLFP)
    with_edges = WLFP(iterations=1, n_bits=256, use_edge_attrs=True, hash_bits=64)
    without = WLFP(iterations=1, n_bits=256, use_edge_attrs=False, hash_bits=64)
    assert not np.array_equal(with_edges.fingerprint(graph), without.fingerprint(graph))


def test_wlfp_neighbor_sorting_is_independent_of_insertion_order():
    original = make_graph(WLFP)
    reordered = nx.Graph()
    reordered.add_nodes_from([(3, {"element": "N"}), (1, {"element": "C"}),
                              (2, {"element": "O"})])
    reordered.add_edges_from([(1, 2), (1, 3)], standard_order=1.0)
    fp = WLFP(iterations=2, n_bits=256, stable_sort_neighbors=True, hash_bits=64)
    assert_array_equal(fp.fingerprint(original), fp.fingerprint(reordered))
