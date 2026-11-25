"""Check transient carriers and invariance of the experimental Flower center."""

import copy
import json
import re

import networkx as nx
import pytest

from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mechanistic_center import (
    ExactClusters,
    mechanistic_center,
    pathway_mtg,
)

INITIAL = "[CH3:1][C:2](=[O:3])[Cl:4].[NH3:5]"
INTERMEDIATE = "[CH3:1][C:2]([O-:3])([Cl:4])[NH3+:5]"
FINAL = "[CH3:1][C:2](=[O:3])[NH3+:5].[Cl-:4]"
STEPS = [INITIAL + ">>" + INTERMEDIATE, INTERMEDIATE + ">>" + FINAL]


def test_restored_carbonyl_and_one_shell_context():
    mtg = pathway_mtg(STEPS)
    center, stats = mechanistic_center(mtg)
    assert set(center) == {2, 3, 4, 5}
    assert stats["additional_active_atoms_over_net"] == 1
    assert stats["returning_edge_witnesses"] == [{"atom_maps": [2, 3], "states": [2.0, 1.0, 2.0]}]
    contextual, context_stats = mechanistic_center(mtg, radius=1)
    assert set(contextual) == {1, 2, 3, 4, 5}
    assert context_stats["context_atoms"] == 1
    assert json.loads(center.edges[2, 3]["label"])["visited"] == [1.0, 2.0]


def test_map_numbering_and_smiles_component_order_do_not_define_classes():
    renamed = []
    for reaction in STEPS:
        renumbered = re.sub(r":(\d+)\]", lambda match: f":{int(match[1]) + 100}]", reaction)
        renamed.append(">>".join(".".join(reversed(side.split("."))) for side in renumbered.split(">>")))
    first, _ = mechanistic_center(pathway_mtg(STEPS), radius=1)
    second, _ = mechanistic_center(pathway_mtg(renamed), radius=1)
    clusters = ExactClusters()
    assert clusters.assign(first) == clusters.assign(second)


def test_transient_explicit_hydrogen_bond_is_kept_in_net_identity():
    water_amine = "[O:1]([H:2])[H:3].[N:4]([H:5])([H:6])[H:7]"
    ions = "[O-:1][H:3].[N+:4]([H:2])([H:5])([H:6])[H:7]"
    graph = pathway_mtg([water_amine + ">>" + ions, ions + ">>" + water_amine])
    center, stats = mechanistic_center(graph)
    assert set(center) == {1, 2, 4}
    assert stats["net_core_atoms"] == 0
    assert stats["transient_formed_edges"] == 1
    assert center.has_edge(2, 4)
    assert json.loads(center.nodes[2]["label"])["element"] == "H"


def test_internal_state_order_and_repetition_are_discarded():
    graph = pathway_mtg(STEPS)
    expanded = copy.deepcopy(graph)
    reordered = copy.deepcopy(graph)
    for _, attrs in expanded.nodes(data=True):
        for field in ("aromatic", "hcount", "charge", "radical", "lone_pairs", "present"):
            start, middle, end = attrs[field]
            attrs[field] = (start, middle, start, end)
    for _, _, attrs in expanded.edges(data=True):
        start, middle, end = attrs["order"]
        attrs["order"] = (start, middle, start, end)
    reordered = copy.deepcopy(expanded)
    for _, attrs in reordered.nodes(data=True):
        for field in ("aromatic", "hcount", "charge", "radical", "lone_pairs", "present"):
            start, middle, extra, end = attrs[field]
            attrs[field] = (start, extra, middle, end)
    for _, _, attrs in reordered.edges(data=True):
        start, middle, extra, end = attrs["order"]
        attrs["order"] = (start, extra, middle, end)
    clusters = ExactClusters()
    first = clusters.assign(mechanistic_center(graph)[0])
    assert first == clusters.assign(mechanistic_center(expanded)[0])
    assert first == clusters.assign(mechanistic_center(reordered)[0])


def test_different_transient_mechanisms_with_same_endpoints_stay_separate():
    graph = pathway_mtg(STEPS)
    alternative = copy.deepcopy(graph)
    alternative.edges[2, 3]["order"] = (2.0, 2.0, 2.0)
    alternative.nodes[3]["charge"] = (0, 0, 0)
    alternative.nodes[3]["lone_pairs"] = (2, 2, 2)
    clusters = ExactClusters()
    assert clusters.assign(mechanistic_center(graph)[0]) != clusters.assign(mechanistic_center(alternative)[0])


def test_inconsistent_intermediate_and_changed_elements_are_rejected():
    with pytest.raises(ValueError, match="states disagree"):
        pathway_mtg([STEPS[0], INITIAL + ">>" + FINAL])
    with pytest.raises(ValueError, match="changes element"):
        pathway_mtg(["[C:1]>>[O:1]", "[O:1]>>[O:1]"])


def test_wl_hash_is_not_accepted_as_proof_of_isomorphism(monkeypatch):
    monkeypatch.setattr(nx, "weisfeiler_lehman_graph_hash", lambda *args, **kwargs: "same-bucket")
    first = nx.path_graph(4)
    second = nx.star_graph(3)
    for graph in (first, second):
        nx.set_node_attributes(graph, "C", "label")
        nx.set_edge_attributes(graph, "single", "label")
    clusters = ExactClusters()
    assert clusters.assign(first) != clusters.assign(second)


def test_shared_intermediate_state_is_converted_once(monkeypatch):
    from synkit.IO import chem_converter
    original = chem_converter.smiles_to_graph
    calls = []

    def observe(smiles, **kwargs):
        calls.append(smiles)
        return original(smiles, **kwargs)

    monkeypatch.setattr(chem_converter, "smiles_to_graph", observe)
    graph = pathway_mtg(STEPS)
    assert len(calls) == len(set(calls)) == 3
    assert set(graph) == {1, 2, 3, 4, 5}
