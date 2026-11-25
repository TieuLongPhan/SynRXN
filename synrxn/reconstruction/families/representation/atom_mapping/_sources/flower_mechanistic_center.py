"""Project SynKit MTGs into mechanistic centers without ordered histories.

This experimental projection keeps initial/final chemical labels and the set
of states observed on each atom and bond. It deliberately forgets time indices,
event multiplicities and global intermediate-state correlations.
"""

from __future__ import annotations

import json
from copy import deepcopy

import networkx as nx

NODE_FIELDS = ("aromatic", "hcount", "charge", "radical", "lone_pairs", "present")


def encode(value) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _side(graph, right):
    nodes = {
        node: (attrs["element"][right], *(attrs[field][right] for field in NODE_FIELDS))
        for node, attrs in graph.nodes(data=True)
    }
    edges = {
        tuple(sorted((left, right_node))): attrs["order"][right]
        for left, right_node, attrs in graph.edges(data=True)
        if attrs["order"][right] != 0
    }
    return nodes, edges


def pathway_mtg(reactions: list[str]) -> nx.Graph:
    """Fuse original mapped steps after checking closed atom pools and continuity."""
    from synkit.Graph.MTG.mtg import MTG
    from synkit.IO.chem_converter import smiles_to_graph
    from synkit.Graph.ITS.its_construction import ITSConstruction

    _require(len(reactions) > 1, "MTG pilot requires at least two changing steps")
    graphs = []
    states = {}
    fields = ["element", "atom_map", *(field for field in NODE_FIELDS if field != "present")]
    for reaction in reactions:
        # In MolToGraph, True uses existing nonzero map numbers as graph IDs.
        # This is necessary for atom correspondence between sides and steps.
        sides = reaction.split(">>")
        _require(len(sides) == 2, "Malformed mapped step")
        for side in sides:
            if side not in states:
                states[side] = smiles_to_graph(
                    side, drop_non_aam=False, use_index_as_atom_map=True, node_attrs=fields,
                    edge_attrs=["order"], include_stereo_descriptors=False,
                )
                _require(states[side] is not None, "Mapped state graph construction failed")
        graph = ITSConstruction().construct(
            deepcopy(states[sides[0]]), deepcopy(states[sides[1]]), node_attrs=fields, edge_attrs=["order"]
        )
        for node, attrs in graph.nodes(data=True):
            _require(attrs["present"] == (True, True), "Step has an open atom pool")
            _require(node > 0 and attrs["atom_map"] == (node, node), "Missing or inconsistent atom-map IDs")
            _require(attrs["element"][0] == attrs["element"][1], "Mapped atom changes element")
        if graphs:
            _require(set(graphs[-1]) == set(graph), "Atom pool changes between steps")
            _require(_side(graphs[-1], 1) == _side(graph, 0), "Consecutive pathway states disagree")
        graphs.append(graph)
    return MTG(graphs).get_mtg()


def growing_centers(reactions: list[str]):
    """Fuse one complete path once; project each prefix without a depth cutoff."""
    full = pathway_mtg(reactions)
    for length in range(2, len(reactions) + 1):
        prefix = nx.Graph()
        for node, attrs in full.nodes(data=True):
            prefix.add_node(node, element=attrs["element"],
                            **{field: attrs[field][:length + 1] for field in NODE_FIELDS})
        for left, right, attrs in full.edges(data=True):
            history = attrs["order"][:length + 1]
            if any(history):
                prefix.add_edge(left, right, order=history)
        center, statistics = mechanistic_center(prefix)
        yield length, center, statistics


def node_states(attrs: dict) -> list[tuple]:
    """Keep correlated local states; normalize only the common H-count offset."""
    values = [attrs[field] for field in NODE_FIELDS]
    count = len(values[0])
    _require(all(len(value) == count for value in values), "Nonrectangular MTG node timeline")
    h_offset = min(attrs["hcount"][0], attrs["hcount"][-1])
    states = []
    for index in range(count):
        state = [value[index] for value in values]
        state[1] -= h_offset
        states.append(tuple(state))
    return states


def mechanistic_center(mtg: nx.Graph, radius: int = 0) -> tuple[nx.Graph, dict]:
    """Retain all changing carriers and induced edges, with optional heavy context.

    Radius zero includes unchanged bonds between active atoms. Positive radii
    add heavy-atom neighbours in the union graph. Mapped hydrogens that change
    remain active atoms; unchanged pendant H atoms do not form the context shell.
    """
    _require(radius >= 0, "Context radius must be nonnegative")
    states = {node: node_states(attrs) for node, attrs in mtg.nodes(data=True)}
    changed_nodes = {node for node, history in states.items() if len(set(history)) > 1}
    net_nodes = {node for node, history in states.items() if history[0] != history[-1]}
    changed_edges = []
    returning_edges = []
    transient_formed = []
    for left, right, attrs in mtg.edges(data=True):
        history = attrs["order"]
        if len(set(history)) > 1:
            changed_edges.append((left, right))
            changed_nodes.update((left, right))
            if history[0] == history[-1]:
                returning_edges.append((left, right))
                if history[0] == 0:
                    transient_formed.append((left, right))
        if history[0] != history[-1]:
            net_nodes.update((left, right))
    selected = set(changed_nodes)
    for _ in range(radius):
        selected.update(
            neighbor for node in tuple(selected) for neighbor in mtg[node] if mtg.nodes[neighbor]["element"] != "H"
        )
    result = nx.Graph()
    for node in sorted(selected):
        history = states[node]
        result.add_node(
            node,
            label=encode(
                {
                    "element": mtg.nodes[node]["element"],
                    "core": node in changed_nodes,
                    "initial": history[0],
                    "final": history[-1],
                    "visited": sorted(set(history)),
                }
            ),
        )
    for left, right, attrs in mtg.subgraph(selected).edges(data=True):
        history = attrs["order"]
        result.add_edge(
            left,
            right,
            label=encode(
                {
                    "initial": history[0],
                    "final": history[-1],
                    "visited": sorted(set(history)),
                }
            ),
        )
    stats = {
        "core_atoms": len(changed_nodes),
        "net_core_atoms": len(net_nodes),
        "additional_active_atoms_over_net": len(changed_nodes - net_nodes),
        "changed_edges": len(changed_edges),
        "returning_edges": len(returning_edges),
        "transient_formed_edges": len(transient_formed),
        "context_atoms": len(selected - changed_nodes),
        "returning_edge_witnesses": [
            {"atom_maps": [left, right], "states": list(mtg.edges[left, right]["order"])}
            for left, right in returning_edges
        ],
    }
    result.graph.update(projection="unordered-local-state-mechanistic-center", context_radius=radius)
    return result, stats


class ExactClusters:
    """Use WL only for buckets; confirm every merge with graph isomorphism."""

    def __init__(self):
        self.graphs = []
        self.buckets = {}

    def assign(self, graph: nx.Graph) -> int:
        signature = nx.weisfeiler_lehman_graph_hash(graph, node_attr="label", edge_attr="label", iterations=3)
        key = (len(graph), graph.number_of_edges(), signature)
        candidates = self.buckets.setdefault(key, [])
        match = nx.algorithms.isomorphism.categorical_node_match("label", None)
        edge_match = nx.algorithms.isomorphism.categorical_edge_match("label", None)
        for index in candidates:
            if nx.is_isomorphic(graph, self.graphs[index], node_match=match, edge_match=edge_match):
                return index
        index = len(self.graphs)
        self.graphs.append(graph)
        candidates.append(index)
        return index
