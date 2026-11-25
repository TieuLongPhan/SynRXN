"""Compare specified atom correspondences without expanding equivalence permutations."""
from __future__ import annotations

from collections import Counter, defaultdict
import json

import networkx as nx


class Constraints:
    """Endpoint atom graphs plus balanced, specified mapping classes."""

    def __init__(self, graph):
        self.graph = nx.Graph()
        self.groups = defaultdict(lambda: [[], []])
        self.membership = {}
        if graph.graph.get('kind', 'its') == 'classes':
            self._classes(graph)
        else:
            self._its(graph)
        self.closed = {g for g, (left, right) in self.groups.items() if left and len(left) == len(right)}
        self.required = {n for g in self.closed for side in self.groups[g] for n in side}
        self.complete = len(self.required) == len(self.graph)
        self.bijective = self.complete and all(len(left) == 1 for left, _ in self.groups.values())
        self.partner = {}
        for group in self.closed:
            left, right = self.groups[group]
            if len(left) == 1:
                self.partner[left[0]], self.partner[right[0]] = right[0], left[0]

    def _atom(self, node, side, element, isotope, state, group=None):
        self.graph.add_node(node, label=json.dumps([side, element, isotope, state], separators=(',', ':')),
                            side=side, identity=(element, isotope))
        if group is not None:
            self.membership[node] = group
            self.groups[group][side].append(node)

    def _its(self, graph):
        nodes = {}
        for index, (number, attrs) in enumerate(graph.nodes(data=True)):
            element, isotope, left, right = json.loads(attrs['label'])
            for side, state in enumerate([left, right]):
                if state is not None:
                    node = index * 2 + side
                    nodes[number, side] = node
                    self._atom(node, side, element, isotope, state, number)
        for a, b, attrs in graph.edges(data=True):
            for side, state in enumerate(json.loads(attrs['label'])):
                if state is not None:
                    self.graph.add_edge(nodes[a, side], nodes[b, side], label=state[0])

    def _classes(self, graph):
        labels = {n: json.loads(a['label']) for n, a in graph.nodes(data=True)}
        for number, label in labels.items():
            if label[0] == 'atom':
                self._atom(number, label[3], label[1], label[2], label[4])
        for number, label in labels.items():
            if label[0] == 'class':
                for atom in graph.neighbors(number):
                    self.membership[atom] = number
                    self.groups[number][labels[atom][3]].append(atom)
        for a, b, attrs in graph.edges(data=True):
            label = json.loads(attrs['label'])
            if label[0] == 'bond':
                self.graph.add_edge(a, b, label=label[1])

    def allowed(self, left, right):
        a, b = self.membership.get(left), self.membership.get(right)
        return a == b if a in self.closed or b in self.closed else True


def compatible_alignment(first, second, alignment):
    """Whether this structural alignment has a common completion of specified pairs."""
    if first.complete and second.complete:
        sides = [Counter(), Counter()]
        for a, b in alignment.items():
            sides[first.graph.nodes[a]['side']][first.membership[a], second.membership[b]] += 1
        return sides[0] == sides[1]
    # Unknown/unrepresented correspondences are not invented. Test feasibility
    # only: every explicitly constrained atom must be coverable by a matching.
    matching = nx.Graph()
    left = [n for n, a in first.graph.nodes(data=True) if a['side'] == 0]
    right = [n for n, a in first.graph.nodes(data=True) if a['side'] == 1]
    required = first.required | {a for a, b in alignment.items() if b in second.required}
    scale = len(first.graph) + 1
    for a in left:
        for b in right:
            if (first.graph.nodes[a]['identity'] == first.graph.nodes[b]['identity']
                    and first.allowed(a, b) and second.allowed(alignment[a], alignment[b])):
                matching.add_edge(a, b, weight=scale * ((a in required) + (b in required)) + 1)
    pairs = nx.max_weight_matching(matching)
    covered = {n for pair in pairs for n in pair}
    return required <= covered


def refines_alignment(first, second, alignment):
    """Certify a complete reference's allowed mappings are contained in the other."""
    if not first.complete:
        return False
    for group in first.closed:
        left, right = first.groups[group]
        for a in left:
            for b in right:
                if not second.allowed(alignment[a], alignment[b]):
                    return False
    return True


class ConstraintMatcher(nx.algorithms.isomorphism.GraphMatcher):
    def __init__(self, first, second, *, refine=False):
        super().__init__(first.graph, second.graph,
                         node_match=lambda a, b: a['label'] == b['label'],
                         edge_match=lambda a, b: a['label'] == b['label'])
        self.first, self.second, self.refine = first, second, refine

    def semantic_feasibility(self, a, b):
        if not super().semantic_feasibility(a, b):
            return False
        partner = self.first.partner.get(a)
        if partner in self.core_1 and not self.second.allowed(b, self.core_1[partner]):
            return False
        partner = self.second.partner.get(b)
        if partner in self.core_2 and not self.first.allowed(a, self.core_2[partner]):
            return False
        if len(self.core_1) + 1 == len(self.first.graph):
            alignment = dict(self.core_1)
            alignment[a] = b
            check = refines_alignment if self.refine else compatible_alignment
            return check(self.first, self.second, alignment)
        return True


def compare(first, second, *, refine=False, distinct=False):
    """Call under the audit's time limit; exhaustive failure certifies incompatibility."""
    a, b = Constraints(first), Constraints(second)
    if distinct and not refine and a.bijective and b.bijective:
        # Distinct exact ITS classes with complete individual maps cannot agree.
        return False
    return ConstraintMatcher(a, b, refine=refine).is_isomorphic()
