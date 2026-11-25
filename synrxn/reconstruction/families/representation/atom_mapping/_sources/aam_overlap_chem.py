"""Stereo-independent reaction identities, partial maps and mapping-class graphs."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
import zlib

import networkx as nx
from rdkit import Chem


def encode(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=True)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def parse_reaction(value):
    # Dative bonds use -> inside an endpoint and are not reaction separators.
    parts = re.split(r'(?<!-)>', value)
    if len(parts) != 3 or not parts[0] or not parts[2]:
        raise ValueError('Expected two nonempty reaction endpoints')
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    mols = tuple(Chem.MolFromSmiles(side, parameters) for side in (parts[0], parts[2]))
    if any(mol is None or not mol.GetNumAtoms() for mol in mols):
        raise ValueError('Unparseable reaction endpoint')
    return mols


def components(mol, *, remove_maps=True, remove_hs=True, keep_stereo=False):
    copy = Chem.Mol(mol)
    if not keep_stereo:
        Chem.RemoveStereochemistry(copy)
    if remove_maps:
        for atom in copy.GetAtoms():
            atom.SetAtomMapNum(0)
    if remove_hs:
        copy = Chem.RemoveHs(copy)
    return Counter(Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
                   for fragment in Chem.GetMolFrags(copy, asMols=True))


def reaction_text(mols, *, remove_hs=True):
    sides = ('.'.join(sorted(components(mol, remove_hs=remove_hs).elements())) for mol in mols)
    return '>>'.join(sides)


def identities(value):
    mols = parse_reaction(value)
    sides = [components(mol) for mol in mols]
    full = '>>'.join('.'.join(sorted(side.elements())) for side in sides)
    common = sides[0] & sides[1]
    active = [side - common for side in sides]
    # Never collapse all identity/transport reactions into the empty reaction.
    if not all(active):
        active, common = sides, Counter()
    normalized = '>>'.join('.'.join(sorted(side.elements())) for side in active)
    ends = normalized.split('>>')
    return {'canonical_rxn': full, 'normalized_rxn': normalized,
            'exact_key': sha(full), 'normalized_key': sha(normalized),
            'undirected_key': sha('>>'.join(sorted(ends))),
            'reverse': ends[0] > ends[1], 'spectator_pairs': sum(common.values())}


def mapping_status(mols, *, heavy=True):
    sides = [[atom for atom in mol.GetAtoms() if not heavy or atom.GetAtomicNum() != 1]
             for mol in mols]
    maps, counts, missing = [], [], False
    for atoms in sides:
        labels = [atom.GetAtomMapNum() for atom in atoms]
        missing |= 0 in labels or not atoms
        values = {}
        for atom in atoms:
            label = atom.GetAtomMapNum()
            if not label:
                continue
            identity = atom.GetAtomicNum(), atom.GetIsotope()
            if label in values and values[label] != identity:
                return 'mapped_atom_identity_mismatch'
            values[label] = identity
        maps.append(values)
        counts.append(Counter(n for n in labels if n))
    common = maps[0].keys() & maps[1].keys()
    if any(maps[0][number] != maps[1][number] for number in common):
        return 'mapped_atom_identity_mismatch'
    if missing or counts[0] != counts[1]:
        return 'partial'
    return 'equivalent' if any(n > 1 for n in counts[0].values()) else 'ok'


def active_molecules(mols):
    fragments = [list(Chem.GetMolFrags(mol, asMols=True)) for mol in mols]
    for side in fragments:
        for mol in side:
            Chem.RemoveStereochemistry(mol)
    labels = [[Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) for mol in side]
              for side in fragments]
    shared = Counter(labels[0]) & Counter(labels[1])
    inventories = [Counter(a.GetAtomMapNum() for m in side for a in m.GetAtoms() if a.GetAtomicNum() != 1)
                   for side in fragments]
    # A repeated class spanning active and unchanged components permits atom
    # exchange. Such a component is not a certified mapped spectator.
    unsafe = set()
    for side, names in zip(fragments, labels):
        for mol, name in zip(side, names):
            counts = Counter(a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
            if 0 in counts or any(counts[n] != total[n] for n in counts for total in inventories):
                unsafe.add(name)
    shared = Counter({name: count for name, count in shared.items() if name not in unsafe})
    active = []
    for side, names in zip(fragments, labels):
        remaining, remove = [], shared.copy()
        for mol, name in zip(side, names):
            if remove[name]:
                remove[name] -= 1
            else:
                remaining.append(mol)
        if not remaining:
            return mols
        result = remaining[0]
        for mol in remaining[1:]:
            result = Chem.CombineMols(result, mol)
        active.append(result)
    return tuple(active)


def reference_info(value):
    mols = parse_reaction(value)
    status = ('generic_structure' if any(a.GetAtomicNum() == 0 for m in mols for a in m.GetAtoms())
              else mapping_status(mols))
    active = active_molecules(mols) if status in {'ok', 'equivalent', 'partial'} else mols
    explicit = mapping_status(mols, heavy=False)
    hydrogen_atoms = sum(a.GetAtomicNum() == 1 for m in mols for a in m.GetAtoms())
    represented = explicit in {'ok', 'equivalent', 'partial'} and hydrogen_atoms > 0
    heavy_labels = [[a.GetAtomMapNum() for a in m.GetAtoms() if a.GetAtomicNum() != 1] for m in mols]
    repeated = any(any(n > 1 for n in Counter(x for x in side if x).values()) for side in heavy_labels)
    missing = sum(side.count(0) for side in heavy_labels)
    return {'mols': mols, 'active': active, 'mapping_status': status,
            'mapping_encoding': 'equivalence_classes' if repeated else 'individual_atoms',
            'unmapped_heavy_atoms': missing,
            'explicit_mapping_status': explicit,
            'target_rxn': reaction_text(mols), 'target_normalized_rxn': reaction_text(active),
            'represented_rxn': reaction_text(mols, remove_hs=False) if represented else '',
            'represented_normalized_rxn': reaction_text(active, remove_hs=False) if represented else '',
            'explicit_h_atoms': hydrogen_atoms}


def atom_state(atom):
    if atom is None:
        return None
    isotopic_h = sorted((a.GetIsotope(), a.GetFormalCharge()) for a in atom.GetNeighbors()
                        if a.GetAtomicNum() == 1 and a.GetIsotope())
    return [atom.GetFormalCharge(), atom.GetNumRadicalElectrons(), atom.GetIsAromatic(),
            atom.GetTotalNumHs(includeNeighbors=True),
            '', isotopic_h]


def equivalence_graph(mols, *, heavy):
    """Keep each endpoint atom; a class node represents a set of allowed correspondences."""
    graph = nx.Graph(kind='classes')
    groups, atom_ids = {}, []
    for side, mol in enumerate(mols):
        indices = {}
        for atom in mol.GetAtoms():
            if heavy and atom.GetAtomicNum() == 1:
                continue
            node = len(graph)
            indices[atom.GetIdx()] = node
            graph.add_node(node, label=encode(['atom', atom.GetAtomicNum(), atom.GetIsotope(), side,
                                              atom_state(atom)]))
            if atom.GetAtomMapNum():
                groups.setdefault(atom.GetAtomMapNum(), [[], []])[side].append(node)
        atom_ids.append(indices)
    for side, mol in enumerate(mols):
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in atom_ids[side] and b in atom_ids[side]:
                graph.add_edge(atom_ids[side][a], atom_ids[side][b], label=encode(['bond', str(bond.GetBondType())]))
    for left, right in groups.values():
        node = len(graph)
        graph.add_node(node, label=encode(['class', len(left), len(right)]))
        for atom in left + right:
            graph.add_edge(node, atom, label=encode(['membership']))
    return graph


def its_graph(mols, *, heavy=True):
    mols = tuple(Chem.Mol(m) for m in mols)
    for mol in mols:
        Chem.RemoveStereochemistry(mol)
    if mapping_status(mols, heavy=heavy) == 'mapped_atom_identity_mismatch':
        raise ValueError('A mapping class mixes elements or isotopes')
    labels = [[a.GetAtomMapNum() for a in m.GetAtoms() if (not heavy or a.GetAtomicNum() != 1)
               and a.GetAtomMapNum()] for m in mols]
    if any(len(side) != len(set(side)) for side in labels):
        return equivalence_graph(mols, heavy=heavy)
    # Unknown atoms have distinct endpoint-only IDs. No missing correspondence is invented.
    for side, mol in enumerate(mols):
        for atom in mol.GetAtoms():
            if not atom.GetAtomMapNum():
                atom.SetIntProp('_unknown_map', -(1 + atom.GetIdx() + side * (mols[0].GetNumAtoms() + 1)))

    def mapped_number(atom):
        return atom.GetAtomMapNum() or atom.GetIntProp('_unknown_map')

    atoms = [{mapped_number(a): a for a in m.GetAtoms() if not heavy or a.GetAtomicNum() != 1} for m in mols]
    graph = nx.Graph()
    for number in sorted(atoms[0].keys() | atoms[1].keys()):
        left, right = atoms[0].get(number), atoms[1].get(number)
        atom = left if left is not None else right
        graph.add_node(number, label=encode([atom.GetAtomicNum(), atom.GetIsotope(),
                                            atom_state(left), atom_state(right)]))
    edges = []
    for mol in mols:
        side = {}
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtom(), bond.GetEndAtom()
            if heavy and (a.GetAtomicNum() == 1 or b.GetAtomicNum() == 1):
                continue
            key = tuple(sorted((mapped_number(a), mapped_number(b))))
            side[key] = [str(bond.GetBondType()), '']
        edges.append(side)
    for a, b in edges[0].keys() | edges[1].keys():
        graph.add_edge(a, b, label=encode([edges[0].get((a, b)), edges[1].get((a, b))]))
    return graph


def pack(graph):
    value = {'kind': graph.graph.get('kind', 'its'), 'nodes': [(n, a['label']) for n, a in graph.nodes(data=True)],
             'edges': [(a, b, x['label']) for a, b, x in graph.edges(data=True)]}
    return zlib.compress(encode(value).encode())


def unpack(value, *, reverse=False):
    data = json.loads(zlib.decompress(value))
    graph = nx.Graph(kind=data.get('kind', 'its'))
    for number, label in data['nodes']:
        if reverse:
            state = json.loads(label)
            if graph.graph['kind'] == 'classes':
                state = (state[:3] + [1 - state[3], state[4]] if state[0] == 'atom'
                         else [state[0], state[2], state[1]])
            else:
                state = state[:2] + [state[3], state[2]]
            label = encode(state)
        graph.add_node(number, label=label)
    for a, b, label in data['edges']:
        if reverse and graph.graph['kind'] != 'classes':
            label = encode(list(reversed(json.loads(label))))
        graph.add_edge(a, b, label=label)
    return graph


class ITSClasses:
    """Bucket by invariants; certify every merged class by exact isomorphism."""

    def __init__(self):
        self.graphs, self.buckets = [], {}

    def assign(self, graph):
        signature = (len(graph), graph.number_of_edges(),
                     nx.weisfeiler_lehman_graph_hash(graph, node_attr='label', edge_attr='label', iterations=3))
        for index in self.buckets.get(signature, []):
            if nx.is_isomorphic(graph, self.graphs[index],
                                node_match=lambda a, b: a['label'] == b['label'],
                                edge_match=lambda a, b: a['label'] == b['label']):
                return index
        index = len(self.graphs)
        self.graphs.append(graph)
        self.buckets.setdefault(signature, []).append(index)
        return index
