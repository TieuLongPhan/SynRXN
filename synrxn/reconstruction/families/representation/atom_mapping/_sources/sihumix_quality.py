"""Conservative, source-independent quality checks for SIHUMIx mappings."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import re

import networkx as nx
from rdkit import Chem, rdBase

SOURCE_LABELS = {
    "Curated by user": "User-curated",
    "From AAM entry in ChemRecon database": "ChemRecon",
    "Generated via RDT": "RDT-generated",
}
QC_FIELDS = (
    "ground_truth",
    "rxn",
    "reaction_group",
    "parseable",
    "concrete_structures",
    "heavy_element_balanced",
    "element_balanced",
    "charge_balanced",
    "heavy_mapping_valid",
    "explicit_mapping_valid",
    "is_identity",
    "quality_issues",
)


def sha_text(text: str) -> str:
    """Return a stable identity without depending on Python's hash seed."""
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def source_label(value: str | None, mapped: bool) -> str:
    """Keep missing mappings unclassified and reject unknown mapping methods."""
    if not mapped:
        return ""
    if value not in SOURCE_LABELS:
        raise ValueError(f"Unrecognized upstream matching_source: {value!r}")
    return SOURCE_LABELS[value]


def atom_inventory(mol) -> Counter:
    """Count nuclei including both graph H atoms and attached H counts."""
    result = Counter()
    for atom in mol.GetAtoms():
        result[atom.GetAtomicNum(), atom.GetIsotope()] += 1
        result[1, 0] += atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
    return +result


def mapping_valid(left, right, *, heavy_only: bool) -> bool:
    """Require a complete element/isotope-preserving atom-map bijection."""
    sides = [[a for a in mol.GetAtoms() if not heavy_only or a.GetAtomicNum() != 1] for mol in (left, right)]
    maps = [{a.GetAtomMapNum(): (a.GetAtomicNum(), a.GetIsotope()) for a in side} for side in sides]
    return (
        bool(sides[0])
        and all(len(side) == len(index) and 0 not in index for side, index in zip(sides, maps))
        and maps[0] == maps[1]
    )


def unmapped_smiles(mol) -> str:
    """Canonicalize endpoints while preserving stereo, isotopes and free H."""
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    copy = Chem.RemoveHs(copy)
    return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)


def _atom_state(atom) -> tuple:
    return (
        atom.GetAtomicNum(),
        atom.GetIsotope(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(includeNeighbors=True),
        atom.GetIsAromatic(),
        atom.GetNumRadicalElectrons(),
        atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "",
    )


def reaction_graph(left, right, *, reverse: bool = False) -> nx.Graph:
    """Represent the heavy-atom correspondence with paired atom/bond states."""
    if reverse:
        left, right = right, left
    nodes = [{a.GetAtomMapNum(): a for a in mol.GetAtoms() if a.GetAtomicNum() != 1} for mol in (left, right)]
    graph = nx.Graph()
    for number in nodes[0]:
        graph.add_node(number, label=repr((_atom_state(nodes[0][number]), _atom_state(nodes[1][number]))))
    bonds = []
    for mol in (left, right):
        bonds.append(
            {
                tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()))): (
                    str(b.GetBondType()),
                    str(b.GetStereo()),
                )
                for b in mol.GetBonds()
                if b.GetBeginAtom().GetAtomicNum() != 1 and b.GetEndAtom().GetAtomicNum() != 1
            }
        )
    for edge in bonds[0].keys() | bonds[1].keys():
        graph.add_edge(*edge, label=repr((bonds[0].get(edge), bonds[1].get(edge))))
    return graph


def _inspect(aam: str) -> dict:
    result = {
        name: False for name in QC_FIELDS if name not in {"ground_truth", "rxn", "reaction_group", "quality_issues"}
    }
    result.update(ground_truth=aam, rxn=re.sub(r":\d+(?=\])", "", aam), reaction_group="", quality_issues=[])
    try:
        parts = aam.split(">>")
        if len(parts) != 2 or not all(parts):
            raise ValueError("reaction must contain two nonempty endpoints")
        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        left, right = (Chem.MolFromSmiles(s, parser) for s in parts)
        if left is None or right is None or not left.GetNumAtoms() or not right.GetNumAtoms():
            raise ValueError("unparseable or empty endpoint")
        for mol in (left, right):
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        result["parseable"] = True
        result["ground_truth"] = ">>".join(Chem.MolToSmiles(m, isomericSmiles=True) for m in (left, right))
        lsm, rsm = unmapped_smiles(left), unmapped_smiles(right)
        result["rxn"] = lsm + ">>" + rsm
        result["reaction_group"] = sha_text("\n".join(sorted((lsm, rsm))))
        result["is_identity"] = lsm == rsm
        concrete = all(a.GetAtomicNum() != 0 for m in (left, right) for a in m.GetAtoms())
        result["concrete_structures"] = concrete
        if not concrete:
            result["quality_issues"].append("generic_structure")
            return result
        li, ri = atom_inventory(left), atom_inventory(right)
        result["heavy_element_balanced"] = Counter({k: v for k, v in li.items() if k[0] != 1}) == Counter(
            {k: v for k, v in ri.items() if k[0] != 1}
        )
        result["element_balanced"] = li == ri
        result["charge_balanced"] = Chem.GetFormalCharge(left) == Chem.GetFormalCharge(right)
        result["heavy_mapping_valid"] = mapping_valid(left, right, heavy_only=True)
        result["explicit_mapping_valid"] = mapping_valid(left, right, heavy_only=False)
        for field, problem in (
            ("element_balanced", "element_imbalance"),
            ("charge_balanced", "charge_imbalance"),
            ("heavy_mapping_valid", "invalid_heavy_atom_mapping"),
        ):
            if not result[field]:
                result["quality_issues"].append(problem)
        if result["heavy_mapping_valid"]:
            result["_graph"] = reaction_graph(left, right, reverse=lsm > rsm)
    except (ValueError, RuntimeError) as exc:
        result["quality_issues"].append("parse_or_normalization_error")
        result["_error"] = str(exc)
    return result


def inspect_mapping(aam: str) -> dict:
    """Retain failed records as audit rows instead of dropping their payload."""
    if not aam:
        return {
            **{k: False for k in QC_FIELDS if k not in {"ground_truth", "rxn", "reaction_group", "quality_issues"}},
            "ground_truth": "",
            "rxn": "",
            "reaction_group": "",
            "quality_issues": ["missing_mapping"],
        }
    with rdBase.BlockLogs():
        return _inspect(aam)


def mapping_classes(quality: dict[str, dict]) -> dict[str, str]:
    """Use WL buckets followed by exact ITS isomorphism, never hash alone."""
    classes = defaultdict(list)
    assigned = {}
    for aam in sorted(quality, key=sha_text):
        record = quality[aam]
        graph = record.get("_graph")
        if graph is None:
            assigned[aam] = ""
            continue
        bucket = record["reaction_group"], nx.weisfeiler_lehman_graph_hash(
            graph, node_attr="label", edge_attr="label", iterations=3
        )
        found = None
        for representative, identifier in classes[bucket]:
            if nx.is_isomorphic(
                graph,
                representative,
                node_match=lambda a, b: a["label"] == b["label"],
                edge_match=lambda a, b: a["label"] == b["label"],
            ):
                found = identifier
                break
        if found is None:
            found = sha_text(record["reaction_group"] + ":" + sha_text(aam))
            classes[bucket].append((graph, found))
        assigned[aam] = found
    return assigned
