"""Transport mapped annotations onto explicitly declared, map-free input graphs.

The task input is the source reaction field. An annotation or mapper output
may reorder atoms and molecules, but cannot silently change chemistry, omit
input components, or supply atoms absent from the declared graphs.
"""

from __future__ import annotations

from synrxn.metrics.reference import _mol, product_multiset, transport_product_origins


def reaction_inputs(reaction: str) -> dict[str, str]:
    """Include both recorded reactants and agents in the mapping source pool."""
    from rdkit import Chem
    parts = reaction.split(">")
    if len(parts) != 3 or not parts[0] or not parts[2]:
        raise ValueError("reaction must contain reactants>agents>products")
    before = ".".join(side for side in parts[:2] if side)
    return {"reactants": Chem.MolToSmiles(_mol(before), isomericSmiles=True),
            "products": Chem.MolToSmiles(_mol(parts[2]), isomericSmiles=True)}


def _attributes(atom):
    return (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(), atom.GetIsAromatic(),
            atom.GetTotalNumHs(includeNeighbors=True))


def _align(source, declared, budget):
    """Find a full, chemically faithful atom-index transport."""
    from rdkit import Chem
    if (source.GetNumAtoms() != declared.GetNumAtoms() or
            Chem.MolToSmiles(source, isomericSmiles=True) != Chem.MolToSmiles(declared, isomericSmiles=True)):
        raise ValueError("annotation changes the declared molecular graph")

    def valid(match):
        return (len(match) == source.GetNumAtoms() and
                all(_attributes(source.GetAtomWithIdx(i)) == _attributes(declared.GetAtomWithIdx(j))
                    for i, j in enumerate(match)))

    first = declared.GetSubstructMatch(source, useChirality=True)
    if valid(first):
        return first
    for match in declared.GetSubstructMatches(source, useChirality=True, uniquify=False, maxMatches=budget):
        if valid(match):
            return match
    raise ValueError("no faithful annotation transport within the declared alignment budget")


def product_origins_from_mapped(reaction: str, inputs: dict[str, str], alignment_budget: int = 4096):
    """Read all product origins while allowing unused or unmapped reactant atoms.

    Returned integers index reactant atoms in the declared input; vector order
    indexes product atoms in that input. Mapping labels are never input features.
    """
    from rdkit import Chem
    if type(alignment_budget) is not int or alignment_budget < 1:
        raise ValueError("alignment budget must be a positive integer")
    parts = reaction.split(">")
    if len(parts) != 3:
        raise ValueError("mapped output must have three reaction fields")
    raw_reactants = ".".join(side for side in parts[:2] if side)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    reactants, products = Chem.MolFromSmiles(raw_reactants, params), Chem.MolFromSmiles(parts[2], params)
    if reactants is None or products is None:
        raise ValueError("mapped reaction contains an invalid molecule")
    reactant_labels = [a.GetAtomMapNum() for a in reactants.GetAtoms()]
    product_labels = [a.GetAtomMapNum() for a in products.GetAtoms()]
    positive = [label for label in reactant_labels if label > 0]
    if len(set(positive)) != len(positive):
        raise ValueError("reactant annotation has ambiguous duplicate positive map labels")
    if (not product_labels or 0 in product_labels or len(set(product_labels)) != len(product_labels) or
            not set(product_labels) <= set(positive)):
        raise ValueError("every product atom requires one unique supplied reactant origin")
    lookup = {label: i for i, label in enumerate(reactant_labels) if label > 0}
    origins = tuple(lookup[label] for label in product_labels)
    for product, reactant in enumerate(origins):
        a, b = reactants.GetAtomWithIdx(reactant), products.GetAtomWithIdx(product)
        if (a.GetAtomicNum(), a.GetIsotope()) != (b.GetAtomicNum(), b.GetIsotope()):
            raise ValueError("mapped atom changes element or isotope")
    for mol in (reactants, products):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    if (product_multiset(inputs["reactants"]) != product_multiset(raw_reactants) or
            product_multiset(inputs["products"]) != product_multiset(parts[2])):
        raise ValueError("annotation changes the declared reaction molecules")
    left = _align(reactants, _mol(inputs["reactants"]), alignment_budget)
    right = _align(products, _mol(inputs["products"]), alignment_budget)
    return transport_product_origins(origins, left, right)
