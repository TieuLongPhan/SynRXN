"""Bounded evaluation with explicit references and all-eligible denominators."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from synrxn.benchmarks.contracts import RESONANCE_PRODUCT_ORIGIN_POLICY, ReferenceRecord, TaskContract


def _mol(smiles: str):
    from rdkit import Chem

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("invalid or empty molecular representation")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def product_multiset(smiles: str | Sequence[str]) -> tuple[str, ...]:
    """Preserve stereo, isotopes, charges and repeated product components."""
    from rdkit import Chem

    components = smiles.split(".") if isinstance(smiles, str) else list(smiles)
    if not components or any(not value.strip() for value in components):
        raise ValueError("empty product component")
    # Accept components containing dots without losing multiplicity.
    normalized = []
    for component in components:
        mol = _mol(component)
        normalized.extend(Chem.MolToSmiles(frag, isomericSmiles=True)
                          for frag in Chem.GetMolFrags(mol, asMols=True))
    return tuple(sorted(normalized))


def transport_correspondence(
    mapping: Sequence[int], reactant_old_to_new: Sequence[int],
    product_old_to_new: Sequence[int],
) -> tuple[int, ...]:
    """Compute q ∘ m ∘ p⁻¹ for validated atom permutations."""
    n = len(mapping)
    for permutation in (mapping, reactant_old_to_new, product_old_to_new):
        if any(type(i) is not int for i in permutation) or sorted(permutation) != list(range(n)):
            raise ValueError("complete maps and transports must be bijective permutations")
    result = [0] * n
    for old_index, new_index in enumerate(reactant_old_to_new):
        result[new_index] = product_old_to_new[mapping[old_index]]
    return tuple(result)


def _valid_mapping(mapping, reactants, products) -> tuple[int, ...]:
    mapping = tuple(mapping)
    n = reactants.GetNumAtoms()
    if products.GetNumAtoms() != n or len(mapping) != n:
        raise ValueError("incomplete atom inventory for a complete correspondence")
    if any(type(i) is not int for i in mapping) or sorted(mapping) != list(range(n)):
        raise ValueError("mapping is not a complete bijection")
    for left, right in enumerate(mapping):
        a, b = reactants.GetAtomWithIdx(left), products.GetAtomWithIdx(right)
        if (a.GetAtomicNum(), a.GetIsotope()) != (b.GetAtomicNum(), b.GetIsotope()):
            raise ValueError("correspondence changes element or isotope identity")
    return mapping


def transport_product_origins(origins, reactant_old_to_new, product_old_to_new) -> tuple[int, ...]:
    """Transport a product-to-reactant injection as r ∘ origin ∘ p⁻¹.

    The reactant and observed-product atom inventories may differ. Every
    product atom still has exactly one distinct source atom; unmatched reactant
    atoms are allowed, and missing product origins are not imputed.
    """
    nr, np = len(reactant_old_to_new), len(product_old_to_new)
    for permutation, size in ((reactant_old_to_new, nr), (product_old_to_new, np)):
        if any(type(i) is not int for i in permutation) or sorted(permutation) != list(range(size)):
            raise ValueError("atom transports must be bijective permutations")
    if (len(origins) != np or any(type(i) is not int or not 0 <= i < nr for i in origins) or
            len(set(origins)) != np):
        raise ValueError("product origins must be a complete injective assignment into reactant atoms")
    result = [0] * np
    for product, reactant in enumerate(origins):
        result[product_old_to_new[product]] = reactant_old_to_new[reactant]
    return tuple(result)


def _valid_product_origins(origins, reactants, products):
    origins = tuple(origins)
    transport_product_origins(origins, range(reactants.GetNumAtoms()), range(products.GetNumAtoms()))
    for product, reactant in enumerate(origins):
        a, b = reactants.GetAtomWithIdx(reactant), products.GetAtomWithIdx(product)
        if (a.GetAtomicNum(), a.GetIsotope()) != (b.GetAtomicNum(), b.GetIsotope()):
            raise ValueError("product correspondence changes element or isotope identity")
    return origins


def _automorphisms(mol, budget: int):
    matches = mol.GetSubstructMatches(
        mol, uniquify=False, useChirality=True, maxMatches=budget + 1)
    if len(matches) > budget:
        raise ValueError("equivalence enumeration exceeds the declared budget")
    # Concrete RDKit molecules are also substructure queries. Explicitly retain
    # only operations preserving the atom attributes required by the policy.

    def attributes(atom):
        return (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(),
                atom.GetIsAromatic(), atom.GetTotalNumHs(includeNeighbors=True))

    return [m for m in matches if all(attributes(mol.GetAtomWithIdx(i)) ==
            attributes(mol.GetAtomWithIdx(j)) for i, j in enumerate(m))]


def mapping_equivalence_keys(inputs: dict[str, str], references, budget: int):
    """Enumerate accepted encodings within a declared finite symmetry budget."""
    reactants, products = _mol(inputs["reactants"]), _mol(inputs["products"])
    left_ops, right_ops = _automorphisms(reactants, budget), _automorphisms(products, budget)
    accepted = set()
    for reference in references:
        reference = _valid_mapping(reference, reactants, products)
        for left in left_ops:
            for right in right_ops:
                accepted.add(transport_correspondence(reference, left, right))

    def equivalence_class(mapping):
        return min(transport_correspondence(mapping, left, right)
                   for left in left_ops for right in right_ops)

    return reactants, products, accepted, equivalence_class


def _resonance_automorphisms(mol, budget):
    """Bounded flags=0 resonance symmetry, without proton or isotope motion.

    RDKit returns Kekulized resonance forms. Restore aromaticity before matching
    the original graph; otherwise aromatic queries can yield no matches at all.
    Atom-index permutations are induced by full isomorphisms to those forms.
    Formal charges and bond orders may redistribute only through the enumerator,
    never by dropping charge/bond constraints from the matching query.
    """
    from rdkit import Chem

    forms = Chem.ResonanceMolSupplier(mol, flags=0, maxStructs=budget + 1)
    if len(forms) > budget:
        raise ValueError("resonance enumeration exceeds the declared budget")
    operations = set()

    def attributes(atom):
        return (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(),
                atom.GetIsAromatic(), atom.GetTotalNumHs(includeNeighbors=True))

    for raw_form in forms:
        form = Chem.Mol(raw_form)
        Chem.SanitizeMol(form)
        matches = form.GetSubstructMatches(mol, uniquify=False, useChirality=True, maxMatches=budget + 1)
        if len(matches) > budget:
            raise ValueError("resonance symmetry enumeration exceeds the declared budget")
        for operation in matches:
            if all(attributes(mol.GetAtomWithIdx(i)) == attributes(form.GetAtomWithIdx(j))
                   for i, j in enumerate(operation)):
                operations.add(operation)
                if len(operations) > budget:
                    raise ValueError("resonance symmetry enumeration exceeds the declared budget")
    if not operations:
        raise ValueError("resonance policy found no identity-preserving operations")
    return sorted(operations)


def product_origin_equivalence_keys(inputs, references, budget, *, resonance=False):
    """Keep graph symmetry separate from substantively different source atoms."""
    reactants, products = _mol(inputs["reactants"]), _mol(inputs["products"])
    operations = _resonance_automorphisms if resonance else _automorphisms
    left_ops, right_ops = operations(reactants, budget), operations(products, budget)
    accepted = set()
    for reference in references:
        reference = _valid_product_origins(reference, reactants, products)
        for left in left_ops:
            for right in right_ops:
                accepted.add(transport_product_origins(reference, left, right))

    def equivalence_class(origins):
        return min(transport_product_origins(origins, left, right)
                   for left in left_ops for right in right_ops)

    return reactants, products, accepted, equivalence_class


def _reference_scoring(instance, reference, contract):
    if contract.task == "forward":
        accepted = {product_multiset(outcome) for outcome in reference.accepted_outcomes}
        return product_multiset, tuple, accepted
    if reference.correspondence_scope != contract.target_scope:
        raise ValueError("task contract/reference correspondence scope mismatch")
    if contract.target_scope == "reported_product_atom_origin":
        reactants, products, accepted, class_key = product_origin_equivalence_keys(
            instance["inputs"], reference.accepted_product_origins, contract.max_automorphisms,
            resonance=contract.equivalence_policy == RESONANCE_PRODUCT_ORIGIN_POLICY)

        def normalize(value):
            return _valid_product_origins(value, reactants, products)

        return normalize, class_key, accepted
    reactants, products, accepted, class_key = mapping_equivalence_keys(
        instance["inputs"], reference.accepted_correspondences, contract.max_automorphisms)

    def normalize(value):
        return _valid_mapping(value, reactants, products)

    return normalize, class_key, accepted


def _score_candidates(instance_id, prediction, normalize, class_key, accepted, contract):
    status = prediction.get("status", "missing")
    if status not in {"ok", "missing", "invalid", "timeout", "abstention", "unsupported", "error"}:
        raise ValueError(f"unknown prediction status: {status!r}")
    candidates = prediction.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("raw candidates must be a list")
    if len(candidates) > contract.raw_candidate_limit:
        raise ValueError("prediction exceeds the declared raw generation budget")
    unique, seen, invalid = [], set(), 0
    if status == "ok":
        for raw in candidates:
            try:
                candidate = normalize(raw)
                # Equivalent mapping encodings are one ranked class.
                identity = class_key(candidate)
                if identity not in seen:
                    seen.add(identity)
                    unique.append(candidate)
            except (ValueError, TypeError, RuntimeError):
                invalid += 1
        if not unique:
            status = "invalid"
    # Invalid first-ranked output is a top-1 failure; later validity does
    # not retroactively improve a method's originally selected answer.
    first_ok = False
    if status == "ok" and candidates:
        try:
            first_ok = normalize(candidates[0]) in accepted
        except (ValueError, TypeError, RuntimeError):
            pass
    return {
        "instance_id": instance_id, "status": status,
        "top1": first_ok,
        "topk": any(value in accepted for value in unique[:contract.top_k]),
        "valid_output": status == "ok", "raw_candidates": len(candidates),
        "unique_outputs": len(unique), "invalid_candidates": invalid,
    }


def evaluate_predictions(
    instances: list[dict[str, Any]], references: list[ReferenceRecord],
    predictions: list[dict[str, Any]], contract: TaskContract,
) -> dict[str, Any]:
    """Score one fixed evaluation population; missing and failed eligible outputs are counted.

    Ranked candidates are deduplicated inside the raw generation budget. No
    retry/refill is performed. Reference problems block execution rather than
    masquerading as prediction failures or silently changing eligibility.
    """
    contract.validate()
    ref_by_id = {r.reference_id: r for r in references}
    pred_by_id = {p["instance_id"]: p for p in predictions}
    instance_ids = {i["instance_id"] for i in instances}
    if len(instance_ids) != len(instances) or len(pred_by_id) != len(predictions):
        raise ValueError("duplicate task instance or prediction identifiers")
    if len(ref_by_id) != len(references):
        raise ValueError("duplicate reference identifiers")
    if set(pred_by_id) - instance_ids:
        raise ValueError("prediction refers to a task instance outside the evaluation population")
    rows, excluded = [], []
    for instance in instances:
        reference = ref_by_id[instance["reference_id"]]
        reference.validate()
        if instance.get("reference_revision") != reference.revision:
            raise ValueError("task instance/reference revision mismatch")
        if instance.get("observation_id") != reference.observation_id:
            raise ValueError("task instance/reference observation mismatch")
        if not reference.task_eligibility.get(contract.task, False):
            excluded.append(instance["instance_id"])
            continue
        if reference.evidence_tier not in contract.evidence_tiers:
            raise ValueError("reference evidence tier is outside the contract")
        contract.validate_inputs(instance["inputs"])
        normalize, class_key, accepted = _reference_scoring(instance, reference, contract)
        rows.append(_score_candidates(instance["instance_id"],
                                      pred_by_id.get(instance["instance_id"], {}),
                                      normalize, class_key, accepted, contract))
    n = len(rows)
    return {
        "task_id": contract.task_id, "task_version": contract.version,
        "eligible_count": n, "excluded_count": len(excluded),
        "top1_accuracy": sum(row["top1"] for row in rows) / n if n else None,
        "topk_accuracy": sum(row["topk"] for row in rows) / n if n else None,
        "valid_output_coverage": sum(row["valid_output"] for row in rows) / n if n else None,
        "failures": dict(Counter(row["status"] for row in rows if row["status"] != "ok")),
        "raw_candidates": sum(row["raw_candidates"] for row in rows),
        "unique_outputs": sum(row["unique_outputs"] for row in rows),
        "scores": rows, "excluded_instances": excluded,
    }
