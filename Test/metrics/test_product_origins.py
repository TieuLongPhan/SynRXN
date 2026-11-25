"""Partial inventories cannot erase atom origin, chirality or missingness."""

from dataclasses import replace

import pytest

from synrxn.metrics.reference import evaluate_predictions, transport_product_origins
from synrxn.benchmarks.contracts import PRODUCT_ORIGIN_POLICY, RESONANCE_PRODUCT_ORIGIN_POLICY, ReferenceRecord, TaskContract
from synrxn.benchmarks.mapping_io import product_origins_from_mapped, reaction_inputs


def contract(policy=PRODUCT_ORIGIN_POLICY, **options):
    return TaskContract("fixture.origins", "1", "mapping", ("reactants", "products"),
                        "reported_product_atom_origin", policy, ("fixture",),
                        status="runnable", **options)


def evaluate(inputs, accepted, candidates, **options):
    reference = ReferenceRecord("ref:1", "obs:1", 1, "source:1", "fixture", "accepted", ("test-fixture",),
                                {"mapping": True}, review_events=({"reviewer": "synthetic fixture"},),
                                correspondence_scope="reported_product_atom_origin",
                                accepted_product_origins=tuple(tuple(m) for m in accepted))
    instance = {"instance_id": "task:1", "reference_id": "ref:1", "reference_revision": 1,
                "observation_id": "obs:1", "inputs": inputs}
    prediction = {"instance_id": "task:1", "status": "ok", "candidates": candidates}
    return evaluate_predictions([instance], [reference], [prediction], contract(**options))


def test_leaving_group_is_allowed_but_every_product_atom_needs_one_origin():
    inputs = {"reactants": "CO.Cl", "products": "CO"}
    assert evaluate(inputs, [(0, 1)], [[0, 1]])["top1_accuracy"] == 1
    for candidate in ([0], [0, 0], [0, 2], [-1, 1], [0, True]):
        result = evaluate(inputs, [(0, 1)], [candidate])
        assert result["eligible_count"] == 1
        assert result["top1_accuracy"] == 0 and result["failures"] == {"invalid": 1}


def test_product_origin_transport_has_different_side_sizes():
    original = (3, 0, 4)
    r, p = (4, 1, 0, 2, 3), (1, 2, 0)
    transported = transport_product_origins(original, r, p)
    for product, reactant in enumerate(original):
        assert transported[p[product]] == r[reactant]
    with pytest.raises(ValueError, match="injective"):
        transport_product_origins((0, 0), (0, 1, 2), (0, 1))


def test_identical_components_are_equivalent_within_the_fixed_raw_budget():
    result = evaluate({"reactants": "O.O", "products": "O"}, [(0,)], [[0], [1]],
                      raw_candidate_limit=2, top_k=2)
    assert result["raw_candidates"] == 2 and result["unique_outputs"] == 1
    assert result["top1_accuracy"] == 1


def test_competing_non_equivalent_precursors_remain_distinct_accepted_classes():
    inputs = {"reactants": "CO.CCO", "products": "CO"}
    result = evaluate(inputs, [(0, 1), (3, 4)], [[0, 1], [3, 4]], raw_candidate_limit=2, top_k=2)
    assert result["unique_outputs"] == 2 and result["top1_accuracy"] == 1
    assert evaluate(inputs, [(0, 1)], [[3, 4]])["top1_accuracy"] == 0


def test_isotopes_explicit_hydrogens_and_chiral_precursors_are_not_collapsed():
    result = evaluate({"reactants": "[H]O.[2H]O", "products": "[2H]O"}, [(2, 3)], [[0, 1]])
    assert result["failures"] == {"invalid": 1}
    inputs = {"reactants": "F[C@H](Cl)Br.F[C@@H](Cl)Br", "products": "F[C@H](Cl)Br"}
    assert evaluate(inputs, [(0, 1, 2, 3)], [[4, 5, 6, 7]])["top1_accuracy"] == 0


def test_scope_cannot_be_changed_by_reusing_a_complete_reference():
    ref = ReferenceRecord("ref:1", "obs:1", 1, "source:1", "fixture", "accepted", ("test-fixture",),
                          {"mapping": True}, accepted_correspondences=((0,),),
                          review_events=({"reviewer": "fixture"},))
    item = {"instance_id": "task:1", "reference_id": "ref:1", "reference_revision": 1,
            "observation_id": "obs:1", "inputs": {"reactants": "O", "products": "O"}}
    with pytest.raises(ValueError, match="scope mismatch"):
        evaluate_predictions([item], [ref], [], contract())
    with pytest.raises(ValueError, match="scope"):
        replace(ref, correspondence_scope="reported_product_atom_origin").validate()


def test_raw_input_graph_cannot_be_replaced_by_the_reference_annotation():
    inputs = reaction_inputs("CO.[Na]>>CO")
    mapped = "[CH3:4][OH:5].[NaH:6]>>[CH3:4][OH:5]"
    with pytest.raises(ValueError, match="declared reaction molecules"):
        product_origins_from_mapped(mapped, inputs)


def test_mapped_output_transport_preserves_agents_and_all_product_atoms():
    inputs = {"reactants": "O.C.Cl", "products": "CO"}
    mapped = "[CH4:8].Cl>[OH2:9]>[CH3:8][OH:9]"
    origins = product_origins_from_mapped(mapped, inputs)
    assert origins == (1, 0)
    with pytest.raises(ValueError, match="every product atom"):
        product_origins_from_mapped("[CH4:8].Cl>[OH2:9]>[CH3:8]O", inputs)
    with pytest.raises(ValueError, match="duplicate"):
        product_origins_from_mapped("[CH4:8].[OH2:8].Cl>>[CH3:8][OH:9]", inputs)


def test_contract_rejects_maps_in_inputs_even_when_the_scorer_could_strip_them():
    with pytest.raises(ValueError, match="map-free"):
        contract().validate_inputs({"reactants": "[CH4:7]", "products": "C"})
    with pytest.raises(ValueError, match="map-free"):
        contract().validate_inputs({"reactants": "[H:7]O", "products": "O"})


@pytest.mark.parametrize("molecule,swapped", [
    ("C[N+](=O)[O-]", [0, 1, 3, 2]),
    ("CC(=O)[O-]", [0, 1, 3, 2]),
    ("O=[N+]([O-])c1ccccc1", [2, 1, 0, 3, 4, 5, 6, 7, 8]),
])
def test_resonance_policy_is_versioned_and_handles_aromatic_queries(molecule, swapped):
    inputs = {"reactants": molecule, "products": molecule}
    identity = list(range(len(swapped)))
    assert evaluate(inputs, [identity], [swapped])["top1_accuracy"] == 0
    result = evaluate(inputs, [identity], [identity, swapped], policy=RESONANCE_PRODUCT_ORIGIN_POLICY,
                      raw_candidate_limit=2, top_k=2)
    assert result["top1_accuracy"] == 1 and result["unique_outputs"] == 1
    assert evaluate(inputs, [identity], [swapped], policy=RESONANCE_PRODUCT_ORIGIN_POLICY)["top1_accuracy"] == 1


def test_resonance_does_not_move_acid_protons_isotopes_or_change_species():
    for molecule in ("CC(=O)O", "C[N+](=[18O])[O-]"):
        inputs = {"reactants": molecule, "products": molecule}
        assert evaluate(inputs, [[0, 1, 2, 3]], [[0, 1, 3, 2]],
                        policy=RESONANCE_PRODUCT_ORIGIN_POLICY)["top1_accuracy"] == 0
    assert evaluate({"reactants": "[NH4+].N", "products": "[NH4+]"}, [[0]], [[1]],
                    policy=RESONANCE_PRODUCT_ORIGIN_POLICY)["top1_accuracy"] == 0
    inputs = {"reactants": "F[C@H](Cl)Br.F[C@@H](Cl)Br", "products": "F[C@H](Cl)Br"}
    assert evaluate(inputs, [[0, 1, 2, 3]], [[4, 5, 6, 7]],
                    policy=RESONANCE_PRODUCT_ORIGIN_POLICY)["top1_accuracy"] == 0


def test_resonance_limits_fail_explicitly_without_scoring_a_truncated_orbit():
    with pytest.raises(ValueError, match="resonance enumeration"):
        evaluate({"reactants": "C[N+](=O)[O-]", "products": "C[N+](=O)[O-]"},
                 [[0, 1, 2, 3]], [[0, 1, 2, 3]], policy=RESONANCE_PRODUCT_ORIGIN_POLICY,
                 max_automorphisms=1)
