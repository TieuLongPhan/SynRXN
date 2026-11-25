"""Tests for failures capable of changing a benchmark conclusion."""

from dataclasses import replace

import pytest

from synrxn.metrics.reference import (
    evaluate_predictions, product_multiset, transport_correspondence,
)
from synrxn.benchmarks.contracts import (
    MAPPING_POLICY, OUTCOME_POLICY, ReferenceRecord, TaskContract,
)


def contract(task="forward", **kwargs):
    return TaskContract(
        "test." + task, "1", task,
        ("reactants", "products") if task == "mapping" else ("reactants",),
        "complete_correspondence" if task == "mapping" else "reported_product_multiset",
        MAPPING_POLICY if task == "mapping" else OUTCOME_POLICY,
        ("test_review",), status="runnable", **kwargs,
    )


def reference(index=0, **kwargs):
    values = dict(reference_id=f"ref:{index}", observation_id=f"obs:{index}",
                  revision=1, source_record_id=f"src:{index}",
                  evidence_tier="test_review", resolution="accepted",
                  evidence_locators=("synthetic-test-fixture",),
                  task_eligibility={"forward": True}, accepted_outcomes=(("CO",),))
    values.update(kwargs)
    return ReferenceRecord(**values)


def instance(index=0, inputs=None):
    return dict(instance_id=f"task:{index}", reference_id=f"ref:{index}",
                reference_revision=1, observation_id=f"obs:{index}",
                inputs=inputs or {"reactants": "C"})


def test_missing_timeout_and_abstention_remain_in_the_denominator():
    refs = [reference(i) for i in range(4)]
    predictions = [dict(instance_id="task:0", status="ok", candidates=["CO"]),
                   dict(instance_id="task:1", status="timeout"),
                   dict(instance_id="task:2", status="abstention")]
    result = evaluate_predictions([instance(i) for i in range(4)], refs, predictions, contract())
    assert result["eligible_count"] == 4
    assert result["top1_accuracy"] == result["valid_output_coverage"] == 0.25
    assert result["failures"] == {"timeout": 1, "abstention": 1, "missing": 1}


def test_alternatives_are_outcomes_not_a_union_of_required_coproducts():
    ref = reference(accepted_outcomes=(("CO", "O"), ("CC",)))
    good = dict(instance_id="task:0", status="ok", candidates=["O.CO"])
    bad = dict(instance_id="task:0", status="ok", candidates=["CO"])
    assert evaluate_predictions([instance()], [ref], [good], contract())["top1_accuracy"] == 1
    assert evaluate_predictions([instance()], [ref], [bad], contract())["top1_accuracy"] == 0
    assert product_multiset("O.O") != product_multiset("O")


def test_stereo_isotopes_and_charge_survive_product_equivalence():
    assert product_multiset("F[C@H](Cl)Br") != product_multiset("F[C@@H](Cl)Br")
    assert product_multiset("[13CH4]") != product_multiset("C")
    assert product_multiset("[NH4+]") != product_multiset("N")
    assert product_multiset("[CH3:99][OH:6]") == product_multiset("CO")


def test_malformed_outcome_arrays_cannot_change_target_semantics():
    with pytest.raises(ValueError, match="multiset"):
        reference(accepted_outcomes=("CO",)).validate()
    with pytest.raises(ValueError, match="nested arrays"):
        ReferenceRecord.from_dict(dict(reference_id="r", observation_id="o", revision=1,
                                       source_record_id="s", evidence_tier="reported", resolution="accepted",
                                       evidence_locators=["source"], task_eligibility={"forward": True},
                                       accepted_outcomes=["CO"]))


def test_equivalent_repeats_do_not_refill_distinct_ranks():
    pred = dict(instance_id="task:0", status="ok", candidates=["C", "[CH4:99]", "CO"])
    result = evaluate_predictions([instance()], [reference()], [pred],
                                  contract(raw_candidate_limit=3, top_k=2))
    assert result["scores"][0]["unique_outputs"] == 2
    assert result["top1_accuracy"] == 0
    assert result["topk_accuracy"] == 1
    pred["candidates"].append("CO")
    with pytest.raises(ValueError, match="budget"):
        evaluate_predictions([instance()], [reference()], [pred],
                             contract(raw_candidate_limit=3, top_k=2))


def test_invalid_first_candidate_does_not_become_top1_success():
    pred = dict(instance_id="task:0", status="ok", candidates=["bad-smiles", "CO"])
    result = evaluate_predictions([instance()], [reference()], [pred], contract())
    assert result["top1_accuracy"] == 0
    assert result["topk_accuracy"] == 1


def test_target_information_cannot_enter_a_forward_contract():
    with pytest.raises(ValueError, match="target-derived"):
        replace(contract(), allowed_inputs=("reactants", "products")).validate()
    with pytest.raises(ValueError, match="outside"):
        evaluate_predictions([instance(inputs={"reactants": "C", "products": "CO"})],
                             [reference()], [], contract())


def test_unresolved_and_unreviewed_mapping_references_cannot_be_qualified():
    with pytest.raises(ValueError, match="unresolved"):
        replace(reference(), resolution="unresolved").validate()
    with pytest.raises(ValueError, match="review"):
        reference(task_eligibility={"mapping": True}, accepted_outcomes=(),
                  accepted_correspondences=((0,),)).validate()
    excluded = reference(task_eligibility={"forward": False}, resolution="unresolved",
                         accepted_outcomes=())
    result = evaluate_predictions([instance()], [excluded], [], contract())
    assert result["eligible_count"] == 0 and result["top1_accuracy"] is None


def test_revision_mismatch_and_duplicate_predictions_fail_execution():
    item = instance()
    item["reference_revision"] = 2
    with pytest.raises(ValueError, match="revision"):
        evaluate_predictions([item], [reference()], [], contract())
    pred = dict(instance_id="task:0", status="missing")
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_predictions([instance()], [reference()], [pred, pred], contract())


def test_reference_transport_is_equivariant():
    original = (2, 0, 1)
    p, q = (1, 2, 0), (2, 1, 0)
    transported = transport_correspondence(original, p, q)
    for i, j in enumerate(original):
        assert transported[p[i]] == q[j]
    with pytest.raises(ValueError, match="permutation"):
        transport_correspondence(original, (0, 0, 2), q)


def test_mapping_symmetry_does_not_erase_isotope_distinctions():
    ref = reference(task_eligibility={"mapping": True}, accepted_outcomes=(),
                    accepted_correspondences=((0, 1),), review_events=({"reviewer": "fixture"},))
    pred = dict(instance_id="task:0", status="ok", candidates=[[1, 0]])
    symmetric = instance(inputs={"reactants": "CC", "products": "CC"})
    assert evaluate_predictions([symmetric], [ref], [pred], contract("mapping"))["top1_accuracy"] == 1
    labelled = instance(inputs={"reactants": "[13CH3]C", "products": "[13CH3]C"})
    result = evaluate_predictions([labelled], [ref], [pred], contract("mapping"))
    assert result["top1_accuracy"] == 0 and result["failures"] == {"invalid": 1}
