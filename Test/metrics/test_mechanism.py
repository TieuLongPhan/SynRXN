"""Check distinctions that can change a mechanism-recovery conclusion."""

from dataclasses import replace
import re

import pytest

from synrxn.metrics.mechanism import evaluate_mechanism_predictions

mechanism = pytest.importorskip("synkit.Mechanism")

REACTION = "[NH3+:2][H:4].[O-:1][CH3:3]>>[NH3:2].[O:1]([CH3:3])[H:4]"
ARROWS = [["LP-/Sigma+", [1], [1, 4]], ["Sigma-/LP+", [2, 4], [2]]]


def reference():
    return mechanism.mechanism_from_legacy_epd(REACTION, ARROWS)


def test_missing_and_malformed_predictions_remain_in_the_denominator():
    record = reference()
    report = evaluate_mechanism_predictions(
        {"ok": [record], "missing": [record], "bad": [record]}, {"ok": [record.to_dict()], "bad": [{}]}, top_k=(1,)
    )
    assert report["cases"] == 3
    assert report["prediction_coverage"] == 2 / 3
    assert report["top_k"]["1"]["event_recovery"] == 1 / 3
    assert report["top_k"]["1"]["strict_validity"] == 1 / 3
    assert "error" in report["case_results"][2]["predictions"][0]


def test_correct_endpoints_without_arrows_do_not_recover_a_mechanism():
    record = reference()
    report = evaluate_mechanism_predictions({"case": [record]}, {"case": [replace(record, steps=())]})
    scores = report["top_k"]["1"]
    assert scores["net_equivalence"] == 1
    assert scores["event_recovery"] == scores["trajectory_recovery"] == scores["strict_validity"] == 0


def test_ranked_recovery_preserves_failed_candidates():
    record = reference()
    report = evaluate_mechanism_predictions({"case": [record]}, {"case": [{}, record]}, top_k=(1, 2))
    assert report["top_k"]["1"]["event_recovery"] == 0
    assert report["top_k"]["2"]["event_recovery"] == 1
    assert len(report["case_results"][0]["predictions"]) == 2


def test_atom_map_renumbering_is_not_a_prediction_error():
    record = reference()
    reaction = re.sub(r":(\d+)\]", lambda match: f":{int(match[1]) + 10}]", REACTION)
    arrows = [
        [action, [value + 10 for value in source], [value + 10 for value in target]]
        for action, source, target in ARROWS
    ]
    prediction = mechanism.mechanism_from_legacy_epd(reaction, arrows)
    report = evaluate_mechanism_predictions({"case": [record]}, {"case": [prediction]}, top_k=(1,))
    assert report["top_k"]["1"]["event_recovery"] == report["top_k"]["1"]["trajectory_recovery"] == 1


def test_dependent_arrow_reordering_is_distinct_even_when_replay_passes():
    record = reference()
    reordered = mechanism.mechanism_from_legacy_epd(REACTION, ARROWS[::-1])
    report = evaluate_mechanism_predictions({"case": [record]}, {"case": [reordered]}, top_k=(1,))
    scores = report["top_k"]["1"]
    assert scores["strict_validity"] == scores["net_equivalence"] == 1
    assert scores["event_recovery"] == scores["trajectory_recovery"] == 0


def test_explicit_reference_alternatives_can_accept_a_different_order():
    record = reference()
    reordered = mechanism.mechanism_from_legacy_epd(REACTION, ARROWS[::-1])
    report = evaluate_mechanism_predictions({"case": [record, reordered]}, {"case": [reordered]}, top_k=(1,))
    assert report["top_k"]["1"]["event_recovery"] == report["top_k"]["1"]["trajectory_recovery"] == 1


def test_independent_event_swaps_preserve_events_but_change_the_ordered_trajectory():
    reactants = "[NH3+:2][H:4].[O-:1][CH3:3].[NH2+:6]([CH3:9])[H:8].[S-:5][CH3:7]"
    products = "[NH3:2].[O:1]([CH3:3])[H:4].[NH2:6][CH3:9].[S:5]([CH3:7])[H:8]"
    second = [["LP-/Sigma+", [5], [5, 8]], ["Sigma-/LP+", [6, 8], [6]]]
    record = mechanism.mechanism_from_legacy_epd(reactants + ">>" + products, ARROWS + second)
    prediction = mechanism.mechanism_from_legacy_epd(reactants + ">>" + products, second + ARROWS)
    report = evaluate_mechanism_predictions({"case": [record]}, {"case": [prediction]}, top_k=(1,))
    scores = report["top_k"]["1"]
    assert scores["strict_validity"] == scores["event_recovery"] == 1
    assert scores["trajectory_recovery"] == 0


def test_prediction_cannot_select_a_relaxed_endpoint_policy():
    record = reference()
    forged = replace(record, metadata={"endpoint_resource_policy": "closed_shell_pair", "closed_shell_atom_maps": [1]})
    report = evaluate_mechanism_predictions({"case": [record]}, {"case": [forged]}, top_k=(1,))
    assert report["case_results"][0]["reference_endpoint_policy"] == {}
    assert report["top_k"]["1"]["event_recovery"] == 1


def test_invalid_reference_fails_instead_of_silently_shrinking_the_cohort():
    with pytest.raises(ValueError, match="reference fails strict replay"):
        evaluate_mechanism_predictions({"case": [replace(reference(), steps=())]}, {})


@pytest.mark.parametrize("ranks", [(), (0,), (-1,), (1.5,), (1, True)])
def test_invalid_top_k_is_rejected(ranks):
    with pytest.raises(ValueError, match="positive integers"):
        evaluate_mechanism_predictions({}, {}, top_k=ranks)


def test_unknown_case_ids_are_rejected():
    with pytest.raises(ValueError, match="absent from references"):
        evaluate_mechanism_predictions({}, {"unknown": []})


def test_empty_cohort_reports_undefined_rates():
    report = evaluate_mechanism_predictions({}, {}, top_k=(1,))
    assert report["cases"] == 0
    assert report["prediction_coverage"] is report["top_k"]["1"]["event_recovery"] is None


def test_delta_consistent_references_require_an_explicit_separate_tier():
    side = "[O:1]=[Mn:2]=[O:3]"
    record = mechanism.mechanism_from_legacy_epd(side + ">>" + side, [])
    certificate = mechanism.MechanismReplayer(validation="strict").replay(record).certificate
    assert certificate.status == "VALID" and certificate.verification_level == "DELTA_CONSISTENT"
    with pytest.raises(ValueError, match="verification level is outside"):
        evaluate_mechanism_predictions({"case": [record]}, {"case": [record]})
    report = evaluate_mechanism_predictions(
        {"case": [record]}, {"case": [record]}, allowed_verification_levels=("DELTA_CONSISTENT",)
    )
    assert report["allowed_verification_levels"] == ["DELTA_CONSISTENT"]
    assert report["top_k"]["1"]["strict_validity"] == 1


def test_unsupported_verification_tiers_are_rejected():
    with pytest.raises(ValueError, match="supported verification levels"):
        evaluate_mechanism_predictions({}, {}, allowed_verification_levels=("ANY",))
