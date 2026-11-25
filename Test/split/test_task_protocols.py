"""Scientific split coverage and leakage/causality regression tests."""

import json

import pytest

from synrxn.split import (
    get_task_split_protocol, get_task_split_protocols, make_task_split,
    split_protocol_coverage,
)


def cohort(n=30):
    return [{"instance_id": f"row-{i}", "reaction_group": f"r{i // 2}",
             "label": i % 3, "timestamp": i} for i in range(n)]


def test_registry_all_41_tasks_in_12_families():
    coverage = split_protocol_coverage()
    assert coverage["complete"], coverage
    assert coverage["registry_task_count"] == coverage["split_protocol_count"] == 41
    assert len({p.family_id for p in get_task_split_protocols().values()}) == 12


def test_registry_detects_missing_and_wrong_family():
    report = split_protocol_coverage({"tasks": [
        {"id": "forward_prediction", "family_id": "wrong"},
        {"id": "new_task", "family_id": "new"},
    ]})
    assert not report["complete"]
    assert report["missing_tasks"] == ["new_task"]
    assert report["family_mismatches"] == ["forward_prediction"]


def test_unknown_task_rejected():
    with pytest.raises(KeyError, match="Unknown scientific task"):
        get_task_split_protocol("typo")


def test_group_split_reproducible_complete_and_group_disjoint():
    records = cohort()
    first = make_task_split("forward_prediction", records, seed=9)
    assert first == make_task_split("forward_prediction", records, seed=9)
    assert first["indices"] != make_task_split("forward_prediction", records, seed=10)["indices"]
    assert sorted(i for part in first["indices"].values() for i in part) == list(range(30))
    for i in range(0, 30, 2):
        assert first["membership"][str(i)] == first["membership"][str(i + 1)]
    assert first["declared_group_fields"] == ["reaction_group"]
    assert not first["empty_partitions"]
    assert len(first["input_metadata_sha256"]) == 64
    json.dumps(first, allow_nan=False)


def test_missing_group_metadata_not_guessed_from_smiles():
    records = [{"instance_id": i, "smiles": "CCO"} for i in range(10)]
    with pytest.raises(ValueError, match="audited metadata"):
        make_task_split("forward_prediction", records)
    with pytest.raises(ValueError, match="scaffold_disjoint"):
        make_task_split("forward_prediction", records, strategy="scaffold_disjoint")


def test_hard_relationships_and_multiple_fields_transitively_union():
    records = cohort()
    for i, record in enumerate(records):
        record["parent_sequence_group"] = f"p{i}"
        record["campaign_id"] = f"c{i}"
    records[0]["parent_sequence_group"] = records[2]["parent_sequence_group"]
    records[2]["campaign_id"] = records[4]["campaign_id"]
    edge = {"left": "row-4", "right": "row-6", "kind": "same_observation", "status": "accepted"}
    result = make_task_split("forward_prediction", records,
                             group_fields=("reaction_group", "parent_sequence_group", "campaign_id"),
                             relationships=[edge])
    assert len({result["membership"][str(i)] for i in range(8)}) == 1
    assert result["applied_hard_relationships"] == 1
    assert "campaign_id" in result["explicit_group_fields"]
    assert "campaign_id" not in result["enforced_intrinsic_fields"]


def test_random_diagnostic_preserves_trajectory_without_imposing_campaign_holdout():
    records = cohort()
    for i, row in enumerate(records):
        row["trajectory_id"] = f"trajectory-{i // 3}"
        row["campaign_id"] = f"campaign-{i // 6}"
    result = make_task_split("reaction_progress_monitoring", records, strategy="random_diagnostic")
    for start in range(0, 30, 3):
        assert len({result["membership"][str(i)] for i in range(start, start + 3)}) == 1
    assert "campaign_id" not in result["enforced_intrinsic_fields"]
    assert any("not an unrestricted row shuffle" in item for item in result["qualification"])


def test_source_and_protein_transfer_require_real_metadata():
    records = cohort()
    with pytest.raises(ValueError, match="protein_disjoint"):
        make_task_split("enzyme_activity", records, strategy="protein_disjoint")
    for i, row in enumerate(records):
        row["source_group"] = f"source-{i // 3}"
        row["protein_cluster"] = f"homology-{i // 3}"
    for strategy in ("source_disjoint", "protein_disjoint"):
        report = make_task_split("enzyme_activity", records, strategy=strategy)
        assert report["component_count"] == 10
        assert not report["empty_partitions"]


def test_native_keeps_labels_normalizes_val_and_reports_missing_validation():
    records = [{"instance_id": i, "native_split": "train" if i < 6 else "test"} for i in range(10)]
    result = make_task_split("forward_prediction", records, strategy="native")
    assert result["counts"] == {"train": 6, "validation": 0, "test": 4}
    assert result["empty_partitions"] == ["validation"]
    assert result["requested_ratios"] is None
    records[4]["native_split"] = "val"
    assert make_task_split("forward_prediction", records, strategy="native")["indices"]["validation"] == [4]


def test_native_rejects_unknown_labels_missing_or_dependence_leak():
    records = cohort()
    for i, row in enumerate(records):
        row["native_split"] = "train" if i < 21 else "test"
    with pytest.raises(ValueError, match="dependence group"):
        make_task_split("forward_prediction", records, strategy="native")
    records[21]["native_split"] = "train"
    records[0]["native_split"] = "external"
    with pytest.raises(ValueError, match="Unknown native partition"):
        make_task_split("forward_prediction", records, strategy="native")
    records[0]["native_split"] = None
    with pytest.raises(ValueError, match="required for every row"):
        make_task_split("forward_prediction", records, strategy="native")


def test_native_rejects_accepted_hard_edge_across_partitions():
    records = [{"instance_id": i, "native_split": "train" if i < 6 else "test"} for i in range(10)]
    with pytest.raises(ValueError, match="dependence group"):
        make_task_split("forward_prediction", records, strategy="native", relationships=[
            {"left": 0, "right": 9, "kind": "shared_parent", "status": "accepted"}])


def test_group_chronological_has_no_future_training_or_group_leak():
    records = cohort()
    report = make_task_split("forward_prediction", records, strategy="group_chronological", timestamp_field="timestamp")
    assert report["temporal_ranges"]["train"][1] < report["temporal_ranges"]["validation"][0]
    assert report["temporal_ranges"]["validation"][1] < report["temporal_ranges"]["test"][0]
    for i in range(0, 30, 2):
        assert report["membership"][str(i)] == report["membership"][str(i + 1)]


def test_interleaved_time_groups_fail_instead_of_leaking():
    records = cohort(6)
    records[0]["reaction_group"] = records[-1]["reaction_group"] = "spans-all"
    with pytest.raises(ValueError, match="temporally separable"):
        make_task_split("forward_prediction", records, strategy="group_chronological", timestamp_field="timestamp")


def test_timestamp_ties_are_whole_and_mixed_time_types_rejected():
    records = [{"instance_id": i, "timestamp": f"2026-01-{1 + i // 2:02d}T00:00:00Z"} for i in range(12)]
    result = make_task_split("forward_prediction", records, strategy="chronological", timestamp_field="timestamp")
    for i in range(0, 12, 2):
        assert result["membership"][str(i)] == result["membership"][str(i + 1)]
    records[0]["timestamp"] = 0
    with pytest.raises(ValueError, match="mix numerical"):
        make_task_split("forward_prediction", records, strategy="chronological", timestamp_field="timestamp")


def test_stratification_never_drops_rare_or_unseen_labels():
    records = cohort()
    records[-1]["label"] = "rare"
    result = make_task_split("reaction_classification", records, strategy="stratified_group", label_field="label")
    assert sum(result["counts"].values()) == len(records)
    assert sum(sum(counts.values()) for counts in result["label_summary"]["counts"].values()) == len(records)
    assert any('"rare"' in missing for missing in result["label_summary"]["missing_from_partition"].values())
    labels = {label for counts in result["label_summary"]["counts"].values() for label in counts}
    assert labels == {'0', '1', '2', '"rare"'}


def test_native_discloses_unseen_test_label():
    records = [{"instance_id": i, "native_split": "train" if i < 5 else "test",
                "label": "a" if i < 5 else "unseen"} for i in range(10)]
    report = make_task_split("reaction_classification", records, strategy="native", label_field="label")
    assert report["label_summary"]["unseen_in_training"]["test"] == ['"unseen"']


@pytest.mark.parametrize("kwargs,match", [
    ({"ratios": (1, 0, 0)}, "At least two"),
    ({"ratios": (.7, .1, .1)}, "sum to one"),
    ({"strategy": "typo"}, "Unknown split"),
    ({"seed": None}, "seed must"),
    ({"group_fields": "reaction_group"}, "sequence of"),
    ({"strategy": "stratified_group"}, "label_field"),
    ({"strategy": "chronological"}, "timestamp_field"),
])
def test_invalid_strategy_inputs_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_task_split("forward_prediction", cohort(), **kwargs)


def test_tiny_or_single_dependency_components_rejected():
    records = cohort(4)
    with pytest.raises(ValueError, match="Only 2 independent components"):
        make_task_split("forward_prediction", records)
    records = cohort()
    for row in records:
        row["campaign_id"] = "one-campaign"
    with pytest.raises(ValueError, match="Only 1 independent components"):
        make_task_split("reaction_success", records, group_fields=("campaign_id",))


def test_missing_intrinsic_is_disclosed_and_known_values_stay_grouped():
    records = cohort()
    records[0]["trajectory_id"] = records[-1]["trajectory_id"] = "known"
    report = make_task_split("forward_prediction", records)
    assert report["incomplete_intrinsic_metadata"] == {"trajectory_id": 28}
    assert report["membership"]["0"] == report["membership"]["29"]
    assert any("unknown rows is unqualified" in item for item in report["qualification"])


def test_provenance_and_metadata_hash_change_with_lineage():
    records = cohort()
    first = make_task_split("forward_prediction", records, provenance={"source_sha256": "a" * 64})
    assert first["source_provenance"]["source_sha256"] == "a" * 64
    records[-1]["reaction_group"] = "new-lineage"
    second = make_task_split("forward_prediction", records)
    assert first["input_metadata_sha256"] != second["input_metadata_sha256"]


def test_duplicate_ids_and_outside_relationships_rejected():
    records = cohort()
    with pytest.raises(ValueError, match="outside"):
        make_task_split("forward_prediction", records, relationships=[
            {"left": "row-0", "right": "absent", "kind": "same_observation", "status": "accepted"}])
    records[-1]["instance_id"] = records[0]["instance_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        make_task_split("forward_prediction", records)


def _unit_records(field="panel_id"):
    """Provide ten repeated units inside one shared acquisition campaign."""
    return [{"instance_id": f"row-{i}", field: f"unit-{i // 2}",
             "campaign_id": "shared-campaign", "source_group": f"source-{i // 2}"}
            for i in range(20)]


@pytest.mark.parametrize("field", [
    "split", "native_split", "derived_split", "proposed_split", "outer_fold",
    "partition", "upstream_split", "custom_fold", "source_partition",
])
def test_assignment_columns_are_not_dependency_ids(field):
    rows = _unit_records("reaction_group")
    for row in rows:
        row[field] = "train"
    with pytest.raises(ValueError, match="assignments, not dependency"):
        make_task_split("forward_prediction", rows, group_fields=(field,))


@pytest.mark.parametrize("task", ["condition_ranking", "sequential_optimization"])
@pytest.mark.parametrize("strategy", ["group", "random_diagnostic", "source_disjoint", "native"])
def test_condition_panels_stay_whole_for_every_assignment_strategy(task, strategy):
    rows = _unit_records()
    for i, row in enumerate(rows):
        row["native_split"] = "train" if i < 12 else "validation" if i < 16 else "test"
    report = make_task_split(task, rows, strategy=strategy)
    for i in range(0, 20, 2):
        assert report["membership"][str(i)] == report["membership"][str(i + 1)]
    assert report["required_independence_unit"]["selected_field"] == "panel_id"
    assert "campaign_id" not in report["enforced_intrinsic_fields"]
    assert not report["empty_partitions"]


@pytest.mark.parametrize("task", ["condition_ranking", "sequential_optimization"])
def test_panel_task_rejects_an_unqualified_reaction_group_alias(task):
    with pytest.raises(ValueError, match="complete measured condition panel"):
        make_task_split(task, _unit_records("reaction_group"), strategy="random_diagnostic")
    report = make_task_split(task, _unit_records("audited_candidate_panel"),
                             strategy="random_diagnostic", group_fields=("audited_candidate_panel",))
    assert report["required_independence_unit"]["resolution"] == "explicit_group_fields"


@pytest.mark.parametrize("task", ["literature_extraction", "reaction_record_curation"])
@pytest.mark.parametrize("strategy", ["random_diagnostic", "source_disjoint", "native"])
def test_document_rows_stay_whole_in_diagnostics_native_and_transfer(task, strategy):
    rows = _unit_records("document_id")
    for i, row in enumerate(rows):
        row["native_split"] = "train" if i < 12 else "test"
    report = make_task_split(task, rows, strategy=strategy)
    assert report["required_independence_unit"]["selected_field"] == "document_id"
    for i in range(0, 20, 2):
        assert report["membership"][str(i)] == report["membership"][str(i + 1)]


@pytest.mark.parametrize("task", ["literature_extraction", "reaction_record_curation"])
def test_document_contract_rejects_absent_lineage_even_for_native(task):
    rows = _unit_records("reaction_group")
    for i, row in enumerate(rows):
        row["native_split"] = "train" if i < 12 else "test"
    with pytest.raises(ValueError, match="source document"):
        make_task_split(task, rows, strategy="native")


def test_native_condition_labels_cannot_override_panel_integrity():
    rows = _unit_records()
    for i, row in enumerate(rows):
        row["native_split"] = "train" if i < 13 else "test"
    with pytest.raises(ValueError, match="dependence group"):
        make_task_split("condition_ranking", rows, strategy="native")


def test_explicit_complete_relationships_can_supply_panels():
    rows = _unit_records("not_a_group_field")
    edges = [{"left": f"row-{i}", "right": f"row-{i + 1}",
              "kind": "shared_parent", "status": "accepted"}
             for i in range(0, 20, 2)]
    report = make_task_split("condition_ranking", rows, relationships=edges)
    assert report["required_independence_unit"]["resolution"] == "accepted_relationships"
    assert report["required_independence_unit"]["accepted_relationship_rows"] == 20
    for i in range(0, 20, 2):
        assert report["membership"][str(i)] == report["membership"][str(i + 1)]
    with pytest.raises(ValueError, match="covering every row"):
        make_task_split("condition_ranking", rows, relationships=edges[:-1])
    self_edges = [{"left": row["instance_id"], "right": row["instance_id"],
                   "kind": "shared_parent", "status": "accepted"} for row in rows]
    with pytest.raises(ValueError, match="covering every row"):
        make_task_split("condition_ranking", rows, relationships=self_edges)


@pytest.mark.parametrize("task,field", [
    ("reactive_potential", "trajectory_id"),
    ("mechanistic_prediction", "mechanism_group"),
    ("ts_geometry", "endpoint_group"),
    ("reaction_network_modeling", "mechanism_id"),
    ("analytical_interpretation", "experiment_group"),
    ("reaction_progress_monitoring", "experiment_group"),
    ("process_control", "episode_id"),
])
def test_transfer_enforces_whole_physical_units_in_addition_to_source(task, field):
    rows = _unit_records(field)
    for i, row in enumerate(rows):
        row["source_group"] = f"source-{i}"
    report = make_task_split(task, rows, strategy="source_disjoint")
    assert report["selected_transfer_field"] == "source_group"
    assert field in report["declared_group_fields"]
    for i in range(0, 20, 2):
        assert report["membership"][str(i)] == report["membership"][str(i + 1)]


def test_campaign_transfer_is_explicit_and_can_be_impossible():
    rows = _unit_records()
    diagnostic = make_task_split("condition_ranking", rows, strategy="random_diagnostic")
    assert diagnostic["component_count"] == 10
    with pytest.raises(ValueError, match="Only 1 independent components"):
        make_task_split("condition_ranking", rows, strategy="campaign_disjoint")
