"""Task views must withhold labels and preserve their declared source semantics."""

import csv
import gzip
import json

from synrxn.data import DatasetCatalog
from synrxn.benchmarks.task_views import definitions, export_task_view, make_instance, task_definition


def metadata(key):
    task, name = key.removesuffix(".csv.gz").split("/")
    return DatasetCatalog().get(task, name)


def test_eight_task_definitions_are_available():
    tasks = definitions()["tasks"]
    assert len(tasks) == len({task["id"] for task in tasks}) == 8


def test_condition_target_middle_field_is_withheld():
    task = task_definition("role_aware_condition")
    row = {"reaction_smiles": "C>CC.O>CO", "bases": "CC", "solvents": "O", "sample_id": "one"}
    instance = make_instance(row, task, metadata("condition_sharealike/ord_hte_condition.csv.gz"), "a" * 64, 0)
    assert instance["inputs"] == {"reactants": "C", "products": "CO"}
    assert instance["targets"]["bases"] == "CC"
    assert instance["target_mask"]["catalysts"] is False


def test_zero_yield_with_missing_product_remains_eligible():
    row = {"reactants": "C", "products": "", "reaction_smiles": "C>>", "yield_pct": "0", "catalyst": "Pd"}
    instance = make_instance(row, task_definition("yield"), metadata("yield/hitea.csv.gz"), "a" * 64, 0)
    assert instance["targets"] == {"yield_pct": 0.0}
    assert instance["inputs"]["products"] is None
    assert instance["inputs"]["context"] == {"catalyst": "Pd"}


def test_reference_tiers_are_not_silently_pooled():
    row = {
        "reaction_smiles": "C>>CO",
        "reference_replay_status": "VALID",
        "verification_level": "DELTA_CONSISTENT",
        "electron_path_json": '[["LP-/Sigma+",[1],[1,2]]]',
        "mechanism_record_json": '{"mapped_reaction":"C>>CO"}',
    }
    task, dataset = task_definition("mechanistic_prediction"), metadata("mechanism/synepd_polar.csv.gz")
    assert make_instance(row, task, dataset, "a" * 64, 0) is None
    row["verification_level"] = "EXACT"
    instance = make_instance(row, task, dataset, "a" * 64, 0)
    assert instance["targets"]["electron_path_json"] == [["LP-/Sigma+", [1], [1, 2]]]
    assert instance["inputs"] == {"mapped_overall_reaction": "C>>CO"}


def test_duplicate_native_ids_remain_distinct_rows_and_shared_groups_stay_together():
    row = {"rxn": "C>>CO", "ec1": "1", "ec2": "2", "ec3": "3", "r_id": "repeated"}
    dataset = metadata("classification/ecreact.csv.gz")
    a = make_instance(row, task_definition("classification"), dataset, "a" * 64, 0)
    b = make_instance(row, task_definition("classification"), dataset, "a" * 64, 1)
    c = make_instance(row, task_definition("clustering"), dataset, "a" * 64, 0)
    assert a["instance_id"] != b["instance_id"]
    assert a["group"] == b["group"] and a["proposed_split"] == b["proposed_split"]
    assert a["observation_id"] == c["observation_id"] and a["instance_id"] != c["instance_id"]


def test_generated_feasibility_provenance_is_not_a_predictor_input():
    row = {
        "reactants": "C",
        "products": "CO",
        "feasible": "0",
        "evidence_tier": "generated_negative",
        "negative_method": "ET",
        "additional_note": "label explanation",
    }
    instance = make_instance(
        row, task_definition("feasibility"), metadata("feasibility/frea_corpus.csv.gz"), "a" * 64, 0
    )
    assert set(instance["inputs"]) == {"reactants", "products"}
    assert instance["targets"]["feasible"] == 0
    assert instance["evidence_tier"] == "generated_negative"


def test_export_pairs_inputs_targets_and_counts_fields_without_summing_targets(tmp_path):
    source = tmp_path / "Data/yield/hitea.csv.gz"
    source.parent.mkdir(parents=True)
    with gzip.open(source, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "reactants", "products", "yield_pct"])
        writer.writerows([["one", "C", "CO", "60"], ["two", "C", "", "0"]])
    output = tmp_path / "view"
    report = export_task_view("yield", "yield/hitea.csv.gz", tmp_path / "Data", output)
    with gzip.open(output / "inputs.jsonl.gz", "rt") as handle:
        inputs = [json.loads(line) for line in handle]
    with gzip.open(output / "targets.jsonl.gz", "rt") as handle:
        targets = [json.loads(line) for line in handle]
    assert report["field_eligible_rows"] == 2 and report["observed_target_counts"] == {"yield_pct": 2}
    assert [row["instance_id"] for row in inputs] == [row["instance_id"] for row in targets]
    assert all("yield_pct" not in str(row["inputs"]) for row in inputs)


def test_classification_review_export_withholds_labels_and_leaves_assignments_open(tmp_path, capsys):
    from synrxn.__main__ import main

    data = tmp_path / "Data"
    key = "classification/schneider_b.csv.gz"
    source = data / key
    source.parent.mkdir(parents=True)
    with gzip.open(source, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["r_id", "rxn", "label", "split"])
        writer.writerows([["one", "CC>>CO", "35", "train"], ["two", "CC>>CO", "35", "test"]])
    output = tmp_path / "review"
    assert main([
        "task", "export", "--task-id", "classification", "--dataset", key,
        "--data-dir", str(data), "--output", str(output), "--no-proposed-split",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["proposed_split_counts"] == {} and report["seed"] is None
    with gzip.open(output / "inputs.jsonl.gz", "rt") as handle:
        inputs = [json.loads(line) for line in handle]
    with gzip.open(output / "membership.jsonl.gz", "rt") as handle:
        memberships = [json.loads(line) for line in handle]
    with gzip.open(output / "targets.jsonl.gz", "rt") as handle:
        targets = [json.loads(line) for line in handle]
    assert all(row["inputs"] == {"reactants": "CC", "products": "CO"} for row in inputs)
    assert all(row["targets"] == {"label": "35"} for row in targets)
    assert [row["instance_id"] for row in inputs] == [row["instance_id"] for row in memberships]
    assert [row["instance_id"] for row in inputs] == [row["instance_id"] for row in targets]
    assert [row["source_split"] for row in memberships] == ["train", "test"]
    assert all(row["proposed_split"] is None for row in memberships)


def test_energy_views_keep_reverse_direction_records_together_without_changing_targets():
    from synrxn.data import DatasetCatalog
    from synrxn.benchmarks.task_views import make_instance, task_definition

    dataset = DatasetCatalog().get("property", "rdb7")
    task = task_definition("energy_barrier")
    left = make_instance(
        {"aam": "[CH3:1][OH:2]>>[CH2:1]=[O:2]", "ea": "20", "split": "train"}, task, dataset, "a" * 64, 0
    )
    right = make_instance(
        {"aam": "[CH2:1]=[O:2]>>[CH3:1][OH:2]", "ea": "15", "split": "train"}, task, dataset, "a" * 64, 1
    )
    assert left["group"] == right["group"]
    assert left["proposed_split"] == right["proposed_split"]
    assert left["targets"]["ea"] == 20 and right["targets"]["ea"] == 15
    assert left["observation_id"] != right["observation_id"]
