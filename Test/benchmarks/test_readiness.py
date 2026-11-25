"""Readiness must not equate target presence or saved metrics with validation."""

import csv
import gzip
import json

from synrxn.benchmarks.readiness import inspect_run, inspect_table, legacy_metric_series, select_tables, target_state


def test_shared_task_views_exclude_deferred_activity_and_aliases():
    records = [
        {"key": "classification/classes.csv.gz", "targets": ["label"]},
        {"key": "classification/ehreact_activity.csv.gz", "targets": ["activity"]},
        {"key": "condition/uspto_500.csv.gz", "targets": ["reagent"]},
    ]
    task = {"selectors": [{"prefix": "classification/", "exclude": [records[1]["key"]]}]}
    assert select_tables(records, task) == [records[0]]
    reagent_task = {"selectors": [{"key": "synthesis/uspto_500.csv.gz"}]}
    assert select_tables(records, reagent_task) == []


def test_zero_yields_are_valid_but_nonfinite_and_out_of_range_values_are_not():
    assert target_state("0", "yield_pct") == "present"
    assert target_state("100", "yield_pct") == "present"
    assert target_state("101", "yield_pct") == "out_of_range"
    assert target_state("inf", "yield_pct") == "nonfinite"
    assert target_state("unknown", "yield_pct") == "not_numeric"
    # No unqualified energy range or unit conversion is invented.
    assert target_state("-12.3", "G_act") == "present"


def test_streamed_field_screens_remain_separate_from_benchmark_eligibility(tmp_path):
    path = tmp_path / "table.csv.gz"
    rows = [
        {"sample_id": "a", "reaction_smiles": "C>>CO", "yield_pct": "0", "upstream_split": "train"},
        {"sample_id": "a", "reaction_smiles": "C>>CO", "yield_pct": "101", "upstream_split": "test"},
        {"sample_id": "b", "reaction_smiles": "C>>", "yield_pct": "50", "upstream_split": "test"},
        {
            "sample_id": "c",
            "reaction_smiles": "not-a-molecule>>also-invalid",
            "yield_pct": "30",
            "upstream_split": "train",
        },
    ]
    with gzip.open(path, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = inspect_table(path, {"row_identifier": "sample_id", "split_column": "upstream_split"}, ["yield_pct"])
    assert report["rows_measured"] == 4
    assert report["duplicate_identifiers"] == 1
    assert report["reaction_shape_failures"] == 1
    assert report["rows_passing_field_screen_by_target"] == {"yield_pct": 3}
    assert report["rows_passing_target_and_both_endpoint_screen"] == 2
    assert report["benchmark_eligible_count"] is None
    assert "Molecular parsing" in report["screen_limit"]


def test_sparse_role_targets_are_counted_per_target_without_inventing_complete_roles(tmp_path):
    path = tmp_path / "roles.csv"
    path.write_text("sample_id,reaction_smiles,catalysts,bases\na,C>>CO,,O\nb,C>>CO,[],\n")
    report = inspect_table(path, {"row_identifier": "sample_id", "split_status": "proposed"}, ["catalysts", "bases"])
    assert report["rows_passing_any_requested_target_screen"] == 1
    assert report["target_field_states"]["catalysts"] == {"missing": 2}
    assert report["benchmark_eligible_count"] is None


def test_bad_result_metrics_are_retained_and_not_promoted_to_reproducibility():
    run = {
        "dataset": "x",
        "fingerprint": "fp",
        "model": "knn",
        "scheme": "cv",
        "seed": 0,
        "mcc": float("nan"),
        "accuracy": 1.2,
        "n_test": -1,
    }
    report = inspect_run(run, "classification")
    assert report["state"] == "incomplete_or_invalid_metric_record"
    assert len(report["issues"]) == 3
    assert report["scientific_reproduction_verified"] is False
    json.dumps(report, allow_nan=False)


def test_metric_only_result_does_not_certify_the_original_cohort():
    run = {
        "dataset": "x",
        "fingerprint": "fp",
        "model": "knn",
        "scheme": "cv",
        "seed": 0,
        "mcc": 0.5,
        "n_train": 80,
        "n_val": 20,
        "n_test": 20,
    }
    report = inspect_run(run, "classification")
    assert report["state"] == "metric_record_present"
    assert report["test_membership_pin_present"] is False
    assert report["data_pin_present"] is False
    assert report["scientific_reproduction_verified"] is False


def test_legacy_numpy_wrapper_is_inventory_only():
    summary = legacy_metric_series({"fp": {"test_mae": {"__ndarray__": True, "data": [1.0, None]}, "fit_time": [1.0]}})
    assert summary["metric_series"] == 1
    assert summary["finite_metric_values"] == 1 and summary["invalid_metric_values"] == 1
    assert summary["benchmark_eligible_count"] is None
