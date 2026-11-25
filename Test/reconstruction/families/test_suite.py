"""Test reproducibility failures and scope accounting using tiny local fixtures."""

import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from synrxn.reconstruction.families import suite as reproduce
from synrxn.reconstruction.paths import physical_path

HERE = physical_path("kdd")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source_record(repo, relative="Data/example/tiny.csv.gz", expected_rows=2):
    """Make two logical records, one containing a physical newline."""
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["r_id", "description"])
        writer.writerow(["r0", "first line\nsecond line"])
        writer.writerow(["r1", "ordinary"])
    payload = path.read_bytes()
    return {
        "namespace": "canonical",
        "key": "example/tiny",
        "path": relative,
        "expected_sha256": hashlib.sha256(payload).hexdigest(),
        "expected_bytes": len(payload),
        "expected_rows": expected_rows,
        "format": "csv.gz",
        "role": "primary",
    }


def small_registry():
    shared = {"namespace": "canonical", "key": "example/tiny", "rows": 2, "role": "primary"}
    overlap = {
        "namespace": "expansion", "key": "C00/overlap", "rows": 1, "role": "overlapping_view"
    }
    status = "not completed by this planning work"
    tasks = [
        {
            "id": "forward", "title": "Forward", "family_id": "transformations",
            "bindings": [copy.deepcopy(shared), overlap], "primary_table_count": 1,
            "primary_source_rows": 2, "legacy_view_ids": [],
            "readiness": "source data; contract pending", "benchmark_result_status": status,
        },
        {
            "id": "reverse", "title": "Reverse", "family_id": "transformations",
            "bindings": [copy.deepcopy(shared)], "primary_table_count": 1,
            "primary_source_rows": 2, "legacy_view_ids": ["old_reverse"],
            "readiness": "export available; models pending", "benchmark_result_status": status,
        },
    ]
    future = {
        "id": "monitoring", "title": "Monitoring", "family_id": "analysis", "bindings": [],
        "readiness": "roadmap; no integrated source", "required_data": "Time series and endpoints",
        "evaluation": "Held-out experiment error", "benchmark_result_status": status,
    }
    return {"tasks": tasks, "future_tasks": [future], "families": [
        {"id": "transformations", "title": "Transformations", "task_ids": ["forward", "reverse"],
         "future_task_ids": []},
        {"id": "analysis", "title": "Analysis", "task_ids": [], "future_task_ids": ["monitoring"]},
    ]}


def test_inventory_resolves_expansion_under_its_own_root_without_double_data(tmp_path):
    write_json(tmp_path / "doc/_static/catalog-data.json", {"datasets": [{
        "id": "aam/tiny", "sha256": "a" * 64, "size": 17, "rows": 3,
    }]})
    write_json(tmp_path / "doc/_static/catalog-sharealike-data.json", {"datasets": [{
        "id": "condition_sharealike/tiny", "sha256": "b" * 64, "size": 19, "rows": 4,
    }]})
    write_json(tmp_path / "Data/expansion/catalog.json", {"tables": [{
        "key": "C00/tiny", "path": "main/C00/tiny.parquet", "sha256": "c" * 64,
        "bytes": 23, "rows": 5, "role": "overlapping_view",
    }]})

    records = reproduce.dataset_inventory(tmp_path)
    by_key = {(item["namespace"], item["key"]): item for item in records}
    assert len(records) == len(by_key) == 3
    canonical = by_key[("canonical", "aam/tiny")]
    assert canonical["path"] == "Data/aam/tiny.csv.gz"
    assert canonical["expected_sha256"] == "a" * 64
    assert (canonical["expected_bytes"], canonical["expected_rows"], canonical["format"]) == (17, 3, "csv.gz")
    expansion = by_key[("expansion", "C00/tiny")]
    assert expansion["path"] == "Data/expansion/main/C00/tiny.parquet"
    assert expansion["expected_sha256"] == "c" * 64
    assert (expansion["expected_bytes"], expansion["expected_rows"], expansion["format"]) == (23, 5, "parquet")
    assert expansion["role"] == "overlapping_view"


def test_hash_audit_preserves_declared_rows_without_claiming_to_recount_them(tmp_path):
    record = source_record(tmp_path)
    audit = reproduce.audit_file(record, tmp_path)
    assert audit["status"] == "verified"
    assert audit["errors"] == []
    assert audit["observed_sha256"] == record["expected_sha256"]
    assert audit["observed_bytes"] == record["expected_bytes"]
    assert audit["observed_rows"] is None
    assert audit["row_count_status"] == "catalog_declared"
    assert audit["path"] == record["path"]
    assert "status" not in record  # Do not mutate the source expectation.


def test_gzip_row_audit_counts_csv_records_including_a_quoted_newline(tmp_path):
    record = source_record(tmp_path)
    audit = reproduce.audit_file(record, tmp_path, check_rows=True)
    assert audit["status"] == "verified"
    assert audit["observed_rows"] == 2
    assert audit["row_count_status"] == "verified"
    assert audit["errors"] == []


def test_row_count_mismatch_fails_even_with_matching_hash_and_size(tmp_path):
    record = source_record(tmp_path, expected_rows=3)
    audit = reproduce.audit_file(record, tmp_path, check_rows=True)
    assert audit["observed_sha256"] == record["expected_sha256"]
    assert audit["observed_rows"] == 2
    assert audit["status"] == audit["row_count_status"] == "failed"
    assert audit["errors"]


def test_changed_source_fails_even_when_file_length_is_unchanged(tmp_path):
    record = source_record(tmp_path)
    path = tmp_path / record["path"]
    changed = bytearray(path.read_bytes())
    changed[-1] ^= 1
    path.write_bytes(changed)
    audit = reproduce.audit_file(record, tmp_path)
    assert audit["observed_bytes"] == record["expected_bytes"]
    assert audit["observed_sha256"] != record["expected_sha256"]
    assert audit["status"] == "failed"
    assert audit["errors"]


def test_incorrect_declared_size_fails_independently_of_hash(tmp_path):
    record = source_record(tmp_path)
    record["expected_bytes"] += 1
    audit = reproduce.audit_file(record, tmp_path)
    assert audit["observed_sha256"] == record["expected_sha256"]
    assert audit["status"] == "failed"
    assert audit["errors"]


def test_missing_source_reports_failure_without_crashing(tmp_path):
    record = source_record(tmp_path)
    (tmp_path / record["path"]).unlink()
    audit = reproduce.audit_file(record, tmp_path)
    assert audit["status"] == "failed"
    assert audit["errors"]


def test_legacy_hash_only_record_is_audited_without_invented_row_evidence(tmp_path):
    record = source_record(tmp_path)
    for field in ("expected_bytes", "expected_rows", "format"):
        del record[field]
    audit = reproduce.audit_file(record, tmp_path)
    assert audit["status"] == "verified"
    assert audit["observed_rows"] is None
    assert audit["row_count_status"] == "not_requested"


@pytest.mark.parametrize("escape", ["parent", "symlink"])
def test_audit_rejects_sources_resolving_outside_repository(tmp_path, escape):
    repo = tmp_path / "repo"
    repo.mkdir()
    record = source_record(tmp_path, relative="outside.csv.gz")
    if escape == "parent":
        record["path"] = "../outside.csv.gz"
    else:
        (repo / "link.csv.gz").symlink_to(tmp_path / "outside.csv.gz")
        record["path"] = "link.csv.gz"
    audit = reproduce.audit_file(record, repo, check_rows=True)
    assert audit["status"] == "failed"
    assert audit["errors"]
    # A valid outside file must not be read and presented as verified local evidence.
    assert audit.get("observed_sha256") in (None, "")


def test_task_matrix_keeps_shared_sources_local_and_roadmap_unavailable():
    registry = small_registry()
    audits = [
        {"namespace": "canonical", "key": "example/tiny", "status": "verified", "errors": []},
        {"namespace": "expansion", "key": "C00/overlap", "status": "verified", "errors": []},
    ]
    matrix = {row["id"]: row for row in reproduce.build_task_matrix(registry, audits)}
    assert set(matrix) == {"forward", "reverse", "monitoring"}
    for name in ("forward", "reverse"):
        assert matrix[name]["data_status"] == "source_backed"
        assert matrix[name]["source_verification"] == "verified"
        # Both tasks use one shared source; overlapping views add no primary table or rows.
        assert matrix[name]["primary_table_count"] == 1
        assert matrix[name]["primary_source_rows"] == 2
        assert matrix[name]["benchmark_status"] == "not completed by this planning work"
    assert matrix["reverse"]["legacy_view_ids"] == ["old_reverse"]
    roadmap = matrix["monitoring"]
    assert roadmap["family_id"] == "analysis"
    assert roadmap["data_status"] == "roadmap"
    assert roadmap["source_verification"] == "not_available"
    assert roadmap["primary_table_count"] == roadmap["primary_source_rows"] == 0
    assert roadmap["benchmark_status"] == "not completed by this planning work"
    assert roadmap["blockers"]


@pytest.mark.parametrize("audits", [[], [
    {"namespace": "canonical", "key": "example/tiny", "status": "failed", "errors": ["hash mismatch"]}
]])
def test_unverified_sources_block_task_verification(audits):
    matrix = {row["id"]: row for row in reproduce.build_task_matrix(small_registry(), audits)}
    assert matrix["reverse"]["source_verification"] == "failed"
    assert matrix["reverse"]["blockers"]
    assert matrix["reverse"]["benchmark_status"] == "not completed by this planning work"


def test_full_registry_matrix_preserves_all_41_definitions_and_roadmap_boundary():
    registry = json.loads((HERE / "task_registry.json").read_text())
    audits = [
        {"namespace": binding["namespace"], "key": binding["key"], "status": "verified", "errors": []}
        for task in registry["tasks"] for binding in task["bindings"]
    ]
    # Deduplicate the shared audit evidence before it enters the matrix.
    audits = list({(item["namespace"], item["key"]): item for item in audits}.values())
    matrix = reproduce.build_task_matrix(registry, audits)
    assert len(matrix) == len({row["id"] for row in matrix}) == 41
    assert len({row["family_id"] for row in matrix}) == 12
    assert sum(row["data_status"] == "source_backed" for row in matrix) == len(registry['tasks'])
    assert sum(row["data_status"] == "roadmap" for row in matrix) == len(registry['future_tasks'])
    assert all(row["benchmark_status"] == "not completed by this planning work" for row in matrix)
    assert all(row["source_verification"] == "not_available" for row in matrix if row["data_status"] == "roadmap")
    assert "multistep_planning" not in {row["id"] for row in matrix}


@pytest.mark.parametrize('format_', ['csv', 'jsonl.gz'])
def test_new_coverage_formats_are_recounted(tmp_path, format_):
    path = tmp_path / ('tiny.' + format_)
    if format_ == 'csv':
        path.write_text('record_id,value\nr0,1\nr1,2\n')
    else:
        with gzip.open(path, 'wt') as handle:
            handle.write('{"record_id":"r0"}\n{"record_id":"r1"}\n')
    record = {'path': path.name, 'format': format_, 'expected_rows': 2,
              'expected_sha256': reproduce.sha(path), 'expected_bytes': path.stat().st_size}
    audit = reproduce.audit_file(record, tmp_path, check_rows=True)
    assert audit['status'] == audit['row_count_status'] == 'verified'
    assert audit['observed_rows'] == 2


def test_artifact_comparison_identifies_match_mismatch_and_missing_outputs(tmp_path):
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    expected.mkdir()
    actual.mkdir()
    for name in ("equal.json", "changed.json", "missing.json"):
        (expected / name).write_text('{"frozen": true}\n')
    (actual / "equal.json").write_bytes((expected / "equal.json").read_bytes())
    (actual / "changed.json").write_text('{"frozen": false}\n')
    results = reproduce.compare_artifacts(expected, actual, ["equal.json", "changed.json", "missing.json"])
    assert len(results) == 3
    assert results[0]["status"] == "verified"
    assert results[0]["errors"] == []
    assert results[1]["status"] == "failed"
    assert results[1]["errors"]
    assert results[2]["status"] == "failed"
    assert results[2]["errors"]


@pytest.mark.parametrize("registry_contents", [None, "{malformed JSON"])
def test_main_persists_failed_manifest_when_registry_is_missing_or_malformed(tmp_path, monkeypatch, registry_contents):
    repo = tmp_path / "repo"
    kdd = repo / "kdd"
    kdd.mkdir(parents=True)
    monkeypatch.setattr(reproduce, "REPO", repo)
    monkeypatch.setattr(reproduce, "KDD", kdd)
    if registry_contents is not None:
        (kdd / "task_registry.json").write_text(registry_contents)
    run = tmp_path / "run"

    result = reproduce.main(["--profile", "scope", "--output-dir", str(run)])

    assert result != 0
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["errors"]
    assert manifest["completed_benchmarks_in_this_run"] == 0
    assert manifest["finished_at"]
    assert manifest["elapsed_seconds"] >= 0


def test_main_retains_readiness_and_original_failure_when_finalization_also_fails(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    kdd = repo / "kdd"
    kdd.mkdir(parents=True)
    monkeypatch.setattr(reproduce, "REPO", repo)
    monkeypatch.setattr(reproduce, "KDD", kdd)
    registry = small_registry()
    registry.update(summary={"scientific_families": 2, "local_data_backed_task_definitions": 2,
                             "future_task_definitions": 1}, source_sha256={}, excluded_tasks=[])
    write_json(kdd / "task_registry.json", registry)
    # The expansion metadata and study protocol are both absent. The latter fails
    # during finalization, after task readiness has already been written.
    run = tmp_path / "run"

    result = reproduce.main(["--profile", "scope", "--output-dir", str(run)])

    assert result != 0
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert len(manifest["errors"]) >= 2
    assert any("catalog.json" in error for error in manifest["errors"])
    assert any("Task-report finalization failed" in error for error in manifest["errors"])
    assert manifest["counts"]["total_task_definitions"] == 3
    readiness = json.loads((run / "task_readiness.json").read_text())
    assert {row["id"] for row in readiness} == {"forward", "reverse", "monitoring"}
    assert {row["source_verification"] for row in readiness if row["data_status"] == "source_backed"} == {"failed"}
    assert {row["source_verification"] for row in readiness if row["data_status"] == "roadmap"} == {"not_available"}
    assert manifest["task_readiness_counts"] == {"source_backed": 2, "roadmap": 1}
    assert manifest["completed_benchmarks_in_this_run"] == 0
    assert "task_readiness.json" in {item["path"] for item in manifest["output_artifacts"]}


def test_main_rejects_nonempty_output_without_changing_existing_files(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    kdd = repo / "kdd"
    kdd.mkdir(parents=True)
    monkeypatch.setattr(reproduce, "REPO", repo)
    monkeypatch.setattr(reproduce, "KDD", kdd)
    run = tmp_path / "existing-run"
    run.mkdir()
    sentinel = run / "sentinel.txt"
    sentinel.write_bytes(b"An existing result must survive.\n")

    with pytest.raises(SystemExit) as error:
        reproduce.main(["--profile", "scope", "--output-dir", str(run)])

    assert error.value.code == 2
    assert sentinel.read_bytes() == b"An existing result must survive.\n"
    assert list(run.iterdir()) == [sentinel]


def test_preparation_validators_write_current_results_without_copying_legacy_reports(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    metadata = repo / "kdd"
    metadata.mkdir(parents=True)
    registry = {"summary": {}, "tasks": [], "future_tasks": [], "excluded_tasks": []}
    write_json(metadata / "task_registry.json", registry)
    write_json(metadata / "study_protocol.json", {})
    for name in reproduce.HITEA_OUTPUTS:
        target = metadata / "preparation" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"source reference\n")
    obsolete = [metadata / "preparation/plan_validation.json",
                metadata / "literature/validation.json"]
    for path in obsolete:
        write_json(path, {"status": "failed", "legacy_result": True})
    original_reports = {path: path.read_bytes() for path in obsolete}
    monkeypatch.setattr(reproduce, "REPO", repo)
    monkeypatch.setattr(reproduce, "KDD", metadata)
    monkeypatch.setattr(reproduce, "metadata_pins", lambda *args: [])
    monkeypatch.setattr(reproduce, "snapshot", lambda *args, **kwargs: [])
    monkeypatch.setattr(reproduce, "environment", lambda *args: {"subprocess_environment": {}})
    monkeypatch.setattr(reproduce, "dataset_inventory", lambda *args: [])
    monkeypatch.setattr(reproduce, "legacy_inventory", lambda *args: [])
    commands = {}

    def run_step(step_id, command, run, manifest, env, timeout):
        commands[step_id] = command
        if "--output" not in command:
            return
        output = Path(command[command.index("--output") + 1])
        if step_id == "build_registry":
            write_json(output / "task_registry.json", registry)
        elif step_id == "prepare_hitea":
            output.mkdir(parents=True)
            for name in reproduce.HITEA_OUTPUTS:
                (output / name).write_bytes((metadata / "preparation" / name).read_bytes())
        else:
            write_json(output, {"status": "passed", "validator": step_id})

    monkeypatch.setattr(reproduce, "execute", run_step)
    run = tmp_path / "run"

    assert reproduce.main(["--profile", "preparation", "--output-dir", str(run)]) == 0

    for step_id, name in (("validate_plan", "protocol_validation.json"),
                          ("validate_literature", "literature_validation.json")):
        output = run / "artifacts" / name
        command = commands[step_id]
        assert command[command.index("--output") + 1] == str(output)
        assert json.loads(output.read_text()) == {"status": "passed", "validator": step_id}
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "passed"
    assert not {"artifacts/plan_validation.json", "artifacts/validation.json"} & {
        item["path"] for item in manifest["output_artifacts"]}
    assert {path: path.read_bytes() for path in obsolete} == original_reports
