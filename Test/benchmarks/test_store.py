import hashlib
import json
import pytest

from synrxn.benchmarks.reference_store import ReferenceStore
from synrxn.__main__ import main


@pytest.fixture()
def store_path(tmp_path):
    assets = {
        "observations.jsonl": [{"observation_id": "obs:1", "source_record_id": "src:1"}],
        "references.jsonl": [{"reference_id": "ref:1", "observation_id": "obs:1", "revision": 1,
                              "source_record_id": "src:1", "evidence_tier": "unresolved",
                              "resolution": "unresolved", "evidence_locators": ["fixture"],
                              "task_eligibility": {"mapping": False}}],
        "task-instances.jsonl": [{"observation_id": "obs:1", "instance_id": "task:1",
                                  "reference_id": "ref:1", "reference_revision": 1}],
    }
    hashes = {}
    for name, values in assets.items():
        path = tmp_path / name
        path.write_text("".join(json.dumps(value) + "\n" for value in values))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "artifact-manifest.json").write_text(json.dumps({"schema_version": "1.0", "files": hashes}))
    return tmp_path


def test_store_resolves_exact_evidence_and_bounds_pages(store_path):
    store = ReferenceStore(store_path)
    assert store.inspect("obs:1")["references"][0]["resolution"] == "unresolved"
    assert store.release_id.startswith("sha256:")
    assert store.list(evidence_tier="unresolved", limit=1)["total"] == 1
    assert store.list(evidence_tier="unknown")["total"] == 0
    with pytest.raises(ValueError, match="limit"):
        store.list(limit=1001)


def test_corrupted_reference_release_is_rejected(store_path):
    (store_path / "references.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        ReferenceStore(store_path)


def test_reference_inspection_cli(store_path, capsys):
    assert main(["references", "--store", str(store_path), "--accession", "obs:1"]) == 0
    assert json.loads(capsys.readouterr().out)["observation"]["source_record_id"] == "src:1"
    assert main(["references", "--store", str(store_path), "--accession", "missing"]) == 2


def _repin(store_path, name, rows):
    path = store_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = store_path / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_valid_checksums_cannot_hide_broken_reference_history(store_path):
    reference = json.loads((store_path / "references.jsonl").read_text())
    reference.update({"revision": 2, "previous_revision": 1})
    _repin(store_path, "references.jsonl", [reference])
    with pytest.raises(ValueError, match="correction history"):
        ReferenceStore(store_path)


def test_reference_history_keeps_old_targets_and_resolves_latest_revision(store_path):
    reference = json.loads((store_path / "references.jsonl").read_text())
    revised = {**reference, "revision": 2, "previous_revision": 1,
               "review_events": [{"reviewer_kind": "ai_assistant", "decision": "remain unresolved"}]}
    _repin(store_path, "references.jsonl", [reference, revised])
    store = ReferenceStore(store_path)
    inspected = store.inspect("obs:1")
    assert [r["revision"] for r in inspected["references"]] == [1, 2]
    assert inspected["latest_references"][0]["revision"] == 2
    _repin(store_path, "references.jsonl", [reference, revised, revised])
    with pytest.raises(ValueError, match="duplicate reference revisions"):
        ReferenceStore(store_path)
