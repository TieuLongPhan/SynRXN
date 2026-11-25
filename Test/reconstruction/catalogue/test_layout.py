"""Scientific navigation must preserve shared identities and release decisions."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from synrxn.reconstruction.catalogue import layout
from synrxn.reconstruction.paths import REPO


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def tiny_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    families = [{"id": "outcomes", "title": "Reaction outcomes"},
                {"id": "conditions", "title": "Reaction conditions"}]
    definitions = [
        ("canonical", "yield/shared", True, "primary", "CC-BY-4.0"),
        ("coverage", "subset/overlap", True, "overlapping_view", "CC-BY-NC-4.0"),
        ("canonical", "synthesis/held", False, "primary", None),
    ]
    tables, rights, bindings = [], [], []
    for index, (namespace, key, admitted, role, license_name) in enumerate(definitions):
        path = repo / f"Data/payload/table_{index}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"record,yield\n{index:03},10\n".encode()
        path.write_bytes(payload)
        pin = hashlib.sha256(payload).hexdigest()
        table = {"namespace": namespace, "key": key, "title": f"Source table {index}",
                 "rows": 1, "role": role, "counting_unit": "measured assay observation",
                 "path": path.relative_to(repo).as_posix(), "sha256": pin,
                 "source_catalogue": "Data/payload/catalog.json",
                 "source_id": f"S{index}", "source_urls": ["https://example.org/source"],
                 "qualification": "A source-specific measurement."}
        tables.append(table)
        rights.append({"namespace": namespace, "key": key, "path": table["path"],
                       "format": "csv", "expected_sha256": pin, "expected_bytes": len(payload),
                       "expected_rows": 1, "license_expression": license_name,
                       "rights_status": "verified" if admitted else "unresolved_grant",
                       "license_obligations": ["Preserve attribution"],
                       "qualification": "Verified source" if admitted else "Hold redistribution"})
        bindings.append({"namespace": namespace, "key": key, "rows": 1, "role": role,
                         "counting_unit": table["counting_unit"],
                         "release_eligible": admitted, "task_specific": index})
    tasks = [
        {"id": "yield_conversion", "family_id": "outcomes", "title": "Predict assay outcomes",
         "question": "What response is measured?", "metrics": ["MAE"],
         "qualification": "Only recorded observations.", "bindings": bindings},
        {"id": "condition_ranking", "family_id": "conditions", "title": "Rank conditions",
         "question": "Which recorded condition performs best?", "metrics": ["NDCG"],
         "qualification": "Requires measured condition groups.", "bindings": [bindings[0]]},
    ]
    registry = {"families": families, "tasks": tasks}
    hierarchy = {"families": [
        {**families[0], "tasks": [{"id": tasks[0]["id"], "tables": tables}]},
        {**families[1], "tasks": [{"id": tasks[1]["id"], "tables": [tables[0]]}]},
    ]}
    release = {"status": "passed", "release_tables": rights[:2], "held_tables": rights[2:]}
    for name, value in zip(layout.INPUTS, (registry, hierarchy, release)):
        _write_json(repo / name, value)
    return repo


def test_hierarchy_preserves_shared_views_rights_and_task_metadata(tiny_repo):
    source_bytes = {p: p.read_bytes() for p in (tiny_repo / "Data/payload").glob("*.csv")}
    result = layout.build_layout(tiny_repo)
    assert result["summary"] == {
        "families": 2, "tasks": 2, "task_table_bindings": 4, "unique_table_views": 3,
        "primary_table_views": 2, "overlapping_table_views": 1,
        "admitted_table_views": 2, "held_table_views": 1, "data_links": 3,
        "held_bindings": 1, "stored_rows_across_unique_views": 3}
    base = tiny_repo / "Data/families"
    outcome = base / "outcomes/yield_conversion/canonical/yield/shared"
    condition = base / "conditions/condition_ranking/canonical/yield/shared"
    assert (outcome / "data.csv").is_file() and not (outcome / "data.csv").is_symlink()
    assert (outcome / "data.csv").resolve() == (condition / "data.csv").resolve()
    assert not os.path.isabs(os.readlink(condition / "data.csv"))
    metadata = json.loads((outcome / "table.json").read_text())
    assert metadata["rights"]["license_expression"] == "CC-BY-4.0"
    assert metadata["rights"]["license_obligations"] == ["Preserve attribution"]
    assert metadata["source"]["source_urls"] == ["https://example.org/source"]
    task = json.loads((base / "conditions/condition_ranking/task.json").read_text())
    assert task["task"]["metrics"] == ["NDCG"]
    assert task["task"]["qualification"] == "Requires measured condition groups."
    held = base / "outcomes/yield_conversion/canonical/synthesis/held"
    assert list(p.name for p in held.iterdir()) == ["table.json"]
    held_metadata = json.loads((held / "table.json").read_text())
    assert held_metadata["data_link"] is None
    assert held_metadata["release_admitted"] is False
    assert held_metadata["rights"]["rights_status"] == "unresolved_grant"
    overlap = base / "outcomes/yield_conversion/coverage/subset/overlap/table.json"
    assert json.loads(overlap.read_text())["rights"]["license_expression"] == "CC-BY-NC-4.0"
    index = json.loads((base / "index.json").read_text())
    assert {entry["metadata"] for entry in index["families"]} == {
        "outcomes/family.json", "conditions/family.json"}
    assert not list(base.rglob("README*"))
    assert {p: p.read_bytes() for p in source_bytes} == source_bytes
    assert layout.verify_layout(tiny_repo)["source_hashes_checked"] == 0


def test_payload_hashes_checked_once_per_admitted_identity(tiny_repo, monkeypatch):
    layout.build_layout(tiny_repo)
    original = layout.digest
    checked = []

    def observe(path):
        if Path(path).name == "data.csv":
            checked.append(Path(path))
        return original(path)

    monkeypatch.setattr(layout, "digest", observe)
    result = layout.verify_layout(tiny_repo, check_hashes=True)
    assert len(checked) == 2
    assert len(set(checked)) == 2
    assert result["source_hashes_checked"] == 2
    assert "not recounted" in result["source_row_counts"]
    payload = tiny_repo / "Data/families/outcomes/yield_conversion/canonical/yield/shared/data.csv"
    content = payload.read_bytes()
    payload.write_bytes(content.replace(b",10", b",99"))
    with pytest.raises(ValueError, match="Source SHA256 mismatch"):
        layout.verify_layout(tiny_repo, check_hashes=True)


def test_spectra_folder_names_preserve_source_identity_and_shared_file(tiny_repo):
    old_key = "yield/shared"
    source_key = "process/nmr_pmg_284_snapshots"
    for name in layout.INPUTS:
        path = tiny_repo / name
        document = json.loads(path.read_text())

        def rename_identity(value):
            if isinstance(value, dict):
                if value.get("namespace") == "canonical" and value.get("key") == old_key:
                    value.update(namespace="coverage", key=source_key)
                for item in value.values():
                    rename_identity(item)
            elif isinstance(value, list):
                for item in value:
                    rename_identity(item)

        rename_identity(document)
        _write_json(path, document)

    layout.build_layout(tiny_repo, check_hashes=True)
    base = tiny_repo / "Data/families"
    folders = [base / family / task / "coverage/process/nmr_pmg_284_spectra"
               for family, task in [("outcomes", "yield_conversion"),
                                    ("conditions", "condition_ranking")]]
    for folder in folders:
        metadata = json.loads((folder / "table.json").read_text())
        assert metadata["key"] == source_key
        assert metadata["display_key"] == "process/nmr_pmg_284_spectra"
        assert not folder.with_name("nmr_pmg_284_snapshots").exists()
        assert (folder / "data.csv").resolve() == folders[0] / "data.csv"
    assert (folders[0] / "data.csv").resolve() == (folders[1] / "data.csv").resolve()


def test_tampered_link_is_detected_and_preserved_on_rebuild(tiny_repo):
    layout.build_layout(tiny_repo)
    link = tiny_repo / "Data/families/conditions/condition_ranking/canonical/yield/shared/data.csv"
    source = link.resolve()
    link.unlink()
    link.symlink_to(source)
    with pytest.raises(ValueError, match="data link was modified"):
        layout.verify_layout(tiny_repo)
    with pytest.raises(ValueError, match="data link was modified"):
        layout.build_layout(tiny_repo)
    assert os.readlink(link) == str(source)


@pytest.mark.parametrize("extra", ["notes.txt", "personal/notes.txt", "empty/"])
def test_rebuild_preserves_unmanaged_entries(tiny_repo, extra):
    layout.build_layout(tiny_repo)
    path = tiny_repo / "Data/families" / extra
    if extra.endswith("/"):
        path.mkdir()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Keep my notes.")
    with pytest.raises(ValueError, match="unmanaged"):
        layout.build_layout(tiny_repo)
    assert path.exists()


def test_modified_metadata_not_overwritten(tiny_repo):
    layout.build_layout(tiny_repo)
    metadata = tiny_repo / "Data/families/outcomes/family.json"
    metadata.write_text("My changes")
    with pytest.raises(ValueError, match="metadata was modified"):
        layout.build_layout(tiny_repo)
    assert metadata.read_text() == "My changes"


@pytest.mark.parametrize("modified", [False, True])
def test_previous_managed_readme_removed_only_when_unmodified(tiny_repo, modified):
    """Retire old generated prose through the same managed-file safety checks."""
    layout.build_layout(tiny_repo)
    base = tiny_repo / "Data/families"
    marker = base / layout.MANIFEST
    manifest = json.loads(marker.read_text())
    readme = base / "outcomes/README.md"
    payload = b"# Previously generated family documentation\n"
    readme.write_bytes(payload)
    manifest["files"]["outcomes/README.md"] = {
        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    _write_json(marker, manifest)
    if modified:
        readme.write_text("Keep my changes.")
        with pytest.raises(ValueError, match="metadata was modified"):
            layout.build_layout(tiny_repo)
        assert readme.read_text() == "Keep my changes."
    else:
        layout.build_layout(tiny_repo)
        assert not list(base.rglob("README*"))
        assert layout.verify_layout(tiny_repo)["status"] == "passed"


def test_clean_managed_rebuild_and_current_input_change(tiny_repo):
    layout.build_layout(tiny_repo)
    marker = tiny_repo / "Data/families" / layout.MANIFEST
    previous = marker.read_bytes()
    layout.build_layout(tiny_repo)
    assert marker.read_bytes() == previous
    path = tiny_repo / layout.INPUTS[0]
    registry = json.loads(path.read_text())
    registry["tasks"][0]["qualification"] = "New source qualification"
    _write_json(path, registry)
    with pytest.raises(ValueError, match="current source metadata"):
        layout.verify_layout(tiny_repo)
    layout.build_layout(tiny_repo)
    metadata = tiny_repo / "Data/families/outcomes/yield_conversion/task.json"
    assert json.loads(metadata.read_text())["task"]["qualification"] == "New source qualification"


def test_place_table_creates_real_output_without_the_release_payload(tiny_repo):
    original = tiny_repo / "Data/payload/table_1.csv"
    built = original.rename(tiny_repo / "rebuilt.csv")
    destination = layout.place_table(built, "outcomes", "yield_conversion", "coverage", "subset/overlap", tiny_repo)
    assert destination == tiny_repo / "Data/families/outcomes/yield_conversion/coverage/subset/overlap/data.csv"
    assert destination.is_file() and not destination.is_symlink()
    assert destination.read_bytes() == built.read_bytes()
    layout.build_layout(tiny_repo, check_hashes=True)
    assert layout.verify_layout(tiny_repo)["materialized_tables"] == 2
    assert not destination.is_symlink()


def test_place_shared_table_in_a_checkout_without_payloads_or_links(tiny_repo):
    layout.build_layout(tiny_repo)
    base = tiny_repo / "Data/families"
    for payload in base.rglob("data.csv"):
        payload.unlink()
    assert layout.verify_layout(tiny_repo, allow_missing=True)["pending_payloads"] == 2
    with pytest.raises(ValueError, match="missing"):
        layout.verify_layout(tiny_repo)
    source = (tiny_repo / "Data/payload/table_0.csv").rename(tiny_repo / "restored.csv")
    (tiny_repo / "Data/payload/table_1.csv").unlink()
    destination = layout.place_table(
        source, "conditions", "condition_ranking", "canonical", "yield/shared", tiny_repo
    )
    owner = base / "outcomes/yield_conversion/canonical/yield/shared/data.csv"
    assert destination.is_symlink()
    assert destination.resolve() == owner
    assert owner.read_bytes() == source.read_bytes()
    assert layout.verify_layout(tiny_repo, allow_missing=True)["pending_payloads"] == 1


def test_escaping_source_and_identity_rejected(tiny_repo, tmp_path):
    outside = tmp_path / "outside.csv"
    outside.write_bytes((tiny_repo / "Data/payload/table_0.csv").read_bytes())
    source = tiny_repo / "Data/payload/table_0.csv"
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(ValueError, match="escaping source"):
        layout.build_layout(tiny_repo)
    source.unlink()
    source.write_bytes(outside.read_bytes())
    registry = json.loads((tiny_repo / layout.INPUTS[0]).read_text())
    registry["tasks"][0]["bindings"][0]["key"] = "../escape"
    _write_json(tiny_repo / layout.INPUTS[0], registry)
    with pytest.raises(ValueError, match="Unsafe relative path"):
        layout.build_layout(tiny_repo)
    with pytest.raises(ValueError, match="subdirectory"):
        layout.build_layout(tiny_repo, tmp_path / "outside-layout")


def test_metadata_disagreement_fails_before_creating_output(tiny_repo):
    path = tiny_repo / layout.INPUTS[1]
    hierarchy = json.loads(path.read_text())
    hierarchy["families"][0]["tasks"][0]["tables"][0]["rows"] = 2
    _write_json(path, hierarchy)
    with pytest.raises(ValueError, match="rows disagreement"):
        layout.build_layout(tiny_repo)
    assert not (tiny_repo / "Data/families").exists()


def test_current_repository_complete_scientific_inventory():
    """Catch dropped task bindings and accidental inclusion of quarantined views."""
    inventory = layout._load(REPO)
    summary = layout._summary(inventory)
    assert (summary["families"], summary["tasks"], summary["task_table_bindings"],
            summary["unique_table_views"]) == (12, 41, 146, 105)
    assert (summary["primary_table_views"], summary["overlapping_table_views"]) == (98, 7)
    assert (summary["admitted_table_views"], summary["held_table_views"],
            summary["data_links"], summary["held_bindings"]) == (104, 1, 144, 2)
    assert summary["stored_rows_across_unique_views"] == 19611896
