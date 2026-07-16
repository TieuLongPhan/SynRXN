import copy
import gzip
import hashlib
import json
from pathlib import Path

from synrxn.build_manifest import analyze_file, build_manifest
from synrxn.verify_manifest import (
    ManifestContractError,
    main,
    manifest_files,
    verification_succeeded,
    verify_with_root,
)
from synrxn.validate_data import validate_catalog


def _artifact(path: Path, key: str) -> dict:
    content = path.read_bytes()
    return {
        "key": key,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _manifest(entry: dict) -> dict:
    return {
        "schema_version": "1.0",
        "generated_at": "2026-01-01T00:00:00Z",
        "dataset": {"files": [entry]},
        "provenance": {},
        "project": {},
    }


def test_analyze_file_counts_records_not_header(tmp_path):
    path = tmp_path / "rows.csv.gz"
    with gzip.open(path, "wt", encoding="utf8") as handle:
        handle.write("r_id,value\n1,a\n2,b\n")
    info = analyze_file(path)
    assert info["rows"] == 2
    assert info["columns"] == ["r_id", "value"]


def test_build_manifest_declares_schema_version(tmp_path):
    data = tmp_path / "Data" / "aam"
    data.mkdir(parents=True)
    (data / "sample.csv").write_text("r_id,rxn\n1,A>>B\n", encoding="utf8")
    manifest = build_manifest(
        tmp_path / "Data",
        meta={"name": "synrxn", "title": "SynRXN", "version": "1.0.0"},
        include_code_manifest=False,
    )
    assert manifest["schema_version"] == "1.0"
    assert manifest["dataset"]["files"][0]["rows"] == 1


def test_contract_rejects_empty_file_list():
    manifest = _manifest({"key": "aam/a.csv", "size": 1, "sha256": "0" * 64})
    manifest["dataset"]["files"] = []
    try:
        manifest_files(manifest)
    except ManifestContractError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("empty artifact list was accepted")


def test_verifier_detects_missing_unexpected_size_and_checksum(tmp_path):
    root = tmp_path / "Data"
    task = root / "aam"
    task.mkdir(parents=True)
    declared = task / "declared.csv"
    declared.write_bytes(b"abc")
    entry = _artifact(declared, "aam/declared.csv")
    manifest = _manifest(entry)

    assert verification_succeeded(verify_with_root(manifest, root))

    (task / "unexpected.csv").write_bytes(b"extra")
    result = verify_with_root(manifest, root)
    assert result["unexpected"] == ["aam/unexpected.csv"]

    declared.write_bytes(b"changed")
    result = verify_with_root(manifest, root, allow_unexpected=True)
    assert result["size_mismatch"]
    assert result["checksum_mismatch"]

    declared.unlink()
    assert verify_with_root(manifest, root)["missing"] == ["aam/declared.csv"]


def test_cli_success_and_contract_failure(tmp_path):
    root = tmp_path / "Data"
    task = root / "aam"
    task.mkdir(parents=True)
    artifact = task / "sample.csv"
    artifact.write_bytes(b"r_id\n1\n")
    manifest = _manifest(_artifact(artifact, "aam/sample.csv"))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf8")

    assert main(["--manifest", str(manifest_path), "--root", str(root), "--quiet"]) == 0

    invalid = copy.deepcopy(manifest)
    invalid.pop("schema_version")
    manifest_path.write_text(json.dumps(invalid), encoding="utf8")
    assert main(["--manifest", str(manifest_path), "--root", str(root)]) == 2


def test_catalog_validation_checks_splits_and_identifiers(tmp_path):
    import yaml

    root = tmp_path / "Data"
    task = root / "classification"
    task.mkdir(parents=True)
    artifact = task / "sample.csv.gz"
    with gzip.open(artifact, "wt", encoding="utf8") as handle:
        handle.write("r_id,rxn,label,split\n1,A>>B,x,train\n2,C>>D,y,test\n")
    key = "classification/sample.csv.gz"
    manifest = _manifest(_artifact(artifact, key))
    manifest["dataset"]["files"][0].update(
        {"rows": 2, "columns": ["r_id", "rxn", "label", "split"]}
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf8")
    metadata = {
        "schema_version": "1.0",
        "column_definitions": {
            column: {"description": column}
            for column in ("r_id", "rxn", "label", "split")
        },
        "datasets": {
            key: {
                "title": "Sample",
                "description": "Sample data",
                "benchmark_role": "Test validation",
                "license": "CC0",
                "citations": ["sample"],
                "targets": ["label"],
                "split_values": ["train", "test"],
                "row_identifier": "r_id",
                "row_identifier_unique": True,
            }
        },
    }
    metadata_path = root / "metadata.yaml"
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf8")
    result = validate_catalog(root, metadata_path, manifest_path)
    assert result["ok"] is True
    assert result["rows_checked"] == 2

    metadata["datasets"][key]["split_values"] = ["train", "valid", "test"]
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf8")
    result = validate_catalog(root, metadata_path, manifest_path)
    assert result["ok"] is False
    assert any("split values differ" in error for error in result["errors"])
