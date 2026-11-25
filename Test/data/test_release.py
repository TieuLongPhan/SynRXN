"""Release boundaries must hold even when local excluded artifacts exist."""

import copy
import gzip
import json
from pathlib import Path

import pytest
import yaml

from synrxn.data.build_catalog import build_catalog
from synrxn.data.build_manifest import build_cff_from_manifest, build_manifest
from synrxn.data import DatasetCatalog
from synrxn.data.release import release_keys
from synrxn.access.query import convert_dataset, convert_release
from synrxn.data.verify_manifest import verify_with_root, verification_succeeded
from synrxn.data.validate import validate_catalog


@pytest.fixture()
def partitioned_data(tmp_path):
    root = tmp_path / "Data"
    records = {}
    for key, partition, license_id in [
        ("yield/main.csv.gz", "main", "MIT"),
        ("yield/pending.csv.gz", "excluded", "LicenseRef-Unknown"),
        ("condition_sharealike/sa.csv.gz", "sharealike", "CC-BY-SA-4.0"),
        ("condition/mirror.csv.gz", "main", "MIT"),
    ]:
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        # Equal gzip bytes are important for byte-identical alias checking.
        path.write_bytes(gzip.compress(b"r_id,rxn\n1,C>>CO\n", mtime=0))
        records[key] = {"redistribution_partition": partition,
                        "license_spdx": license_id}
    records["condition/mirror.csv.gz"]["alias_of"] = "yield/main.csv.gz"
    (root / "metadata.yaml").write_text(yaml.safe_dump({"datasets": records}))
    return root, records


def test_manifest_partitions_exclude_pending_and_mirror(partitioned_data):
    root, _ = partitioned_data
    for partition, expected in [("main", "yield/main.csv.gz"),
                                ("sharealike", "condition_sharealike/sa.csv.gz")]:
        manifest = build_manifest(root, {"version": "test"}, partition=partition,
                                  candidate=True, include_code_manifest=False)
        datasets = [f["key"] for f in manifest["dataset"]["files"]
                    if f["key"].endswith(".csv.gz")]
        assert datasets == [expected]
        assert verification_succeeded(verify_with_root(manifest, root))


def test_candidate_citation_does_not_claim_a_deposit_or_release_date(partitioned_data):
    root, _ = partitioned_data
    manifest = build_manifest(root, {"version": "1.5.0"}, candidate=True,
                              include_code_manifest=False)
    overrides = {"doi": "10.5281/zenodo.17297258",
                 "url": "https://doi.org/10.5281/zenodo.17297258"}
    cff = build_cff_from_manifest(manifest, overrides)
    assert "doi" not in cff
    assert "date-released" not in cff
    assert "url" not in cff
    assert "Unpublished" in cff["message"]


def test_modified_alias_fails_instead_of_silently_deduplicating(partitioned_data):
    root, _ = partitioned_data
    (root / "condition/mirror.csv.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="alias differs"):
        build_manifest(root, {}, include_code_manifest=False)


def test_held_catalogue_record_uses_metadata_without_reading_payload(partitioned_data):
    root, records = partitioned_data
    path = root / "yield/pending.csv.gz"
    records["yield/pending.csv.gz"]["catalogue_metadata"] = {
        "rows": 1, "size": path.stat().st_size, "sha256": "1" * 64,
        "columns": ["r_id", "rxn"],
    }
    metadata = root / "metadata.yaml"
    metadata.write_text(yaml.safe_dump({"column_definitions": {}, "datasets": records}))
    path.unlink()
    manifest = root.parent / "manifest.json"
    manifest.write_text(json.dumps(
        build_manifest(root, {"version": "test"}, include_code_manifest=False)
    ))
    catalogue = build_catalog(root, metadata, manifest, root.parent / "catalog.json")
    held = next(item for item in catalogue["datasets"] if item["id"] == "yield/pending")
    assert held["rows"] == 1
    assert held["redistribution_partition"] == "excluded"
    assert not any(held[key] for key in ("sample", "reaction_text", "target_summaries", "depiction"))


def test_verifier_rejects_an_injected_excluded_member(partitioned_data):
    root, _ = partitioned_data
    manifest = build_manifest(root, {}, include_code_manifest=False)
    import hashlib

    path = root / "yield/pending.csv.gz"
    manifest["dataset"]["files"].append({
        "key": "yield/pending.csv.gz", "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
    result = verify_with_root(manifest, root)
    assert result["policy_errors"]
    assert not verification_succeeded(result)


def test_unregistered_new_dataset_is_not_hidden(partitioned_data):
    root, _ = partitioned_data
    manifest = build_manifest(root, {}, include_code_manifest=False)
    (root / "selectivity").mkdir()
    (root / "selectivity/unregistered.csv").write_text("id\n1\n")
    result = verify_with_root(manifest, root)
    assert "selectivity/unregistered.csv" in result["unexpected"]


def test_partition_and_alias_inconsistency_is_rejected(partitioned_data):
    _, records = partitioned_data
    bad = copy.deepcopy(records)
    bad["yield/main.csv.gz"]["license_spdx"] = "CC-BY-SA-4.0"
    with pytest.raises(ValueError, match="license/partition"):
        release_keys(bad)
    bad = copy.deepcopy(records)
    bad["condition/mirror.csv.gz"]["alias_of"] = "yield/pending.csv.gz"
    with pytest.raises(ValueError, match="crosses"):
        release_keys(bad)


def test_canonical_catalog_has_separate_views():
    catalog = DatasetCatalog()
    main = catalog.for_release("main").list()
    sa = catalog.for_release("sharealike").list()
    assert len(main) == 46
    assert len(sa) == 2
    assert all(d.redistribution_partition == "main" for d in main)
    assert all(d.redistribution_partition == "sharealike" for d in sa)
    assert catalog.get("condition", "uspto_500").alias_of == "synthesis/uspto_500.csv.gz"
    assert not any(
        d.name in {"suzuki_miyaura", "da"} or d.alias_of for d in main
    )
    assert catalog.get("synthesis", "da").redistribution_partition == "excluded"
    assert catalog.for_release("main").get("condition", "uspto_500").task.value == "synthesis"
    assert catalog.get("condition_sharealike", "openexp").has_split is False
    assert catalog.get("condition", "orderly_condition").has_split is True


def test_proposed_memberships_validate_without_claiming_published_splits(tmp_path):
    root = tmp_path / "Data"
    path = root / "condition_sharealike/sample.csv.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(gzip.compress(b"sample_id,synrxn_split\n1,train\n2,test\n", mtime=0))
    metadata = {"schema_version": "1.0", "column_definitions": {
        "sample_id": {}, "synrxn_split": {}}, "datasets": {
        "condition_sharealike/sample.csv.gz": {
            "title": "Sample", "description": "Source split test", "benchmark_role": "Workflow probe",
            "license": "CC BY-SA 4.0", "license_spdx": "CC-BY-SA-4.0",
            "redistribution_partition": "sharealike", "citations": ["fixture"],
            "split_values": ["train", "test"], "split_column": "synrxn_split", "split_status": "proposed",
            "row_identifier": "sample_id", "row_identifier_unique": True}}}
    registry = root / "metadata.yaml"
    registry.write_text(yaml.safe_dump(metadata))
    manifest = build_manifest(root, {"version": "test"}, partition="sharealike", include_code_manifest=False)
    manifest_path = tmp_path / "manifest.json"
    import json

    manifest_path.write_text(json.dumps(manifest))
    result = validate_catalog(root, registry, manifest_path)
    assert result["ok"], result
    assert result["rows_checked"] == 2


def test_parquet_export_cannot_convert_excluded_dataset(tmp_path):
    item = DatasetCatalog().get("yield", "suzuki_miyaura")
    with pytest.raises(ValueError, match="excluded"):
        convert_dataset(Path("does-not-exist"), tmp_path / "x.parquet", item)


def test_parquet_export_selects_only_its_partition(tmp_path, monkeypatch):
    from synrxn.access.query import ParquetArtifact

    full = DatasetCatalog()
    seen = []

    def fake_convert(source, output, dataset):
        seen.append(dataset)
        return ParquetArtifact(f"{dataset.task.value}/{dataset.name}", output,
                               0, "0" * 64, "1" * 64, ())

    monkeypatch.setattr("synrxn.access.query.convert_dataset", fake_convert)
    convert_release(Path("Data"), tmp_path / "main", full)
    assert len(seen) == 46
    assert all(d.redistribution_partition == "main" and not d.alias_of for d in seen)
    seen.clear()
    convert_release(Path("Data"), tmp_path / "sa", full, partition="sharealike")
    assert len(seen) == 2
    assert all(d.is_sharealike for d in seen)


def test_parquet_export_refuses_mixed_existing_output(tmp_path):
    stray = tmp_path / "condition_sharealike/old.parquet"
    stray.parent.mkdir()
    stray.write_bytes(b"old derivative")
    with pytest.raises(ValueError, match="non-member"):
        convert_release(Path("Data"), tmp_path)
