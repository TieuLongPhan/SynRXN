"""Public readers retain logical identities after physical storage relocation."""
import gzip
import json

import pytest
import yaml

from synrxn.data.build_catalog import build_catalog
from synrxn.data.build_manifest import build_manifest
from synrxn.data import DataLoader, LocalSource
from synrxn.reconstruction import paths
from synrxn.data.validate import validate_catalog
from synrxn.data.verify_manifest import verify_with_root, verification_succeeded


def test_relocated_tables_keep_release_keys_and_do_not_duplicate_family_links(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPO", tmp_path)
    data = tmp_path / "Data"
    stored = data / "tables/canonical/property/tiny.csv.gz"
    stored.parent.mkdir(parents=True)
    stored.write_bytes(gzip.compress(b"r_id,rxn,ea\n1,C>>CO,1.5\n2,CO>>C,2.5\n", mtime=0))
    family = data / "families/physics/barriers/tiny/data.csv.gz"
    family.parent.mkdir(parents=True)
    family.symlink_to(stored)
    historical_bundle = data / "sources/previous/mcsa/native-source.tar.gz"
    historical_bundle.parent.mkdir(parents=True)
    historical_bundle.write_bytes(b"historical native input; not a canonical release member")
    key = "property/tiny.csv.gz"
    metadata = {
        "schema_version": "1.0",
        "column_definitions": {name: {} for name in ("r_id", "rxn", "ea")},
        "datasets": {key: {
            "title": "Tiny", "description": "Fixture", "benchmark_role": "Storage migration",
            "license": "CC BY 4.0", "license_spdx": "CC-BY-4.0",
            "redistribution_partition": "main", "citations": ["fixture"],
            "targets": ["ea"], "row_identifier": "r_id", "row_identifier_unique": True,
        }},
    }
    metadata_path = data / "metadata.yaml"
    metadata_path.write_text(yaml.safe_dump(metadata))
    (data / "README.md").write_text("Repository storage organization guide")
    source = LocalSource(data)
    assert source.available_names("property") == ["tiny"]
    assert source.resolve("property", "tiny") == stored
    loaded = DataLoader("property", source="local", data_dir=data, cache_dir=None).load("tiny")
    assert len(loaded) == 2
    manifest = build_manifest(data, {"version": "fixture"}, include_code_manifest=False)
    keys = [row["key"] for row in manifest["dataset"]["files"]]
    assert keys.count(key) == 1
    assert "README.md" not in keys
    assert not any(name.startswith(("tables/", "families/", "sources/previous/")) for name in keys)
    assert verification_succeeded(verify_with_root(manifest, data))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assert validate_catalog(data, metadata_path, manifest_path)["ok"]
    catalog = build_catalog(data, metadata_path, manifest_path, tmp_path / "catalog.json")
    assert catalog["datasets"][0]["id"] == "property/tiny"
    assert catalog["datasets"][0]["rows"] == 2
    extra = stored.with_name("unregistered.csv")
    extra.write_text("r_id\n3\n")
    assert verify_with_root(manifest, data)["unexpected"] == ["property/unregistered.csv"]


def test_checkout_local_loader_keeps_existing_arguments():
    source = LocalSource(paths.REPO / "Data")
    assert source.resolve("aam", "ecoli") == (
        paths.REPO / "Data/families/representation/atom_mapping/canonical/aam/ecoli/data.csv.gz"
    )
    assert len(DataLoader("aam", source="local", data_dir=paths.REPO / "Data", cache_dir=None).load("ecoli", nrows=2)) == 2


@pytest.mark.parametrize("family,payload,target_col,expected_targets", [
    ("classification", b"r_id,label\n1,0\n2,1\n", None, [0, 1]),
    ("property", b"r_id,energy\n1,-1.25\n2,2.75\n", "energy", [-1.25, 2.75]),
])
def test_saved_baseline_feature_reads_follow_archived_storage_without_fitting(
        tmp_path, monkeypatch, family, payload, target_col, expected_targets):
    import numpy as np
    from synrxn.baseline import classification, property as property_baseline

    monkeypatch.setattr(paths, "REPO", tmp_path)
    monkeypatch.chdir(tmp_path)
    expected = np.array([[1, 2], [3, 4]])
    for feature in ("drfp", "rxnfp"):
        output = tmp_path / f"archive/benchmarks/features/{feature}/{family}/{feature}_tiny.npz"
        output.parent.mkdir(parents=True)
        np.savez(output, fps=expected)
    source = tmp_path / "Data/tables/canonical" / family / "tiny.csv.gz"
    source.parent.mkdir(parents=True)
    source.write_bytes(gzip.compress(payload, mtime=0))
    if family == "classification":
        drfp, rxnfp, targets = classification.get_data("tiny")
    else:
        drfp, rxnfp, targets = property_baseline.get_data("tiny", target_col=target_col)
    assert np.array_equal(drfp, expected)
    assert np.array_equal(rxnfp, expected)
    assert targets.tolist() == expected_targets
    assert not (tmp_path / "Data/Benchmark").exists()
