"""Check reproducible AAM exports and the integrity of source-to-target joins."""

import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from synrxn.reconstruction.families.representation.atom_mapping._sources import aam_benchmark as builder
from synrxn.reconstruction.families.representation.atom_mapping._sources import aam_sources as sources
from synrxn.reconstruction.paths import physical_path


@pytest.fixture
def local_source(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    records = [
        {
            "r_id": "0",
            "reactions": "C>>C",
            "ground_truth": "[CH4:1]>>[CH4:1]",
            "rdt": "old prediction",
            "original_id": "native:1",
            "split": "validation",
            "ground_truth_variants": '["[CH4:1]>>[CH4:1]"]',
        },
        {
            "r_id": "1",
            "reactions": "N>>N",
            "ground_truth": "[NH3:2]>>[NH3:2]",
            "rdt": "other prediction",
            "original_id": "native:2",
            "split": "train",
            "ground_truth_variants": '["[NH3:2]>>[NH3:2]"]',
        },
    ]
    product = root / "frozen.csv.gz"
    with gzip.open(product, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lock = root / "source.json"
    lock.write_text(json.dumps({"source_url": "https://example.invalid/frozen", "qualification": "candidate"}))
    entry = {
        "name": "fixture",
        "mode": "published_product",
        "rows": 2,
        "source_product": "frozen.csv.gz",
        "product_sha256": sources.digest(product),
        "source_rows_sha256": builder.source_digest(records),
        "product_columns": list(records[0]),
        "archive_member": "Data/aam/fixture.csv.gz",
        "source_lock": {"path": "source.json", "sha256": sources.digest(lock)},
        "evidence": [],
        "rights": {"license_expression": "LicenseRef-Unknown", "partition": "excluded"},
        "qualification": "Unverified candidate mappings",
    }
    registry = {"published_archive": {"url": "https://example.invalid/archive.zip"}}
    args = SimpleNamespace(output=tmp_path / "output", fetch=False, rebuild_sources=False)
    return root, entry, registry, args, records


def test_build_preserves_source_targets_variants_and_provenance(local_source):
    root, entry, registry, args, records = local_source
    report = builder.build_dataset(entry, registry, args, root)
    output = args.output / "fixture"
    columns, rows = builder.read_rows(output / "aam.csv.gz")
    assert columns == ["r_id", "ground_truth", "rxn", "original_id", "role", "ground_truth_variants"]
    assert [row["role"] for row in rows] == ["val", "train"]
    assert [row["ground_truth"] for row in rows] == [row["ground_truth"] for row in records]
    assert [row["ground_truth_variants"] for row in rows] == [row["ground_truth_variants"] for row in records]
    with gzip.open(output / "provenance.jsonl.gz", "rt") as handle:
        witnesses = [json.loads(line) for line in handle]
    assert [w["source_record"] for w in witnesses] == records
    assert report["redistribution_partition"] == "excluded"
    first = builder.file_manifest(output)
    builder.build_dataset(entry, registry, args, root)
    assert builder.file_manifest(output) == first


def test_source_tampering_leaves_existing_export_intact(local_source):
    root, entry, registry, args, _ = local_source
    builder.build_dataset(entry, registry, args, root)
    output = args.output / "fixture"
    before = builder.file_manifest(output)
    cached = physical_path("Data/raw/aam/fixture/published.csv.gz", root)
    cached.write_bytes(cached.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Source checksum mismatch"):
        builder.build_dataset(entry, registry, args, root)
    assert builder.file_manifest(output) == before
    assert builder.verify_dataset(output, entry)["source_records_verified"]


def test_pinned_center_annotations_extend_rows_without_changing_source_targets(local_source):
    root, entry, registry, args, records = local_source
    annotation = root / 'centers.csv.gz'
    with gzip.open(annotation, 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['r_id', 'center_cluster_id', 'source_lines'])
        writer.writeheader()
        writer.writerows([{'r_id': row['r_id'], 'center_cluster_id': 'center-0', 'source_lines': '[1,2]'}
                          for row in records])
    entry['center_annotations'] = {'path': 'centers.csv.gz', 'sha256': sources.digest(annotation)}
    builder.build_dataset(entry, registry, args, root)
    _, rows = builder.read_rows(args.output / 'fixture' / 'aam.csv.gz')
    assert [r['ground_truth'] for r in rows] == [r['ground_truth'] for r in records]
    assert {r['center_cluster_id'] for r in rows} == {'center-0'}
    assert [r['role'] for r in rows] == ['val', 'train']
    assert builder.verify_dataset(args.output / 'fixture', entry, root)['source_records_verified']
    annotation.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        builder.verify_dataset(args.output / 'fixture', entry, root)


def test_verify_rejects_changed_targets_even_with_recomputed_manifest(local_source):
    root, entry, registry, args, records = local_source
    builder.build_dataset(entry, registry, args, root)
    output = args.output / "fixture"
    records[0]["ground_truth"] = "[C:9]>>[C:9]"
    columns, rows = builder.normalize(entry["product_columns"], records)
    builder.write_tables(output, columns, rows, records)
    marker = output / "manifest.json"
    manifest = json.loads(marker.read_text())
    manifest["files"] = builder.file_manifest(output)
    sources.write_json(marker, manifest)
    with pytest.raises(ValueError, match="Source records differ from pinned selection"):
        builder.verify_dataset(output, entry)


def test_restore_from_pinned_archive_without_frozen_table(local_source, tmp_path):
    root, entry, registry, _, _ = local_source
    upstream = tmp_path / "upstream.zip"
    with zipfile.ZipFile(upstream, "w") as package:
        package.write(root / entry["source_product"], "repo-v1/" + entry["archive_member"])
    (root / entry["source_product"]).unlink()
    registry["published_archive"] = {
        "url": upstream.as_uri(),
        "path": "Data/raw/aam/release.zip",
        "sha256": sources.digest(upstream),
    }
    restored = sources.published_product(entry, registry, fetch=True, root=root)
    assert sources.digest(restored) == entry["product_sha256"]
    assert not list(restored.parent.glob("*.part"))


@pytest.mark.parametrize("kind", ["duplicate", "alias", "unknown_role"])
def test_conflicting_source_fields_are_rejected(kind):
    row = {"r_id": "0", "ground_truth": "[C:1]>>[C:1]", "rxn": "C>>C", "original_id": "native"}
    if kind == "alias":
        row["reactions"] = "N>>N"
    elif kind == "unknown_role":
        row["role"] = "unrecognized"
    records = [row, row] if kind == "duplicate" else [row]
    with pytest.raises(ValueError):
        builder.normalize(list(row), records)


def test_unmanaged_output_is_preserved(local_source):
    root, entry, registry, args, _ = local_source
    output = args.output / "fixture"
    output.mkdir(parents=True)
    (output / "important.txt").write_text("keep")
    with pytest.raises(ValueError, match="unmanaged directory"):
        builder.build_dataset(entry, registry, args, root)
    assert (output / "important.txt").read_text() == "keep"


def test_selected_inventory_has_one_name_per_dataset():
    registry = sources.read_registry()
    names = {entry["name"] for entry in registry["datasets"]}
    assert len(names) == 11 and registry["expected_rows"] == 101480
    assert {"flower", "sihumix", "metambd_cyclic"} <= names
    assert "metamdb" not in names
    assert not names.intersection(registry["retired_datasets"])
    assert all(not Path(entry["source_lock"]["path"]).is_absolute() for entry in registry["datasets"])


def test_moved_native_recipe_verifies_packaged_code_without_archive(tmp_path):
    entry = next(item for item in sources.read_registry()["datasets"] if item["name"] == "flower")
    manifest_path = Path("synrxn/reconstruction/provenance/migration.json")
    checkout_manifest = tmp_path / manifest_path
    checkout_manifest.parent.mkdir(parents=True)
    checkout_manifest.write_bytes((sources.REPO / manifest_path).read_bytes())
    manifest = json.loads(checkout_manifest.read_text())
    paths = {item["historical_path"]: item["module_path"]
             for item in (*manifest["entries"], *manifest["relocated_modules"])}
    for pin in entry["native_recipe"]["implementation"]:
        relative = paths.get(pin["path"], pin["path"])
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((sources.REPO / relative).read_bytes())
    sources.check_native_implementation(entry, tmp_path)

    migrated = tmp_path / paths["scripts/build_flower_aam.py"]
    migrated.write_bytes(migrated.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="Source checksum mismatch"):
        sources.check_native_implementation(entry, tmp_path)
