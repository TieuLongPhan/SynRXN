import gzip
import hashlib
import json
from pathlib import Path

import yaml

from synrxn.build_catalog import build_catalog


def test_catalog_asset_is_bounded_and_derived_from_manifest(tmp_path):
    data_dir = tmp_path / "Data"
    source = data_dir / "property" / "tiny.csv.gz"
    source.parent.mkdir(parents=True)
    with gzip.open(source, "wt", encoding="utf8", newline="") as handle:
        handle.write("r_id,rxn,ea,split\n1,CC>>CO,1.5,train\n2,CO>>C,,test\n")
    metadata = {
        "schema_version": "1.0",
        "column_definitions": {
            "r_id": {"role": "identifier", "description": "Row identifier."},
            "rxn": {"role": "reaction", "description": "Reaction SMILES."},
            "ea": {"role": "target", "description": "Activation energy."},
            "split": {"role": "partition", "description": "Published split."},
        },
        "datasets": {
            "property/tiny.csv.gz": {
                "title": "Tiny property set",
                "description": "Fixture",
                "benchmark_role": "Test catalog generation",
                "license": "CC BY 4.0",
                "citations": ["fixture2026"],
                "targets": ["ea"],
                "split_values": ["train", "test"],
                "row_identifier": "r_id",
                "row_identifier_unique": True,
            }
        },
    }
    metadata_path = data_dir / "metadata.yaml"
    metadata_path.write_text(yaml.safe_dump(metadata), encoding="utf8")
    manifest = {
        "schema_version": "1.0",
        "generated_at": "2026-07-16T00:00:00Z",
        "dataset": {
            "version": "1.0.0",
            "doi": "10.0000/fixture",
            "files": [
                {
                    "key": "property/tiny.csv.gz",
                    "size": source.stat().st_size,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "rows": 2,
                    "columns": ["r_id", "rxn", "ea", "split"],
                }
            ],
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf8")
    output = tmp_path / "catalog.json"

    catalog = build_catalog(
        data_dir, metadata_path, manifest_path, output, reaction_dir=None
    )

    assert output.is_file()
    script_output = output.with_suffix(".js")
    assert script_output.is_file()
    assert script_output.read_text(encoding="utf8").startswith("window.SYNRXN_CATALOG = ")
    assert catalog["release"]["dataset_count"] == 1
    record = catalog["datasets"][0]
    assert record["rows"] == 2
    assert len(record["sample"]) == 2
    assert record["sha256"] == manifest["dataset"]["files"][0]["sha256"]
    assert record["target_summaries"]["ea"]["mean"] == 1.5
    assert record["column_metadata"]["ea"]["logical_type"] == "number"
    assert record["column_metadata"]["ea"]["nullable"] is True


def test_published_catalog_covers_every_dataset():
    catalog_path = Path("doc/_static/catalog-data.json")
    catalog = json.loads(catalog_path.read_text(encoding="utf8"))
    assert len(catalog["datasets"]) == 33, catalog_path
    assert len({item["id"] for item in catalog["datasets"]}) == 33
    for item in catalog["datasets"]:
        assert len(item["sample"]) <= 5
        if item["depiction"]:
            assert (Path("doc") / item["depiction"]).is_file()


def test_catalog_script_waits_for_the_document_body():
    script = Path("doc/_static/catalog.js").read_text(encoding="utf8")
    assert "DOMContentLoaded" in script
    assert "initializeCatalog" in script
    assert "window.SYNRXN_CATALOG" in script
