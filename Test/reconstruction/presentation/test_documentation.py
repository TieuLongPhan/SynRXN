"""Public examples must follow the current table-level redistribution review."""
import copy
import json
from types import SimpleNamespace

import pytest

from synrxn.reconstruction.paths import REPO
from synrxn.reconstruction.presentation.documentation import (
    _write_exports, canonical_catalog, catalog_citations,
)


def test_current_terms_replace_historical_terms_without_publishing_held_examples():
    catalogue = {
        "release": {"version": "1.5.0"},
        "datasets": [
            {"id": "aam/example", "license": "CC BY 4.0", "sample": [{"rxn": "CC>>C=C"}]},
            {"id": "synthesis/held", "license": "CC BY 3.0",
             "sample": [{"rxn": "restricted example"}], "depiction": "_static/held.svg",
             "reaction_text": "restricted example", "target_summaries": {"product": "source value"}},
        ],
    }
    original = copy.deepcopy(catalogue)
    release = {
        "release_tables": [{"namespace": "canonical", "key": "aam/example",
                            "license_expression": "MIT", "partition": "main",
                            "rights_status": "verified_redistributable"}],
        "held_tables": [{"namespace": "canonical", "key": "synthesis/held",
                         "license_expression": None, "partition": "main",
                         "rights_status": "upstream_database_terms_unverified"}],
    }
    admitted, held = canonical_catalog(catalogue, release)["datasets"]
    assert admitted["license"] == "MIT"
    assert admitted["release_eligible"] is True
    assert admitted["sample"] == original["datasets"][0]["sample"]
    assert held["release_eligible"] is False
    assert held["license"] == "Unresolved"
    assert not any(held[field] for field in ("sample", "depiction", "reaction_text", "target_summaries"))
    assert catalogue == original


def test_unreviewed_table_cannot_inherit_an_old_public_license():
    catalogue = {"release": {}, "datasets": [{"id": "property/unreviewed", "license": "CC BY 4.0"}]}
    with pytest.raises(KeyError):
        canonical_catalog(catalogue, {"release_tables": [], "held_tables": []})


def test_browser_citations_are_formatted_from_the_source_bibliography(tmp_path):
    bibliography = tmp_path / "refs.bib"
    bibliography.write_text(
        '@article{source2026, author={Doe, Jane}, title={{Reaction} data}, '
        'journal={Example Journal}, year={2026}, doi={10.0000/example}}', encoding="utf-8")
    citation = catalog_citations(bibliography)["source2026"]
    assert "Jane Doe" in citation["text"]
    assert "Reaction data" in citation["text"]
    assert "{" not in citation["text"]
    assert citation["doi"] == "10.0000/example"


def test_site_exports_match_main_database_scope(tmp_path):
    app = SimpleNamespace(srcdir=REPO / "doc", outdir=tmp_path,
                          builder=SimpleNamespace(format="html"))
    _write_exports(app, exception=None)
    static = tmp_path / "_static"
    report = json.loads((static / "families.json").read_text())
    summary = report["summary"]
    assert report["release_scope"] == "main"
    assert (summary["families"], summary["tasks"], summary["admitted_table_views"]) == (12, 40, 102)
    assert summary["metadata_only_table_views"] == 1
    table_views = [table for family in report["families"] for task in family["tasks"]
                   for table in task["tables"]]
    assert {table["key"] for table in table_views if not table["release_eligible"]} == {"synthesis/da"}
    assert not any(table["key"].startswith("feasibility/") for table in table_views)
    outcomes = next(family for family in report["families"] if family["id"] == "outcomes")
    assert {task["id"] for task in outcomes["tasks"]} == {"yield_conversion", "reaction_success"}
    assert outcomes["primary_table_count"] == 9
    for filename, admitted in (("catalog-data.json", 44), ("catalog-sharealike-data.json", 2)):
        catalogue = json.loads((static / filename).read_text())
        datasets = catalogue["datasets"]
        assert sum(table["release_eligible"] for table in datasets) == admitted
        assert catalogue["release"]["dataset_count"] == len(datasets)
        assert not any(table["id"].startswith("feasibility/") for table in datasets)
