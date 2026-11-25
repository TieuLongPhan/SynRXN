"""Release gates: evidence pins, upstream restrictions and task-field isolation."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from synrxn.reconstruction.coverage import rights, verify_release

from synrxn.reconstruction.paths import REPO, physical_path, source_path


class ReleaseGateFixtures(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.grant = self.repo / "grant.json"
        self.grant.write_text(json.dumps({"id": 42, "metadata": {"license": {"id": "mit-license"}}}))
        self.source = {"id": "fixture", "license": "MIT", "redistribution_allowed": True,
                       "rights": {"grant_url": "https://zenodo.org/api/records/42", "grant_snapshot": {
                           "path": "grant.json", "sha256": hashlib.sha256(self.grant.read_bytes()).hexdigest()}}}
        self.catalog = {"sources": [self.source]}
        self.table = {"id": "qualified", "source_id": "fixture", "license": "MIT", "release_eligible": True}

    def test_valid_pinned_source_grant_admitted(self):
        self.assertTrue(rights.source_has_evidence(self.source, self.catalog, self.repo))
        self.assertTrue(verify_release.release_decision(self.table, self.source, self.catalog, self.repo)["release_eligible"])

    def test_missing_checksum_cannot_clear_a_declared_license(self):
        source = copy.deepcopy(self.source)
        del source["rights"]["grant_snapshot"]["sha256"]
        self.assertFalse(rights.source_has_evidence(source, self.catalog, self.repo))
        self.assertFalse(verify_release.release_decision(self.table, source, self.catalog, self.repo)["release_eligible"])

    def test_changed_grant_bytes_fail_closed(self):
        self.grant.write_text(json.dumps({"id": 42, "metadata": {"license": {"id": "cc-by-nc-4.0"}}}))
        self.assertFalse(rights.source_has_evidence(self.source, self.catalog, self.repo))

    def test_primary_record_identity_must_match_cited_grant(self):
        source = copy.deepcopy(self.source)
        source["rights"]["grant_url"] = "https://zenodo.org/records/999"
        self.assertFalse(rights.source_has_evidence(source, self.catalog, self.repo))

    def test_crossref_vor_grant_is_limited_to_same_publication_bundled_data(self):
        doi = "10.1039/example"
        self.grant.write_text(json.dumps({"message-type": "work", "message": {
            "DOI": doi, "license": [{"content-version": "vor", "delay-in-days": 0,
                                      "URL": "http://creativecommons.org/licenses/by/3.0/"}]}}))
        source = {"license": "CC-BY-3.0", "rights": {
            "scope": "publisher_bundled_supplementary_data", "publication_doi": doi,
            "grant_url": "https://api.crossref.org/works/" + doi,
            "grant_snapshot": {"path": "grant.json", "sha256": hashlib.sha256(self.grant.read_bytes()).hexdigest()}}}
        self.assertTrue(rights.source_has_evidence(source, {}, self.repo))
        for field, value in [("scope", "external_linked_repository"), ("publication_doi", "10.1039/other")]:
            changed = copy.deepcopy(source)
            changed["rights"][field] = value
            self.assertFalse(rights.source_has_evidence(changed, {}, self.repo))
        changed = copy.deepcopy(source)
        changed["rights"]["grant_url"] = "https://api.crossref.org/works/10.1039/other"
        changed["rights"]["publication_doi"] = "10.1039/other"
        self.assertFalse(rights.source_has_evidence(changed, {}, self.repo))
        self.assertFalse(rights._grant_matches(self.grant.read_bytes(), source["rights"]["grant_url"], "CC-BY-4.0"))

    def test_a_global_bio_report_does_not_grant_an_unrelated_source(self):
        (self.repo / "global-rights.json").write_text("{}")
        unrelated = {"id": "unrelated", "license": "MIT"}
        catalog = {"rights_audit_path": "global-rights.json", "source_lock_sha256": "f" * 64}
        self.assertFalse(rights.source_has_evidence(unrelated, catalog, self.repo))

    def test_software_only_grant_does_not_clear_data(self):
        source = copy.deepcopy(self.source)
        source["rights"]["scope"] = "software_only"
        self.assertFalse(rights.source_has_evidence(source, self.catalog, self.repo))

    def test_restricted_legal_heading_cannot_be_cleared_by_a_ccby_link(self):
        text = b"Attribution-NonCommercial 4.0 International. Related license: https://creativecommons.org/licenses/by/4.0/"
        self.assertFalse(rights._grant_matches(text, "https://raw.githubusercontent.com/owner/data/" + "a" * 40 + "/LICENSE", "CC-BY-4.0"))

    def test_explicit_restrictions_and_quarantine_override_other_evidence(self):
        for field, value in [("release_eligible", False), ("license", "CC-BY-NC-4.0"), ("license", "MIT; academic use only")]:
            table = {**self.table, field: value}
            self.assertFalse(verify_release.release_decision(table, self.source, self.catalog, self.repo)["release_eligible"])
        self.assertFalse(verify_release.release_decision(self.table, {**self.source, "redistribution_allowed": False}, self.catalog, self.repo)["release_eligible"])

    def test_sharealike_stays_in_its_own_partition(self):
        self.grant.write_text(json.dumps({"id": 42, "metadata": {"license": {"id": "cc-by-sa-4.0"}}}))
        source = copy.deepcopy(self.source)
        source["license"] = "CC-BY-SA-4.0"
        source["rights"]["grant_snapshot"]["sha256"] = hashlib.sha256(self.grant.read_bytes()).hexdigest()
        decision = verify_release.release_decision({**self.table, "license": "CC-BY-SA-4.0"}, source, self.catalog, self.repo)
        self.assertTrue(decision["release_eligible"])
        self.assertEqual(decision["partition"], "sharealike")

    def test_raw_carrier_false_is_preserved_in_release_manifest(self):
        metadata = self.repo / "kdd/coverage/legacy_rights.json"
        metadata.parent.mkdir(parents=True)
        metadata.write_text('{"tables": []}')
        table_path = self.repo / "qualified.csv"
        table_path.write_text("source_record_id,input,target,group\n1,C,1,A\n")
        carrier = self.repo / "mixed.csv"
        carrier.write_text("restricted mixed source carrier\n")
        source = {**self.source, "assets": [{"path": "mixed.csv", "sha256": hashlib.sha256(carrier.read_bytes()).hexdigest(),
                                             "redistribution_allowed": False, "license": "MIT"}]}
        table = {**self.table, "path": "qualified.csv", "sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
                 "rows": 1, "source_url": "https://zenodo.org/records/42", "task_ids": ["prediction"],
                 "counting_unit": "native record", "reference_evidence": "native observation", "qualification": "fixture"}
        catalog = {"sources": [source], "tables": [table]}
        catalog_path = self.repo / "Data/coverage_expansion/fixture/catalog.json"
        catalog_path.parent.mkdir(parents=True)
        catalog_path.write_text(json.dumps(catalog))
        with patch.object(verify_release, "dataset_inventory", return_value=[]):
            result = verify_release.build(self.repo, check_rows=False, check_assets=False)
        self.assertEqual(len(result["release_tables"]), 1)
        self.assertFalse(result["release_native_assets"])


class ContractGates(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.table = {"path": "data.csv", "rows": 2, "task_ids": ["prediction"], "task_fields": {
            "prediction": {"inputs": ["input"], "targets": ["target"], "group_fields": ["native_group"]}}}

    def test_missing_target_in_one_row_cannot_hide_behind_column_union(self):
        (self.repo / "data.csv").write_text("source_record_id,input,target,native_group\n1,C,0,A\n2,N,,B\n")
        self.assertEqual(verify_release.verify_contract(self.table, self.repo)["status"], "failed")

    def test_missing_native_group_rejected(self):
        (self.repo / "data.csv").write_text("source_record_id,input,target,native_group\n1,C,0,A\n2,N,1,\n")
        self.assertEqual(verify_release.verify_contract(self.table, self.repo)["status"], "failed")

    def test_logical_target_must_not_be_an_input(self):
        row = {"sample_id": "1", "inputs_json": '{"yield": 42}', "targets_json": '{"yield":42}',
               "native_group": "A", "provenance_json": '{"source_locator":"native:1","source_sha256":"' + "a" * 64 + '"}'}
        table = {"path": "data.csv", "rows": 1, "task_ids": ["prediction"], "storage_layout": "json_objects",
                 "task_fields": {"prediction": {"inputs": ["inputs_json"], "targets": ["targets_json"],
                    "input_keys": ["yield"], "target_keys": ["yield"], "group_fields": ["native_group"]}}}
        with patch.object(verify_release, "rows_of", return_value=iter([row])):
            result = verify_release.verify_contract(table, self.repo)
        self.assertEqual(result["status"], "failed")

    def test_recorded_zero_is_a_valid_target(self):
        (self.repo / "data.csv").write_text("source_record_id,input,target,native_group\n1,C,0,A\n2,N,1,B\n")
        self.assertEqual(verify_release.verify_contract(self.table, self.repo)["status"], "verified")


class ActualCatalogGates(unittest.TestCase):
    def test_declared_only_legacy_tables_are_held(self):
        legacy = json.loads(physical_path("kdd/coverage/legacy_rights.json", repo=REPO).read_text())
        declared = [table for table in legacy["tables"] if table["status"] == "declared_only"]
        # Counts can decrease when new primary grants are verified; declaring a
        # license alone must never change any remaining table's release status.
        self.assertTrue(all(table["release_eligible"] is False for table in declared))
        self.assertTrue(all(table["remaining_qualification"] or table["reason"] for table in declared))

    def test_actual_primary_grants_and_composite_sources_resolve(self):
        for branch, _, catalog in verify_release.branches(REPO):
            sources = {source["id"]: source for source in catalog["sources"]}
            for table in catalog["tables"]:
                if table.get("release_eligible") is False:
                    continue
                self.assertIn(table["source_id"], sources, branch + ":" + table["id"])
                for source_id in table.get("source_ids", [table["source_id"]]):
                    self.assertIn(source_id, sources)
                self.assertTrue(rights.table_has_evidence(table, sources[table["source_id"]], catalog, REPO), branch + ":" + table["id"])

    def test_rate_target_derivatives_and_other_metadata_are_withheld(self):
        catalog = json.loads((source_path(REPO / "Data/coverage_expansion/physics_rates/catalog.json")).read_text())
        for table in catalog["tables"]:
            for task_id, contract in table["task_fields"].items():
                if task_id == "reaction_rates":
                    self.assertFalse({"A_low", "Ea_low", "A_high", "Ea_high", "barrier", "k_298", "log10_k_298"} & set(contract["inputs"]))
        bio = json.loads((source_path(REPO / "Data/coverage_expansion/bio/catalog.json")).read_text())
        for table in bio["tables"]:
            for contract in table["task_fields"].values():
                self.assertFalse(set(contract["inputs"]) & {"target_evidence", "source_record_id", "native_record_id", "source_sha256"})

    def test_bio_mixed_raw_assets_stay_outside_redistribution(self):
        lock = json.loads(physical_path("kdd/coverage/adapters/bio_sources.json", repo=REPO).read_text())
        for asset in lock["assets"]:
            if "C15--datasets--" in asset["path"] or asset["path"].endswith("test_dataset_with_similarity_idx.csv"):
                self.assertFalse(asset["redistribution_allowed"])

    def test_bio_chain_cannot_substitute_an_unrelated_or_unpinned_grant(self):
        catalog = json.loads((source_path(REPO / "Data/coverage_expansion/bio/catalog.json")).read_text())
        table = next(t for t in catalog["tables"] if t["id"] == "reactzyme_retrieval")
        source = next(s for s in catalog["sources"] if s["id"] == table["source_id"])
        self.assertTrue(rights.table_has_evidence(table, source, catalog, REPO))
        changed = copy.deepcopy(table)
        changed["license_chain"][-1]["evidence"] = "https://zenodo.org/records/99999999"
        self.assertFalse(rights.table_has_evidence(changed, source, catalog, REPO))
        self.assertFalse(rights.table_has_evidence(table, source, {**catalog, "source_lock_sha256": "0" * 64}, REPO))
        self.assertFalse(rights.table_has_evidence(table, source, {**catalog, "rights_audit_sha256": "0" * 64}, REPO))


if __name__ == "__main__":
    unittest.main()
