"""Evidence-backed legacy rights grading, including actual NC conflict."""
import hashlib
import json
import unittest

from synrxn.reconstruction.paths import REPO, physical_path


class LegacyRightsAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads(physical_path("kdd/coverage/legacy_rights.json", repo=REPO).read_text())
        cls.evidence = {r["id"]: r for r in cls.audit["evidence"]}

    def test_every_original_table_has_a_separate_rights_decision(self):
        rows = self.audit["tables"]
        self.assertEqual(len(rows), 65)
        self.assertEqual(sum(r["namespace"] == "canonical" for r in rows), 49)
        self.assertEqual(sum(r["namespace"] == "expansion" for r in rows), 16)
        self.assertEqual(len({(r["namespace"], r["key"]) for r in rows}), 65)
        self.assertTrue(all(r["evidence_ids"] and r["reason"] and r["data_sha256"] for r in rows))
        for row in rows:
            if row["status"] in {"declared_only", "license_conflict", 'original_grant_unverified', 'upstream_database_terms_unverified'}:
                self.assertFalse(row["release_eligible"])
            else:
                self.assertTrue(row["status"].startswith("verified_"))

    def test_snapshots_are_exact_and_article_data_restrictions_are_not_laundered(self):
        for evidence in self.evidence.values():
            if evidence["status"] == "captured":
                self.assertEqual(hashlib.sha256(physical_path(evidence["path"], repo=REPO).read_bytes()).hexdigest(), evidence["sha256"])
        native = json.loads(physical_path(self.evidence["autotst_kinetic_si_grant"]["path"], repo=REPO).read_text())
        self.assertEqual(native["license"]["name"], "CC BY-NC 4.0")
        lograte = next(r for r in self.audit["tables"] if r["key"] == "property/lograte")
        self.assertEqual(lograte["status"], "verified_noncommercial_grant")
        self.assertTrue(lograte["release_eligible"])
        self.assertFalse(lograte['commercial_use_allowed'])
        self.assertEqual(lograte['release_license_expression'], 'CC-BY-NC-4.0')
        self.assertEqual(lograte['partition'], 'noncommercial')
        for name in ["pmechdb_paper_data_terms", "rmechdb_paper_data_terms"]:
            body = physical_path(self.evidence[name]["path"], repo=REPO).read_text()
            self.assertIn("NonCommercial-NoDerivs", body)

    def test_open_base_curated_grants_match_both_frozen_bytes_and_exact_product_notice(self):
        notice_hash = self.evidence["synrxn_product_notice"]["sha256"]
        for row in self.audit["tables"]:
            if row["status"] != "verified_curated_product_grant_with_open_base":
                continue
            origin = row["published_product_evidence"]
            self.assertTrue(origin["bytes_identical_to_published_release"])
            self.assertEqual(origin["member_sha256"], row["data_sha256"])
            self.assertEqual(origin["data_license_sha256"], notice_hash)
        lowe = json.loads(physical_path(self.evidence["lowe_original_data_grant"]["path"], repo=REPO).read_text())
        self.assertEqual(lowe["license"]["name"], "CC0")

    def test_pnas_original_table_permission_is_scoped_and_cannot_be_relabelled_cc(self):
        from synrxn.reconstruction.coverage.rights import _grant_matches
        from synrxn.reconstruction.coverage.license_policy import PNAS_EDUCATIONAL
        row = next(r for r in self.audit['tables'] if r['key'] == 'property/phosphatase')
        self.assertTrue(row['release_eligible'])
        self.assertEqual(row['release_license_expression'], 'CC-BY-4.0 AND ' + PNAS_EDUCATIONAL)
        self.assertEqual(row['partition'], 'noncommercial_educational')
        self.assertFalse(row['commercial_use_allowed'])
        grants = [self.evidence[name] for name in row['grant_source_evidence_ids']]
        self.assertTrue(any(_grant_matches(physical_path(e['path'], repo=REPO).read_bytes(), e['url'], PNAS_EDUCATIONAL) for e in grants))
        self.assertTrue(all(not _grant_matches(physical_path(e['path'], repo=REPO).read_bytes(), e['url'], 'CC-BY-NC-4.0') for e in grants))
        self.assertEqual(row['previous_audit_decision']['status'], 'original_grant_unverified')


if __name__ == "__main__": unittest.main()
