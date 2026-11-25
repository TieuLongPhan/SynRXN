"""Check that impurity references are real source annotations, never proxies."""
import gzip
import json
import unittest

import pandas as pd

from synrxn.reconstruction.families.selectivity.impurity_prediction import _source_tables as impurity


def read(table):
    with gzip.open(impurity.ROOT / table["path"], "rt") as file:
        return [json.loads(line) for line in file]


class ImpurityCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((impurity.OUT / "catalog.json").read_text())
        cls.tables = {t["id"]: t for t in cls.catalog["tables"]}
        cls.phase_rows = read(cls.tables["solid_state_reported_impurity_phases"])

    def test_native_organic_side_product_quantities_are_preserved(self):
        table = self.tables["pba_benzaldehyde_side_product"]
        original = pd.read_csv(impurity.RAW / "extracted" / impurity.PBA_MEMBER).set_index("Index")
        rows = read(table)
        self.assertEqual(len(rows), 756)
        self.assertEqual(len({r["native_index"] for r in rows}), 756)
        for row in rows:
            expected = float(original.loc[row["native_index"], "Yield_of_benzaldehyde (%)"])
            self.assertEqual(row["side_product_yield_percent"]["O=Cc1ccccc1"], expected)
            self.assertEqual(row["source_product_codes"], {"reactant": "20a", "intended_product": "20c", "side_product": "20b"})
        inputs = table["task_fields"]["impurity_prediction"]["inputs"]
        self.assertNotIn("Yield_of_styrene_oxide (%)", inputs)
        self.assertNotIn("Selectivity_of_styrene_oxide (%)", inputs)
        self.assertNotIn("side_product_yield_percent", inputs)

    def test_phase_identity_qualification_and_no_false_negative_cohort(self):
        audit = json.loads((impurity.OUT / "solid_state_qualification.json").read_text())
        self.assertEqual(audit["total_native_records"], 80806)
        self.assertEqual(audit["native_reported_impurity_phase_records"], 18869)
        self.assertEqual(audit["no_reported_impurity_phase_excluded_not_negative"], 61937)
        self.assertEqual(len(self.phase_rows), audit["eligible_records"])
        self.assertEqual(len(self.phase_rows), 7261)
        self.assertEqual(audit["eligible_unique_original_dois"], 6300)
        for row in self.phase_rows:
            self.assertTrue(row["reported_impurity_formulas"])
            self.assertFalse(set(row["reported_impurity_formulas"]) & set(row["intended_target_formulas"]))
            self.assertNotIn("unknown", row["reported_impurity_formulas"])
            self.assertTrue(row["source_publication_doi"].startswith("10."))
            self.assertIn("literature_reported_experimental", row["reference_kind"])

    def test_phase_references_match_native_explicit_annotations(self):
        with gzip.open(impurity.RAW / "SS_rxns_80806.json.gz", "rt") as file:
            original = json.load(file)
        for row in self.phase_rows:
            native = original[row["native_index"]]
            observed = sorted(set(filter(None, (impurity.concrete_formula(v) for v in native["impurity_phase"]))))
            self.assertEqual(observed, row["reported_impurity_formulas"])
            self.assertEqual(native["DOI"], row["source_publication_doi"])
            self.assertTrue(native["target_reaction"])

    def test_conditions_have_no_result_text_or_computed_phase_ids(self):
        for row in self.phase_rows:
            self.assertNotIn("mp_id", row)
            self.assertNotIn("mp_eabovehull", row)
            for context in row["synthesis_condition_context"]:
                self.assertEqual(set(context), {"act_type", "temperature", "duration"})
                for measurement in context["temperature"] + context["duration"]:
                    self.assertTrue(set(measurement) <= {"min", "max", "values", "units"})
        contract = self.tables["solid_state_reported_impurity_phases"]["task_fields"]["impurity_prediction"]
        self.assertFalse(set(contract["inputs"]) & set(contract["targets"]))

    def test_licences_and_hash_provenance(self):
        licences = {source["id"]: source["license"] for source in self.catalog["sources"]}
        self.assertEqual(licences, {"ROBOWSKI_IMPURITY2025": "MIT", "SOLIDSTATE_IMPURITY2025": "CC-BY-4.0"})
        for source in self.catalog["sources"]:
            self.assertEqual(impurity.sha(impurity.ROOT / source["license_evidence"]), source["license_evidence_sha256"])
        for table in self.tables.values():
            self.assertEqual(impurity.sha(impurity.ROOT / table["path"]), table["sha256"])
            for asset in table["source_assets"]:
                self.assertEqual(impurity.sha(impurity.ROOT / asset["path"]), asset["sha256"])
                self.assertEqual(asset["license"], table["license"])
                self.assertTrue(asset["url"].startswith("https://"))
                self.assertTrue(asset["redistribution_allowed"])
        notice = impurity.ROOT / self.catalog["attribution_path"]
        self.assertEqual(impurity.sha(notice), self.catalog["attribution_sha256"])
        self.assertIn("Yaroslav I. Sobolev", notice.read_text())
        self.assertIn("Sanghoon Lee", notice.read_text())


if __name__ == "__main__":
    unittest.main()
