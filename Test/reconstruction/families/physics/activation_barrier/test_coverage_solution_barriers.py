"""Validate the physical target, native cell lineage and solvent qualification."""
import csv
import json
import unittest

import openpyxl

from synrxn.reconstruction.coverage.rights import source_has_evidence

from synrxn.reconstruction.families.physics.activation_barrier import _solution_barriers as module


class SolutionBarrierQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((module.OUT / "catalog.json").read_text())
        cls.table = cls.catalog["tables"][0]
        with (module.ROOT / cls.table["path"]).open() as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.workbook = openpyxl.load_workbook(module.RAW / "QM_dataset.xlsx", data_only=True, read_only=True)

    def test_actual_native_data_grant_and_complete_source_workbook(self):
        source = self.catalog["sources"][0]
        self.assertTrue(source_has_evidence(source, self.catalog, module.ROOT))
        self.assertEqual(source["data_license_spdx"], "CC-BY-4.0")
        self.assertEqual(self.workbook["pred vs exp"].max_row - 2, 970)
        self.assertEqual(module.digest(module.ROOT / self.table["path"]), self.table["sha256"])
        self.assertEqual(len(self.rows), 4514)
        report = json.loads((module.OUT / "qualification.json").read_text())
        self.assertEqual(report["paper_reported_computed_values"], 4513)
        self.assertFalse(report["count_discrepancy_resolved"])

    def test_absolute_gibbs_targets_are_not_relative_barriers_rates_or_gas_energies(self):
        native_rows = list(self.workbook["pred vs exp"].values)
        seen_ids = set()
        for row in self.rows:
            column = openpyxl.utils.column_index_from_string(row["source_excel_column"]) - 1
            original = native_rows[int(row["source_excel_row"]) - 1]
            self.assertGreaterEqual(column, 8)
            self.assertLessEqual(column, 15)
            self.assertEqual(float(row["activation_gibbs_kcal_per_mol"]), original[column])
            self.assertEqual(row["solvent_name"], original[4])
            self.assertEqual(float(row["temperature_K"]), 298.15)
            self.assertEqual(row["phase"], "solution")
            self.assertEqual(row["target_quantity"], "Delta_G_dagger")
            self.assertEqual(row["reference_kind"], "computed_native_cached_solution_activation_gibbs")
            self.assertNotIn(row["source_record_id"], seen_ids)
            seen_ids.add(row["source_record_id"])

    def test_hypothetical_solvents_and_conflicting_native_contexts_remain_flagged(self):
        hypothetical = [row for row in self.rows if row["hypothetical_solvent"] == "True"]
        self.assertEqual(len(hypothetical), 54)
        self.assertTrue(all(row["solvent_identity_qualified"] == "False" for row in hypothetical))
        duplicates = [row for row in self.rows if row["duplicate_context"] == "True"]
        self.assertEqual({int(row["source_entry"]) for row in duplicates}, {408, 531})
        self.assertEqual(len({row["reaction_group"] for row in duplicates}), 1)
        self.assertEqual({row["solvent_name"] for row in duplicates}, {"Diiodomethane"})
        self.assertEqual(len({row["reaction_group"] for row in self.rows}), 30)
        self.assertTrue(all("smiles" not in column.lower() for column in self.table["columns"]))

    def test_inputs_exclude_physical_targets_and_source_identifiers(self):
        fields = self.table["task_fields"]["activation_barrier"]
        self.assertEqual(fields["targets"], ["activation_gibbs_kcal_per_mol"])
        self.assertTrue(set(fields["inputs"]).isdisjoint({"activation_gibbs_kcal_per_mol", "source_entry", "source_record_id", "source_excel_row", "reference_state_correction_kcal_per_mol"}))
        self.assertIn("method", fields["inputs"])
        self.assertIn("solvent_name", fields["inputs"])
        self.assertIn("reaction_group", fields["group_fields"])


if __name__ == "__main__":
    unittest.main()
