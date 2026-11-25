"""Native rate quantity, lineage, and grouping checks for the allowed replacement."""
import csv
import json
import math
import unittest

from synrxn.reconstruction.families.physics.reaction_rates import _quantumpioneer as module


class QuantumPioneerQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((module.OUT / "catalog.json").read_text())

    def test_complete_native_release_is_not_a_truncated_prefix(self):
        table = self.catalog["tables"][0]
        source = self.catalog["sources"][0]
        self.assertTrue(source["redistribution_allowed"])
        self.assertEqual(source["data_license_spdx"], "CC-BY-4.0")
        for name, expected in module.ASSETS.items():
            path = module.RAW / name
            self.assertEqual(path.stat().st_size, expected["bytes"])
            self.assertEqual(module.digest(path, "md5"), expected["md5"])
        self.assertEqual(module.digest(module.ROOT / table["path"]), table["sha256"])

    def test_every_rate_target_matches_native_value_and_documented_unit(self):
        table = self.catalog["tables"][0]
        self.assertEqual(table["rows"], 166158)
        self.assertEqual(table["target_units"]["k_298"], "m^3/(mol*s)")
        groups = {}
        with (module.ROOT / table["path"]).open() as prepared, (module.RAW / "quantumpioneer_kinetics_dataset.csv").open() as native:
            outputs, originals = csv.DictReader(prepared), csv.DictReader(native)
            count = 0
            for count, (row, original) in enumerate(zip(outputs, originals, strict=True), 1):
                self.assertEqual(row["reaction_smiles"], original["rxn_smi"])
                self.assertEqual(int(row["source_row_index"]), count - 1)
                self.assertEqual(float(row["k_298"]), float(original["k_298"]))
                self.assertGreater(float(row["k_298"]), 0)
                self.assertAlmostEqual(float(row["log10_k_298"]), math.log10(float(original["k_298"])), places=12)
                self.assertEqual(row["reference_kind"], "computed_transition_state_theory")
                for key in ["barrier", "deltaHrxn", "deltaHrxn298", "deltaGrxn298", "A_low", "Ea_low", "A_high", "Ea_high"]:
                    self.assertEqual(float(row[key]), float(original[key]))
                group = row["reactant_pair_group"]
                groups.setdefault(group, row["proposed_split"])
                self.assertEqual(groups[group], row["proposed_split"])
            self.assertEqual(count, 166158)
        self.assertEqual(len(groups), table["parent_groups"])

    def test_rate_fit_and_thermo_labels_never_become_rate_prediction_inputs(self):
        fields = self.catalog["tables"][0]["task_fields"]
        rate_inputs = set(fields["reaction_rates"]["inputs"])
        self.assertTrue(rate_inputs.isdisjoint({"k_298", "log10_k_298", "A_low", "Ea_low", "A_high", "Ea_high", "barrier", "deltaHrxn", "deltaGrxn298"}))
        self.assertEqual(fields["activation_barrier"]["targets"], ["barrier"])
        self.assertEqual(set(fields["reaction_energetics"]["targets"]), {"deltaHrxn", "deltaHrxn298", "deltaGrxn298"})


if __name__ == "__main__": unittest.main()
