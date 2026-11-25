"""Guards for mechanism-source identity, reference isolation and scoped usage."""
from copy import deepcopy
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from synrxn.reconstruction.families.mechanisms.mechanistic_prediction import _mcsa_views as mcsa
from synrxn.reconstruction.coverage.rights import table_has_evidence


class MCSACoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((mcsa.OUT / "catalog.json").read_text())
        csv.field_size_limit(100_000_000)
        path = mcsa.ROOT / mcsa.COMPANION / "Data/mechanism/mcsa_complete_reactions_observed.csv.gz"
        with gzip.open(path, "rt", newline="") as handle:
            cls.native = next(csv.DictReader(handle))

    def test_targets_and_native_certificates_are_withheld(self):
        for table in self.catalog["tables"]:
            fields = table["task_fields"]["mechanistic_prediction"]
            self.assertEqual(fields["inputs"], ["reaction_smiles"])
            self.assertTrue(set(fields["targets"]) <= set(fields["withheld_reference_columns"]))
            for metadata in ["source_context_json", "native_index_json", "source_ids_json", "verification_level", "development_exposed"]:
                self.assertIn(metadata, fields["withheld_reference_columns"])
                self.assertNotIn(metadata, fields["inputs"])
            self.assertEqual(table["native_split_status"], "unassigned")
            self.assertFalse(table["independent_dataset"])
            self.assertEqual(table["role"], "primary" if table["id"] == "mcsa_observed_mechanisms" else "overlapping_view")

    def test_same_native_proposal_keeps_group_across_views(self):
        lock = json.loads(mcsa.LOCK.read_text())
        rows = []
        for view in mcsa.VIEWS:
            path = f"{mcsa.COMPANION}/Data/mechanism/mcsa_{view}.csv.gz"
            with gzip.open(mcsa.ROOT / path, "rt", newline="") as handle:
                native = next(csv.DictReader(handle))
            rows.append(mcsa.qualified_row(native, view, mcsa.pin_for(lock, path)["sha256"]))
        self.assertEqual(len({r["enzyme_group"] for r in rows}), 1)
        self.assertEqual(len({r["mechanism_group"] for r in rows}), 1)
        route_transitions = set(json.loads(rows[0]["native_transition_groups_json"]))
        self.assertTrue(set(json.loads(rows[-1]["native_transition_groups_json"])) <= route_transitions)
        self.assertTrue(all(r["native_split"] == "unassigned" for r in rows))

    def test_no_promotion_of_reconstructed_or_delta_consistent_cases(self):
        native = deepcopy(self.native)
        native["initial_participant_context"] = "mechanism_derived_component_preconditions"
        with self.assertRaisesRegex(ValueError, "Reconstructed participants"):
            mcsa.qualified_row(native, "complete_reactions_observed", "0" * 64)
        native = deepcopy(self.native)
        native["verification_level"] = "DELTA_CONSISTENT"
        row = mcsa.qualified_row(native, "complete_reactions_observed", "0" * 64)
        self.assertFalse(row["primary_replay_eligible"])
        self.assertFalse(row["preferred_primary_reference"])
        native["native_index_json"] = json.dumps({"source_ids": [999999]})
        with self.assertRaisesRegex(ValueError, "identities differ"):
            mcsa.qualified_row(native, "complete_reactions_observed", "0" * 64)

    def test_actual_source_grant_and_frozen_dependency_pins(self):
        source = self.catalog["sources"][0]
        for asset in source["assets"]:
            if asset["license"] == "Apache-2.0":
                self.assertFalse(asset["redistribution_allowed"])
                self.assertFalse(asset["raw_assets_redistribution_allowed"])
        for table in self.catalog["tables"]:
            self.assertTrue(table_has_evidence(table, source, self.catalog, mcsa.ROOT))
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            (repo / "source.txt").write_text("new content")
            lock = {"assets": [{"path": "source.txt", "bytes": 11, "sha256": "0" * 64}]}
            with self.assertRaisesRegex(ValueError, "Missing/changed"):
                mcsa.verify_assets(lock, repo)


if __name__ == "__main__":
    unittest.main()
