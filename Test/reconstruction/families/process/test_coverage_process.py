"""Substantive integrity checks for acquired analytical/process task views."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from synrxn.reconstruction.families.process import _source_tables as process


def records(table):
    with gzip.open(process.ROOT / table["path"], "rt") as file:
        return [json.loads(line) for line in file]


class ProcessCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((process.OUT / "catalog.json").read_text())
        cls.tables = {t["id"]: t for t in cls.catalog["tables"]}

    def test_headerless_physical_and_generated_rows_are_distinguished(self):
        self.assertEqual(self.tables["fcc_lab_source_full"]["rows"], 10000)
        self.assertEqual(self.tables["fcc_pilot_test"]["rows"], 15)
        self.assertEqual(self.tables["fcc_pilot_train"]["rows"], 48)
        self.assertEqual(self.tables["fcc_pilot_validation"]["rows"], 12)
        experiment = records(self.tables["fcc_pilot_test"])[0]
        self.assertEqual(experiment["reference_kind"], "measured_pilot_experiment")
        self.assertEqual(experiment["product_bulk_properties"][4], experiment["coke_mass_percent"])
        self.assertEqual(len(experiment["feed_molecular_composition"]), 129)
        generated = records(self.tables["fcc_lab_source_full"])[0]
        self.assertEqual(generated["reference_kind"], "native_mechanistic_simulation")
        self.assertEqual(len(generated["product_molecular_composition"]), 129)

    def test_native_views_do_not_inflate_independent_experiments(self):
        self.assertEqual(self.tables["reac_discovery_cs1"]["rows"], 55)
        self.assertEqual(self.tables["reac_discovery_cs2"]["rows"], 58)
        self.assertFalse(any("m1" in t for t in self.tables))
        self.assertIn("1016", self.catalog["excluded_assets"][0]["reason"])
        self.assertIn("982", self.catalog["excluded_assets"][0]["reason"])
        self.assertEqual(self.tables["fcc_pilot_train"]["independent_groups"], 15)

    def test_nmr_native_reference_alignment_and_normalization(self):
        expected = {"saccharose_hydrolysis": 1023, "pmg_284": 999, "pmg_287": 1000}
        for native, count in expected.items():
            table = self.tables[f"nmr_{native}_snapshots"]
            self.assertEqual(table["rows"], count)
            self.assertEqual(table["readiness"], "derived_analytical_reference_from_licensed_measurements")
            rows = records(table)
            self.assertEqual([r["spectrum_column"] for r in rows], list(range(1, count + 1)))
            for row in rows:
                refs = np.maximum(row["reference_integrals"], 0)
                np.testing.assert_allclose(refs / refs.sum(), row["reference_area_fractions"], rtol=1e-14)
                self.assertIn("derived_trapezoid", row["reference_kind"])
                self.assertNotIn("reference_path", row)
            forecasts = records(self.tables[f"nmr_{native}_progress"])
            self.assertEqual(len(forecasts), count - 1)
            for current, following in zip(forecasts, rows[1:]):
                self.assertGreater(current["target_elapsed_seconds"], current["elapsed_seconds"])
                self.assertEqual(current["target_native_spectrum_column"], current["spectrum_column"] + 1)
                self.assertEqual(current["next_reference_area_fractions"], following["reference_area_fractions"])

    def test_spectrum_loader_never_reads_a_future_column(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savetxt(root / "signals.csv", [[1, 11, 999999], [2, np.nan, 999999], [3, 13, 999999]])
            with mock.patch.object(process, "ROOT", root):
                x, y = process.load_spectrum({"spectrum_path": "signals.csv", "spectrum_column": 1})
            np.testing.assert_array_equal(x, [1, 3])
            np.testing.assert_array_equal(y, [11, 13])

    def test_integration_does_not_bridge_disjoint_windows_or_impute_zeros(self):
        ppm = np.array([1., 2., 3., 100., 101.])
        signal = np.array([1., np.nan, 1., 2., 2.])
        result = process.integrate_spectrum(ppm, signal, [[[1., 3.], [100., 101.]]])
        # Area2underfirstwindow +area2undersecond; no97ppm bridge.
        np.testing.assert_allclose(result, [4.])

    def test_lowfield_profiles_are_four_groups_and_past_only(self):
        table = self.tables["lowfield_nmr_yield_progress"]
        self.assertEqual(table["rows"], 236)
        rows = records(table)
        self.assertEqual(len({r["trajectory_id"] for r in rows}), 4)
        for row in rows:
            self.assertGreater(row["target_elapsed_minutes"], row["elapsed_minutes"])
        inputs = table["task_fields"]["reaction_progress_monitoring"]["inputs"]
        self.assertNotIn("next_yield_percent", inputs)

    def test_cstr_oracle_replays_published_model_transitions(self):
        table = self.tables["cstr_control_transitions"]
        rows = records(table)
        self.assertEqual(len(rows), 600)
        for row in rows[::50]:
            reproduced = process.cstr_step(row["state"], row["action"], row["duration_hours"], row["alpha"], row["beta"])
            np.testing.assert_allclose(reproduced, row["next_state"], rtol=1e-10, atol=1e-11)
            self.assertIn("derived_ODE_simulation", row["reference_kind"])
        with self.assertRaises(ValueError):
            process.cstr_step([.8, .5, 130, 125], [101, -3000])
        with self.assertRaises(ValueError):
            process.cstr_step([.8, .5, 130, 125], [40, -8501])

    def test_provenance_and_task_input_contracts(self):
        expected = {"analytical_interpretation", "reaction_progress_monitoring", "reactor_scale_modeling", "process_control", "process_impact_assessment"}
        self.assertTrue(expected <= set(self.catalog["task_ids"]))
        for table in self.tables.values():
            self.assertEqual(process.sha(process.ROOT / table["path"]), table["sha256"])
            self.assertTrue(table["qualification"])
            self.assertTrue(table["reference_evidence"])
            for asset in table["source_assets"]:
                self.assertTrue((process.ROOT / asset["path"]).exists())
                self.assertTrue(asset["url"].startswith("https://"))
                self.assertNotIn("Mnova", asset["license"])
                self.assertTrue(asset["redistribution_allowed"])
            for contract in table["task_fields"].values():
                self.assertFalse(set(contract["inputs"]) & set(contract["targets"]))
                self.assertTrue(contract["group_fields"])
                self.assertTrue(contract["metrics"])
        for source in self.catalog["sources"]:
            self.assertEqual(process.sha(process.ROOT / source["license_evidence"]), source["license_evidence_sha256"])
            self.assertTrue(source["license_evidence_url"].startswith("https://"))
            self.assertTrue(source["authors"])
        notice = process.ROOT / self.catalog["attribution_path"]
        self.assertEqual(process.sha(notice), self.catalog["attribution_sha256"])
        self.assertIn("LGPL code changes", notice.read_text())


if __name__ == "__main__":
    unittest.main()
