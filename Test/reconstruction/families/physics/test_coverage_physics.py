"""Qualification checks for physical-source adapters and their actual data."""
from __future__ import annotations

import csv
import json
import unittest

import numpy as np
from scipy.interpolate import CubicSpline

from synrxn.reconstruction.families.physics import _source_tables as physics


class PhysicsQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = physics.OUT / "catalog.json"
        cls.catalog = json.loads(path.read_text())
        cls.tables = {t["id"]: t for t in cls.catalog["tables"]}

    def read(self, identifier):
        with (physics.ROOT / self.tables[identifier]["path"]).open() as handle:
            return list(csv.DictReader(handle))

    def test_every_contract_has_disjoint_inputs_and_targets_and_pinned_provenance(self):
        for table in self.tables.values():
            self.assertEqual(physics.digest(physics.ROOT / table["path"]), table["sha256"])
            self.assertEqual(len(self.read(table["id"])), table["rows"])
            self.assertEqual(len(table["source_sha256"]), 64)
            self.assertTrue(table["source_url"].startswith("https://"))
            for contract in table["task_fields"].values():
                self.assertFalse(set(contract["inputs"]) & set(contract["targets"]))
                self.assertTrue(set(contract["inputs"] + contract["targets"] + contract["group_fields"]) <= set(table["columns"]))

    def test_actual_data_grants_and_notices_allow_redistribution(self):
        for source in self.catalog["sources"]:
            self.assertIn(source["data_license_spdx"], {"MIT", "CC-BY-4.0"})
            self.assertTrue(source["redistribution_allowed"])
            self.assertTrue(source["creators"])
            rights = source["rights"]
            for key in ["grant_snapshot", "license_snapshot", "notice", "distributed_license_notice"]:
                record = rights[key]
                self.assertEqual(physics.digest(physics.ROOT / record["path"]), record["sha256"])
            if source["id"] != "ChemSINDy2026":
                declaration = json.loads((physics.ROOT / rights["grant_snapshot"]["path"]).read_text())
                self.assertTrue(declaration["files"])
                self.assertEqual(declaration["license"]["name"], "MIT" if source["id"] == "Transition1x2022" else "CC BY 4.0")

    def test_complete_orgreact_triplets_preserve_geometry_and_missing_context(self):
        rows = self.read("orgreact_ts_geometry")
        self.assertEqual(len(rows), 9649)
        self.assertEqual(len({r["source_record_id"] for r in rows}), len(rows))
        for row in rows:
            n = len(json.loads(row["elements"]))
            for role in ["reactant", "product", "ts"]:
                xyz = np.asarray(json.loads(row[f"{role}_positions_A"]))
                self.assertEqual(xyz.shape, (n, 3))
                self.assertTrue(np.isfinite(xyz).all())
            self.assertEqual(row["charge"], "")
            self.assertEqual(row["multiplicity"], "")

    def test_kinetic_references_are_explicit_derived_simulations_with_conservation(self):
        rows = self.read("chemsindy_reference_trajectories")
        self.assertEqual(len(rows), 14)
        self.assertEqual(sum(len(json.loads(r["time_values"])) for r in rows), 580)
        for row in rows:
            self.assertEqual(row["source_reference_kind"], "derived_simulation")
            y = np.asarray(json.loads(row["concentration_trajectories"]))
            self.assertGreaterEqual(float(y.min()), -1e-8)
            np.testing.assert_allclose(y[0], json.loads(row["initial_concentrations"]), atol=1e-12)
            # Substrate and catalyst totals follow actual reaction stoichiometry.
            substrate = y[:, 0] + y[:, 1] + y[:, 3]
            catalyst = y[:, 2] + y[:, 3]
            if row["mechanism_id"] == "M20":
                substrate += y[:, 5]
                catalyst += y[:, 4] + y[:, 5]
            np.testing.assert_allclose(substrate, substrate[0], rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(catalyst, catalyst[0], rtol=1e-8, atol=1e-8)

    def test_rate_parameters_recover_from_observed_trajectories_without_exact_derivatives(self):
        rows = self.read("chemsindy_reference_trajectories")
        for model in ["M1", "M20"]:
            design, observed_changes = [], []
            for row in rows:
                if row["mechanism_id"] != model:
                    continue
                reactants = np.asarray(json.loads(row["reactant_stoichiometry"]))
                products = np.asarray(json.loads(row["product_stoichiometry"]))
                constants = np.asarray(json.loads(row["rate_constants"]))
                observations = np.asarray(json.loads(row["concentration_trajectories"]))
                times = np.asarray(json.loads(row["time_values"]))
                # Infer from finite observed time series, without reading hidden
                # derivatives or calling the reference solver's RHS.
                monomials = np.prod(observations[:, None, :] ** reactants[None, :, :], axis=2)
                integral = CubicSpline(times, monomials).antiderivative()
                cumulative = integral(times) - integral(times[0])
                for j in range(1, len(times)):
                    design.append((products - reactants).T * cumulative[j][None, :])
                    observed_changes.append(observations[j] - observations[0])
            matrix = np.concatenate(design, axis=0)
            self.assertEqual(np.linalg.matrix_rank(matrix), len(constants))
            estimate = np.linalg.lstsq(matrix, np.concatenate(observed_changes), rcond=None)[0]
            # Coarse original 30/50-point native example grids introduce
            # quadrature error; both examples recover each constant within1%.
            np.testing.assert_allclose(estimate, constants, rtol=0.01, atol=1e-5)

    def test_transition1x_retains_native_splits_trajectory_boundaries_and_input_whitelist(self):
        rows = self.read("transition1x_reactive_trajectories")
        table = self.tables["transition1x_reactive_trajectories"]
        self.assertGreater(len(rows), 10000)
        self.assertGreater(table["atomic_configurations"], 9_000_000)
        self.assertEqual(sum(int(r["configuration_count"]) for r in rows), table["atomic_configurations"])
        formula_splits = {}
        for row in rows:
            self.assertIn(row["native_split"], ["train", "val", "test"])
            formula_splits.setdefault(row["formula"], row["native_split"])
            self.assertEqual(formula_splits[row["formula"]], row["native_split"])
        for row in [rows[0], rows[len(rows)//2], rows[-1]]:
            inputs = physics.load_reactive_configuration(row, 0)
            self.assertEqual(set(inputs), {"atomic_numbers", "positions_A", "theory"})
            self.assertEqual(len(inputs["atomic_numbers"]), len(inputs["positions_A"]))
            loaded_inputs, targets = physics.load_reactive_configuration(row, int(row["configuration_count"])-1, include_targets=True)
            self.assertTrue(np.isfinite(targets["energy_eV"]))
            self.assertEqual(len(loaded_inputs["positions_A"]), len(targets["forces_eV_per_A"]))
            with self.assertRaises(IndexError):
                physics.load_reactive_configuration(row, int(row["configuration_count"]))


if __name__ == "__main__":
    unittest.main()
