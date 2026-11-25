"""Checks for scientific label semantics and provenance in biocatalysis views."""
import json
from pathlib import Path
import pickle
import io
import unittest
import zipfile

import numpy as np
import pandas as pd

from synrxn.reconstruction.families.biocatalysis import _source_tables as bio


class BioSemantics(unittest.TestCase):
    def test_native_residue_index_and_categories(self):
        self.assertEqual(bio.decode_sites("[[1], [3, 4], [5]]", "[0, 1, 2]", 5), [1, 0, 2, 2, 3])
        with self.assertRaises(ValueError):
            bio.decode_sites("[[0]]", "[1]", 4)
        with self.assertRaises(ValueError):
            bio.decode_sites("[[4, 5]]", "[1]", 4)

    def test_untrusted_pickle_classes_refused(self):
        with self.assertRaises(pickle.UnpicklingError):
            bio.BuiltinsOnlyUnpickler(io.BytesIO(pickle.dumps(Path("/tmp")))).load()
        plain = {1: ("C.O", "MAA")}
        self.assertEqual(bio.BuiltinsOnlyUnpickler(io.BytesIO(pickle.dumps(plain))).load(), plain)

    def test_sequence_qualification(self):
        self.assertEqual(bio.sequence("M A C\nD*"), "MACD")
        self.assertEqual(bio.sequence("?"), "")
        self.assertEqual(bio.sequence("MK-12"), "")

    def test_fasta_header_arrow_is_not_a_new_record(self):
        text = ">sp|P12345|X (1->5) enzyme\nMAC\nDE\n>sp|Q54321|Y\nMKTA\n"
        self.assertEqual(list(bio.fasta_sequences(text)), [("P12345", "MACDE"), ("Q54321", "MKTA")])

    def test_grouping_closes_transitive_links(self):
        frame = pd.DataFrame({"paper": ["A", "A", "B", "C"], "parent": ["X", "Y", "Y", "Z"]})
        grouped = bio.close_split_components(frame, ["paper", "parent"], "test")
        self.assertEqual(grouped.iloc[:3]["split_component"].nunique(), 1)
        self.assertNotEqual(grouped.iloc[0]["split_component"], grouped.iloc[3]["split_component"])


class BioArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = bio.OUT / "catalog.json"
        if not path.exists():
            raise unittest.SkipTest("Build acquired source views through synrxn.reconstruction families")
        cls.catalog = json.loads(path.read_text())
        cls.tables = {x["id"]: x for x in cls.catalog["tables"]}

    def test_assets_counts_provenance_and_contracts(self):
        self.assertEqual(set(self.catalog["summary"]["task_ids"]),
                         {"enzyme_retrieval", "mutation_effects", "active_sites", "enzyme_kinetic_parameters", "reaction_component_design"})
        bio.verify_assets(json.loads(bio.LOCK.read_text()))
        for table in self.tables.values():
            path = bio.ROOT / table["path"]
            self.assertEqual(bio.sha256(path), table["sha256"])
            frame = pd.read_parquet(path)
            self.assertEqual(len(frame), table["rows"])
            self.assertFalse(frame["source_record_id"].duplicated().any())
            self.assertTrue(frame["source_sha256"].str.fullmatch("[a-f0-9]{64}").all())
            for contract in table["task_fields"].values():
                self.assertFalse(set(contract["inputs"]) & set(contract["targets"]))
                self.assertLessEqual(set(contract["inputs"] + contract["targets"] + contract["group_fields"]), set(frame))

    def test_retrieval_positives_and_native_partitions(self):
        frame = pd.read_parquet(bio.ROOT / self.tables["reactzyme_retrieval"]["path"])
        self.assertEqual(len(frame), 178463)
        self.assertEqual(frame["positive_reference"].unique().tolist(), [1])
        self.assertEqual(frame["native_split"].value_counts().to_dict(), {"train_validation": 169724, "test": 8739})
        self.assertIn("no manufactured negatives", self.tables["reactzyme_retrieval"]["qualification"])

    def test_release_kinetics_excludes_mixed_source_aggregates(self):
        for parameter in ["kcat", "km", "ki"]:
            table = self.tables[f"brenda_record_{parameter}"]
            frame = pd.read_parquet(bio.ROOT / table["path"])
            self.assertEqual(frame["parameter"].unique().tolist(), [parameter])
            self.assertTrue(frame["kinetic_source"].eq("BRENDA").all())
            self.assertTrue(np.isfinite(frame["target_log10_brenda_record"]).all())
            self.assertTrue(np.allclose(frame["target_log10_brenda_record"], np.log10(frame["native_brenda_record_value"])))
            self.assertFalse(any("aggregate" in c or c in {"log10kcat_max", "log10km_mean", "log10ki_mean"} for c in frame))
            self.assertTrue(frame.groupby("native_cluster_40")["derived_split"].nunique().le(1).all())
            inputs = table["task_fields"]["enzyme_kinetic_parameters"]["inputs"]
            self.assertNotIn("native_measurement_value", inputs)
            native = pd.read_csv(bio.ROOT / table["source_asset_path"], dtype=str, keep_default_na=False)
            for record in frame.iloc[[0, len(frame) // 2, len(frame) - 1]].to_dict("records"):
                index = int(record["source_record_id"].rsplit(":csvrow:", 1)[1])
                self.assertEqual(native.iloc[index]["sequence_source"], "brenda")
                self.assertAlmostEqual(record["target_log10_brenda_record"], float(native.iloc[index]["log10_value"]))

    def test_active_site_labels_and_join_grouping(self):
        frame = pd.read_parquet(bio.ROOT / self.tables["mcsa_reference_active_sites"]["path"])
        self.assertTrue(frame["native_split"].eq("not_provided").all())
        self.assertTrue(frame.groupby("mechanism_group")["derived_split"].nunique().le(1).all())
        self.assertTrue(frame.groupby("sequence_group")["derived_split"].nunique().le(1).all())
        for seq, encoded, residues in zip(frame["enzyme_sequence"], frame["residue_labels_json"], frame["catalytic_residue_evidence_json"]):
            labels = json.loads(encoded)
            self.assertEqual(len(seq), len(labels))
            self.assertLessEqual(set(labels), {0, 1})
            for residue in json.loads(residues):
                self.assertEqual(seq[residue["position_1based"] - 1], residue["amino_acid"])
                self.assertEqual(labels[residue["position_1based"] - 1], 1)

    def test_release_set_and_quarantine_have_explicit_rights(self):
        self.assertTrue(all(t["release_eligible"] and t["license_expression"] for t in self.tables.values()))
        for table in self.tables.values():
            expected = not table["id"].startswith("brenda_record_")
            self.assertEqual(table["raw_assets_redistribution_allowed"], expected)
        quarantine = self.catalog["quarantined_tables"]
        self.assertEqual({t["id"] for t in quarantine}, {"catpred_kcat", "catpred_km", "catpred_ki", "easifa_rhea_active_sites"})
        self.assertTrue(all(not t["release_eligible"] and "/quarantine/" in t["path"] for t in quarantine))
        for table in quarantine:
            self.assertFalse((bio.OUT / (table["id"] + ".parquet")).exists())
        rights = json.loads((bio.OUT / "rights_audit.json").read_text())
        self.assertEqual(set(rights["release_table_ids"]), set(self.tables))
        self.assertIn("Non-Commercial Purpose License", (bio.RAW / "sabio-current-app.js").read_text())
        locked = json.loads(bio.LOCK.read_text())
        for asset in locked["assets"]:
            if "C15--datasets--" in asset["path"] or asset["path"].endswith("test_dataset_with_similarity_idx.csv"):
                self.assertFalse(asset["redistribution_allowed"])
        self.assertTrue((bio.OUT / "ATTRIBUTION.txt").is_file())
        self.assertEqual((bio.OUT / "licenses/CatPred-MIT.txt").read_bytes(), (bio.RAW / "C15--LICENSE").read_bytes())

    def test_inverse_design_uses_explicit_catalytic_endpoints(self):
        table = self.tables["enzeng_inverse_biocatalyst_design"]
        frame = pd.read_parquet(bio.ROOT / table["path"])
        self.assertTrue(frame["source_collection"].eq("ARNLD").all())
        self.assertLessEqual(set(frame["desired_endpoint_kind"]), {"total_turnover_number", "reaction_activity_percent_as_reported"})
        self.assertTrue(np.allclose(frame["desired_native_response"], frame["measured_response_native"]))
        self.assertTrue(frame["desired_native_response"].ge(0).all())
        self.assertEqual(table["task_fields"]["reaction_component_design"]["targets"], ["variant_sequence"])
        self.assertNotIn("variant_sequence", table["task_fields"]["reaction_component_design"]["inputs"])
        with zipfile.ZipFile(bio.RAW / "C13--Data.zip") as archive:
            for record in frame.iloc[[0, len(frame) // 2, len(frame) - 1]].to_dict("records"):
                native = pd.read_csv(archive.open(record["source_member"]), dtype=str, keep_default_na=False)
                original = native.iloc[record["native_csv_row_index"]]
                self.assertAlmostEqual(record["measured_response_native"], float(original[record["native_endpoint_column"]]))

    def test_mutation_delta_is_source_derived_and_references_do_not_cross_splits(self):
        frame = pd.read_parquet(bio.ROOT / self.tables["enzeng_mutation_effects"]["path"])
        self.assertTrue(np.allclose(frame["fitness_delta_native"], frame["fitness_value_native"] - frame["parent_fitness_median_native"]))
        self.assertTrue(frame["parent_reference_count"].gt(0).all())
        self.assertTrue(frame["parent_sequence"].ne(frame["variant_sequence"]).all())
        self.assertTrue(frame.groupby("reference_doi")["derived_split"].nunique().le(1).all())
        self.assertTrue(frame.groupby("parent_sequence_group")["derived_split"].nunique().le(1).all())
        self.assertTrue(frame["reference_doi"].ne("").all())
        self.assertTrue(frame["assay_technique"].ne("").all())

    def test_mutation_measurements_trace_to_native_variant_and_parent_rows(self):
        frame = pd.read_parquet(bio.ROOT / self.tables["enzeng_mutation_effects"]["path"])
        sample = frame.iloc[[0, len(frame) // 2, len(frame) - 1]]
        with zipfile.ZipFile(bio.RAW / "C13--Data.zip") as archive:
            for row in sample.to_dict("records"):
                native = pd.read_csv(archive.open(row["source_member"]), dtype=str, keep_default_na=False)
                original = native.iloc[row["native_csv_row_index"]]
                self.assertEqual(row["native_record_id"], original["id"])
                self.assertAlmostEqual(row["fitness_value_native"], float(original["fitness_value"]))
                self.assertEqual(row["variant_sequence"], bio.sequence(original["aa_sequence"]))
                parent_values = []
                for reference in json.loads(row["parent_reference_record_ids_json"]):
                    parent_idx = int(reference.rsplit(":csvrow:", 1)[1])
                    parent = native.iloc[parent_idx]
                    self.assertEqual(parent["amino_acid_substitutions"], "#PARENT#")
                    self.assertEqual(row["parent_sequence"], bio.sequence(parent["aa_sequence"]))
                    parent_values.append(float(parent["fitness_value"]))
                self.assertAlmostEqual(row["parent_fitness_median_native"], float(np.median(parent_values)))


if __name__ == "__main__":
    unittest.main()
