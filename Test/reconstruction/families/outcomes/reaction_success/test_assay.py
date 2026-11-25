"""Safety checks for recovering split metadata by native reaction identity."""

from copy import deepcopy
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from synrxn.reconstruction.families.outcomes.reaction_success._bh_hte_ood import (
    CANONICAL_PATH,
    CANONICAL_SHA256,
    OUTPUT_COLUMNS,
    SUCCESS_COLUMNS,
    decode_assignments,
    find_source,
    join_native_splits,
    load_native,
    native_reaction,
    seed_summary,
    success_label_summary,
    success_target_rows,
    verify_hash,
)


def example_rows(record_id="0"):
    original = {
        "": record_id, "Source": "example cohort", "Yield": "12.345678901234",
        "labels": "1", "Aryl SMILES": "Brc1ccccc1", "Amine SMILES": "CN",
        "Catalyst SMILES": "[Pd]", "Base SMILES": "O", "Solvent SMILES": "CO",
        "Product SMILES": "CNc1ccccc1",
    }
    for seed in range(10):
        original[f"iteration_{seed} cluster"] = str(seed)
        original[f"iteration_{seed} train_test"] = "['train', 'test', 'train', 'train']"
    canonical = {
        "sample_id": f"sample:{record_id}", "source_record_id": record_id,
        "source_index": "0", "source": original["Source"],
        "raw_reaction_smiles": native_reaction(original),
        "yield_pct": "12.3456789012",
        "source_metadata_json": json.dumps({"upstream_labels": "1"}),
    }
    return original, canonical


class NativeSplitTests(unittest.TestCase):
    def test_original_split_strings_preserved_and_targets_not_exported(self):
        original, canonical = example_rows()
        output, excluded = join_native_splits([original], [canonical])
        self.assertEqual(excluded, [])
        self.assertEqual(tuple(output[0]), OUTPUT_COLUMNS)
        for seed in range(10):
            self.assertEqual(output[0][f"iteration_{seed} train_test"], original[f"iteration_{seed} train_test"])
        self.assertNotIn("yield_pct", output[0])
        self.assertNotIn("labels", output[0])

    def test_matching_ids_cannot_hide_changed_chemistry_target_or_cohort(self):
        original, canonical = example_rows()
        changes = {
            "raw_reaction_smiles": "C>>CC",
            "yield_pct": "99",
            "source": "a different experimental cohort",
            "source_index": "1",
            "source_metadata_json": '{"upstream_labels":"0"}',
        }
        for column, changed in changes.items():
            with self.subTest(column=column):
                corrupted = {**canonical, column: changed}
                with self.assertRaises(ValueError):
                    join_native_splits([original], [corrupted])

    def test_ambiguous_native_or_canonical_ids_rejected(self):
        original, canonical = example_rows()
        with self.assertRaisesRegex(ValueError, "duplicate native"):
            join_native_splits([original, deepcopy(original)], [canonical])
        with self.assertRaisesRegex(ValueError, "duplicate canonical"):
            join_native_splits([original], [canonical, deepcopy(canonical)])

    def test_bad_assignments_rejected_without_evaluating_code(self):
        for value in (
            "['train', 'test']",
            "['test', 'test', 'train', 'train']",
            "['train', 'train', 'train', 'validation']",
            "__import__('os').getcwd()",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_assignments(value)

    def test_native_clusters_cannot_cross_fold_boundaries(self):
        rows = []
        for cluster in range(12):
            labels = ["test" if cycle == cluster // 3 else "train" for cycle in range(4)]
            rows.append({"iteration_0 cluster": str(cluster), "iteration_0 train_test": repr(labels)})
        summary = seed_summary(rows, 0)
        self.assertEqual(summary["cycles"][0]["counts"], {"test": 3, "train": 9})
        rows.append({"iteration_0 cluster": "0", "iteration_0 train_test": "['train', 'test', 'train', 'train']"})
        with self.assertRaisesRegex(ValueError, "crosses a fold"):
            seed_summary(rows, 0)

    def test_corrupted_cached_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_bytes(b"original captured source")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            verify_hash(path, expected)
            path.write_bytes(b"changed captured source")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                verify_hash(path, expected)

    def test_binary_targets_preserve_native_label_at_threshold_and_below(self):
        for value in ("10", "7.33"):
            with self.subTest(yield_pct=value):
                original, canonical = example_rows()
                original["Yield"] = value
                canonical["yield_pct"] = value
                splits, _ = join_native_splits([original], [canonical])
                targets = success_target_rows([original], splits)
                self.assertEqual(tuple(targets[0]), SUCCESS_COLUMNS)
                self.assertEqual(targets[0]["success"], "1")
                self.assertEqual(targets[0]["source_record_id"], original[""])
                self.assertNotIn("Yield", targets[0])
                self.assertNotIn("raw_reaction_smiles", targets[0])
                audit = success_label_summary([original])
                self.assertEqual(audit["threshold_comparison"]["yield_gt_10"]["disagreement_rows"], 1)
                self.assertEqual(audit["threshold_comparison"]["yield_ge_10"]["disagreement_rows"], int(value != "10"))

    def test_invalid_native_binary_target_is_rejected(self):
        original, canonical = example_rows()
        original["labels"] = "uncertain"
        canonical["source_metadata_json"] = '{"upstream_labels":"uncertain"}'
        splits, _ = join_native_splits([original], [canonical])
        with self.assertRaisesRegex(ValueError, "Invalid native binary"):
            success_target_rows([original], splits)
        with self.assertRaisesRegex(ValueError, "Invalid native binary"):
            success_label_summary([original])

    def test_pinned_native_labels_counts_and_source_threshold_exceptions(self):
        original = load_native(find_source())
        verify_hash(CANONICAL_PATH, CANONICAL_SHA256)
        with gzip.open(CANONICAL_PATH, "rt", encoding="utf-8", newline="") as handle:
            canonical = list(csv.DictReader(handle))
        splits, excluded = join_native_splits(original, canonical)
        targets = success_target_rows(original, splits)
        self.assertEqual(len(targets), 27438)
        self.assertEqual(Counter(row["success"] for row in targets), {"0": 15532, "1": 11906})
        self.assertEqual(len(excluded), 59)
        self.assertEqual(Counter(row["labels"] for row in excluded), {"1": 59})
        self.assertEqual([row["sample_id"] for row in targets], [row["sample_id"] for row in canonical])
        by_id = {row[""]: row for row in original}
        self.assertTrue(all(row["success"] == by_id[row["source_record_id"]]["labels"] for row in targets))
        raw_audit = success_label_summary(original)
        self.assertEqual(raw_audit["native_label_counts"], {"0": 15532, "1": 11965})
        comparisons = raw_audit["threshold_comparison"]
        self.assertEqual(comparisons["yield_gt_10"], {
            "disagreement_rows": 183,
            "disagreement_rows_by_source": {
                "AZ ELN 2023": 28, "Dreher 2016": 2, "HiTEA 2024": 16,
                "JNJ HTE 2024": 111, "Merck 2025": 1, "Roche 2023": 25,
            },
        })
        self.assertEqual(comparisons["yield_ge_10"], {
            "disagreement_rows": 28,
            "disagreement_rows_by_source": {"AZ ELN 2023": 28},
        })


if __name__ == "__main__":
    unittest.main()
