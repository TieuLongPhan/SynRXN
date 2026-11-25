"""Verify the distinctions between reaction context, substrate pairs and substrates."""
import copy

import numpy as np
import pytest

from synrxn.statistics.split_audit import audit_partitions, label_audit, partition_statistics, population


def row(aryl, amine, catalyst="cat1", label="1", value="25"):
    return {"Aryl SMILES": aryl, "Amine SMILES": amine, "Catalyst SMILES": catalyst,
            "Base SMILES": "base", "Solvent SMILES": "solvent", "labels": label,
            "Yield": value, "Source": "source"}


def test_context_novelty_does_not_imply_novel_pairs_or_novel_substrates():
    rows = [row("A", "N"), row("A", "N", catalyst="cat2"), row("B", "N", label="0"),
            row("C", "M"), row("A", "N", label="0")]
    before = copy.deepcopy(rows)
    result = partition_statistics(population(rows), [False, True, True, True, True])
    assert (result["train_rows"], result["test_rows"]) == (1, 4)
    assert result["unseen_context_rows"] == 3
    assert result["unseen_pair_rows"] == 2
    assert result["both_substrates_unseen_rows"] == 1
    assert result["positive_fraction"] == .5
    assert rows == before


def test_random_controls_keep_each_native_test_size_and_do_not_use_labels():
    rows = [row(str(i), str(i)) for i in range(8)]
    splits = [{"iteration_0 train_test": repr(["test" if cycle == i % 4 else "train" for cycle in range(4)])}
              for i in range(8)]
    first = audit_partitions(population(rows), splits, seed=77, iterations=1)
    assert first == audit_partitions(population(rows), splits, seed=77, iterations=1)
    changed_seed = audit_partitions(population(rows), splits, seed=78, iterations=1)
    assert any(before["test_membership_sha256"] != after["test_membership_sha256"]
               for before, after in zip(first[1::2], changed_seed[1::2], strict=True))
    inverted = [dict(item, labels="0") for item in rows]
    second = audit_partitions(population(inverted), splits, seed=77, iterations=1)
    for before, after in zip(first, second, strict=True):
        for key in ("test_rows", "test_membership_sha256", "unseen_context_rows", "unseen_pair_rows",
                    "both_substrates_unseen_rows"):
            assert before[key] == after[key]
    for native, random in zip(first[::2], first[1::2], strict=True):
        assert native["test_rows"] == random["test_rows"] == 2
        assert native["train_rows"] == random["train_rows"] == 6


def test_threshold_audit_counts_both_directions_without_changing_the_targets():
    rows = [row("A", "N", value="10"), row("A", "N", value="9"),
            row("A", "N", label="0", value="10")]
    before = copy.deepcopy(rows)
    result = label_audit(rows)[0]
    assert result["rows"] == 3
    assert result["yield_gt_10_disagreements"] == 2
    assert result["yield_ge_10_disagreements"] == 2
    assert rows == before


@pytest.mark.parametrize("mask", [[True], [True, True], [False, False]])
def test_invalid_partitions_are_rejected(mask):
    with pytest.raises(ValueError, match="Partition"):
        partition_statistics(population([row("A", "N"), row("B", "M")]), np.array(mask))


def test_missing_identity_cannot_be_counted_as_a_novel_substrate():
    with pytest.raises(ValueError, match="Missing chemical identity"):
        population([row("", "N")])
