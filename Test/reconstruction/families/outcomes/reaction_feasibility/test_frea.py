"""FREA import must preserve labels, tiers, original splits and source identity."""

import csv
import gzip

import pytest

from synrxn.reconstruction.families.outcomes.reaction_feasibility._frea_source import build_frea, digest


def source(tmp_path, header, rows, name="test.csv", split="test"):
    path = tmp_path / name
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return {"path": path, "sha256": digest(path), "rows": len(rows), "split": split}


def read(path):
    with gzip.open(path, "rt") as handle:
        return list(csv.DictReader(handle))


def test_expert_labels_and_notes_remain_expert_judgments(tmp_path):
    pin = source(
        tmp_path,
        ["reactant", "product", "feasibility", "failure_reasons"],
        [["C", "CO", "0", "expert reason"], ["CO", "C", "1", ""]],
    )
    out = tmp_path / "expert.csv.gz"
    report = build_frea([pin], out, dataset="frea_expert", expert=True)
    rows = read(out)
    assert report["rows"] == 2
    assert {r["evidence_tier"] for r in rows} == {"expert_judgment"}
    assert rows[0]["failure_reasons"] == "expert reason"
    assert rows[0]["reaction_smiles"] == "C>>CO"
    assert rows[0]["upstream_split"] == "test"


def test_generated_corpus_keeps_label_origin_and_original_splits(tmp_path):
    train = source(
        tmp_path,
        ["reactants", "product", "method", "feasible"],
        [["C", "CO", "ET", "0"], ["CO", "C", "no_method", "1"]],
        name="train.csv",
        split="train",
    )
    test = source(tmp_path, ["reactants", "product", "method", "feasible"], [["C", "CO", "RR", "0"]])
    out = tmp_path / "corpus.csv.gz"
    build_frea([train, test], out, dataset="frea_corpus", expert=False)
    rows = read(out)
    assert rows[0]["evidence_tier"] == "generated_negative" and rows[0]["negative_method"] == "ET"
    assert rows[1]["evidence_tier"] == "upstream_positive" and rows[1]["negative_method"] == ""
    assert rows[0]["exact_reaction_group"] == rows[2]["exact_reaction_group"]
    assert rows[0]["sample_id"] != rows[2]["sample_id"]
    assert [r["upstream_split"] for r in rows] == ["train", "train", "test"]


def test_source_checksum_failure_leaves_no_output(tmp_path):
    pin = source(tmp_path, ["reactant", "product", "feasibility"], [["C", "CO", "1"]])
    pin["sha256"] = "0" * 64
    out = tmp_path / "output.csv.gz"
    with pytest.raises(ValueError, match="checksum mismatch"):
        build_frea([pin], out, dataset="frea_expert", expert=True)
    assert not out.exists()


def test_invalid_label_is_rejected_without_silent_row_loss(tmp_path):
    pin = source(tmp_path, ["reactant", "product", "feasibility"], [["C", "CO", "maybe"]])
    out = tmp_path / "output.csv.gz"
    with pytest.raises(ValueError, match="invalid FREA binary label"):
        build_frea([pin], out, dataset="frea_expert", expert=True)
    assert not out.exists() and not list(tmp_path.glob("*.tmp"))


def test_repeated_build_preserves_compressed_bytes(tmp_path):
    pin = source(tmp_path, ["reactant", "product", "feasibility"], [["C", "CO", "1"]])
    paths = [tmp_path / "one.csv.gz", tmp_path / "two.csv.gz"]
    for path in paths:
        build_frea([pin], path, dataset="frea_expert", expert=True)
    assert paths[0].read_bytes() == paths[1].read_bytes()
