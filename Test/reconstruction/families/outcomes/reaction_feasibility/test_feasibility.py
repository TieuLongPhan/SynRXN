import csv
import gzip

from synrxn.reconstruction.families.outcomes.reaction_feasibility._normalize import build_ursa_plausibility
from synrxn.reconstruction.families.outcomes.yield_conversion._normalize import sha256_path


def _rows(path):
    with gzip.open(path, "rt", encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_ursa_builder_preserves_expert_labels_and_is_deterministic(tmp_path):
    source = tmp_path / "ursa.csv"
    source.write_text(
        "reaction_smiles,expert_score,chemcensor_score,chemcensor_score_bin\n"
        "C.O>>CO,1,2.0,1\n"
        "CC>>C,0,0.0,0\n",
        encoding="utf8",
    )
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    report = build_ursa_plausibility(source, first, expected_sha256=None)
    build_ursa_plausibility(source, second, expected_sha256=None)
    rows = _rows(first)

    assert report["accepted_rows"] == 2
    assert report["label_counts"] == {"0": 1, "1": 1}
    assert sha256_path(first) == sha256_path(second)
    assert rows[0]["feasible"] == "1"
    assert rows[0]["upstream_split"] == "test"
    assert rows[0]["license_spdx"] == "Apache-2.0"
    assert rows[0]["experimental"] == "false"
