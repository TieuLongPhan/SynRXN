import csv
import gzip
import json
import zipfile

from synrxn.reconstruction.families.selectivity.impurity_prediction._normalize import build_seismiq_simulated_impurities
from synrxn.reconstruction.families.outcomes.yield_conversion._normalize import sha256_path


def _rows(path):
    with gzip.open(path, "rt", encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_seismiq_builder_marks_candidates_as_non_ground_truth(tmp_path):
    source = tmp_path / "simulated.zip"
    payload = [
        {
            "Starting Materials": ["C", "O"],
            "Predicted Impurities": ["CO", "C=O"],
        }
    ]
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "simulated_impurities/example1.json", json.dumps(payload)
        )
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    report = build_seismiq_simulated_impurities(
        source, first, expected_sha256=None
    )
    build_seismiq_simulated_impurities(
        source, second, expected_sha256=None
    )
    rows = _rows(first)

    assert report["accepted_rows"] == 2
    assert sha256_path(first) == sha256_path(second)
    assert rows[0]["candidate_status"] == "predicted_candidate"
    assert rows[0]["structure_valid"] == "true"
    assert rows[0]["observed"] == ""
    assert rows[0]["experimental"] == "false"
    assert rows[0]["license_spdx"] == "CC-BY-4.0"
