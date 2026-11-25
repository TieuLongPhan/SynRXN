import csv
import gzip
import io
import json
import tarfile

import pandas as pd
import pytest

from synrxn.reconstruction.families.conditions.condition_recovery._normalize import (
    build_openexp_condition,
    build_ord_hte_condition,
    build_orderly_condition,
)
from synrxn.reconstruction.families.outcomes.yield_conversion._normalize import sha256_path


def _frame(index, *, temperature=25.0, reported_yield=80.0):
    return pd.DataFrame(
        {
            "original_index": [index],
            "agent_000": ["O"],
            "agent_001": [None],
            "agent_002": [None],
            "date_of_experiment": [pd.NaT],
            "extracted_from_file": ["ord_dataset-fixture"],
            "grant_date": [pd.Timestamp("2020-01-02")],
            "is_mapped": [True],
            "procedure_details": ["Fixture procedure."],
            "product_000": ["CO"],
            "reactant_000": ["C"],
            "reactant_001": [None],
            "rxn_str": ["[CH4:1]>O>[CH3:1]O"],
            "rxn_time": [2.0],
            "solvent_000": ["CCO"],
            "solvent_001": [None],
            "temperature": [temperature],
            "yield_000": [reported_yield],
        }
    )


def _rows(path):
    with gzip.open(path, "rt", encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_orderly_builder_preserves_splits_missingness_and_quality(tmp_path):
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    _frame(1).to_parquet(train, index=False)
    _frame(2, temperature=7580, reported_yield=101).to_parquet(
        test, index=False
    )
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"

    report = build_orderly_condition(
        train,
        test,
        first,
        expected_train_sha256=None,
        expected_test_sha256=None,
    )
    build_orderly_condition(
        train,
        test,
        second,
        expected_train_sha256=None,
        expected_test_sha256=None,
    )
    rows = _rows(first)

    assert report["accepted_rows"] == 2
    assert report["quarantined_rows"] == 0
    assert report["split_counts"] == {"train": 1, "test": 1}
    assert report["quality_counts"] == {
        "temperature_outside_minus196_500": 1,
        "yield_outside_0_100": 1,
    }
    assert sha256_path(first) == sha256_path(second)
    assert [row["upstream_split"] for row in rows] == ["train", "test"]
    assert rows[0]["agent_2"] == ""
    assert rows[0]["temperature_plausible"] == "true"
    assert rows[1]["temperature_plausible"] == "false"
    assert rows[1]["yield_valid_0_100"] == "false"
    assert rows[0]["license_spdx"] == "CC-BY-4.0"


def test_openexp_builder_preserves_actions_and_sharealike(tmp_path):
    payload = [
        {
            "index": 1,
            "REACTANT": ["C", "O"],
            "PRODUCT": ["CO"],
            "CATALYST": ["[Pd]"],
            "SOLVENT": ["CCO"],
            "actions": "ADD $1$ ; STIR for @1@ ; YIELD $-1$.",
            "source": "Fixture procedure.",
            "extracted_molecules": {"C": "$1$"},
            "extracted_temperature": {"80 °C": "#1#"},
            "extracted_duration": {"2 hours": "@1@"},
            "molecules": {"methane": "C"},
            "score": 0.9,
        }
    ]
    source = tmp_path / "OpenExp.tar"
    encoded = json.dumps(payload).encode()
    with tarfile.open(source, "w") as archive:
        info = tarfile.TarInfo("OpenExp.json")
        info.size = len(encoded)
        archive.addfile(info, io.BytesIO(encoded))
    output = tmp_path / "openexp.csv.gz"

    report = build_openexp_condition(source, output, expected_sha256=None)
    row = _rows(output)[0]

    assert report["accepted_rows"] == 1
    assert row["action_sequence"].startswith("ADD")
    assert row["temperature_c"] == "80"
    assert row["time_hours"] == "2"
    assert row["license_spdx"] == "CC-BY-SA-4.0"
    assert row["synrxn_split"] in {"train", "validation", "test"}


def test_ord_builder_requires_complete_allowlist(tmp_path):
    pytest.importorskip("ord_schema")
    with pytest.raises(ValueError, match="missing allowlisted ORD datasets"):
        build_ord_hte_condition(
            [], tmp_path / "ord.csv.gz", expected_sha256={}
        )
