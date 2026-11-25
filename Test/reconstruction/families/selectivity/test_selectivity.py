import csv
import gzip
import json
from pathlib import Path

from synrxn.reconstruction.paths import physical_path

from synrxn.reconstruction.families.selectivity._normalize import (
    build_hitea_chiral_resolution,
    build_regiosqm_eas,
    sha256_path,
)


def _rows(path):
    with gzip.open(path, "rt", encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_hitea_selectivity_keeps_measured_ee_and_rejects_zero_yield(tmp_path):
    source = tmp_path / "hitea.csv"
    fields = [
        "Dataset entry number",
        "REACTION_ID",
        "RXN_SMILES",
        "PRODUCT_STRUCTURE",
        "Product_Yield_PCT_Area_UV",
        "Product_Selectivity",
        "KeyWord_STD",
        "Reaction_T",
        "Reaction_Time_hrs",
        "Solvent_1_Name",
        "Reagent_1_Short_Hand",
        "Reagent_2_Short_Hand",
        "SCREEN_ID",
        "NOTEBOOK_ID",
        "PCAT_CMPD_ID",
    ]
    with source.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "REACTION_ID": "ee-1",
                "RXN_SMILES": "CCO.O>>CCO",
                "PRODUCT_STRUCTURE": "CCO",
                "Product_Yield_PCT_Area_UV": "42",
                "Product_Selectivity": "-62.5",
                "KeyWord_STD": "CHIRAL SALT RESOLUTION",
                "Reaction_T": "25",
                "Solvent_1_Name": "MEK",
            }
        )
        writer.writerow(
            {
                "REACTION_ID": "no-product",
                "RXN_SMILES": ">>CCO",
                "PRODUCT_STRUCTURE": "CCO",
                "Product_Yield_PCT_Area_UV": "0",
                "Product_Selectivity": "0",
                "KeyWord_STD": "CHIRAL SALT RESOLUTION",
            }
        )
        writer.writerow(
            {
                "REACTION_ID": "yield-only",
                "RXN_SMILES": "CC>>CO",
                "PRODUCT_STRUCTURE": "CO",
                "Product_Yield_PCT_Area_UV": "50",
                "Product_Selectivity": "10",
                "KeyWord_STD": "ALKYLATION",
            }
        )

    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    report = build_hitea_chiral_resolution(source, first, expected_sha256=None)
    build_hitea_chiral_resolution(source, second, expected_sha256=None)

    assert report["raw_rows"] == 3
    assert report["task_candidate_rows"] == 2
    assert report["accepted_rows"] == 1
    assert report["quarantine_reasons"] == {"zero_reported_product_yield": 1}
    assert sha256_path(first) == sha256_path(second)
    row = _rows(first)[0]
    assert row["selectivity_value"] == "-62.5"
    assert row["selectivity_type"] == "enantiomeric_excess"
    assert row["selectivity_unit"] == "percent_ee"
    assert row["measurement_method"] == "chiral SFC-MS"
    assert row["license_spdx"] == "MIT"


def test_regiosqm_builder_preserves_raw_atom_indices_and_multiple_sites(tmp_path):
    source = tmp_path / "regio"
    source.mkdir()
    (source / "compounds_smiles.csv").write_text(
        "comp1 c1ccccc1 0,2,4,\n", encoding="utf8"
    )
    (source / "compounds_doi.csv").write_text(
        "Compound, Reference\ncomp1,10.1000/example\n", encoding="utf8"
    )
    (source / "compounds_yield.csv").write_text(
        "comp1 80% (NBS)\n", encoding="utf8"
    )

    output = tmp_path / "regiosqm.csv.gz"
    report = build_regiosqm_eas(source, output, expected_sha256=None)
    row = _rows(output)[0]

    assert report["accepted_rows"] == 1
    assert report["observed_labels"] == 3
    assert row["candidate_site_indices"] == "0;1;2;3;4;5"
    assert row["observed_site_indices"] == "0;2;4"
    assert row["observed_site_count"] == "3"
    assert row["site_index_basis"].startswith("zero_based")
    assert row["reference"] == "10.1000/example"
    assert '"reported_yield_and_electrophile_raw":"80% (NBS)"' in row[
        "source_metadata_json"
    ]
    assert row["license_spdx"] == "MIT"


def test_regiosqm_builder_rejects_non_aromatic_observed_site(tmp_path):
    source = tmp_path / "regio"
    source.mkdir()
    (source / "compounds_smiles.csv").write_text("comp1 CCO 0,\n", encoding="utf8")
    (source / "compounds_doi.csv").write_text(
        "Compound, Reference\ncomp1,example\n", encoding="utf8"
    )
    (source / "compounds_yield.csv").write_text("comp1 10%\n", encoding="utf8")
    output = tmp_path / "regiosqm.csv.gz"
    report = build_regiosqm_eas(source, output, expected_sha256=None)
    assert report["accepted_rows"] == 0
    assert report["quarantined_rows"] == 1
    assert report["quarantine_reasons"] == {
        "observed_site_is_not_aromatic_carbon_hydrogen": 1
    }


def test_working_selectivity_artifacts_match_audit_report():
    report = json.loads(
        Path("Data/metadata/sources/selectivity/build_report.json").read_text(
            encoding="utf8"
        )
    )
    reports = {item["dataset"]: item for item in report["datasets"]}
    hitea_path = physical_path("Data/selectivity/hitea_chiral_resolution.csv.gz")
    regio_path = physical_path("Data/selectivity/regiosqm_eas.csv.gz")
    hitea = _rows(hitea_path)
    regio = _rows(regio_path)

    assert len(hitea) == reports["hitea_chiral_resolution"]["accepted_rows"] == 50
    assert len(regio) == reports["regiosqm_eas"]["accepted_rows"] == 535
    assert sum(int(row["observed_site_count"]) for row in regio) == 604
    assert all(float(row["reported_yield_pct"]) > 0 for row in hitea)
    assert all(-100 <= float(row["selectivity_value"]) <= 100 for row in hitea)
    assert sha256_path(hitea_path) == reports["hitea_chiral_resolution"][
        "output_sha256"
    ]
    assert sha256_path(regio_path) == reports["regiosqm_eas"]["output_sha256"]
