import csv
import gzip
import zipfile

from synrxn.reconstruction.families.outcomes.yield_conversion._normalize import (
    USPTO_MEMBERS,
    build_buchwald_hartwig_ood,
    build_hitea_yield,
    build_suzuki_miyaura,
    build_uspto_yield,
    sha256_path,
)


def _rows(path):
    with gzip.open(path, "rt", encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_uspto_builder_is_deterministic_and_preserves_splits(tmp_path):
    archive = tmp_path / "uspto.zip"
    header = (
        "\tmyID\tSource\tTarget\tCanonicalizedReaction\t"
        "OriginalReaction\tYield\n"
    )
    with zipfile.ZipFile(archive, "w") as output:
        for index, (split, member) in enumerate(USPTO_MEMBERS.items(), start=1):
            row = (
                f"{index}\tID{index:08d}\ttoken source\ttoken target\t"
                f"C{index}>O>CO{index}\t[C:{index}]>>[C:{index}]O\t{90 + index}\n"
            )
            output.writestr(member, header + row)

    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    report = build_uspto_yield(archive, first, expected_sha256=None)
    build_uspto_yield(archive, second, expected_sha256=None)

    assert report["accepted_rows"] == 3
    assert report["quarantined_rows"] == 0
    assert sha256_path(first) == sha256_path(second)
    rows = _rows(first)
    assert [row["upstream_split"] for row in rows] == [
        "test",
        "train",
        "validation",
    ]
    assert rows[0]["yield_measurement_type"] == "patent_reported_curated"
    assert rows[0]["license_spdx"] == "MIT"


def test_hitea_builder_removes_atom_maps_and_keeps_context(tmp_path):
    source = tmp_path / "hitea.csv"
    fields = [
        "Dataset entry number",
        "REACTION_ID",
        "RXN_SMILES",
        "Product_Yield_PCT_Area_UV",
        "Product_Yield_Mass_Ion_Count",
        "Product_Selectivity",
        "Reaction_T",
        "Reaction_Time_hrs",
        "Solvent_1_Name",
        "ReactionClass",
        "SCREEN_ID",
        "NOTEBOOK_ID",
        "KeyWord_STD",
        "ReactionGroup",
        "PCAT_CMPD_ID",
        "Product_UV_Analysis_Wavelength_nm",
        "catalyst_1_ID_1_SMILES",
        "Catalyst_1_ID_2_SMILES",
        "Catalyst_2_ID_1_SMILES",
        "Catalyst_2_ID_2_SMILES",
        "Catalyst_1_Short_Hand",
        "Catalyst_2_Short_Hand",
        "Reagent_1_Short_Hand",
        "Reagent_2_Short_Hand",
    ]
    with source.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "Dataset entry number": "7",
                "REACTION_ID": "RXN-7",
                "RXN_SMILES": "[CH3:1][OH:2]>>[CH2:1]=[O:2]",
                "Product_Yield_PCT_Area_UV": "42.5",
                "Product_Selectivity": "#ERROR",
                "Reaction_T": "25",
                "Reaction_Time_hrs": "2",
                "Solvent_1_Name": "MeCN",
                "ReactionClass": "Oxidation",
                "SCREEN_ID": "screen-1",
                "NOTEBOOK_ID": "book-1",
                "Reagent_1_Short_Hand": "base",
            }
        )

    output = tmp_path / "hitea.csv.gz"
    report = build_hitea_yield(source, output, expected_sha256=None)
    row = _rows(output)[0]
    assert report["accepted_rows"] == 1
    assert ":1" not in row["canonical_reaction_smiles"]
    assert row["temperature_c"] == "25"
    assert row["screen_id"] == "screen-1"
    assert row["selectivity"] == ""
    assert '"selectivity_raw":"#ERROR"' in row["source_metadata_json"]
    assert row["license_spdx"] == "MIT"


def test_bh_builder_quarantines_invalid_yield(tmp_path):
    source = tmp_path / "bh.csv.gz"
    fields = [
        "sample_id",
        "source_row_id",
        "source_index",
        "raw_reaction_smiles",
        "yield_pct",
        "yield_measurement_type",
        "substrate",
        "amine",
        "product",
        "catalyst",
        "catalyst_name",
        "base",
        "base_name",
        "solvent",
        "solvent_name",
        "source",
        "provenance",
        "labels",
    ]
    with gzip.open(source, "wt", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id, value in (("valid", "0"), ("bad", "101")):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "source_row_id": sample_id,
                    "yield_pct": value,
                    "yield_measurement_type": "hte",
                    "substrate": "BrC",
                    "amine": "N",
                    "product": "CN",
                    "source": "test",
                }
            )

    output = tmp_path / "bh-output.csv.gz"
    report = build_buchwald_hartwig_ood(
        source, output, expected_sha256=None
    )
    assert report["raw_rows"] == 2
    assert report["accepted_rows"] == 1
    assert report["quarantined_rows"] == 1
    assert report["quarantine_reasons"] == {"yield_outside_0_100": 1}
    assert report["zero_yields"] == 1
    row = _rows(output)[0]
    assert row["sample_id"] == "valid"
    assert row["reactants"] == "BrC.N"
    assert ";" not in row["reaction_smiles"]


def test_suzuki_builder_normalizes_workbook_and_marks_unknown_license(
    tmp_path,
):
    openpyxl = __import__("openpyxl")
    source = tmp_path / "suzuki.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(
        [
            "Reaction_No",
            "Reactant_1_Name",
            "Reactant_1_Short_Hand",
            "Reactant_1_eq",
            "Reactant_1_mmol",
            "Reactant_2_Name",
            "Reactant_2_eq",
            "Catalyst_1_Short_Hand",
            "Catalyst_1_eq",
            "Ligand_Short_Hand",
            "Ligand_eq",
            "Reagent_1_Short_Hand",
            "Reagent_1_eq",
            "Solvent_1_Short_Hand",
            "Product_Yield_PCT_Area_UV",
            "Product_Yield_Mass_Ion_Count",
        ]
    )
    sheet.append(
        [
            1,
            "6-chloroquinoline",
            "1a, 6-Cl-Q",
            1,
            0.0004,
            "2a, Boronic Acid",
            1,
            "Pd(OAc)2",
            0.0625,
            "P(tBu)3",
            0.125,
            "NaOH",
            2.5,
            "MeCN",
            42.5,
            6262,
        ]
    )
    workbook.save(source)

    output = tmp_path / "suzuki.csv.gz"
    report = build_suzuki_miyaura(source, output, expected_sha256=None)
    row = _rows(output)[0]

    assert report["accepted_rows"] == 1
    assert report["quarantined_rows"] == 0
    assert report["redistributable"] is False
    assert row["sample_id"] == "suzuki_miyaura:1"
    assert row["license_spdx"] == "LicenseRef-Unknown"
    assert row["yield_pct"] == "42.5"
    assert row["reaction_smiles"].count(">") == 2
    assert row["canonical_reaction_smiles"].count(">>") == 1
