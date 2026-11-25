"""Deterministic builders for SynRXN reaction-yield datasets."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import shutil
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


YIELD_COLUMNS = (
    "sample_id",
    "dataset",
    "source_record_id",
    "source_index",
    "raw_reaction_smiles",
    "reaction_smiles",
    "canonical_reaction_smiles",
    "reactants",
    "reagents",
    "products",
    "yield_pct",
    "yield_measurement_type",
    "measurement_method",
    "temperature_c",
    "time_hours",
    "solvent",
    "catalyst",
    "base",
    "selectivity",
    "mass_ion_count",
    "reaction_class",
    "screen_id",
    "notebook_id",
    "upstream_split",
    "source",
    "citation",
    "license_spdx",
    "license_url",
    "provenance",
    "duplicate_group",
    "chemical_split_group",
    "conditions_json",
    "source_metadata_json",
)

BH_NORMALIZED_SHA256 = (
    "623a0b58f4155d7438e8515e0075b3cbc776bf8567de8dd80d59f2c2027519aa"
)
USPTO_ARCHIVE_SHA256 = (
    "e83c1a7b80e1855a45e30b834f02ab8540eadf66caa0b688609d2a9d00bb1783"
)
HITEA_SOURCE_SHA256 = (
    "9d3b35b8606538df7fc8dd9104b6dd43188ddc8dc43e3d9d3972e10a621ef889"
)
SUZUKI_MIYAURA_SOURCE_SHA256 = (
    "a869e020ba31bd5676c67a4791c3b7384711b5216de6af444b8cd0a24c284640"
)

SUZUKI_REACTANT_1_SMILES = {
    "6-chloroquinoline": "C1=C(Cl)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC",
    "6-Bromoquinoline": "C1=C(Br)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC",
    "6-triflatequinoline": (
        "C1C2C(=NC=CC=2)C=CC=1OS(C(F)(F)F)(=O)=O.CCC1=CC(=CC=C1)CC"
    ),
    "6-Iodoquinoline": "C1=C(I)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC",
    "6-quinoline-boronic acid hydrochloride": (
        "C1C(B(O)O)=CC=C2N=CC=CC=12.Cl.O"
    ),
    "Potassium quinoline-6-trifluoroborate": (
        "[B-](C1=CC2=C(C=C1)N=CC=C2)(F)(F)F.[K+].O"
    ),
    "6-Quinolineboronic acid pinacol ester": (
        "B1(OC(C(O1)(C)C)(C)C)C2=CC3=C(C=C2)N=CC=C3.O"
    ),
}

SUZUKI_REACTANT_2_SMILES = {
    "2a, Boronic Acid": "CC1=CC=C2C(C=NN2C3OCCCC3)=C1B(O)O",
    "2b, Boronic Ester": (
        "CC1=CC=C2C(C=NN2C3OCCCC3)=C1B4OC(C)(C)C(C)(C)O4"
    ),
    "2c, Trifluoroborate": (
        "CC1=CC=C2C(C=NN2C3OCCCC3)=C1[B-](F)(F)F.[K+]"
    ),
    "2d, Bromide": "CC1=CC=C2C(C=NN2C3OCCCC3)=C1Br",
}

SUZUKI_CATALYST_SMILES = {"Pd(OAc)2": "CC(=O)O~CC(=O)O~[Pd]"}

SUZUKI_LIGAND_SMILES = {
    "": "",
    "P(tBu)3": "CC(C)(C)P(C(C)(C)C)C(C)(C)C",
    "P(Ph)3": "c3c(P(c1ccccc1)c2ccccc2)cccc3",
    "AmPhos": "CC(C)(C)P(C1=CC=C(C=C1)N(C)C)C(C)(C)C",
    "P(Cy)3": "C1(CCCCC1)P(C2CCCCC2)C3CCCCC3",
    "P(o-Tol)3": "CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C",
    "CataCXium A": "CCCCP(C12CC3CC(C1)CC(C3)C2)C45CC6CC(C4)CC(C6)C5",
    "SPhos": "COc1cccc(c1c2ccccc2P(C3CCCCC3)C4CCCCC4)OC",
    "dtbpf": (
        "CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C."
        "CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.[Fe]"
    ),
    "XPhos": (
        "P(c2ccccc2c1c(cc(cc1C(C)C)C(C)C)C(C)C)(C3CCCCC3)C4CCCCC4"
    ),
    "dppf": (
        "C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3."
        "C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.[Fe+2]"
    ),
    "Xantphos": (
        "O6c1c(cccc1P(c2ccccc2)c3ccccc3)"
        "C(c7cccc(P(c4ccccc4)c5ccccc5)c67)(C)C"
    ),
    "None": "",
}

SUZUKI_REAGENT_SMILES = {
    "": "",
    "NaOH": "[OH-].[Na+]",
    "NaHCO3": "[Na+].OC([O-])=O",
    "CsF": "[F-].[Cs+]",
    "K3PO4": "[K+].[K+].[K+].[O-]P([O-])([O-])=O",
    "KOH": "[K+].[OH-]",
    "LiOtBu": "[Li+].[O-]C(C)(C)C",
    "Et3N": "CCN(CC)CC",
    "None": "",
}

SUZUKI_SOLVENT_SMILES = {
    "MeCN": "CC#N.O",
    "THF": "C1CCOC1.O",
    "DMF": "CN(C)C=O.O",
    "MeOH": "CO.O",
    "MeOH/H2O_V2 9:1": "CO.O",
    "THF_V2": "C1CCOC1.O",
}

SUZUKI_PRODUCT_SMILES = (
    "C1=C(C2=C(C)C=CC3N(C4OCCCC4)N=CC2=3)C=CC2=NC=CC=C12"
)

USPTO_MEMBERS = {
    "test": (
        "US_patents_1976-Sep2016_1product_reactions_yield_ok_"
        "cropped_data_test.csv"
    ),
    "train": (
        "US_patents_1976-Sep2016_1product_reactions_yield_ok_"
        "cropped_data_train.csv"
    ),
    "validation": (
        "US_patents_1976-Sep2016_1product_reactions_yield_ok_"
        "cropped_data_valid.csv"
    ),
}


def sha256_path(path: Path) -> str:
    """Return the SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf8")).hexdigest()
    return f"{prefix}:{digest}"


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _number(value: Any) -> str:
    text = _clean(value)
    if not text:
        return ""
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return format(number, ".12g")


def _optional_number(value: Any) -> str:
    """Normalize an optional number, leaving non-numeric source text unset."""
    try:
        return _number(value)
    except (TypeError, ValueError):
        return ""


def _yield(value: Any) -> str:
    text = _number(value)
    if not text:
        raise ValueError("yield is missing")
    number = float(text)
    if number < 0 or number > 100:
        raise ValueError(f"yield outside 0-100: {number}")
    return text


def _json(values: Mapping[str, Any]) -> str:
    cleaned = {key: _clean(value) for key, value in values.items() if _clean(value)}
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"))


def _join(
    values: Iterator[Any] | list[Any] | tuple[Any, ...],
    *,
    separator: str = ";",
) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean(value)
        if cleaned and cleaned not in seen:
            result.append(cleaned)
            seen.add(cleaned)
    return separator.join(result)


def _reaction_parts(reaction: str) -> tuple[str, str, str]:
    """Return reactants, reagents, and products from a reaction SMILES."""
    if reaction.count(">") == 2:
        reactants, reagents, products = reaction.split(">")
    elif reaction.count(">>") == 1:
        reactants, products = reaction.split(">>")
        reagents = ""
    else:
        raise ValueError("reaction does not contain one valid reaction separator")
    if not reactants or not products:
        raise ValueError("reaction has empty reactants or products")
    return reactants, reagents, products


def _canonical_unmapped_side(smiles: str) -> str:
    from rdkit import Chem

    output: list[str] = []
    for component in smiles.split("."):
        molecule = Chem.MolFromSmiles(component)
        if molecule is None:
            raise ValueError(f"invalid molecular SMILES: {component!r}")
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        output.append(
            Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        )
    return ".".join(sorted(output))


def canonical_unmapped_reaction(reaction: str) -> tuple[str, str, str]:
    """Canonicalize a two-part mapped reaction and remove atom maps."""
    if reaction.count(">>") != 1:
        raise ValueError("expected a two-part mapped reaction")
    reactants, products = reaction.split(">>")
    canonical_reactants = _canonical_unmapped_side(reactants)
    canonical_products = _canonical_unmapped_side(products)
    return (
        f"{canonical_reactants}>>{canonical_products}",
        canonical_reactants,
        canonical_products,
    )


@contextmanager
def deterministic_csv(path: Path) -> Iterator[csv.DictWriter]:
    """Write a reproducible gzip-compressed CSV with a fixed timestamp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0
            ) as compressed:
                with io.TextIOWrapper(
                    compressed, encoding="utf8", newline=""
                ) as text:
                    writer = csv.DictWriter(
                        text,
                        fieldnames=YIELD_COLUMNS,
                        lineterminator="\n",
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                    yield writer
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _stats(
    *,
    dataset: str,
    source_path: Path,
    output_path: Path,
    raw_rows: int,
    accepted_rows: int,
    quarantined_rows: int,
    zero_yields: int,
    identifiers: set[str],
    duplicates: set[str],
    chemical_groups: set[str],
    quarantine_reasons: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "source_file": Path(source_path).name,
        "source_sha256": sha256_path(source_path),
        "output_file": Path(output_path).name,
        "output_sha256": sha256_path(output_path),
        "raw_rows": raw_rows,
        "accepted_rows": accepted_rows,
        "quarantined_rows": quarantined_rows,
        "zero_yields": zero_yields,
        "unique_source_record_ids": len(identifiers),
        "unique_duplicate_groups": len(duplicates),
        "unique_chemical_split_groups": len(chemical_groups),
        "quarantine_reasons": dict(sorted((quarantine_reasons or {}).items())),
    }


def build_buchwald_hartwig_ood(
    source_path: Path,
    output_path: Path,
    *,
    expected_sha256: str | None = BH_NORMALIZED_SHA256,
) -> dict[str, Any]:
    """Project the frozen BH OOD table into the common yield schema."""
    source_path = Path(source_path)
    if expected_sha256 and sha256_path(source_path) != expected_sha256:
        raise ValueError("Buchwald-Hartwig OOD source checksum mismatch")

    raw_rows = accepted = quarantined = zero_yields = 0
    identifiers: set[str] = set()
    duplicates: set[str] = set()
    chemical_groups: set[str] = set()
    quarantine_reasons: dict[str, int] = {}
    with gzip.open(source_path, "rt", encoding="utf8", newline="") as source:
        reader = csv.DictReader(source)
        with deterministic_csv(output_path) as writer:
            for row in reader:
                raw_rows += 1
                try:
                    record_id = _clean(row.get("source_row_id"))
                    if not record_id:
                        raise ValueError("source row identifier is missing")
                    yield_pct = _yield(row.get("yield_pct"))
                    reactants = _join(
                        iter((row.get("substrate"), row.get("amine"))),
                        separator=".",
                    )
                    products = _clean(row.get("product"))
                    if not reactants or not products:
                        raise ValueError("reactants or product are missing")
                    reagents = _join(
                        iter(
                            (
                                row.get("catalyst"),
                                row.get("base"),
                                row.get("solvent"),
                            )
                        ),
                        separator=".",
                    )
                    reaction = f"{reactants}>{reagents}>{products}"
                    canonical_reactants = _canonical_unmapped_side(reactants)
                    canonical_products = _canonical_unmapped_side(products)
                    transformation = (
                        f"{canonical_reactants}>>{canonical_products}"
                    )
                    conditions = _json(
                        {
                            "catalyst": row.get("catalyst"),
                            "catalyst_name": row.get("catalyst_name"),
                            "base": row.get("base"),
                            "base_name": row.get("base_name"),
                            "solvent": row.get("solvent"),
                            "solvent_name": row.get("solvent_name"),
                        }
                    )
                    duplicate_group = _hash(
                        "experiment", f"{reaction}|{conditions}"
                    )
                    chemical_group = _hash("reaction", transformation)
                    output = {
                        "sample_id": _clean(row.get("sample_id")),
                        "dataset": "buchwald_hartwig_ood",
                        "source_record_id": record_id,
                        "source_index": _clean(row.get("source_index")),
                        "raw_reaction_smiles": _clean(
                            row.get("raw_reaction_smiles")
                        ),
                        "reaction_smiles": reaction,
                        "canonical_reaction_smiles": transformation,
                        "reactants": reactants,
                        "reagents": reagents,
                        "products": products,
                        "yield_pct": yield_pct,
                        "yield_measurement_type": _clean(
                            row.get("yield_measurement_type")
                        ),
                        "measurement_method": "CAD or isolated yield; harmonized upstream",
                        "solvent": _clean(row.get("solvent")),
                        "catalyst": _clean(row.get("catalyst")),
                        "base": _clean(row.get("base")),
                        "reaction_class": "Buchwald-Hartwig C-N coupling",
                        "source": _clean(row.get("source")),
                        "citation": "10.1038/s43588-026-01017-6",
                        "license_spdx": "Apache-2.0",
                        "license_url": (
                            "https://github.com/schwallergroup/bh-hte-ood/"
                            "blob/a2770413fce269b49a35fb4b2e6471f25632d2ef/"
                            "LICENSE.txt"
                        ),
                        "provenance": _clean(row.get("provenance")),
                        "duplicate_group": duplicate_group,
                        "chemical_split_group": chemical_group,
                        "conditions_json": conditions,
                        "source_metadata_json": _json(
                            {"upstream_labels": row.get("labels")}
                        ),
                    }
                    if not output["sample_id"]:
                        raise ValueError("sample identifier is missing")
                except (TypeError, ValueError) as exc:
                    quarantined += 1
                    reason = (
                        "yield_outside_0_100"
                        if str(exc).startswith("yield outside 0-100")
                        else "invalid_record"
                    )
                    quarantine_reasons[reason] = (
                        quarantine_reasons.get(reason, 0) + 1
                    )
                    continue
                writer.writerow(output)
                accepted += 1
                zero_yields += float(yield_pct) == 0
                identifiers.add(record_id)
                duplicates.add(duplicate_group)
                chemical_groups.add(chemical_group)

    return _stats(
        dataset="buchwald_hartwig_ood",
        source_path=source_path,
        output_path=output_path,
        raw_rows=raw_rows,
        accepted_rows=accepted,
        quarantined_rows=quarantined,
        zero_yields=zero_yields,
        identifiers=identifiers,
        duplicates=duplicates,
        chemical_groups=chemical_groups,
        quarantine_reasons=quarantine_reasons,
    )


def build_uspto_yield(
    archive_path: Path,
    output_path: Path,
    *,
    expected_sha256: str | None = USPTO_ARCHIVE_SHA256,
) -> dict[str, Any]:
    """Normalize the MIT-licensed cropped USPTO yield archive."""
    archive_path = Path(archive_path)
    if expected_sha256 and sha256_path(archive_path) != expected_sha256:
        raise ValueError("USPTO yield archive checksum mismatch")

    raw_rows = accepted = quarantined = zero_yields = 0
    identifiers: set[str] = set()
    duplicates: set[str] = set()
    chemical_groups: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        missing = set(USPTO_MEMBERS.values()) - set(archive.namelist())
        if missing:
            raise ValueError(f"USPTO archive members missing: {sorted(missing)}")
        with deterministic_csv(output_path) as writer:
            for split, member in USPTO_MEMBERS.items():
                with archive.open(member) as binary:
                    with io.TextIOWrapper(
                        binary, encoding="utf-8-sig", newline=""
                    ) as text:
                        reader = csv.DictReader(text, delimiter="\t")
                        for row in reader:
                            raw_rows += 1
                            try:
                                record_id = _clean(row.get("myID"))
                                source_index = _clean(row.get(""))
                                reaction = _clean(row.get("CanonicalizedReaction"))
                                reactants, reagents, products = _reaction_parts(
                                    reaction
                                )
                                yield_pct = _yield(row.get("Yield"))
                                if not record_id:
                                    raise ValueError("source record identifier is missing")
                                transformation = f"{reactants}>>{products}"
                                duplicate_group = _hash("reaction", reaction)
                                chemical_group = _hash("reaction", transformation)
                                output = {
                                    "sample_id": f"uspto_yield:{split}:{record_id}",
                                    "dataset": "uspto_yield",
                                    "source_record_id": record_id,
                                    "source_index": source_index,
                                    "raw_reaction_smiles": _clean(
                                        row.get("OriginalReaction")
                                    ),
                                    "reaction_smiles": reaction,
                                    "canonical_reaction_smiles": reaction,
                                    "reactants": reactants,
                                    "reagents": reagents,
                                    "products": products,
                                    "yield_pct": yield_pct,
                                    "yield_measurement_type": "patent_reported_curated",
                                    "measurement_method": "text-mined yield curated by Minidis",
                                    "upstream_split": split,
                                    "source": "USPTO 1976-Sep2016",
                                    "citation": "10.6084/m9.figshare.14414039.v1",
                                    "license_spdx": "MIT",
                                    "license_url": "https://opensource.org/license/mit",
                                    "provenance": (
                                        "Minidis yield curation of the Schwaller/Lowe "
                                        "USPTO reaction corpus"
                                    ),
                                    "duplicate_group": duplicate_group,
                                    "chemical_split_group": chemical_group,
                                    "conditions_json": "{}",
                                    "source_metadata_json": "{}",
                                }
                            except (TypeError, ValueError):
                                quarantined += 1
                                continue
                            writer.writerow(output)
                            accepted += 1
                            zero_yields += float(yield_pct) == 0
                            identifiers.add(record_id)
                            duplicates.add(duplicate_group)
                            chemical_groups.add(chemical_group)

    return _stats(
        dataset="uspto_yield",
        source_path=archive_path,
        output_path=output_path,
        raw_rows=raw_rows,
        accepted_rows=accepted,
        quarantined_rows=quarantined,
        zero_yields=zero_yields,
        identifiers=identifiers,
        duplicates=duplicates,
        chemical_groups=chemical_groups,
    )


def build_hitea_yield(
    source_path: Path,
    output_path: Path,
    *,
    expected_sha256: str | None = HITEA_SOURCE_SHA256,
) -> dict[str, Any]:
    """Normalize the full MIT-licensed HiTEA reaction table."""
    source_path = Path(source_path)
    if expected_sha256 and sha256_path(source_path) != expected_sha256:
        raise ValueError("HiTEA source checksum mismatch")

    raw_rows = accepted = quarantined = zero_yields = 0
    identifiers: set[str] = set()
    duplicates: set[str] = set()
    chemical_groups: set[str] = set()
    with source_path.open("r", encoding="utf8", newline="") as source:
        reader = csv.DictReader(source)
        with deterministic_csv(output_path) as writer:
            for row in reader:
                raw_rows += 1
                try:
                    record_id = _clean(row.get("REACTION_ID"))
                    if not record_id:
                        raise ValueError("reaction identifier is missing")
                    raw_reaction = _clean(row.get("RXN_SMILES"))
                    reaction, reactants, products = canonical_unmapped_reaction(
                        raw_reaction
                    )
                    yield_pct = _yield(row.get("Product_Yield_PCT_Area_UV"))
                    catalyst = _join(
                        iter(
                            (
                                row.get("catalyst_1_ID_1_SMILES"),
                                row.get("Catalyst_1_ID_2_SMILES"),
                                row.get("Catalyst_2_ID_1_SMILES"),
                                row.get("Catalyst_2_ID_2_SMILES"),
                            )
                        )
                    )
                    reagents = _join(
                        iter(
                            (
                                row.get("Reagent_1_Short_Hand"),
                                row.get("Reagent_2_Short_Hand"),
                            )
                        )
                    )
                    conditions = _json(
                        {
                            "solvent": row.get("Solvent_1_Name"),
                            "catalyst_1": row.get("Catalyst_1_Short_Hand"),
                            "catalyst_1_smiles": row.get(
                                "catalyst_1_ID_1_SMILES"
                            ),
                            "catalyst_2": row.get("Catalyst_2_Short_Hand"),
                            "catalyst_2_smiles": row.get(
                                "Catalyst_2_ID_1_SMILES"
                            ),
                            "reagent_1": row.get("Reagent_1_Short_Hand"),
                            "reagent_2": row.get("Reagent_2_Short_Hand"),
                            "temperature_c": row.get("Reaction_T"),
                            "time_hours": row.get("Reaction_Time_hrs"),
                        }
                    )
                    duplicate_group = _hash(
                        "experiment", f"{reaction}|{conditions}"
                    )
                    chemical_group = _hash("reaction", reaction)
                    output = {
                        "sample_id": f"hitea:{record_id}",
                        "dataset": "hitea",
                        "source_record_id": record_id,
                        "source_index": _clean(row.get("Dataset entry number")),
                        "raw_reaction_smiles": raw_reaction,
                        "reaction_smiles": reaction,
                        "canonical_reaction_smiles": reaction,
                        "reactants": reactants,
                        "reagents": reagents,
                        "products": products,
                        "yield_pct": yield_pct,
                        "yield_measurement_type": "uv_area_assay",
                        "measurement_method": "Product_Yield_PCT_Area_UV",
                        "temperature_c": _number(row.get("Reaction_T")),
                        "time_hours": _number(row.get("Reaction_Time_hrs")),
                        "solvent": _clean(row.get("Solvent_1_Name")),
                        "catalyst": catalyst,
                        "selectivity": _optional_number(
                            row.get("Product_Selectivity")
                        ),
                        "mass_ion_count": _number(
                            row.get("Product_Yield_Mass_Ion_Count")
                        ),
                        "reaction_class": _clean(row.get("ReactionClass")),
                        "screen_id": _clean(row.get("SCREEN_ID")),
                        "notebook_id": _clean(row.get("NOTEBOOK_ID")),
                        "source": "HiTEA",
                        "citation": "HiTEA repository release",
                        "license_spdx": "MIT",
                        "license_url": (
                            "https://github.com/emmaking-smith/HiTEA/blob/"
                            "ccc8cce274e70429498be829dd3bc57754f62265/LICENSE"
                        ),
                        "provenance": "HiTEA 8_SEPT_APPROVED_full_dataset.csv",
                        "duplicate_group": duplicate_group,
                        "chemical_split_group": chemical_group,
                        "conditions_json": conditions,
                        "source_metadata_json": _json(
                            {
                                "keyword": row.get("KeyWord_STD"),
                                "reaction_group": row.get("ReactionGroup"),
                                "product_id": row.get("PCAT_CMPD_ID"),
                                "uv_wavelength_nm": row.get(
                                    "Product_UV_Analysis_Wavelength_nm"
                                ),
                                "selectivity_raw": row.get("Product_Selectivity"),
                            }
                        ),
                    }
                except (TypeError, ValueError):
                    quarantined += 1
                    continue
                writer.writerow(output)
                accepted += 1
                zero_yields += float(yield_pct) == 0
                identifiers.add(record_id)
                duplicates.add(duplicate_group)
                chemical_groups.add(chemical_group)

    return _stats(
        dataset="hitea",
        source_path=source_path,
        output_path=output_path,
        raw_rows=raw_rows,
        accepted_rows=accepted,
        quarantined_rows=quarantined,
        zero_yields=zero_yields,
        identifiers=identifiers,
        duplicates=duplicates,
        chemical_groups=chemical_groups,
    )


def build_suzuki_miyaura(
    source_path: Path,
    output_path: Path,
    *,
    expected_sha256: str | None = SUZUKI_MIYAURA_SOURCE_SHA256,
) -> dict[str, Any]:
    """Normalize the Perera Suzuki--Miyaura supplementary workbook."""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "openpyxl is required to process the Suzuki-Miyaura workbook"
        ) from exc

    source_path = Path(source_path)
    if expected_sha256 and sha256_path(source_path) != expected_sha256:
        raise ValueError("Suzuki-Miyaura source checksum mismatch")

    workbook = load_workbook(source_path, read_only=True, data_only=True)
    try:
        if "Sheet1" not in workbook.sheetnames:
            raise ValueError("Suzuki-Miyaura workbook is missing Sheet1")
        rows = workbook["Sheet1"].iter_rows(values_only=True)
        header = next(rows, None)
        if not header:
            raise ValueError("Suzuki-Miyaura workbook is empty")
        columns = [str(value) if value is not None else "" for value in header]
        required = {
            "Reaction_No",
            "Reactant_1_Name",
            "Reactant_2_Name",
            "Catalyst_1_Short_Hand",
            "Ligand_Short_Hand",
            "Reagent_1_Short_Hand",
            "Solvent_1_Short_Hand",
            "Product_Yield_PCT_Area_UV",
        }
        missing = required - set(columns)
        if missing:
            raise ValueError(
                f"Suzuki-Miyaura workbook columns missing: {sorted(missing)}"
            )

        raw_rows = accepted = quarantined = zero_yields = 0
        identifiers: set[str] = set()
        duplicates: set[str] = set()
        chemical_groups: set[str] = set()
        quarantine_reasons: dict[str, int] = {}
        with deterministic_csv(output_path) as writer:
            for source_index, values in enumerate(rows, start=2):
                raw_rows += 1
                row = {
                    key: value for key, value in zip(columns, values) if key
                }
                try:
                    record_id = _clean(row.get("Reaction_No"))
                    if not record_id:
                        raise ValueError("reaction number is missing")
                    yield_pct = _yield(row.get("Product_Yield_PCT_Area_UV"))

                    reactant_1 = SUZUKI_REACTANT_1_SMILES[
                        _clean(row.get("Reactant_1_Name"))
                    ]
                    reactant_2 = SUZUKI_REACTANT_2_SMILES[
                        _clean(row.get("Reactant_2_Name"))
                    ]
                    catalyst = SUZUKI_CATALYST_SMILES[
                        _clean(row.get("Catalyst_1_Short_Hand"))
                    ]
                    ligand = SUZUKI_LIGAND_SMILES[
                        _clean(row.get("Ligand_Short_Hand"))
                    ]
                    base = SUZUKI_REAGENT_SMILES[
                        _clean(row.get("Reagent_1_Short_Hand"))
                    ]
                    solvent = SUZUKI_SOLVENT_SMILES[
                        _clean(row.get("Solvent_1_Short_Hand"))
                    ]

                    reactants = _canonical_unmapped_side(
                        _join(iter((reactant_1, reactant_2)), separator=".")
                    )
                    reagent_text = _join(
                        iter((catalyst, ligand, base, solvent)), separator="."
                    )
                    reagents = _canonical_unmapped_side(reagent_text)
                    products = _canonical_unmapped_side(SUZUKI_PRODUCT_SMILES)
                    reaction = f"{reactants}>{reagents}>{products}"
                    transformation = f"{reactants}>>{products}"
                    conditions = _json(
                        {
                            "catalyst": row.get("Catalyst_1_Short_Hand"),
                            "catalyst_equiv": row.get("Catalyst_1_eq"),
                            "ligand": row.get("Ligand_Short_Hand"),
                            "ligand_equiv": row.get("Ligand_eq"),
                            "base": row.get("Reagent_1_Short_Hand"),
                            "base_equiv": row.get("Reagent_1_eq"),
                            "solvent": row.get("Solvent_1_Short_Hand"),
                        }
                    )
                    duplicate_group = _hash(
                        "experiment", f"{reaction}|{conditions}"
                    )
                    chemical_group = _hash("reaction", transformation)
                    output = {
                        "sample_id": f"suzuki_miyaura:{record_id}",
                        "dataset": "suzuki_miyaura",
                        "source_record_id": record_id,
                        "source_index": str(source_index - 2),
                        "raw_reaction_smiles": "",
                        "reaction_smiles": reaction,
                        "canonical_reaction_smiles": transformation,
                        "reactants": reactants,
                        "reagents": reagents,
                        "products": products,
                        "yield_pct": yield_pct,
                        "yield_measurement_type": "uv_area_assay",
                        "measurement_method": "Product_Yield_PCT_Area_UV",
                        "solvent": _canonical_unmapped_side(solvent),
                        "catalyst": _canonical_unmapped_side(
                            _join(iter((catalyst, ligand)), separator=".")
                        ),
                        "base": _canonical_unmapped_side(base),
                        "mass_ion_count": _optional_number(
                            row.get("Product_Yield_Mass_Ion_Count")
                        ),
                        "reaction_class": "Suzuki-Miyaura C-C coupling",
                        "source": "Perera et al. Science 2018 Data File S1",
                        "citation": "10.1126/science.aap9112",
                        "license_spdx": "LicenseRef-Unknown",
                        "license_url": "",
                        "provenance": (
                            "Structures reconstructed from source labels "
                            "using the rxn4chemistry/rxn_yields mappings"
                        ),
                        "duplicate_group": duplicate_group,
                        "chemical_split_group": chemical_group,
                        "conditions_json": conditions,
                        "source_metadata_json": _json(
                            {
                                "reactant_1_name": row.get("Reactant_1_Name"),
                                "reactant_1_short_hand": row.get(
                                    "Reactant_1_Short_Hand"
                                ),
                                "reactant_1_equiv": row.get("Reactant_1_eq"),
                                "reactant_1_mmol": row.get("Reactant_1_mmol"),
                                "reactant_2_name": row.get("Reactant_2_Name"),
                                "reactant_2_equiv": row.get("Reactant_2_eq"),
                            }
                        ),
                    }
                except (KeyError, TypeError, ValueError) as exc:
                    quarantined += 1
                    reason = (
                        "yield_outside_0_100"
                        if str(exc).startswith("yield outside 0-100")
                        else "invalid_record"
                    )
                    quarantine_reasons[reason] = (
                        quarantine_reasons.get(reason, 0) + 1
                    )
                    continue
                writer.writerow(output)
                accepted += 1
                zero_yields += float(yield_pct) == 0
                identifiers.add(record_id)
                duplicates.add(duplicate_group)
                chemical_groups.add(chemical_group)
    finally:
        workbook.close()

    report = _stats(
        dataset="suzuki_miyaura",
        source_path=source_path,
        output_path=output_path,
        raw_rows=raw_rows,
        accepted_rows=accepted,
        quarantined_rows=quarantined,
        zero_yields=zero_yields,
        identifiers=identifiers,
        duplicates=duplicates,
        chemical_groups=chemical_groups,
        quarantine_reasons=quarantine_reasons,
    )
    report["release_status"] = "pending_permission"
    report["redistributable"] = False
    return report


def copy_if_identical(source: Path, destination: Path) -> None:
    """Copy an artifact without rewriting it when source and destination match."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_path(source) == sha256_path(destination):
        return
    shutil.copyfile(source, destination)
