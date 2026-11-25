"""Deterministic builders for experimental reaction-selectivity datasets."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


SELECTIVITY_COLUMNS = (
    "sample_id",
    "dataset",
    "source_record_id",
    "source_index",
    "raw_reaction_smiles",
    "reaction_smiles",
    "canonical_reaction_smiles",
    "substrate_smiles",
    "canonical_substrate_smiles",
    "reactants",
    "products",
    "candidate_site_indices",
    "observed_site_indices",
    "observed_site_count",
    "site_index_basis",
    "selectivity_value",
    "selectivity_type",
    "selectivity_unit",
    "selectivity_sign_convention",
    "stereochemistry_complete",
    "reported_yield_pct",
    "measurement_method",
    "temperature_c",
    "time_hours",
    "solvent",
    "reagents",
    "reaction_class",
    "screen_id",
    "reference",
    "citation",
    "license_spdx",
    "license_url",
    "provenance",
    "duplicate_group",
    "chemical_split_group",
    "conditions_json",
    "source_metadata_json",
)

HITEA_SOURCE_SHA256 = (
    "9d3b35b8606538df7fc8dd9104b6dd43188ddc8dc43e3d9d3972e10a621ef889"
)
REGIOSQM_SOURCE_SHA256 = {
    "compounds_smiles.csv": (
        "256b852c51685f6baa50288cf5acdcebdd67629640ea2c72087e0c50ed452585"
    ),
    "compounds_doi.csv": (
        "48b170949032b5279a1ec3a57ab0d0bd5a970fe057c8ec4eb04b8366cefc0cd1"
    ),
    "compounds_yield.csv": (
        "92d80f6813aeb8bde18685e019b0d63fd8addefc3c2589cfe813635d9ac5ad87"
    ),
}


def sha256_path(path: Path) -> str:
    """Return the SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _json(values: Mapping[str, Any]) -> str:
    cleaned = {key: _clean(value) for key, value in values.items() if _clean(value)}
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"))


def _hash(prefix: str, value: str) -> str:
    return f"{prefix}:{hashlib.sha256(value.encode('utf8')).hexdigest()}"


def _canonical_smiles(smiles: str) -> str:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(_clean(smiles))
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"invalid molecular SMILES: {smiles!r}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _canonical_reaction(reaction: str) -> tuple[str, str, str]:
    if reaction.count(">>") != 1:
        raise ValueError("expected one reaction separator")
    raw_reactants, raw_products = reaction.split(">>")
    if not raw_reactants or not raw_products:
        raise ValueError("reaction has an empty side")
    reactants = ".".join(
        sorted(_canonical_smiles(component) for component in raw_reactants.split("."))
    )
    products = ".".join(
        sorted(_canonical_smiles(component) for component in raw_products.split("."))
    )
    canonical = f"{reactants}>>{products}"
    return canonical, reactants, products


@contextmanager
def deterministic_csv(path: Path) -> Iterator[csv.DictWriter]:
    """Write a reproducible gzip CSV with a fixed timestamp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0
            ) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf8", newline="") as text:
                    writer = csv.DictWriter(
                        text,
                        fieldnames=SELECTIVITY_COLUMNS,
                        lineterminator="\n",
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                    yield writer
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _report(
    *,
    dataset: str,
    source: Path,
    output: Path,
    raw_rows: int,
    task_candidate_rows: int,
    accepted_rows: int,
    quarantined_rows: int,
    quarantine_reasons: Mapping[str, int],
    observed_labels: int,
    identifiers: set[str],
    duplicate_groups: set[str],
    chemical_groups: set[str],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "source_file": Path(source).name,
        "source_sha256": sha256_path(source) if Path(source).is_file() else "",
        "output_file": Path(output).name,
        "output_sha256": sha256_path(output),
        "output_bytes": Path(output).stat().st_size,
        "raw_rows": raw_rows,
        "task_candidate_rows": task_candidate_rows,
        "excluded_non_task_rows": raw_rows - task_candidate_rows,
        "accepted_rows": accepted_rows,
        "quarantined_rows": quarantined_rows,
        "quarantine_reasons": dict(sorted(quarantine_reasons.items())),
        "observed_labels": observed_labels,
        "unique_source_records": len(identifiers),
        "unique_duplicate_groups": len(duplicate_groups),
        "unique_chemical_groups": len(chemical_groups),
    }


def build_hitea_chiral_resolution(
    source_path: Path,
    output_path: Path,
    *,
    expected_sha256: str | None = HITEA_SOURCE_SHA256,
) -> dict[str, Any]:
    """Extract measured product ee from HiTEA chiral-salt resolutions.

    The supplementary methods define ``Product_Selectivity`` for these rows as
    product percent ee measured by chiral SFC-MS. Rows with zero in the
    upstream product-yield field are not treated as measured zero-ee results.
    """
    source_path = Path(source_path)
    if expected_sha256 and sha256_path(source_path) != expected_sha256:
        raise ValueError("HiTEA source checksum mismatch")

    raw_rows = candidate_rows = accepted = quarantined = observed_labels = 0
    reasons: Counter[str] = Counter()
    identifiers: set[str] = set()
    duplicate_groups: set[str] = set()
    chemical_groups: set[str] = set()

    with source_path.open("r", encoding="utf8", newline="") as source:
        reader = csv.DictReader(source)
        with deterministic_csv(output_path) as writer:
            for source_index, row in enumerate(reader):
                raw_rows += 1
                if _clean(row.get("KeyWord_STD")).upper() != "CHIRAL SALT RESOLUTION":
                    continue
                candidate_rows += 1
                try:
                    record_id = _clean(row.get("REACTION_ID"))
                    if not record_id:
                        raise ValueError("missing_identifier")
                    reported_yield = _number(row.get("Product_Yield_PCT_Area_UV"))
                    if not reported_yield or float(reported_yield) <= 0:
                        raise ValueError("zero_reported_product_yield")
                    ee = _number(row.get("Product_Selectivity"))
                    if not ee:
                        raise ValueError("missing_product_ee")
                    if not -100 <= float(ee) <= 100:
                        raise ValueError("product_ee_outside_minus100_100")

                    substrate_raw = _clean(row.get("PRODUCT_STRUCTURE"))
                    substrate = _canonical_smiles(substrate_raw)
                    raw_reaction = _clean(row.get("RXN_SMILES"))
                    reaction = reactants = ""
                    products = substrate
                    try:
                        reaction, reactants, products = _canonical_reaction(raw_reaction)
                    except ValueError:
                        # Some upstream resolution records omit the racemic input
                        # structure. The measured product structure is still retained.
                        pass

                    reagents = ";".join(
                        value
                        for value in (
                            _clean(row.get("Reagent_1_Short_Hand")),
                            _clean(row.get("Reagent_2_Short_Hand")),
                        )
                        if value
                    )
                    temperature = _number(row.get("Reaction_T"))
                    time_hours = _number(row.get("Reaction_Time_hrs"))
                    solvent = _clean(row.get("Solvent_1_Name"))
                    conditions = _json(
                        {
                            "solvent": solvent,
                            "reagent_1": row.get("Reagent_1_Short_Hand"),
                            "reagent_2": row.get("Reagent_2_Short_Hand"),
                            "temperature_c": temperature,
                            "time_hours": time_hours,
                        }
                    )
                    duplicate_group = _hash(
                        "experiment", f"{substrate}|{conditions}|{ee}|{reported_yield}"
                    )
                    chemical_group = _hash("substrate", substrate)
                    writer.writerow(
                        {
                            "sample_id": f"hitea_chiral_resolution:{record_id}",
                            "dataset": "hitea_chiral_resolution",
                            "source_record_id": record_id,
                            "source_index": source_index,
                            "raw_reaction_smiles": raw_reaction,
                            "reaction_smiles": reaction,
                            "canonical_reaction_smiles": reaction,
                            "substrate_smiles": substrate_raw,
                            "canonical_substrate_smiles": substrate,
                            "reactants": reactants,
                            "products": products,
                            "selectivity_value": ee,
                            "selectivity_type": "enantiomeric_excess",
                            "selectivity_unit": "percent_ee",
                            "selectivity_sign_convention": (
                                "upstream_signed_product_ee_favored_enantiomer_undefined"
                            ),
                            "stereochemistry_complete": "false",
                            "reported_yield_pct": reported_yield,
                            "measurement_method": "chiral SFC-MS",
                            "temperature_c": temperature,
                            "time_hours": time_hours,
                            "solvent": solvent,
                            "reagents": reagents,
                            "reaction_class": "chiral salt resolution",
                            "screen_id": _clean(row.get("SCREEN_ID")),
                            "reference": "10.1038/s41557-023-01393-w",
                            "citation": "King-Smith et al., Nature Chemistry (2024)",
                            "license_spdx": "MIT",
                            "license_url": (
                                "https://github.com/emmaking-smith/HiTEA/blob/"
                                "ccc8cce274e70429498be829dd3bc57754f62265/LICENSE"
                            ),
                            "provenance": (
                                "HiTEA 8_SEPT_APPROVED_full_dataset.csv at commit "
                                "ccc8cce274e70429498be829dd3bc57754f62265"
                            ),
                            "duplicate_group": duplicate_group,
                            "chemical_split_group": chemical_group,
                            "conditions_json": conditions,
                            "source_metadata_json": _json(
                                {
                                    "dataset_entry_number": row.get(
                                        "Dataset entry number"
                                    ),
                                    "notebook_id": row.get("NOTEBOOK_ID"),
                                    "product_id": row.get("PCAT_CMPD_ID"),
                                    "product_ee_raw": row.get("Product_Selectivity"),
                                    "reported_yield_raw": row.get(
                                        "Product_Yield_PCT_Area_UV"
                                    ),
                                }
                            ),
                        }
                    )
                except (TypeError, ValueError) as exc:
                    quarantined += 1
                    reason = str(exc) or "invalid_record"
                    reasons[reason] += 1
                    continue

                accepted += 1
                observed_labels += 1
                identifiers.add(record_id)
                duplicate_groups.add(duplicate_group)
                chemical_groups.add(chemical_group)

    return _report(
        dataset="hitea_chiral_resolution",
        source=source_path,
        output=output_path,
        raw_rows=raw_rows,
        task_candidate_rows=candidate_rows,
        accepted_rows=accepted,
        quarantined_rows=quarantined,
        quarantine_reasons=reasons,
        observed_labels=observed_labels,
        identifiers=identifiers,
        duplicate_groups=duplicate_groups,
        chemical_groups=chemical_groups,
    )


def _space_table(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf8").splitlines():
        cleaned = line.replace("\u00a0", " ").strip()
        if not cleaned:
            continue
        identifier, value = cleaned.split(maxsplit=1)
        if identifier in values:
            raise ValueError(f"duplicate source identifier: {identifier}")
        values[identifier] = value.strip()
    return values


def build_regiosqm_eas(
    source_dir: Path,
    output_path: Path,
    *,
    expected_sha256: Mapping[str, str] | None = REGIOSQM_SOURCE_SHA256,
) -> dict[str, Any]:
    """Normalize literature-observed EAS sites from db-regioselectivity."""
    from rdkit import Chem

    source_dir = Path(source_dir)
    paths = {name: source_dir / name for name in REGIOSQM_SOURCE_SHA256}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if expected_sha256 and sha256_path(path) != expected_sha256[name]:
            raise ValueError(f"RegioSQM source checksum mismatch: {name}")

    smiles_rows: list[tuple[str, str, list[int], str]] = []
    for line in paths["compounds_smiles.csv"].read_text(encoding="utf8").splitlines():
        cleaned = line.replace("\u00a0", " ").strip()
        if not cleaned:
            continue
        identifier, smiles, raw_sites = cleaned.split(maxsplit=2)
        sites = [int(value) for value in re.findall(r"\d+", raw_sites)]
        if not sites:
            raise ValueError(f"missing observed site: {identifier}")
        smiles_rows.append((identifier, smiles, sites, raw_sites))

    with paths["compounds_doi.csv"].open(
        "r", encoding="utf8", newline=""
    ) as handle:
        references = {
            _clean(row.get("Compound")): _clean(row.get("Reference"))
            for row in csv.DictReader(handle, skipinitialspace=True)
        }
    yields = _space_table(paths["compounds_yield.csv"])
    identifiers_in_smiles = {row[0] for row in smiles_rows}
    if identifiers_in_smiles != set(references) or identifiers_in_smiles != set(yields):
        raise ValueError("RegioSQM source tables contain different identifiers")

    accepted = observed_labels = 0
    reasons: Counter[str] = Counter()
    identifiers: set[str] = set()
    duplicate_groups: set[str] = set()
    chemical_groups: set[str] = set()
    with deterministic_csv(output_path) as writer:
        for source_index, (record_id, raw_smiles, sites, raw_sites) in enumerate(
            smiles_rows
        ):
            try:
                molecule = Chem.MolFromSmiles(raw_smiles)
                if molecule is None or molecule.GetNumAtoms() == 0:
                    raise ValueError("invalid_substrate_smiles")
                candidate_sites = [
                    atom.GetIdx()
                    for atom in molecule.GetAtoms()
                    if atom.GetSymbol() == "C"
                    and atom.GetIsAromatic()
                    and atom.GetTotalNumHs() > 0
                ]
                if not set(sites).issubset(candidate_sites):
                    raise ValueError("observed_site_is_not_aromatic_carbon_hydrogen")
                canonical = Chem.MolToSmiles(
                    molecule, canonical=True, isomericSmiles=True
                )
                site_text = ";".join(str(value) for value in sites)
                candidate_text = ";".join(str(value) for value in candidate_sites)
                reference = references[record_id]
                duplicate_group = _hash("site_set", f"{canonical}|{site_text}")
                chemical_group = _hash("substrate", canonical)
                writer.writerow(
                    {
                        "sample_id": f"regiosqm_eas:{record_id}",
                        "dataset": "regiosqm_eas",
                        "source_record_id": record_id,
                        "source_index": source_index,
                        "substrate_smiles": raw_smiles,
                        "canonical_substrate_smiles": canonical,
                        "candidate_site_indices": candidate_text,
                        "observed_site_indices": site_text,
                        "observed_site_count": len(sites),
                        "site_index_basis": (
                            "zero_based_atom_order_in_upstream_substrate_smiles"
                        ),
                        "selectivity_type": "regioselective_site_set",
                        "selectivity_unit": "atom_index",
                        "selectivity_sign_convention": "not_applicable",
                        "measurement_method": (
                            "literature-reported electrophilic aromatic substitution product"
                        ),
                        "reaction_class": "electrophilic aromatic substitution",
                        "reference": reference,
                        "citation": "Kromann et al., Chemical Science (2018)",
                        "license_spdx": "MIT",
                        "license_url": (
                            "https://github.com/jensengroup/db-regioselectivity/blob/"
                            "f734ce480c61608c636c5d4c0e9866f6bcbbb9a9/LICENSE"
                        ),
                        "provenance": (
                            "db-regioselectivity commit "
                            "f734ce480c61608c636c5d4c0e9866f6bcbbb9a9"
                        ),
                        "duplicate_group": duplicate_group,
                        "chemical_split_group": chemical_group,
                        "conditions_json": "{}",
                        "source_metadata_json": _json(
                            {
                                "observed_sites_raw": raw_sites,
                                "reported_yield_and_electrophile_raw": yields[
                                    record_id
                                ],
                                "reference": reference,
                            }
                        ),
                    }
                )
            except (TypeError, ValueError) as exc:
                reasons[str(exc) or "invalid_record"] += 1
                continue
            accepted += 1
            observed_labels += len(sites)
            identifiers.add(record_id)
            duplicate_groups.add(duplicate_group)
            chemical_groups.add(chemical_group)

    quarantined = len(smiles_rows) - accepted
    report = _report(
        dataset="regiosqm_eas",
        source=paths["compounds_smiles.csv"],
        output=output_path,
        raw_rows=len(smiles_rows),
        task_candidate_rows=len(smiles_rows),
        accepted_rows=accepted,
        quarantined_rows=quarantined,
        quarantine_reasons=reasons,
        observed_labels=observed_labels,
        identifiers=identifiers,
        duplicate_groups=duplicate_groups,
        chemical_groups=chemical_groups,
    )
    report["source_files"] = {
        name: {"sha256": sha256_path(path), "bytes": path.stat().st_size}
        for name, path in paths.items()
    }
    return report
