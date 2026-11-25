"""Deterministic builders for SynRXN reaction-condition datasets."""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import re
from functools import lru_cache
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

from synrxn.reconstruction.families.outcomes.yield_conversion._normalize import _canonical_unmapped_side, _clean, _hash, sha256_path

CONDITION_COLUMNS = (
    "sample_id",
    "dataset",
    "source_record_id",
    "source_index",
    "raw_reaction_smiles",
    "reaction_smiles",
    "canonical_reaction_smiles",
    "reactants",
    "products",
    "agents",
    "reagents",
    "catalysts",
    "ligands",
    "bases",
    "additives",
    "agent_1",
    "agent_2",
    "agent_3",
    "solvents",
    "solvent_1",
    "solvent_2",
    "temperature_c",
    "temperature_plausible",
    "time_hours",
    "pressure_bar",
    "atmosphere",
    "stirring",
    "flow",
    "photochemistry",
    "electrochemistry",
    "reported_yield",
    "yield_valid_0_100",
    "condition_completeness",
    "condition_mask_json",
    "conditions_json",
    "action_sequence",
    "action_score",
    "procedure_details",
    "date_of_experiment",
    "grant_date",
    "source_dataset_id",
    "upstream_split",
    "synrxn_split",
    "source",
    "citation",
    "license_spdx",
    "license_url",
    "provenance",
    "duplicate_group",
    "chemical_split_group",
    "condition_group",
    "source_metadata_json",
)


ORD_HTE_ALLOWLIST = {
    "ord_dataset-805ad863feef48579d95d86a728035f4": 50688,
    "ord_dataset-47eaacc46c3a4487bbdf99adb1a15e41": 47015,
    "ord_dataset-5c9a10329a8a48968d18879a48bb8ab2": 9632,
    "ord_dataset-1ec2807f02fa4beda27d2a81b86eb843": 1227,
    "ord_dataset-2be11f57f3304e678ea8469baf6dd1bc": 1152,
    "ord_dataset-3b8a2ef300e145468579027f206a3ac8": 1440,
    "ord_dataset-c5b00523487a4211a194160edf45e9ab": 1430,
    "ord_dataset-ac78456835404910b3a4c840248b6ac9": 1728,
    "ord_dataset-b440f8c90b6343189093770060fc4098": 1152,
    "ord_dataset-7acd6ad2bf4d4cff841cad008ab726d5": 960,
    "ord_dataset-675eddcaa6674ce3ae61e79bbc1e1c08": 1536,
    "ord_dataset-7d8f5fd922d4497d91cb81489b052746": 1536,
    "ord_dataset-9b8aa9a7835143ef8ce3f70abfab7545": 1762,
    "ord_dataset-3b5db90e337942ea886b8f5bc5e3aa72": 450,
    "ord_dataset-eeba974d3c284aed86d1c1d442260a1e": 376,
    "ord_dataset-d26118acda314269becc35db5c22dc59": 1728,
}

ORD_DATA_COMMIT = "05fe1a62cb36d6bc3c07592ac1bd4ff1a4d8e43c"
ORD_HTE_SHA256 = dict(
    zip(
        ORD_HTE_ALLOWLIST,
        (
            "31b7c4762145e5dad973050373cd967bdd85caf3a9858767c83a17fed6c4af36",
            "7e09c98aa540e15d44959f03028590278bf508650950778568fd264d3a7535c4",
            "a2ea1363cb895b590ae765e918f98f90d243d6caba763ed74543ae62e503b0b0",
            "b1fd81bb3034644117b7d07e0471ede7370597bb4bd2433dd4b5c20bf02fad99",
            "c7d17ab4af3551a4b2883c6e18c1f3a49b6c19d30d37fad762802aeeb275d948",
            "f8438f4490541ecd38d92208decc48ff33424d52772cc42701b7eb886fc14b8a",
            "73c04ce2acaf6cf2b6b3ecda8c623d2a321435f1f5830aedb0a0f1afd562bdc4",
            "52a9d0084abc051a792bfb71501af03ec13bdabc24a7db4e33b7223ddf8c884c",
            "d289e2e9fbb77df61faadeb65cfadfbee0216b05a7d209b3ae3b3aca04a9993a",
            "b6dfe8e151a396c5332668c015bce764d23c667240e443495990205c66e4a1e5",
            "1ecc5469785e55997f244eb5f8d6dbc78029cbb233bd6a140b5f02d5d01860a1",
            "3d657d1eda5703b58b627565894890269b50b18e987709e70fd7968d83e59101",
            "195765643b4aaad79d9ed82de6737a4855b9f2716e0bc0aa10cff950b7604d39",
            "fce2d70c28db7100b3ca5c3f298484daf4c4043c11a18da066db16ad2e36012c",
            "dbf6dae9521463c542fa8abccd2a078742c8052c0af11db842d200d94403fac6",
            "0a600968a24eff0a440e30685b2cb7b62030693605574105eb57b7f07329700b",
        ),
        strict=True,
    )
)
OPENEXP_SHA256 = (
    "00eee069440638130d8341175a2ce65e284b8d1cb43261e1e7e543ff94f3fe1c"
)

ORDERLY_TRAIN_SHA256 = (
    "ac77033d91faa606d6cbe53d520d718b39d90ad4d708e7fab131bf656c520a25"
)
ORDERLY_TEST_SHA256 = (
    "d9b2b8a34960db23904c06965113969c051565eb7379973f40a544f592897c55"
)

ORDERLY_COLUMNS = (
    "original_index",
    "agent_000",
    "agent_001",
    "agent_002",
    "date_of_experiment",
    "extracted_from_file",
    "grant_date",
    "is_mapped",
    "procedure_details",
    "product_000",
    "reactant_000",
    "reactant_001",
    "rxn_str",
    "rxn_time",
    "solvent_000",
    "solvent_001",
    "temperature",
    "yield_000",
)


def _optional_number(value: Any) -> str:
    """Format a finite optional numeric value without imputing missing data."""
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return format(number, ".12g")


def _date(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        if isinstance(value, datetime):
            return value.date().isoformat()
        return value.isoformat()
    text = _clean(value)
    return text[:10] if text else ""


def _json(values: Mapping[str, Any]) -> str:
    normalized = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (datetime, date)):
            value = _date(value)
        text = _clean(value)
        if text:
            normalized[key] = text
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _canonical_slot(value: Any) -> str:
    text = _clean(value)
    return _canonical_unmapped_side(text) if text else ""


@lru_cache(maxsize=32768)
def _cached_canonical_slot(text: str) -> str:
    try:
        return _canonical_unmapped_side(text)
    except (TypeError, ValueError):
        return text


def _safe_canonical_slot(value: Any) -> str:
    """Canonicalize supported structures and preserve the others."""
    text = _clean(value)
    return _cached_canonical_slot(text) if text else ""


def _combine_slots(*values: str) -> str:
    """Return an order-invariant dot-separated molecular set."""
    components = []
    for value in values:
        if value:
            components.extend(value.split("."))
    return ".".join(sorted(set(components)))


@contextmanager
def deterministic_condition_csv(path: Path) -> Iterator[csv.DictWriter]:
    """Write a reproducible gzip CSV using the condition schema."""
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
                        fieldnames=CONDITION_COLUMNS,
                        lineterminator="\n",
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                    yield writer
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _quiet_rdkit_warnings() -> Iterator[None]:
    """Suppress noisy normalization warnings while retaining RDKit errors."""
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.warning")
        RDLogger.EnableLog("rdApp.error")


def _iter_parquet(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pyarrow is required to process ORDerly parquet"
        ) from exc

    parquet = pq.ParquetFile(path)
    missing = set(ORDERLY_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"ORDerly columns missing: {sorted(missing)}")
    batches = parquet.iter_batches(batch_size=8192, columns=ORDERLY_COLUMNS)
    for batch in batches:
        yield from batch.to_pylist()


def build_orderly_condition(
    train_path: Path,
    test_path: Path,
    output_path: Path,
    *,
    expected_train_sha256: str | None = ORDERLY_TRAIN_SHA256,
    expected_test_sha256: str | None = ORDERLY_TEST_SHA256,
) -> dict[str, Any]:
    """Normalize the CC BY 4.0 ORDerly-condition v4 benchmark."""
    train_path = Path(train_path)
    test_path = Path(test_path)
    output_path = Path(output_path)
    train_sha256 = sha256_path(train_path)
    test_sha256 = sha256_path(test_path)
    if expected_train_sha256 and train_sha256 != expected_train_sha256:
        raise ValueError("ORDerly-condition train checksum mismatch")
    if expected_test_sha256 and test_sha256 != expected_test_sha256:
        raise ValueError("ORDerly-condition test checksum mismatch")

    raw_rows = accepted = quarantined = 0
    split_counts = {"train": 0, "test": 0}
    missing_counts = {
        "agent_1": 0,
        "agent_2": 0,
        "agent_3": 0,
        "solvent_1": 0,
        "solvent_2": 0,
        "temperature_c": 0,
        "time_hours": 0,
        "reported_yield": 0,
    }
    quality_counts = {
        "temperature_outside_minus196_500": 0,
        "yield_outside_0_100": 0,
    }
    quarantine_reasons: dict[str, int] = {}
    identifiers: set[str] = set()
    duplicates: set[str] = set()
    chemical_groups: set[str] = set()
    condition_groups: set[str] = set()

    output_context = deterministic_condition_csv(output_path)
    with _quiet_rdkit_warnings(), output_context as writer:
        for split, source_path in (("train", train_path), ("test", test_path)):
            for source_index, row in enumerate(_iter_parquet(source_path)):
                raw_rows += 1
                try:
                    record_id = _clean(row.get("original_index"))
                    if not record_id:
                        raise ValueError("original_index is missing")
                    if record_id in identifiers:
                        raise ValueError("duplicate original_index")

                    reactant_1 = _canonical_slot(row.get("reactant_000"))
                    reactant_2 = _canonical_slot(row.get("reactant_001"))
                    product = _canonical_slot(row.get("product_000"))
                    if not reactant_1 or not product:
                        raise ValueError("reactant or product is missing")
                    reactants = _combine_slots(reactant_1, reactant_2)
                    products = product

                    agent_1 = _canonical_slot(row.get("agent_000"))
                    agent_2 = _canonical_slot(row.get("agent_001"))
                    agent_3 = _canonical_slot(row.get("agent_002"))
                    solvent_1 = _canonical_slot(row.get("solvent_000"))
                    solvent_2 = _canonical_slot(row.get("solvent_001"))
                    agents = _combine_slots(agent_1, agent_2, agent_3)
                    solvents = _combine_slots(solvent_1, solvent_2)
                    reagents = _combine_slots(agents, solvents)
                    reaction = f"{reactants}>{reagents}>{products}"
                    transformation = f"{reactants}>>{products}"

                    temperature = _optional_number(row.get("temperature"))
                    time_hours = _optional_number(row.get("rxn_time"))
                    reported_yield = _optional_number(row.get("yield_000"))
                    temperature_number = (
                        float(temperature) if temperature else None
                    )
                    yield_number = (
                        float(reported_yield) if reported_yield else None
                    )
                    temperature_plausible = (
                        ""
                        if temperature_number is None
                        else str(-196 <= temperature_number <= 500).lower()
                    )
                    yield_valid = (
                        ""
                        if yield_number is None
                        else str(0 <= yield_number <= 100).lower()
                    )
                    if temperature_plausible == "false":
                        quality_counts["temperature_outside_minus196_500"] += 1
                    if yield_valid == "false":
                        quality_counts["yield_outside_0_100"] += 1

                    condition_values = {
                        "agent_1": agent_1,
                        "agent_2": agent_2,
                        "agent_3": agent_3,
                        "solvent_1": solvent_1,
                        "solvent_2": solvent_2,
                        "temperature_c": temperature,
                        "time_hours": time_hours,
                    }
                    mask = dict(
                        (key, bool(value))
                        for key, value in condition_values.items()
                    )
                    completeness = format(
                        sum(mask.values()) / len(mask), ".6f"
                    )
                    condition_text = json.dumps(
                        condition_values, sort_keys=True, separators=(",", ":")
                    )
                    condition_group = _hash("condition", condition_text)
                    duplicate_group = _hash(
                        "experiment", f"{reaction}|{condition_text}"
                    )
                    chemical_group = _hash("reaction", transformation)
                    source_dataset_id = _clean(row.get("extracted_from_file"))
                    output = {
                        "sample_id": f"orderly_condition:{record_id}",
                        "dataset": "orderly_condition",
                        "source_record_id": record_id,
                        "source_index": source_index,
                        "raw_reaction_smiles": _clean(row.get("rxn_str")),
                        "reaction_smiles": reaction,
                        "canonical_reaction_smiles": transformation,
                        "reactants": reactants,
                        "products": products,
                        "agents": agents,
                        "agent_1": agent_1,
                        "agent_2": agent_2,
                        "agent_3": agent_3,
                        "solvents": solvents,
                        "solvent_1": solvent_1,
                        "solvent_2": solvent_2,
                        "temperature_c": temperature,
                        "temperature_plausible": temperature_plausible,
                        "time_hours": time_hours,
                        "reported_yield": reported_yield,
                        "yield_valid_0_100": yield_valid,
                        "condition_completeness": completeness,
                        "condition_mask_json": json.dumps(
                            mask, sort_keys=True, separators=(",", ":")
                        ),
                        "procedure_details": _clean(
                            row.get("procedure_details")
                        ),
                        "date_of_experiment": _date(
                            row.get("date_of_experiment")
                        ),
                        "grant_date": _date(row.get("grant_date")),
                        "source_dataset_id": source_dataset_id,
                        "upstream_split": split,
                        "source": "ORDerly-condition v4",
                        "citation": "10.1021/acs.jcim.4c00292",
                        "license_spdx": "CC-BY-4.0",
                        "license_url": (
                            "https://creativecommons.org/licenses/by/4.0/"
                        ),
                        "provenance": (
                            "10.6084/m9.figshare.23298467.v4; "
                            f"{source_path.name}"
                        ),
                        "duplicate_group": duplicate_group,
                        "chemical_split_group": chemical_group,
                        "condition_group": condition_group,
                        "source_metadata_json": _json(
                            {
                                "is_mapped": row.get("is_mapped"),
                                "extracted_from_file": source_dataset_id,
                            }
                        ),
                    }
                except (TypeError, ValueError) as exc:
                    quarantined += 1
                    reason = (
                        "duplicate_source_record_id"
                        if str(exc) == "duplicate original_index"
                        else "invalid_structure_or_required_field"
                    )
                    quarantine_reasons[reason] = (
                        quarantine_reasons.get(reason, 0) + 1
                    )
                    continue

                writer.writerow(output)
                accepted += 1
                split_counts[split] += 1
                identifiers.add(record_id)
                duplicates.add(duplicate_group)
                chemical_groups.add(chemical_group)
                condition_groups.add(condition_group)
                for key, value in {
                    **condition_values,
                    "reported_yield": reported_yield,
                }.items():
                    missing_counts[key] += not bool(value)

    return {
        "dataset": "orderly_condition",
        "source_files": [train_path.name, test_path.name],
        "source_sha256": {
            "train": train_sha256,
            "test": test_sha256,
        },
        "output_file": output_path.name,
        "output_sha256": sha256_path(output_path),
        "raw_rows": raw_rows,
        "accepted_rows": accepted,
        "quarantined_rows": quarantined,
        "split_counts": split_counts,
        "missing_counts": missing_counts,
        "quality_counts": quality_counts,
        "quarantine_reasons": dict(sorted(quarantine_reasons.items())),
        "unique_source_record_ids": len(identifiers),
        "unique_duplicate_groups": len(duplicates),
        "unique_chemical_split_groups": len(chemical_groups),
        "unique_condition_groups": len(condition_groups),
        "license_spdx": "CC-BY-4.0",
    }


def _stable_split(group: str) -> str:
    """Assign a deterministic, group-safe 80/10/10 split."""
    bucket = int(_hash("split", group).split(":", 1)[1][:8], 16) % 10
    if bucket < 8:
        return "train"
    return "validation" if bucket == 8 else "test"


def _quantity_value(quantity: Mapping[str, Any], target: str) -> str:
    """Convert common ORD temperature, time, and pressure units."""
    if not quantity or "value" not in quantity:
        return ""
    value = float(quantity["value"])
    units = _clean(quantity.get("units")).upper()
    if target == "temperature":
        if units == "KELVIN":
            value -= 273.15
        elif units == "FAHRENHEIT":
            value = (value - 32) * 5 / 9
        elif units not in {"CELSIUS", ""}:
            return ""
    elif target == "time":
        factors = {
            "SECOND": 1 / 3600,
            "MINUTE": 1 / 60,
            "HOUR": 1,
            "DAY": 24,
        }
        if units not in factors:
            return ""
        value *= factors[units]
    elif target == "pressure":
        factors = {
            "BAR": 1,
            "PASCAL": 1e-5,
            "KILOPASCAL": 0.01,
            "MEGAPASCAL": 10,
            "PSI": 0.0689475729,
            "TORR": 0.00133322368,
        }
        if units not in factors:
            return ""
        value *= factors[units]
    return _optional_number(value)


def _identifier_value(identifiers: list[Mapping[str, Any]], kind: str) -> str:
    for identifier in identifiers:
        if _clean(identifier.get("type")).upper() == kind:
            return _clean(identifier.get("value"))
    return ""


def _ord_role(input_name: str, component: Mapping[str, Any]) -> str:
    role = _clean(component.get("reaction_role")).upper()
    name = input_name.lower()
    if role not in {"REACTANT", "SOLVENT", "PRODUCT"}:
        for category in ("ligand", "base", "additive"):
            if category in name:
                return category
    if role == "REAGENT":
        for category in ("catalyst",):
            if category in name:
                return category
    if role:
        return role.lower()
    for category in ("catalyst", "ligand", "base", "additive", "solvent"):
        if category in name:
            return category
    return "reagent"


def _ord_product_and_yield(
    outcomes: list[Mapping[str, Any]],
) -> tuple[list[str], str, str]:
    products: list[str] = []
    reported_yield = ""
    time_hours = ""
    for outcome in outcomes:
        if not time_hours:
            time_hours = _quantity_value(
                outcome.get("reaction_time", {}), "time"
            )
        for product in outcome.get("products", []):
            if product.get("is_desired_product") is False:
                continue
            smiles = _identifier_value(
                product.get("identifiers", []), "SMILES"
            )
            if smiles:
                products.append(_safe_canonical_slot(smiles))
            if not reported_yield:
                for measurement in product.get("measurements", []):
                    if _clean(measurement.get("type")).upper() != "YIELD":
                        continue
                    percentage = measurement.get("percentage", {})
                    reported_yield = _optional_number(percentage.get("value"))
                    if reported_yield:
                        break
    return products, reported_yield, time_hours


def _ord_row(
    reaction_id: str,
    reaction: Mapping[str, Any],
    dataset_id: str,
    source_index: int,
) -> dict[str, Any]:
    roles: dict[str, list[str]] = {
        "reactant": [],
        "solvent": [],
        "reagent": [],
        "catalyst": [],
        "ligand": [],
        "base": [],
        "additive": [],
    }
    for input_name, reaction_input in reaction.get("inputs", {}).items():
        for component in reaction_input.get("components", []):
            smiles = _identifier_value(
                component.get("identifiers", []), "SMILES"
            )
            if not smiles:
                continue
            role = _ord_role(input_name, component)
            if role in {"product", "internal_standard", "workup"}:
                continue
            if role not in roles:
                role = "reagent"
            roles[role].append(_safe_canonical_slot(smiles))

    products_list, reported_yield, time_hours = _ord_product_and_yield(
        reaction.get("outcomes", [])
    )
    reactants = _combine_slots(*roles["reactant"])
    products = _combine_slots(*products_list)
    if not reactants or not products:
        raise ValueError("reactant or product is missing")

    catalysts = _combine_slots(*roles["catalyst"])
    ligands = _combine_slots(*roles["ligand"])
    bases = _combine_slots(*roles["base"])
    additives = _combine_slots(*roles["additive"])
    reagents = _combine_slots(*roles["reagent"])
    solvents = _combine_slots(*roles["solvent"])
    agents = _combine_slots(catalysts, ligands, bases, additives, reagents)
    middle = _combine_slots(agents, solvents)
    transformation = f"{reactants}>>{products}"
    reaction_smiles = f"{reactants}>{middle}>{products}"

    conditions = reaction.get("conditions", {})
    temperature = _quantity_value(
        conditions.get("temperature", {}).get("setpoint", {}), "temperature"
    )
    pressure = _quantity_value(
        conditions.get("pressure", {}).get("setpoint", {}), "pressure"
    )
    atmosphere = _clean(
        conditions.get("pressure", {}).get("atmosphere", {}).get("type")
    )
    stirring = _clean(conditions.get("stirring", {}).get("type"))
    flow = _clean(conditions.get("flow", {}).get("type"))
    photochemistry = _clean(conditions.get("illumination", {}).get("type"))
    electrochemistry = "true" if conditions.get("electrochemistry") else ""
    temperature_number = float(temperature) if temperature else None
    yield_number = float(reported_yield) if reported_yield else None

    condition_values = {
        "catalysts": catalysts,
        "ligands": ligands,
        "bases": bases,
        "additives": additives,
        "reagents": reagents,
        "solvents": solvents,
        "temperature_c": temperature,
        "time_hours": time_hours,
    }
    mask = {key: bool(value) for key, value in condition_values.items()}
    conditions_text = json.dumps(
        conditions, sort_keys=True, separators=(",", ":")
    )
    condition_text = json.dumps(
        condition_values, sort_keys=True, separators=(",", ":")
    )
    chemical_group = _hash("reaction", transformation)
    identifiers = reaction.get("identifiers", [])
    raw_reaction = _identifier_value(identifiers, "REACTION_SMILES")
    reaction_type = _identifier_value(identifiers, "REACTION_TYPE")
    provenance = reaction.get("provenance", {})
    return {
        "sample_id": f"ord_hte_condition:{reaction_id}",
        "dataset": "ord_hte_condition",
        "source_record_id": reaction_id,
        "source_index": source_index,
        "raw_reaction_smiles": raw_reaction,
        "reaction_smiles": reaction_smiles,
        "canonical_reaction_smiles": transformation,
        "reactants": reactants,
        "products": products,
        "agents": agents,
        "reagents": reagents,
        "catalysts": catalysts,
        "ligands": ligands,
        "bases": bases,
        "additives": additives,
        "agent_1": catalysts,
        "agent_2": reagents,
        "agent_3": _combine_slots(ligands, bases, additives),
        "solvents": solvents,
        "solvent_1": solvents,
        "temperature_c": temperature,
        "temperature_plausible": (
            ""
            if temperature_number is None
            else str(-196 <= temperature_number <= 500).lower()
        ),
        "time_hours": time_hours,
        "pressure_bar": pressure,
        "atmosphere": atmosphere,
        "stirring": stirring,
        "flow": flow,
        "photochemistry": photochemistry,
        "electrochemistry": electrochemistry,
        "reported_yield": reported_yield,
        "yield_valid_0_100": (
            ""
            if yield_number is None
            else str(0 <= yield_number <= 100).lower()
        ),
        "condition_completeness": format(
            sum(mask.values()) / len(mask), ".6f"
        ),
        "condition_mask_json": json.dumps(
            mask, sort_keys=True, separators=(",", ":")
        ),
        "conditions_json": conditions_text,
        "procedure_details": _clean(conditions.get("details")),
        "source_dataset_id": dataset_id,
        "synrxn_split": _stable_split(chemical_group),
        "source": "Open Reaction Database",
        "citation": "10.1021/jacs.1c09820",
        "license_spdx": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "provenance": (
            f"ord-data@{ORD_DATA_COMMIT}; "
            f"Hugging Face parquet config {dataset_id}"
        ),
        "duplicate_group": _hash(
            "experiment", f"{reaction_smiles}|{condition_text}"
        ),
        "chemical_split_group": chemical_group,
        "condition_group": _hash("condition", condition_text),
        "source_metadata_json": json.dumps(
            {
                "identifiers": identifiers,
                "provenance": provenance,
                "reaction_type": reaction_type,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def build_ord_hte_condition(
    source_paths: list[Path],
    output_path: Path,
    *,
    expected_sha256: Mapping[str, str] | None = ORD_HTE_SHA256,
) -> dict[str, Any]:
    """Build the non-overlapping, CC-BY-SA ORD condition allowlist."""
    try:
        import pyarrow.parquet as pq
        from google.protobuf.json_format import MessageToDict
        from ord_schema.proto import reaction_pb2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pyarrow and ord-schema>=0.8.3 are required for ORD condition data"
        ) from exc

    paths_by_id: dict[str, Path] = {}
    source_hashes: dict[str, str] = {}
    for source_path in map(Path, source_paths):
        metadata = pq.read_metadata(source_path).metadata or {}
        dataset_id = metadata.get(b"ord.dataset_id", b"").decode()
        if dataset_id not in ORD_HTE_ALLOWLIST:
            raise ValueError(
                f"ORD dataset is not allowlisted: {dataset_id or source_path}"
            )
        if dataset_id in paths_by_id:
            raise ValueError(f"duplicate ORD dataset: {dataset_id}")
        digest = sha256_path(source_path)
        if expected_sha256 and digest != expected_sha256.get(dataset_id):
            raise ValueError(f"ORD checksum mismatch: {dataset_id}")
        paths_by_id[dataset_id] = source_path
        source_hashes[dataset_id] = digest
    missing = set(ORD_HTE_ALLOWLIST) - set(paths_by_id)
    if missing:
        raise ValueError(
            f"missing allowlisted ORD datasets: {sorted(missing)}"
        )

    raw_rows = accepted = quarantined = 0
    dataset_counts: dict[str, int] = {}
    dataset_raw_counts: dict[str, int] = {}
    split_counts = {"train": 0, "validation": 0, "test": 0}
    quarantine_reasons: dict[str, int] = {}
    identifiers: set[str] = set()
    with (
        _quiet_rdkit_warnings(),
        deterministic_condition_csv(output_path) as writer,
    ):
        for dataset_id in ORD_HTE_ALLOWLIST:
            table = pq.ParquetFile(paths_by_id[dataset_id])
            count = 0
            raw_count = 0
            for batch in table.iter_batches(
                batch_size=4096, columns=["reaction_id", "reaction"]
            ):
                batch_rows = batch.to_pylist()
                for source_index, source_row in enumerate(
                    batch_rows, start=raw_count
                ):
                    raw_rows += 1
                    raw_count += 1
                    reaction_id = _clean(source_row.get("reaction_id"))
                    try:
                        if not reaction_id or reaction_id in identifiers:
                            raise ValueError(
                                "missing or duplicate reaction_id"
                            )
                        message = reaction_pb2.Reaction.FromString(
                            source_row["reaction"]
                        )
                        reaction = MessageToDict(
                            message, preserving_proto_field_name=True
                        )
                        row = _ord_row(
                            reaction_id, reaction, dataset_id, source_index
                        )
                    except (TypeError, ValueError) as exc:
                        reason = (
                            "missing_or_duplicate_id"
                            if "reaction_id" in str(exc)
                            else "invalid_structure_or_required_field"
                        )
                        quarantine_reasons[reason] = (
                            quarantine_reasons.get(reason, 0) + 1
                        )
                        quarantined += 1
                        continue
                    writer.writerow(row)
                    identifiers.add(reaction_id)
                    accepted += 1
                    count += 1
                    split_counts[row["synrxn_split"]] += 1
            dataset_counts[dataset_id] = count
            dataset_raw_counts[dataset_id] = raw_count
            if raw_count != ORD_HTE_ALLOWLIST[dataset_id]:
                raise ValueError(
                    f"ORD raw row count mismatch for {dataset_id}: "
                    f"{raw_count} != {ORD_HTE_ALLOWLIST[dataset_id]}"
                )

    return {
        "dataset": "ord_hte_condition",
        "source_commit": ORD_DATA_COMMIT,
        "source_sha256": source_hashes,
        "output_file": Path(output_path).name,
        "output_sha256": sha256_path(output_path),
        "raw_rows": raw_rows,
        "accepted_rows": accepted,
        "quarantined_rows": quarantined,
        "dataset_counts": dataset_counts,
        "dataset_raw_counts": dataset_raw_counts,
        "split_counts": split_counts,
        "quarantine_reasons": dict(sorted(quarantine_reasons.items())),
        "license_spdx": "CC-BY-SA-4.0",
    }


_TEMPERATURE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?([CFK])?", re.I)
_DURATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)",
    re.I,
)


def _openexp_temperature(values: Mapping[str, Any]) -> str:
    for text in values:
        match = _TEMPERATURE_RE.search(text)
        if not match:
            continue
        value = float(match.group(1))
        unit = (match.group(2) or "C").upper()
        if unit == "F":
            value = (value - 32) * 5 / 9
        elif unit == "K":
            value -= 273.15
        return _optional_number(value)
    return ""


def _openexp_duration(values: Mapping[str, Any]) -> str:
    for text in values:
        match = _DURATION_RE.search(text)
        if not match:
            continue
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith(("second", "sec")):
            value /= 3600
        elif unit.startswith(("minute", "min")):
            value /= 60
        elif unit.startswith("day"):
            value *= 24
        return _optional_number(value)
    return ""


def build_openexp_condition(
    source_path: Path,
    output_path: Path,
    *,
    expected_sha256: str | None = OPENEXP_SHA256,
) -> dict[str, Any]:
    """Normalize OpenExp and retain structured and source procedures."""
    source_path = Path(source_path)
    digest = sha256_path(source_path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("OpenExp archive checksum mismatch")
    import tarfile

    with tarfile.open(source_path) as archive:
        member = archive.getmember("OpenExp.json")
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError("OpenExp.json is missing from archive")
        records = json.load(io.TextIOWrapper(handle, encoding="utf8"))

    raw_rows = accepted = quarantined = 0
    split_counts = {"train": 0, "validation": 0, "test": 0}
    quarantine_reasons: dict[str, int] = {}
    identifiers: set[str] = set()
    with (
        _quiet_rdkit_warnings(),
        deterministic_condition_csv(output_path) as writer,
    ):
        for source_index, record in enumerate(records):
            raw_rows += 1
            try:
                record_id = _clean(record.get("index"))
                if not record_id or record_id in identifiers:
                    raise ValueError("missing or duplicate index")
                reactants = _combine_slots(
                    *(
                        _safe_canonical_slot(x)
                        for x in record.get("REACTANT", [])
                    )
                )
                products = _combine_slots(
                    *(
                        _safe_canonical_slot(x)
                        for x in record.get("PRODUCT", [])
                    )
                )
                catalysts = _combine_slots(
                    *(
                        _safe_canonical_slot(x)
                        for x in record.get("CATALYST", [])
                    )
                )
                solvents = _combine_slots(
                    *(
                        _safe_canonical_slot(x)
                        for x in record.get("SOLVENT", [])
                    )
                )
                if not reactants or not products:
                    raise ValueError("reactant or product is missing")
                middle = _combine_slots(catalysts, solvents)
                transformation = f"{reactants}>>{products}"
                reaction_smiles = f"{reactants}>{middle}>{products}"
                temperature = _openexp_temperature(
                    record.get("extracted_temperature", {})
                )
                time_hours = _openexp_duration(
                    record.get("extracted_duration", {})
                )
                temperature_number = (
                    float(temperature) if temperature else None
                )
                chemical_group = _hash("reaction", transformation)
                condition_values = {
                    "catalysts": catalysts,
                    "solvents": solvents,
                    "temperature_c": temperature,
                    "time_hours": time_hours,
                }
                mask = {
                    key: bool(value) for key, value in condition_values.items()
                }
                condition_text = json.dumps(
                    condition_values, sort_keys=True, separators=(",", ":")
                )
                row = {
                    "sample_id": f"openexp:{record_id}",
                    "dataset": "openexp",
                    "source_record_id": record_id,
                    "source_index": source_index,
                    "reaction_smiles": reaction_smiles,
                    "canonical_reaction_smiles": transformation,
                    "reactants": reactants,
                    "products": products,
                    "agents": catalysts,
                    "catalysts": catalysts,
                    "agent_1": catalysts,
                    "solvents": solvents,
                    "solvent_1": solvents,
                    "temperature_c": temperature,
                    "temperature_plausible": (
                        ""
                        if temperature_number is None
                        else str(-196 <= temperature_number <= 500).lower()
                    ),
                    "time_hours": time_hours,
                    "condition_completeness": format(
                        sum(mask.values()) / len(mask), ".6f"
                    ),
                    "condition_mask_json": json.dumps(
                        mask, sort_keys=True, separators=(",", ":")
                    ),
                    "conditions_json": json.dumps(
                        {
                            "extracted_temperature": record.get(
                                "extracted_temperature", {}
                            ),
                            "extracted_duration": record.get(
                                "extracted_duration", {}
                            ),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "action_sequence": _clean(record.get("actions")),
                    "action_score": _optional_number(record.get("score")),
                    "procedure_details": _clean(record.get("source")),
                    "source_dataset_id": "OpenExp.tar",
                    "synrxn_split": _stable_split(chemical_group),
                    "source": "OpenExp",
                    "citation": "10.18653/v1/2024.findings-acl.318",
                    "license_spdx": "CC-BY-SA-4.0",
                    "license_url": (
                        "https://creativecommons.org/licenses/by-sa/4.0/"
                    ),
                    "provenance": (
                        "OSF file 3dv4k; OpenExp.tar v1 (2024-06-03)"
                    ),
                    "duplicate_group": _hash(
                        "experiment", f"{reaction_smiles}|{condition_text}"
                    ),
                    "chemical_split_group": chemical_group,
                    "condition_group": _hash("condition", condition_text),
                    "source_metadata_json": json.dumps(
                        {
                            "extracted_molecules": record.get(
                                "extracted_molecules", {}
                            ),
                            "molecules": record.get("molecules", {}),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            except (TypeError, ValueError):
                quarantined += 1
                quarantine_reasons["invalid_structure_or_required_field"] = (
                    quarantine_reasons.get(
                        "invalid_structure_or_required_field", 0
                    )
                    + 1
                )
                continue
            writer.writerow(row)
            identifiers.add(record_id)
            accepted += 1
            split_counts[row["synrxn_split"]] += 1

    return {
        "dataset": "openexp",
        "source_file": source_path.name,
        "source_sha256": digest,
        "output_file": Path(output_path).name,
        "output_sha256": sha256_path(output_path),
        "raw_rows": raw_rows,
        "accepted_rows": accepted,
        "quarantined_rows": quarantined,
        "split_counts": split_counts,
        "quarantine_reasons": dict(sorted(quarantine_reasons.items())),
        "license_spdx": "CC-BY-SA-4.0",
        "split_note": (
            "SynRXN deterministic transformation-group 80/10/10 split; "
            "not the unreleased random paper split"
        ),
    }
