"""Deterministic builders for impurity-candidate datasets."""

from __future__ import annotations

import csv
import gzip
import io
import json
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from synrxn.reconstruction.families.outcomes.yield_conversion._normalize import _canonical_unmapped_side, _hash, sha256_path


SEISMIQ_SOURCE_SHA256 = (
    "4063cb914b07e7ab3239a3be878cfd1debf0dfa2a94d1bef3fb0aa9700b88fd5"
)
SEISMIQ_SOURCE_DOI = "10.5281/zenodo.16438770"

IMPURITY_COLUMNS = (
    "sample_id",
    "dataset",
    "source_record_id",
    "source_index",
    "query_id",
    "raw_reaction_smiles",
    "reaction_smiles",
    "canonical_reaction_smiles",
    "starting_materials",
    "impurity_candidate",
    "candidate_rank",
    "candidate_status",
    "structure_valid",
    "observed",
    "experimental",
    "upstream_split",
    "synrxn_split",
    "source",
    "citation",
    "license_spdx",
    "license_url",
    "provenance",
    "duplicate_group",
    "chemical_split_group",
    "source_metadata_json",
)


@contextmanager
def deterministic_impurity_csv(path: Path) -> Iterator[csv.DictWriter]:
    """Write the fixed impurity schema with reproducible gzip metadata."""
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
                        fieldnames=IMPURITY_COLUMNS,
                        extrasaction="ignore",
                        lineterminator="\n",
                    )
                    writer.writeheader()
                    yield writer
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _quiet_rdkit_errors() -> Iterator[None]:
    """Suppress expected parse errors while retaining validity flags."""
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.error")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.error")


def build_seismiq_simulated_impurities(
    source_path: Path,
    output_path: Path,
    *,
    expected_sha256: str | None = SEISMIQ_SOURCE_SHA256,
) -> dict[str, Any]:
    """Flatten SEISMiQ's released predicted-impurity example lists.

    These rows are candidate enumerations, not observed impurity labels. The
    distinction is retained explicitly in every output row.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    source_sha256 = sha256_path(source_path)
    if expected_sha256 and source_sha256 != expected_sha256:
        raise ValueError("SEISMiQ source checksum mismatch")

    raw_rows = accepted = quarantined = 0
    invalid_structures = 0
    query_counts: dict[str, int] = {}
    quarantine_reasons: dict[str, int] = {}
    identifiers: set[str] = set()
    duplicate_groups: set[str] = set()
    chemical_groups: set[str] = set()

    with zipfile.ZipFile(source_path) as archive, _quiet_rdkit_errors():
        members = sorted(
            member
            for member in archive.namelist()
            if member.startswith("simulated_impurities/example")
            and member.endswith(".json")
        )
        if not members:
            raise ValueError("SEISMiQ archive has no example JSON files")
        with deterministic_impurity_csv(output_path) as writer:
            source_index = 0
            for member in members:
                query_id = Path(member).stem
                payload = json.loads(archive.read(member))
                if not isinstance(payload, list) or len(payload) != 1:
                    raise ValueError(f"unexpected SEISMiQ payload: {member}")
                record = payload[0]
                starting = record.get("Starting Materials")
                candidates = record.get("Predicted Impurities")
                if not isinstance(starting, list) or not isinstance(
                    candidates, list
                ):
                    raise ValueError(f"invalid SEISMiQ fields: {member}")
                canonical_starting = _canonical_unmapped_side(
                    ".".join(str(value) for value in starting)
                )
                query_counts[query_id] = 0
                for rank, candidate in enumerate(candidates, start=1):
                    raw_rows += 1
                    try:
                        canonical_candidate = _canonical_unmapped_side(
                            str(candidate)
                        )
                        structure_valid = "true"
                    except (TypeError, ValueError):
                        canonical_candidate = str(candidate).strip()
                        structure_valid = "false"
                        invalid_structures += 1
                    canonical_reaction = (
                        f"{canonical_starting}>>{canonical_candidate}"
                    )
                    record_id = f"{query_id}:{rank}"
                    if record_id in identifiers:
                        quarantine_reasons["duplicate source identifier"] = (
                            quarantine_reasons.get(
                                "duplicate source identifier", 0
                            )
                            + 1
                        )
                        quarantined += 1
                        continue
                    chemical_group = _hash("query", canonical_starting)
                    duplicate_group = _hash(
                        "impurity_candidate", canonical_reaction
                    )
                    writer.writerow(
                        {
                            "sample_id": f"seismiq_simulated:{record_id}",
                            "dataset": "seismiq_simulated",
                            "source_record_id": record_id,
                            "source_index": source_index,
                            "query_id": query_id,
                            "raw_reaction_smiles": (
                                f"{'.'.join(map(str, starting))}>>{candidate}"
                            ),
                            "reaction_smiles": canonical_reaction,
                            "canonical_reaction_smiles": canonical_reaction,
                            "starting_materials": canonical_starting,
                            "impurity_candidate": canonical_candidate,
                            "candidate_rank": rank,
                            "candidate_status": "predicted_candidate",
                            "structure_valid": structure_valid,
                            "observed": "",
                            "experimental": "false",
                            "upstream_split": "reference",
                            "synrxn_split": "reference",
                            "source": "SEISMiQ simulated impurities",
                            "citation": SEISMIQ_SOURCE_DOI,
                            "license_spdx": "CC-BY-4.0",
                            "license_url": (
                                "https://creativecommons.org/licenses/by/4.0/"
                            ),
                            "provenance": (
                                f"Zenodo {SEISMIQ_SOURCE_DOI}; "
                                "simulated_impurities.zip"
                            ),
                            "duplicate_group": duplicate_group,
                            "chemical_split_group": chemical_group,
                            "source_metadata_json": json.dumps(
                                {
                                    "archive_member": member,
                                    "candidate_is_ground_truth": False,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        }
                    )
                    source_index += 1
                    accepted += 1
                    query_counts[query_id] += 1
                    identifiers.add(record_id)
                    duplicate_groups.add(duplicate_group)
                    chemical_groups.add(chemical_group)

    if expected_sha256 and raw_rows != 2541:
        raise ValueError(
            f"SEISMiQ candidate count mismatch: {raw_rows} != 2541"
        )
    return {
        "dataset": "seismiq_simulated",
        "source_file": source_path.name,
        "source_doi": SEISMIQ_SOURCE_DOI,
        "source_sha256": source_sha256,
        "output_file": output_path.name,
        "output_sha256": sha256_path(output_path),
        "raw_rows": raw_rows,
        "accepted_rows": accepted,
        "quarantined_rows": quarantined,
        "query_counts": query_counts,
        "split_counts": {"reference": accepted},
        "unique_source_record_ids": len(identifiers),
        "unique_duplicate_groups": len(duplicate_groups),
        "unique_chemical_split_groups": len(chemical_groups),
        "quarantine_reasons": dict(sorted(quarantine_reasons.items())),
        "invalid_structure_rows_retained": invalid_structures,
        "candidate_status": "predicted_candidate_not_ground_truth",
        "license_spdx": "CC-BY-4.0",
    }
