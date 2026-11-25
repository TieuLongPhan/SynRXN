"""Deterministic builders for reaction-feasibility benchmarks."""

from __future__ import annotations

import csv
import gzip
import io
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from synrxn.reconstruction.families.outcomes.yield_conversion._normalize import _hash, _reaction_parts, canonical_unmapped_reaction, sha256_path


URSA_SOURCE_REVISION = "6733b1149e99b0986f8b627c4d0c3df5efb16852"
URSA_SOURCE_SHA256 = (
    "052ca46969e3a17927b9e6ba148e318d700d67dc7eacde64011187fa3e95e652"
)

FEASIBILITY_COLUMNS = (
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
    "feasible",
    "expert_score",
    "plausibility_score",
    "plausibility_score_binary",
    "label_method",
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
def deterministic_feasibility_csv(path: Path) -> Iterator[csv.DictWriter]:
    """Write the fixed feasibility schema with reproducible gzip metadata."""
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
                        fieldnames=FEASIBILITY_COLUMNS,
                        extrasaction="ignore",
                        lineterminator="\n",
                    )
                    writer.writeheader()
                    yield writer
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_ursa_plausibility(
    source_path: Path,
    output_path: Path,
    *,
    expected_sha256: str | None = URSA_SOURCE_SHA256,
) -> dict[str, Any]:
    """Normalize the expert-labelled URSA reaction-plausibility benchmark."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    source_sha256 = sha256_path(source_path)
    if expected_sha256 and source_sha256 != expected_sha256:
        raise ValueError("URSA source checksum mismatch")

    raw_rows = accepted = quarantined = 0
    label_counts = {"0": 0, "1": 0}
    quarantine_reasons: dict[str, int] = {}
    identifiers: set[str] = set()
    duplicate_groups: set[str] = set()
    chemical_groups: set[str] = set()

    with source_path.open("r", encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "reaction_smiles",
            "expert_score",
            "chemcensor_score",
            "chemcensor_score_bin",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"URSA columns missing: {sorted(missing)}")
        with deterministic_feasibility_csv(output_path) as writer:
            for source_index, record in enumerate(reader):
                raw_rows += 1
                try:
                    raw_reaction = str(record["reaction_smiles"]).strip()
                    expert_score = str(record["expert_score"]).strip()
                    plausibility_binary = str(
                        record["chemcensor_score_bin"]
                    ).strip()
                    if expert_score not in {"0", "1"}:
                        raise ValueError("invalid expert label")
                    if plausibility_binary not in {"0", "1"}:
                        raise ValueError("invalid model label")
                    plausibility_score = float(record["chemcensor_score"])
                    if plausibility_score < -1 or plausibility_score > 5:
                        raise ValueError("plausibility score outside -1-5")
                    canonical, reactants, products = (
                        canonical_unmapped_reaction(raw_reaction)
                    )
                    _, reagents, _ = _reaction_parts(raw_reaction)
                    record_id = str(source_index)
                    if record_id in identifiers:
                        raise ValueError("duplicate source identifier")
                except (TypeError, ValueError) as exc:
                    reason = str(exc) or "invalid record"
                    quarantine_reasons[reason] = (
                        quarantine_reasons.get(reason, 0) + 1
                    )
                    quarantined += 1
                    continue

                chemical_group = _hash("reaction", canonical)
                duplicate_group = _hash(
                    "feasibility", f"{canonical}|{expert_score}"
                )
                row = {
                    "sample_id": f"ursa_plausibility:{record_id}",
                    "dataset": "ursa_plausibility",
                    "source_record_id": record_id,
                    "source_index": source_index,
                    "raw_reaction_smiles": raw_reaction,
                    "reaction_smiles": raw_reaction,
                    "canonical_reaction_smiles": canonical,
                    "reactants": reactants,
                    "reagents": reagents,
                    "products": products,
                    "feasible": expert_score,
                    "expert_score": expert_score,
                    "plausibility_score": format(plausibility_score, ".12g"),
                    "plausibility_score_binary": plausibility_binary,
                    "label_method": (
                        "binary judgement by expert organic chemists"
                    ),
                    "experimental": "false",
                    "upstream_split": "test",
                    "synrxn_split": "test",
                    "source": "URSA benchmarking sets",
                    "citation": "arXiv:2602.03554",
                    "license_spdx": "Apache-2.0",
                    "license_url": (
                        "https://www.apache.org/licenses/LICENSE-2.0"
                    ),
                    "provenance": (
                        "insilicomedicine/URSA-benchmarking-sets@"
                        f"{URSA_SOURCE_REVISION}"
                    ),
                    "duplicate_group": duplicate_group,
                    "chemical_split_group": chemical_group,
                    "source_metadata_json": json.dumps(
                        {"chemcensor_score_version": "1.2.0"},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
                writer.writerow(row)
                accepted += 1
                label_counts[expert_score] += 1
                identifiers.add(record_id)
                duplicate_groups.add(duplicate_group)
                chemical_groups.add(chemical_group)

    if expected_sha256 and raw_rows != 1000:
        raise ValueError(f"URSA raw row count mismatch: {raw_rows} != 1000")
    if expected_sha256 and label_counts != {"0": 500, "1": 500}:
        raise ValueError(f"URSA label balance mismatch: {label_counts}")
    return {
        "dataset": "ursa_plausibility",
        "source_file": source_path.name,
        "source_revision": URSA_SOURCE_REVISION,
        "source_sha256": source_sha256,
        "output_file": output_path.name,
        "output_sha256": sha256_path(output_path),
        "raw_rows": raw_rows,
        "accepted_rows": accepted,
        "quarantined_rows": quarantined,
        "label_counts": label_counts,
        "split_counts": {"test": accepted},
        "unique_source_record_ids": len(identifiers),
        "unique_duplicate_groups": len(duplicate_groups),
        "unique_chemical_split_groups": len(chemical_groups),
        "quarantine_reasons": dict(sorted(quarantine_reasons.items())),
        "license_spdx": "Apache-2.0",
    }
