"""Read-only evidence checks for a seven-task database readiness audit.

Field screening and the presence of saved metrics do not establish benchmark
eligibility, scientific correctness, or independently reproducible experiments.
"""

from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

from synrxn.reconstruction.paths import physical_path

NUMERIC_TARGETS = {
    "yield_pct",
    "ea",
    "G_act",
    "dh",
    "G_r",
    "lograte",
    "Conversion",
    "temperature_c",
    "time_hours",
    "pressure_bar",
}
REACTION_FIELDS = ("reaction_smiles", "rxn", "aam", "reactions", "reaction", "original_reactions")


def stable_bytes(path: Path) -> tuple[bytes, str]:
    """Reject evidence changed while it is being read."""
    path = physical_path(Path(path).expanduser().resolve())
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"evidence changed during read: {path}")
    return data, hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    path = physical_path(Path(path).expanduser().resolve())
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"evidence changed during hashing: {path}")
    return digest.hexdigest()


def present(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in {"", "nan", "none", "null", "[]", "{}"}


def target_state(value: str | None, target: str) -> str:
    if not present(value):
        return "missing"
    if target not in NUMERIC_TARGETS:
        return "present"
    try:
        number = float(value)
    except (ValueError, TypeError):
        return "not_numeric"
    if not math.isfinite(number):
        return "nonfinite"
    if target == "yield_pct" and not 0 <= number <= 100:
        return "out_of_range"
    return "present"


def matches_selector(record: dict, selector: dict) -> bool:
    key = record["key"]
    if "key" in selector and key != selector["key"]:
        return False
    if "prefix" in selector and not key.startswith(selector["prefix"]):
        return False
    if key in selector.get("exclude", []):
        return False
    required = selector.get("require_targets")
    return not required or bool(set(required).intersection(record.get("targets", [])))


def select_tables(records: list[dict], task: dict) -> list[dict]:
    return [record for record in records if any(matches_selector(record, s) for s in task["selectors"])]


def inspect_table(path: Path, metadata: dict, targets: list[str], progress=None) -> dict:
    """Stream a table once, reporting screens rather than evaluation-ready datasets."""
    path = physical_path(Path(path).expanduser().resolve())
    digest = file_digest(path)
    before = path.stat()
    opener = gzip.open if path.suffix == ".gz" else open
    csv.field_size_limit(16 * 1024 * 1024)
    states = {target: Counter() for target in targets}
    screened = Counter()
    endpoint_screened = Counter()
    splits = Counter()
    missing_id = duplicate_id = malformed_shape = any_target = endpoint_and_target = rows = 0
    identifiers = set()
    id_column = metadata.get("row_identifier", "r_id")
    split_column = metadata.get("split_column", "split")
    with opener(path, "rt", encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        reaction_column = next((name for name in REACTION_FIELDS if name in columns), None)
        for row in reader:
            rows += 1
            identifier = row.get(id_column)
            if not present(identifier):
                missing_id += 1
            elif identifier in identifiers:
                duplicate_id += 1
            else:
                identifiers.add(identifier)
            reaction = row.get(reaction_column, "") if reaction_column else ""
            parts = reaction.split(">") if isinstance(reaction, str) else []
            shape_ok = len(parts) == 3 and bool(parts[0].strip()) and bool(parts[2].strip())
            malformed_shape += not shape_ok
            row_has_target = False
            for target in targets:
                state = target_state(row.get(target), target)
                states[target][state] += 1
                if state == "present":
                    screened[target] += 1
                    row_has_target = True
                    if shape_ok:
                        endpoint_screened[target] += 1
            any_target += row_has_target
            endpoint_and_target += row_has_target and shape_ok
            if split_column in columns:
                splits[row.get(split_column) or "missing"] += 1
            if progress and rows % 100000 == 0:
                progress(rows)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"table changed during screening: {path}")
    return {
        "sha256": digest,
        "rows_measured": rows,
        "columns": columns,
        "reaction_column": reaction_column,
        "reaction_shape_failures": malformed_shape,
        "row_identifier": id_column,
        "missing_identifiers": missing_id,
        "duplicate_identifiers": duplicate_id,
        "identifier_uniqueness_declared": metadata.get("row_identifier_unique"),
        "unique_nonempty_identifiers": len(identifiers),
        "target_field_states": {k: dict(v) for k, v in states.items()},
        "rows_passing_field_screen_by_target": dict(screened),
        "rows_passing_any_requested_target_screen": any_target,
        "rows_passing_target_and_both_endpoint_screen": endpoint_and_target,
        "rows_passing_target_and_both_endpoint_screen_by_target": dict(endpoint_screened),
        "split_column": split_column if split_column in columns else None,
        "split_status": metadata.get("split_status", "published" if splits else "not_present"),
        "split_counts": dict(splits),
        "benchmark_eligible_count": None,
        "screen_limit": "Target presence/type/range is reported separately from having both recorded "
        "reaction endpoints. A missing product does not itself invalidate a yield measurement or "
        "reactant-only task. Molecular parsing, featurization, rare-class filters, context adequacy and "
        "independence are not certified. The field-screen population is not the benchmark-eligible population.",
    }


def index_source_locks(data_root: Path) -> dict[str, list[dict]]:
    index = {}
    for path in sorted(physical_path(data_root / "sources").rglob("*.json")):
        raw, digest = stable_bytes(path)
        record = json.loads(raw)
        key = record.get("dataset_key") if isinstance(record, dict) else None
        if not key:
            continue
        retrieval_hash = record.get("source_sha256")
        files = record.get("files", [])
        if not retrieval_hash and isinstance(files, list):
            retrieval_hash = bool(files) and all(isinstance(f, dict) and f.get("sha256") for f in files)
        version = record.get("source_commit_or_version") or record.get("source_commit") or record.get("source_version")
        index.setdefault(key, []).append(
            {
                "path": str(path.relative_to(data_root.parent)),
                "sha256": digest,
                "lock_status": record.get("lock_status", "native_lock_without_status_field"),
                "source_partition": record.get("source_partition"),
                "declared_license": record.get("upstream_license") or record.get("license_spdx"),
                "license_scope_note": record.get("license_scope"),
                "source_url": record.get("source_url") or record.get("source_repository"),
                "version_recorded": bool(version),
                "retrieval_digest_recorded": bool(retrieval_hash),
                "outstanding_fields": record.get("outstanding_fields", []),
                "retrofit_note": record.get("retrofit_note"),
                "meaning": "This audit inspects the stored provenance assertions; it does not newly "
                "verify upstream rights or download the source files.",
            }
        )
    return index


def inspect_run(record: dict, kind: str) -> dict:
    """Describe saved evidence without upgrading metrics into reproducibility."""
    if not isinstance(record, dict):
        return {"kind": kind, "state": "unrecognized_json", "issues": ["run record is not an object"]}
    required = (
        ("dataset", "fingerprint", "model", "scheme", "seed")
        if kind == "classification"
        else ("dataset", "fingerprint", "method")
    )
    if not all(name in record for name in required):
        return {"kind": kind, "state": "unrecognized_json", "issues": ["missing run identity fields"]}
    metrics = {
        name: value
        for name, value in record.items()
        if name in {"mcc", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"}
        or "__ari" in name
        or "__ami" in name
        or "__nmi" in name
    }
    issues = []
    for name, value in metrics.items():
        if type(value) not in {int, float} or not math.isfinite(value):
            issues.append("invalid numeric metric: " + name)
        elif any(token in name for token in ("accuracy", "_f1", "__nmi")) and not 0 <= value <= 1:
            issues.append("metric outside [0, 1]: " + name)
        elif any(token in name for token in ("mcc", "__ari", "__ami")) and not -1 <= value <= 1:
            issues.append("metric outside [-1, 1]: " + name)
    for name in ("n_train", "n_val", "n_test", "n_unfeaturised_test", "n"):
        if name in record and (type(record[name]) is not int or record[name] < 0):
            issues.append("invalid population count: " + name)
    identity = {name: record[name] for name in required}
    identity["study"] = record.get("study")
    identity["param"] = record.get("param")
    state = "metric_record_present" if metrics and not issues else "incomplete_or_invalid_metric_record"
    # Keep bad values inspectable in strict JSON rather than aborting the audit
    # or silently dropping a failed result record.
    metrics = {
        name: str(value) if type(value) is float and not math.isfinite(value) else value
        for name, value in metrics.items()
    }
    return {
        "kind": kind,
        "state": state,
        "identity": identity,
        "metrics": metrics,
        "population_counts": {
            k: record[k] for k in ("n_train", "n_val", "n_test", "n_unfeaturised_test", "n") if k in record
        },
        "test_membership_pin_present": bool(record.get("test_membership_sha256")),
        "data_pin_present": bool(record.get("dataset_sha256") or record.get("data_sha256")),
        "issues": issues,
        "scientific_reproduction_verified": False,
    }


def legacy_metric_series(record: dict) -> dict:
    """Inventory list and ndarray-wrapper summaries without executing pickle."""
    series = valid_values = invalid_values = 0
    for values in record.values():
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            if not name.startswith("test_"):
                continue
            data = value.get("data") if isinstance(value, dict) else value
            if not isinstance(data, list):
                continue
            series += 1
            for item in data:
                if type(item) in {int, float} and math.isfinite(item):
                    valid_values += 1
                else:
                    invalid_values += 1
    return {
        "metric_series": series,
        "finite_metric_values": valid_values,
        "invalid_metric_values": invalid_values,
        "evidence_tier": "legacy_metric_summary",
        "benchmark_eligible_count": None,
        "scientific_reproduction_verified": False,
    }
