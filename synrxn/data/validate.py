"""Validate SynRXN catalog metadata against release data and a manifest."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, TextIO

try:
    import yaml
except ImportError:  # pragma: no cover - exercised by clean-install checks
    yaml = None  # type: ignore

from synrxn.data.verify_manifest import DATA_TASKS, ManifestContractError, manifest_files
from synrxn.data.release import artifact_keys, artifact_path, release_keys
from synrxn.reconstruction.paths import physical_path

CATALOG_SCHEMA_VERSION = "1.0"


def load_catalog(path: Path) -> Dict[str, Any]:
    """Load and minimally validate the YAML metadata catalog."""
    if yaml is None:
        raise RuntimeError("PyYAML is required for catalog validation; install synrxn[maintenance]")
    data = yaml.safe_load(physical_path(path.resolve()).read_text(encoding="utf8")) or {}
    if not isinstance(data, dict):
        raise ValueError("catalog root must be a mapping")
    if data.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(f"unsupported catalog schema_version {data.get('schema_version')!r}")
    if not isinstance(data.get("column_definitions"), dict):
        raise ValueError("catalog.column_definitions must be a mapping")
    if not isinstance(data.get("datasets"), dict) or not data["datasets"]:
        raise ValueError("catalog.datasets must be a non-empty mapping")
    return data


def _open_csv(path: Path) -> TextIO:
    csv.field_size_limit(16 * 1024 * 1024)
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf8", errors="replace", newline="")
    return path.open("r", encoding="utf8", errors="replace", newline="")


def _manifest_index(manifest_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if manifest_path is None:
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    return {entry["key"]: entry for entry in manifest_files(manifest)}


def _release_files(data_dir: Path) -> set[str]:
    return {key for task in DATA_TASKS for key in artifact_keys(data_dir, task)
            if key.endswith((".csv", ".csv.gz"))}


def validate_catalog(
    data_dir: Path,
    metadata_path: Path,
    manifest_path: Optional[Path] = None,
    quick: bool = False,
) -> Dict[str, Any]:
    """Validate catalog coverage, schemas, identifiers, splits, and manifest facts."""
    catalog = load_catalog(metadata_path)
    datasets: Dict[str, Dict[str, Any]] = catalog["datasets"]
    column_definitions = catalog["column_definitions"]
    manifest = _manifest_index(manifest_path)
    actual_files = _release_files(data_dir)
    actual_files.update(key for key in datasets if artifact_path(data_dir, key).is_file())
    if manifest_path is not None:
        envelope = json.loads(manifest_path.read_text(encoding="utf8"))
        partition = envelope.get("dataset", {}).get("partition", "main")
        selected = release_keys(datasets, partition)
        actual_files = {
            key
            for key in actual_files
            if key in selected
            or key not in datasets
            and key.split("/", 1)[0].endswith("_sharealike") == (partition == "sharealike")
        }
        datasets = {key: value for key, value in datasets.items() if key in selected}
    declared_files = set(datasets)
    errors: list[str] = []
    warnings: list[str] = []
    checked_rows = 0

    for key in sorted(actual_files - declared_files):
        errors.append(f"missing metadata: {key}")
    for key in sorted(declared_files - actual_files):
        errors.append(f"metadata points to missing artifact: {key}")
    if manifest:
        manifest_dataset_files = {
            key for key in manifest if key.split("/", 1)[0] in DATA_TASKS and key.endswith((".csv", ".csv.gz"))
        }
        for key in sorted(actual_files - set(manifest)):
            errors.append(f"artifact absent from manifest: {key}")
        for key in sorted(manifest_dataset_files - actual_files):
            errors.append(f"manifest points to missing artifact: {key}")

    for key in sorted(actual_files & declared_files):
        metadata = datasets[key]
        for field in ("title", "description", "benchmark_role", "license", "citations"):
            if metadata.get(field) in (None, "", []):
                errors.append(f"{key}: missing required metadata field {field}")

        path = artifact_path(data_dir, key)
        with _open_csv(path) as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            if not columns:
                errors.append(f"{key}: missing CSV header")
                continue

            for column in columns:
                if column not in column_definitions:
                    errors.append(f"{key}: column {column!r} has no catalog definition")

            targets = metadata.get("targets") or []
            for field in ("target_units", "target_kinds"):
                values = metadata.get(field, {})
                if not isinstance(values, dict) or any(
                    key not in targets or not isinstance(value, str) or not value.strip()
                    for key, value in values.items()
                ):
                    errors.append(f"{key}: invalid per-target {field}")
            for target in targets:
                if target not in columns:
                    errors.append(f"{key}: target {target!r} is absent from CSV columns")

            identifier = metadata.get("row_identifier")
            if not identifier or identifier not in columns:
                errors.append(f"{key}: row_identifier {identifier!r} is absent")

            declared_splits = set(str(value) for value in metadata.get("split_values", []))
            split_column = metadata.get("split_column", "split")
            split_status = metadata.get("split_status", "published")
            if split_status not in {"published", "proposed", "none"}:
                errors.append(f"{key}: invalid split_status {split_status!r}")
            if split_status == "none" and (declared_splits or split_column in columns):
                errors.append(f"{key}: split_status none contradicts recorded split fields")
            if declared_splits and split_column not in columns:
                errors.append(f"{key}: split_values declared but {split_column} column is absent")
            if split_column in columns and not declared_splits:
                errors.append(f"{key}: split column exists but split_values are not declared")

            manifest_entry = manifest.get(key)
            if manifest_entry and manifest_entry.get("columns") != columns:
                errors.append(f"{key}: manifest columns differ from the CSV header")

            if quick:
                continue

            row_count = 0
            null_identifiers = 0
            duplicate_identifiers = 0
            seen_identifiers: set[str] = set()
            actual_splits: set[str] = set()
            for row in reader:
                row_count += 1
                value = row.get(identifier or "", "")
                if value is None or not str(value).strip():
                    null_identifiers += 1
                elif metadata.get("row_identifier_unique", False):
                    if value in seen_identifiers:
                        duplicate_identifiers += 1
                    else:
                        seen_identifiers.add(value)
                if split_column in columns:
                    split = row.get(split_column)
                    if split is not None and str(split).strip():
                        actual_splits.add(str(split))

            checked_rows += row_count
            if null_identifiers:
                errors.append(f"{key}: {null_identifiers} empty row identifiers")
            if duplicate_identifiers:
                errors.append(f"{key}: {duplicate_identifiers} duplicate values in unique identifier {identifier}")
            if declared_splits != actual_splits:
                errors.append(
                    f"{key}: split values differ; declared={sorted(declared_splits)} " f"actual={sorted(actual_splits)}"
                )
            if manifest_entry and manifest_entry.get("rows") != row_count:
                errors.append(f"{key}: manifest rows={manifest_entry.get('rows')} actual={row_count}")

    return {
        "ok": not errors,
        "artifacts_checked": len(actual_files & declared_files),
        "rows_checked": checked_rows,
        "errors": errors,
        "warnings": warnings,
    }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--metadata", type=Path, default=Path("Data/metadata.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("manifest.json"))
    parser.add_argument("--quick", action="store_true", help="Validate headers without scanning rows")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = validate_catalog(
            args.data_dir.resolve(),
            args.metadata.resolve(),
            args.manifest.resolve() if args.manifest else None,
            quick=args.quick,
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
        ManifestContractError,
    ) as exc:
        print(f"catalog validation failed: {exc}", file=sys.stderr)
        return 2
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(
        f"Catalog validation: artifacts={result['artifacts_checked']} "
        f"rows={result['rows_checked']} errors={len(result['errors'])} "
        f"warnings={len(result['warnings'])}"
    )
    for error in result["errors"]:
        print(f"ERROR: {error}")
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
