"""Build and verify the selected AAM benchmark using pinned source recipes.

This export preserves source rows and chemical strings; it standardizes field
names and moves historical mapper predictions into the provenance sidecar.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile

from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import (
    REGISTRY,
    REPO,
    checked_file,
    digest,
    prepare_source,
    processor_default,
    read_registry,
    require,
    source_metadata,
    write_json,
)

from synrxn.reconstruction.paths import repo_path

SCHEMA = "synrxn.aam-benchmark.v1"
OUTPUT = repo_path("Data/tables/development/aam")
CORE = ("r_id", "ground_truth", "rxn", "original_id")
ALIASES = {"reactions": "rxn", "split": "role"}
PREDICTIONS = {"rxn_mapper", "graphormer", "local_mapper", "rdt"}
ROLES = {"train": "train", "valid": "val", "validation": "val", "val": "val", "test": "test"}


def canonical_line(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with gzip.open(path, "rt", encoding="utf8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "Source table has no header")
        require(len(reader.fieldnames) == len(set(reader.fieldnames)), "Duplicate source columns")
        return reader.fieldnames, list(reader)


def source_digest(rows: list[dict]) -> str:
    checksum = hashlib.sha256()
    for row in rows:
        checksum.update(canonical_line(row))
    return checksum.hexdigest()


def normalized_row(source: dict) -> dict:
    """Rename fields only; preserve mappings, hydrogens, row IDs and metadata."""
    row = {}
    for key, value in source.items():
        if key in PREDICTIONS:
            continue
        target = ALIASES.get(key, key)
        require(target not in row or row[target] == value, f"Conflicting source aliases: {target}")
        row[target] = value
    require(all(row.get(key) for key in CORE), "Missing AAM core value")
    if "role" in row:
        require(row["role"] in ROLES, f"Unknown source role: {row['role']}")
        row["role"] = ROLES[row["role"]]
    return row


def normalize(columns: list[str], sources: list[dict]) -> tuple[list[str], list[dict]]:
    fields = list(CORE)
    for key in columns:
        target = ALIASES.get(key, key)
        if key not in PREDICTIONS and target not in fields:
            fields.append(target)
    rows = [normalized_row(row) for row in sources]
    require(len({row["r_id"] for row in rows}) == len(rows), "Duplicate dataset row IDs")
    return fields, rows


def extend_annotations(entry, columns, rows, root=REPO):
    """Join pinned derived grouping metadata without changing frozen source fields."""
    if 'center_annotations' not in entry:
        return columns, rows
    path = checked_file(entry['center_annotations'], root)
    extra_columns, annotations = read_rows(path)
    require(extra_columns[0] == 'r_id', 'Center annotations have no row identifier')
    require(not (set(extra_columns[1:]) & set(columns)), 'Center annotation overwrites a source field')
    by_id = {row['r_id']: row for row in annotations}
    require(len(by_id) == len(annotations) == len(rows), 'Duplicate or missing center annotations')
    require(set(by_id) == {row['r_id'] for row in rows}, 'Center annotation row IDs changed')
    return columns + extra_columns[1:], [{**row, **by_id[row['r_id']]} for row in rows]


def write_tables(folder: Path, columns: list[str], rows: list[dict], sources: list[dict]) -> None:
    with (folder / "aam.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=columns, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    with (folder / "provenance.jsonl.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for index, (row, source) in enumerate(zip(rows, sources), start=1):
                zipped.write(canonical_line({"r_id": row["r_id"], "source_row": index, "source_record": source}))


def copy_evidence(folder: Path, entry: dict, source: Path, root: Path) -> None:
    licenses = folder / "licenses"
    licenses.mkdir()
    for evidence in entry["evidence"]:
        path = checked_file(evidence, root)
        shutil.copyfile(path, licenses / path.name)
    if 'center_annotations' in entry:
        shutil.copyfile(checked_file(entry['center_annotations'], root), folder / 'center_annotations.csv.gz')
    if entry["name"] == "flower":
        shutil.copyfile(source.parent / "provenance.jsonl.gz", folder / "native_provenance.jsonl.gz")
        if (source.parent / "segments.csv").is_file():
            shutil.copyfile(source.parent / "segments.csv", folder / "segments.csv")
    elif entry["name"] == "sihumix":
        shutil.copyfile(source.parent / "source-data.tar.gz", folder / "native_source.tar.gz")
    elif "native_provenance" in entry:
        native = source.parent / "native_provenance.jsonl.gz"
        if not native.is_file():
            native = checked_file(entry["native_provenance"], root)
        shutil.copyfile(native, folder / "native_provenance.jsonl.gz")
        if entry["name"] in {"metamdb", "metambd_cyclic"}:
            for name in ("native_report.json", "rejected.csv.gz"):
                shutil.copyfile(source.parent / name, folder / name)


def file_manifest(folder: Path) -> dict:
    return {
        path.relative_to(folder).as_posix(): digest(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path != folder / "manifest.json"
    }


def verify_dataset(folder: Path, entry: dict, root: Path = REPO) -> dict:
    """Check payloads and every row against the registry-pinned source records."""
    manifest = json.loads((folder / "manifest.json").read_text())
    require(manifest["schema"] == SCHEMA, "Unknown AAM output schema")
    require(manifest["dataset_key"] == "aam/" + entry["name"], "Dataset identity mismatch")
    require(file_manifest(folder) == manifest["files"], f"Output checksum mismatch: {entry['name']}")
    metadata = json.loads((folder / "source.json").read_text())
    require(metadata["source_product_sha256"] == entry["product_sha256"], "Product source pin changed")
    require(metadata["source_lock_sha256"] == entry["source_lock"]["sha256"], "Source lock pin changed")
    require(metadata["source_mode"] == entry["mode"], "Source reconstruction mode changed")
    require(metadata["rights"] == entry["rights"], "Dataset rights changed")
    for evidence in entry["evidence"]:
        license_file = folder / "licenses" / Path(evidence["path"]).name
        require(digest(license_file) == evidence["sha256"], "Pinned license evidence changed")
    columns, rows = read_rows(folder / "aam.csv.gz")
    with gzip.open(folder / "provenance.jsonl.gz", "rt", encoding="utf8") as handle:
        witnesses = [json.loads(line) for line in handle]
    require(len(rows) == len(witnesses) == entry["rows"] == manifest["rows"], "AAM row count mismatch")
    sources = [witness["source_record"] for witness in witnesses]
    require(source_digest(sources) == entry["source_rows_sha256"], "Source records differ from pinned selection")
    expected_columns, expected_rows = normalize(entry["product_columns"], sources)
    expected_columns, expected_rows = extend_annotations(entry, expected_columns, expected_rows, root)
    require(columns == expected_columns and rows == expected_rows, "AAM schema or source data changed")
    for index, (row, witness) in enumerate(zip(rows, witnesses), start=1):
        require(witness["r_id"] == row["r_id"] and witness["source_row"] == index, "Provenance join changed")
    return {
        "dataset": entry["name"],
        "rows": len(rows),
        "columns": columns,
        "by_role": dict(Counter(row["role"] for row in rows if "role" in row)),
        "license_expression": entry["rights"]["license_expression"],
        "redistribution_partition": entry["rights"]["partition"],
        "source_records_verified": True,
    }


def build_dataset(entry: dict, registry: dict, args, root: Path = REPO) -> dict:
    """Prepare a complete verified dataset before replacing a managed output."""
    source = prepare_source(entry, registry, args, root)
    columns, sources = read_rows(source)
    require(columns == entry["product_columns"], "Source columns changed")
    require(len(sources) == entry["rows"], "Source row count changed")
    require(source_digest(sources) == entry["source_rows_sha256"], "Source row selection changed")
    columns, rows = normalize(columns, sources)
    columns, rows = extend_annotations(entry, columns, rows, root)
    output = args.output / entry["name"]
    if output.exists():
        marker = output / "manifest.json"
        require(
            marker.is_file() and json.loads(marker.read_text()).get("schema") == SCHEMA,
            f"Refusing to replace an unmanaged directory: {output}",
        )
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".aam-build-", dir=args.output) as temporary:
        folder = Path(temporary)
        write_tables(folder, columns, rows, sources)
        copy_evidence(folder, entry, source, root)
        write_json(folder / "source.json", source_metadata(entry, source, registry, root))
        (folder / "NOTICE.txt").write_text(
            f"SynRXN AAM dataset: {entry['name']}\n"
            f"Applicable terms: {entry['rights']['license_expression']}\n"
            f"Redistribution partition: {entry['rights']['partition']}\n"
            "Source citations, scope limitations and rights evidence: source.json and licenses/.\n"
            "Changes: reactions -> rxn; split -> role; validation/valid -> val, where present.\n"
            "Historical mapper predictions are retained in provenance.jsonl.gz, outside task inputs.\n"
            "Source rows, mapping targets, reaction strings and native split assignments are preserved.\n",
            encoding="utf8",
        )
        write_json(
            folder / "manifest.json",
            {
                "schema": SCHEMA,
                "dataset_key": "aam/" + entry["name"],
                "rows": len(rows),
                "files": file_manifest(folder),
            },
        )
        report = verify_dataset(folder, entry, root)
        backup = args.output / ("." + entry["name"] + ".previous")
        require(not backup.exists(), f"Unresolved previous export: {backup}")
        if output.exists():
            output.rename(backup)
        try:
            folder.rename(output)
        except BaseException:
            if backup.exists():
                backup.rename(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--datasets", nargs="+", help="Dataset names; default: all selected datasets")
    parser.add_argument("--list", action="store_true", help="List selected datasets without building")
    parser.add_argument("--verify", action="store_true", help="Verify existing exports without source processing")
    parser.add_argument("--fetch", action="store_true", help="Fetch pinned public archives when needed")
    parser.add_argument("--rebuild-sources", action="store_true", help="Rerun native source recipes before exporting")
    parser.add_argument(
        "--processor-python", default=processor_default(), help="Python for native chemistry processing"
    )
    parser.add_argument("--flower-work-root", type=Path, help="Fresh path reruns Flower processing from original text")
    args = parser.parse_args(argv)
    registry = read_registry(args.registry)
    names = [entry["name"] for entry in registry["datasets"]]
    selected = args.datasets or names
    require(len(selected) == len(set(selected)), "Dataset requested more than once")
    require(set(selected) <= set(names), f"Unknown datasets: {sorted(set(selected) - set(names))}")
    entries = [entry for entry in registry["datasets"] if entry["name"] in selected]
    if args.list:
        print(
            json.dumps([{key: entry[key] for key in ("name", "rows", "mode", "rights")} for entry in entries], indent=2)
        )
        return 0
    args.output = args.output.resolve()
    reports = []
    for entry in entries:
        report = (
            verify_dataset(args.output / entry["name"], entry) if args.verify else build_dataset(entry, registry, args)
        )
        reports.append(report)
        print(f"{entry['name']}: {report['rows']:,} rows verified", flush=True)
    summary = {"schema": SCHEMA, "datasets": reports, "rows": sum(report["rows"] for report in reports)}
    if not args.verify:
        # A subset run must not pretend to describe the full pool.
        name = "manifest.json" if len(entries) == len(names) else "last-build.json"
        write_json(args.output / name, summary)
    print(json.dumps({"datasets": len(reports), "rows": summary["rows"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
