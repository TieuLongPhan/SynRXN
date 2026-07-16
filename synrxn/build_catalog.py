"""Build static catalog JSON and reaction previews for the documentation UI."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO

import yaml

from .verify_manifest import (
    manifest_files,
    verification_succeeded,
    verify_with_root,
)

REACTION_COLUMNS = ("rxn", "aam", "reactions", "rsmi", "reaction")


def _open_csv(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf8", errors="replace", newline="")
    return path.open("r", encoding="utf8", errors="replace", newline="")


def _short(value: Any, limit: int = 320) -> Any:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _target_summary(values: list[str]) -> dict[str, Any]:
    present = [value for value in values if str(value).strip()]
    numeric = []
    for value in present:
        try:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError
            numeric.append(number)
        except (TypeError, ValueError):
            numeric = []
            break
    if numeric and len(numeric) == len(present):
        return {
            "kind": "numeric",
            "count": len(numeric),
            "min": min(numeric),
            "max": max(numeric),
            "mean": sum(numeric) / len(numeric),
        }
    counts = Counter(present)
    return {
        "kind": "categorical",
        "count": len(present),
        "unique": len(counts),
        "top": [{"value": value, "count": count} for value, count in counts.most_common(8)],
    }


def summarize_csv(
    path: Path, targets: Iterable[str], sample_size: int = 5
) -> dict[str, Any]:
    """Compute bounded UI samples and lightweight data summaries."""
    targets = tuple(targets)
    target_values = {target: [] for target in targets}
    split_counts: Counter[str] = Counter()
    null_counts: Counter[str] = Counter()
    sample = []
    rows = 0
    first_reaction = None
    with _open_csv(path) as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        for row in reader:
            rows += 1
            if first_reaction is None:
                reaction_column = next(
                    (column for column in REACTION_COLUMNS if column in columns), None
                )
                if reaction_column and row.get(reaction_column):
                    first_reaction = str(row[reaction_column])
            if len(sample) < sample_size:
                sample.append({column: _short(row.get(column)) for column in columns})
            for column in columns:
                if row.get(column) is None or not str(row.get(column)).strip():
                    null_counts[column] += 1
            split = row.get("split")
            if split is not None and str(split).strip():
                split_counts[str(split)] += 1
            for target in targets:
                value = row.get(target)
                if value is not None:
                    target_values[target].append(value)
    return {
        "rows": rows,
        "columns": columns,
        "sample": sample,
        "split_counts": dict(sorted(split_counts.items())),
        "null_counts": {column: null_counts[column] for column in columns},
        "target_summaries": {
            target: _target_summary(values) for target, values in target_values.items()
        },
        "first_reaction": first_reaction,
    }


def _write_reaction_svg(reaction_smiles: str, output: Path) -> bool:
    try:
        from rdkit.Chem import Draw, rdChemReactions

        reaction = rdChemReactions.ReactionFromSmarts(reaction_smiles, useSmiles=True)
        if reaction is None:
            return False
        svg = Draw.ReactionToImage(reaction, subImgSize=(220, 150), useSVG=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(svg, encoding="utf8")
        return True
    except Exception:
        return False


def build_catalog(
    data_dir: Path,
    metadata_path: Path,
    manifest_path: Path,
    output_path: Path,
    reaction_dir: Optional[Path] = None,
) -> dict[str, Any]:
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    verification = verify_with_root(manifest, data_dir)
    if not verification_succeeded(verification):
        raise RuntimeError("catalog assets require a fully verified release manifest")
    manifest_index = {entry["key"]: entry for entry in manifest_files(manifest)}
    records = []
    for key, description in sorted(metadata["datasets"].items()):
        task, filename = key.split("/", 1)
        name = filename.removesuffix(".csv.gz").removesuffix(".csv")
        artifact = manifest_index[key]
        summary = summarize_csv(data_dir / key, description.get("targets") or [])
        reaction_column = next(
            (column for column in REACTION_COLUMNS if column in summary["columns"]), None
        )
        depiction = None
        reaction_text = None
        if reaction_column and summary["sample"]:
            reaction_text = summary["first_reaction"]
            if reaction_dir and reaction_text:
                svg_name = f"{task}-{name}.svg"
                if _write_reaction_svg(str(reaction_text), reaction_dir / svg_name):
                    depiction = f"_static/catalog-reactions/{svg_name}"
        records.append(
            {
                "id": f"{task}/{name}",
                "task": task,
                "name": name,
                **description,
                "rows": summary["rows"],
                "size": artifact["size"],
                "sha256": artifact["sha256"],
                "columns": summary["columns"],
                "column_metadata": {
                    column: {
                        **metadata["column_definitions"].get(column, {}),
                        "logical_type": (
                            "number"
                            if summary["target_summaries"].get(column, {}).get("kind")
                            == "numeric"
                            else "string"
                        ),
                        "nullable": summary["null_counts"].get(column, 0) > 0,
                        "unit": metadata["column_definitions"].get(column, {}).get(
                            "unit"
                        ),
                    }
                    for column in summary["columns"]
                },
                "sample": summary["sample"],
                "split_counts": summary["split_counts"],
                "null_counts": summary["null_counts"],
                "target_summaries": summary["target_summaries"],
                "reaction_column": reaction_column,
                "reaction_text": reaction_text,
                "depiction": depiction,
            }
        )
    catalog = {
        "schema_version": "1.0",
        "release": {
            "version": manifest["dataset"]["version"],
            "doi": manifest["dataset"]["doi"],
            "generated_at": manifest["generated_at"],
            "dataset_count": len(records),
            "integrity": "verified",
        },
        "datasets": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_json = json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"
    output_path.write_text(catalog_json, encoding="utf8")
    # The catalog is also consumed from standalone HTML previews, where browsers
    # commonly block file:// fetch requests.  Publish a script form alongside the
    # JSON so the page has immediate, same-document access to the same payload.
    script_payload = (
        catalog_json.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    output_path.with_suffix(".js").write_text(
        f"window.SYNRXN_CATALOG = {script_payload}", encoding="utf8"
    )
    return catalog


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--metadata", type=Path, default=Path("Data/metadata.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("manifest.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("doc/_static/catalog-data.json")
    )
    parser.add_argument(
        "--reaction-dir", type=Path, default=Path("doc/_static/catalog-reactions")
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    catalog = build_catalog(
        args.data_dir.resolve(),
        args.metadata.resolve(),
        args.manifest.resolve(),
        args.output.resolve(),
        args.reaction_dir.resolve() if args.reaction_dir else None,
    )
    print(f"Catalog UI data: {len(catalog['datasets'])} datasets -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
