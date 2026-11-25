"""Stream pinned FREA data while preserving the meaning of its labels."""

from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import io
from pathlib import Path
from typing import Any, Sequence

FREA_COLUMNS = (
    "sample_id", "dataset", "reaction_smiles", "reactants", "products", "feasible",
    "evidence_tier", "label_method", "negative_method", "upstream_split", "source_record_id",
    "source_index", "exact_reaction_group", "additional_note", "failure_reasons", "difficulty",
)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_frea(sources: Sequence[dict], output: Path, *, dataset: str, expert: bool) -> dict[str, Any]:
    """Build a complete source view, without chemical repair or invented memberships.

    Expert judgments and upstream generated supervision remain separate tables.
    The grouping key hashes exact recorded reaction text, not canonical chemistry.
    """
    if not sources or len({s["split"] for s in sources}) != len(sources):
        raise ValueError("provide source files with distinct declared upstream splits")
    csv.field_size_limit(16 * 1024 * 1024)
    for source in sources:
        path = Path(source["path"])
        if digest(path) != source["sha256"]:
            raise ValueError("FREA source checksum mismatch: " + path.name)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    labels, tiers, splits, methods = (Counter() for _ in range(4))
    count = 0
    source_reports = []
    try:
        with temporary.open("wb") as raw, \
                gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=1) as zipped, \
                io.TextIOWrapper(zipped, encoding="utf8", newline="") as text:
            writer = csv.DictWriter(text, fieldnames=FREA_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for source in sources:
                path = Path(source["path"])
                before = path.stat()
                rows = 0
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    required = {"reactant", "product", "feasibility"} if expert else \
                        {"reactants", "product", "feasible", "method"}
                    if not required.issubset(reader.fieldnames or ()):
                        raise ValueError("FREA source columns missing: " + path.name)
                    for index, record in enumerate(reader):
                        label = record["feasibility" if expert else "feasible"].strip()
                        if label not in {"0", "1"}:
                            raise ValueError(f"invalid FREA binary label in {path.name}:{index}")
                        reactants = record["reactant" if expert else "reactants"]
                        products = record["product"]
                        if not reactants.strip() or not products.strip():
                            raise ValueError(f"missing recorded FREA endpoint in {path.name}:{index}")
                        reaction = reactants + ">>" + products
                        tier = "expert_judgment" if expert else \
                            "upstream_positive" if label == "1" else "generated_negative"
                        method = record.get("method", "")
                        identifier = f'{dataset}:{source["sha256"][:12]}:{source["split"]}:{index}'
                        writer.writerow({
                            "sample_id": identifier, "dataset": dataset, "reaction_smiles": reaction,
                            "reactants": reactants, "products": products, "feasible": label,
                            "evidence_tier": tier, "label_method": "expert_judgment" if expert else "upstream_label",
                            "negative_method": method if not expert and label == "0" else "",
                            "upstream_split": source["split"], "source_record_id": f'{path.name}:{index}',
                            "source_index": index,
                            "exact_reaction_group": "raw-" + hashlib.sha256(reaction.encode()).hexdigest(),
                            "additional_note": record.get("additional_note", ""),
                            "failure_reasons": record.get("failure_reasons", ""),
                            "difficulty": record.get("difficulty", ""),
                        })
                        count += 1
                        rows += 1
                        labels[label] += 1
                        tiers[tier] += 1
                        splits[source["split"]] += 1
                        methods[method or "not_recorded"] += 1
                        if count % 500000 == 0:
                            print(f"{dataset}: normalized {count:,} source rows", flush=True)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != \
                        (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError("FREA source changed during import")
                if rows != source["rows"]:
                    raise ValueError("FREA source row count differs from its pin")
                source_reports.append({"source_file": path.name, "sha256": source["sha256"],
                                       "upstream_split": source["split"], "rows": rows})
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"dataset": dataset, "rows": count, "columns": list(FREA_COLUMNS), "source_files": source_reports,
            "sha256": digest(output), "label_counts": dict(labels), "evidence_tiers": dict(tiers),
            "upstream_splits": dict(splits), "source_methods": dict(methods),
            "chemical_parsing_performed": False, "quarantined_rows": 0,
            "grouping": "SHA-256 of exact recorded reaction text; no claim of chemical/source independence",
            "membership": "Complete upstream expert test or corpus train/test, without custom benchmark subsampling"}
