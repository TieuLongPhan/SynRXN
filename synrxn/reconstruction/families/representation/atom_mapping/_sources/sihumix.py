"""Build the single selected sihumix AAM dataset from its pinned upstream source.

Keep original mappings for concrete, two-sided reactions. Hydrogen imbalance
is retained as metadata; hydrogen and proton completion are disabled.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import tarfile
import tempfile
import urllib.request
import zlib

from rdkit import Chem

from synrxn.reconstruction.paths import REPO, repo_path

from synrxn.reconstruction.families.representation.atom_mapping._sources.sihumix_quality import QC_FIELDS, atom_inventory, inspect_mapping, mapping_classes, sha_text, source_label

COMMIT = "c2c225efd8cc4489632f5a4044d81754265598d6"
ARCHIVE_SHA256 = "fd7db3d1ff81f777159b497d87051963324ad84dedf3f493fb9415253740ecc9"
METADATA = REPO / "Data/metadata/development/sihumix"
DEFAULT_ARCHIVE = repo_path("Data/raw/sihumix", COMMIT, "source.tar.gz")
DEFAULT_OUTPUT = repo_path("Data/tables/development/sihumix")
SCHEMA = "synrxn.sihumix-aam.v1"
KEY = "aam/sihumix"
SOURCE_URL = "https://github.com/NoraBeier/cAAMsGEMs_of_SIHUMIx"
COLUMNS = [
    "r_id",
    "ground_truth",
    "rxn",
    "original_id",
    "source",
    "organism",
    "reaction_name",
    "license",
    "upstream_source",
    "reference_tier",
    "validation_status",
    "original_aam",
    "mapping_variant",
    "reaction_group",
    "mapping_equivalence",
    "parseable",
    "concrete_structures",
    "heavy_element_balanced",
    "element_balanced",
    "charge_balanced",
    "heavy_mapping_valid",
    "explicit_mapping_valid",
    "is_transport",
    "is_transport_1to1",
    "is_identity",
    "structural_qc_pass",
    "exclusion_reasons",
    "source_file",
    "source_pointer",
    "hydrogen_balance_status",
    "hydrogen_dH",
    "hydrogen_dQ",
]
TIERS = {
    "User-curated": "upstream_user_curated",
    "ChemRecon": "upstream_database_mapping",
    "RDT-generated": "upstream_generated_mapping",
    "": "mapping_unavailable",
}


def sha_file(path: Path) -> str:
    """Hash complete file bytes for acquisition and output verification."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def acquire_archive(archive: Path, source: dict) -> None:
    """Acquire the metadata-pinned URL atomically, rejecting changed bytes."""
    if archive.is_file():
        if sha_file(archive) != source["archive_sha256"]:
            raise ValueError("Source archive SHA256 mismatch")
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=archive.parent, prefix=archive.name + ".", suffix=".part", delete=False
        ) as raw:
            temporary = Path(raw.name)
            with urllib.request.urlopen(source["archive_url"], timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    raw.write(chunk)
        if sha_file(temporary) != source["archive_sha256"]:
            raise ValueError("Source archive SHA256 mismatch")
        temporary.replace(archive)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def runtime_versions() -> dict:
    """Record chemistry, graph, interpreter and compression dependencies."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "zlib_compile": zlib.ZLIB_VERSION,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        **{name: importlib.metadata.version(name) for name in ("rdkit", "networkx", "numpy", "Pillow")},
    }


def _cell(value):
    if isinstance(value, (list, dict, bool)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Write deterministic gzip CSV, including stable empty-table headers."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=columns, lineterminator="\n")
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: _cell(row.get(key, "")) for key in columns})
    temporary.replace(path)


def write_json(path: Path, value: dict) -> None:
    """Write readable metadata with explicit source and counting policies."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _native_members(archive: Path) -> dict[str, bytes]:
    """Read dataset files in memory; never extract or run upstream code."""
    members = {}
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            if not member.isfile() or "/" not in member.name:
                continue
            relative = member.name.split("/", 1)[1]
            if relative == "README.md" or relative.startswith("data/"):
                members[relative] = source.extractfile(member).read()
    if "README.md" not in members or not any(
        p.startswith("data/SIHUMIx_cAAMsGEMs/") and p.endswith(".json") for p in members
    ):
        raise ValueError("Archive lacks the dataset README or mapped models")
    grant = b"This dataset is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0)."
    if grant not in members["README.md"]:
        raise ValueError("Pinned upstream dataset license declaration is absent")
    return members


def _write_native(path: Path, members: dict[str, bytes]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as target:
                for name, data in sorted(members.items()):
                    info = tarfile.TarInfo(name)
                    info.size, info.mtime, info.mode = len(data), 0, 0o644
                    target.addfile(info, io.BytesIO(data))


def _selected(aam: str, checked: dict) -> str:
    if not aam:
        return "missing_mapping"
    parts = aam.split(">>")
    if len(parts) == 2 and not all(parts):
        return "one_sided_boundary"
    if len(parts) != 2 or not checked["parseable"]:
        return "parse_or_sanitization_failed"
    if not checked["concrete_structures"]:
        return "generic_structure"
    return ""


def _hydrogen(aam: str) -> dict:
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    mols = [Chem.MolFromSmiles(side, parser) for side in aam.split(">>")]
    inventories = [atom_inventory(mol) for mol in mols]
    h = [Counter({iso: n for (z, iso), n in inv.items() if z == 1}) for inv in inventories]
    return {
        "hydrogen_balance_status": "balanced" if h[0] == h[1] else "imbalanced",
        "hydrogen_dH": sum(h[1].values()) - sum(h[0].values()),
        "hydrogen_dQ": Chem.GetFormalCharge(mols[1]) - Chem.GetFormalCharge(mols[0]),
    }


def prepare_records(members: dict[str, bytes]) -> tuple[list[dict], dict]:
    """Select concrete two-sided source records without repairing their mappings."""
    raw, quality, models = [], {}, {}
    for path, payload in sorted(members.items()):
        if not path.startswith("data/SIHUMIx_cAAMsGEMs/") or not path.endswith(".json"):
            continue
        model = json.loads(payload)
        organism = Path(path).stem
        models[organism] = len(model["reactions"])
        for identifier, record in sorted(model["reactions"].items()):
            aam = (record.get("selected_matching") or {}).get("aam") or ""
            if not isinstance(aam, str):
                raise ValueError(f"{path}/{identifier}: AAM must be a string or null")
            quality.setdefault(aam, None)
            raw.append((path, organism, identifier, record, aam))
    for aam in quality:
        quality[aam] = inspect_mapping(aam)
    classes = mapping_classes(quality)
    rows, excluded, hydrogen = [], Counter(), {}
    for path, organism, identifier, record, aam in raw:
        checked = quality[aam]
        label = source_label(record.get("matching_source"), bool(aam))
        reason = _selected(aam, checked)
        if reason:
            excluded[reason] += 1
            continue
        if aam not in hydrogen:
            hydrogen[aam] = _hydrogen(aam)
        reasons = list(checked["quality_issues"])
        if record.get("is_transport"):
            reasons.append("transport_reaction")
        if checked["is_identity"]:
            reasons.append("identity_reaction")
        rows.append(
            {
                **{key: checked[key] for key in QC_FIELDS if key != "quality_issues"},
                **hydrogen[aam],
                "r_id": f"sihumix:{organism}:{identifier}",
                "original_id": identifier,
                "source": label,
                "organism": organism,
                "reaction_name": record.get("name", ""),
                "license": "CC-BY-4.0",
                "upstream_source": record.get("matching_source") or "",
                "reference_tier": TIERS[label],
                "validation_status": (
                    "qc_failed"
                    if checked["quality_issues"]
                    else (
                        "reference_excluded_transport_or_identity" if reasons else "mechanical_qc_passed_review_pending"
                    )
                ),
                "original_aam": aam,
                "mapping_variant": sha_text(aam),
                "mapping_equivalence": classes[aam],
                "is_transport": bool(record.get("is_transport")),
                "is_transport_1to1": bool(record.get("is_transport_1to1")),
                "structural_qc_pass": not reasons,
                "exclusion_reasons": sorted(set(reasons)),
                "source_file": path,
                "source_pointer": "/reactions/" + identifier.replace("~", "~0").replace("/", "~1"),
            }
        )
    return rows, {"models": models, "raw_records": len(raw), "excluded_by_reason": dict(excluded)}


def build_bundle(archive: Path, output: Path, source: dict, license_text: bytes) -> dict:
    """Build the single sihumix table and byte-preserved provenance package."""
    if sha_file(archive) != source["archive_sha256"]:
        raise ValueError("Source archive SHA256 mismatch")
    if hashlib.sha256(license_text).hexdigest() != source["license_text_sha256"]:
        raise ValueError("License text SHA256 mismatch")
    members = _native_members(archive)
    if hashlib.sha256(members["README.md"]).hexdigest() != source["grant_snapshot_sha256"]:
        raise ValueError("Upstream dataset grant snapshot SHA256 mismatch")
    marker = output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()).get("schema") != SCHEMA:
            raise ValueError("Nonempty output is not a managed sihumix dataset")
        allowed = {
            "aam.csv.gz",
            "source-data.tar.gz",
            "UPSTREAM_README.txt",
            "LICENSE-CC-BY-4.0.txt",
            "source.json",
            "NOTICE.txt",
            "build_report.json",
            "manifest.json",
        }
        if {p.name for p in output.iterdir()} - allowed:
            raise ValueError("Unexpected files in the managed sihumix output")
    rows, inventory = prepare_records(members)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "aam.csv.gz", rows, COLUMNS)
    _write_native(output / "source-data.tar.gz", members)
    (output / "UPSTREAM_README.txt").write_bytes(members["README.md"])
    (output / "LICENSE-CC-BY-4.0.txt").write_bytes(license_text)
    write_json(output / "source.json", source)
    (output / "NOTICE.txt").write_text(
        f"Source: {source['repository']}\nCommit: {source['commit']}\nAttribution: {source['attribution']}\n"
        "Dataset license: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/\n\n"
        "Changes: named sihumix; added dataset identifiers, mapping provenance and QC metadata;\n"
        "canonicalized mapped targets and unmapped inputs. Excluded missing mappings, one-sided\n"
        "boundaries, parser/sanitization failures and generic structures. Original mapping strings,\n"
        "source reaction identifiers and native dataset files are retained. Hydrogen-imbalanced\n"
        "records remain unchanged. No proton addition, hydrogen expansion or mapping repair.\n"
        "This is source-supplied mapping data; no independent chemical validation is claimed.\n"
        "Native source-data.tar.gz preserves dataset files and README; upstream software is excluded.\n"
    )
    report = {
        "schema": SCHEMA,
        "dataset_key": KEY,
        "status": "selected_original_mapping_collection",
        "source": source,
        "rows": len(rows),
        "inventory": inventory,
        "mapped_by_source": dict(Counter(r["source"] for r in rows)),
        "hydrogen_balance": dict(Counter(r["hydrogen_balance_status"] for r in rows)),
        "distinct_original_mapping_strings": len({r["original_aam"] for r in rows}),
        "hydrogen_completion": False,
        "proton_additions": 0,
        "selection": "Concrete, parseable, two-sided mappings; retain balance and mapping QC failures as flags.",
        "counting_policy": "Original organism/reaction memberships; no deduplication or split assignment.",
        "mapping_fields": {
            "original_aam": "exact source string",
            "ground_truth": "canonical source mapping",
            "rxn": "canonical unmapped input",
        },
        "versions": runtime_versions(),
        "implementation_sha256": {
            "sihumix.py": sha_file(Path(__file__)),
            "sihumix_quality.py": sha_file(Path(__file__).with_name("sihumix_quality.py")),
        },
        "native_dataset_files": {
            name: hashlib.sha256(payload).hexdigest() for name, payload in sorted(members.items())
        },
    }
    write_json(output / "build_report.json", report)
    write_json(
        marker,
        {
            "schema": SCHEMA,
            "dataset_key": KEY,
            "rows": len(rows),
            "source_commit": source["commit"],
            "license_spdx": "CC-BY-4.0",
            "files": [
                {"file": p.name, "bytes": p.stat().st_size, "sha256": sha_file(p)}
                for p in sorted(output.iterdir())
                if p.is_file() and p.name != "manifest.json"
            ],
        },
    )
    return report


def _preserved_members(output: Path, report: dict) -> dict:
    """Check every preserved native payload before interpreting model records."""
    members = {}
    with tarfile.open(output / "source-data.tar.gz", "r:gz") as archive:
        for name, digest in report["native_dataset_files"].items():
            payload = archive.extractfile(name).read()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError(f"Native source payload mismatch: {name}")
            members[name] = payload
    return members


def _check_payloads(output: Path, manifest: dict) -> None:
    """Reject changed payloads and extra exports in the one-dataset directory."""
    expected = {"manifest.json"} | {entry["file"] for entry in manifest["files"]}
    if {path.name for path in output.iterdir()} != expected:
        raise ValueError("Unexpected or missing files in sihumix output")
    for entry in manifest["files"]:
        path = output / entry["file"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha_file(path) != entry["sha256"]:
            raise ValueError(f"Bundle payload mismatch: {entry['file']}")


def verify_bundle(output: Path) -> dict:
    """Verify one table, original source joins and the no-repair selection policy."""
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("dataset_key") != KEY:
        raise ValueError("Not the selected sihumix dataset")
    _check_payloads(output, manifest)
    with gzip.open(output / "aam.csv.gz", "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise ValueError("sihumix table schema mismatch")
        rows = list(reader)
    by_id = {r["r_id"]: r for r in rows}
    if len(by_id) != len(rows) or len(rows) != manifest["rows"]:
        raise ValueError("Duplicate or missing memberships")
    report = json.loads((output / "build_report.json").read_text())
    quality, expected = {}, set()
    members = _preserved_members(output, report)
    for name, payload in members.items():
        if not name.startswith("data/SIHUMIx_cAAMsGEMs/") or not name.endswith(".json"):
            continue
        organism = Path(name).stem
        for identifier, native in json.loads(payload)["reactions"].items():
            aam = (native.get("selected_matching") or {}).get("aam") or ""
            if aam not in quality:
                quality[aam] = inspect_mapping(aam)
            checked = quality[aam]
            if _selected(aam, checked):
                continue
            rid = f"sihumix:{organism}:{identifier}"
            expected.add(rid)
            row = by_id.get(rid)
            if row is None or row["original_aam"] != aam or row["original_id"] != identifier:
                raise ValueError("Native mapping or source identifier changed")
            if row["source"] != source_label(native.get("matching_source"), True):
                raise ValueError("Native source label changed")
            if row["ground_truth"] != checked["ground_truth"] or row["rxn"] != checked["rxn"]:
                raise ValueError("Source endpoint or atom correspondence changed")
            if row["license"] != "CC-BY-4.0":
                raise ValueError("Source license field changed")
    if expected != set(by_id):
        raise ValueError("Dataset selection differs from the native source policy")
    return {
        "dataset_key": KEY,
        "verified_files": len(manifest["files"]),
        "records": len(rows),
        "all_source_mappings_and_memberships_verified": True,
        "hydrogen_completion": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--fetch", action="store_true", help="Download the checksum-pinned upstream commit if needed")
    parser.add_argument("--verify", action="store_true", help="Verify the selected sihumix dataset")
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify_bundle(args.output), indent=2))
        return 0
    source = json.loads((args.metadata / "source.json").read_text())
    if not args.archive.is_file() and not args.fetch:
        parser.error("Pinned archive absent; use --fetch or provide --archive")
    if args.fetch:
        acquire_archive(args.archive, source)
    report = build_bundle(args.archive, args.output, source, (args.metadata / "LICENSE-CC-BY-4.0.txt").read_bytes())
    print(
        json.dumps(
            {
                "dataset_key": KEY,
                "output": str(args.output),
                "records": report["rows"],
                "verification": verify_bundle(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
