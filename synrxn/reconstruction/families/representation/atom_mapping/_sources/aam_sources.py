"""Acquire pinned AAM source products and dispatch the original native recipes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

from synrxn.reconstruction.paths import REPO, physical_path

REGISTRY = REPO / "Data/metadata/reconstruction/aam/registry.json"


def digest(path: Path) -> str:
    """Hash complete file bytes without retaining the payload in memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf8")


def read_registry(path: Path = REGISTRY) -> dict:
    registry = json.loads(path.read_text(encoding="utf8"))
    require(registry["schema"] == "synrxn.aam-sources.v1", "Unknown AAM source registry")
    names = [entry["name"] for entry in registry["datasets"]]
    require(len(names) == len(set(names)), "Duplicate dataset names in registry")
    require(
        sum(entry["rows"] for entry in registry["datasets"]) == registry["expected_rows"], "Registry count mismatch"
    )
    return registry


def checked_file(pin: dict, root: Path = REPO) -> Path:
    path = physical_path(pin["path"], root).resolve()
    require(path.is_relative_to(root.resolve()), f"Source outside repository: {pin['path']}")
    require(path.is_file(), f"Missing source: {path}")
    require(digest(path) == pin["sha256"], f"Source checksum mismatch: {pin['path']}")
    return path


def download(pin: dict, root: Path = REPO) -> Path:
    """Acquire an archive atomically and reject changed upstream bytes."""
    target = root / pin["path"]
    if target.is_file():
        return checked_file(pin, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            request = urllib.request.Request(pin["url"], headers={"User-Agent": "SynRXN-pinned-AAM-rebuild"})
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, handle, 1024 * 1024)
        require(digest(temporary) == pin["sha256"], "Downloaded archive checksum mismatch")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def published_product(entry: dict, registry: dict, fetch: bool, root: Path = REPO) -> Path:
    """Restore exact historical curated products, without rerunning a mapper."""
    target = physical_path(Path("Data/raw/aam", entry["name"], "published.csv.gz"), root)
    if target.is_file():
        require(digest(target) == entry["product_sha256"], f"Source checksum mismatch: {target}")
        return target
    frozen = physical_path(entry["source_product"], root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if frozen.is_file():
        require(digest(frozen) == entry["product_sha256"], f"Frozen source checksum mismatch: {frozen}")
        shutil.copyfile(frozen, target)
        return target
    archive_pin = registry["published_archive"]
    archive = root / archive_pin["path"]
    if fetch:
        archive = download(archive_pin, root)
    else:
        require(archive.is_file(), f"Missing {entry['name']} source product; use --fetch for the pinned release")
        checked_file(archive_pin, root)
    temporary = target.with_suffix(".part")
    try:
        with zipfile.ZipFile(archive) as package:
            suffix = entry["archive_member"]
            members = [name for name in package.namelist() if name == suffix or name.endswith("/" + suffix)]
            require(len(members) == 1, f"Archive member missing or ambiguous: {suffix}")
            with package.open(members[0]) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
        require(digest(temporary) == entry["product_sha256"], f"Published member checksum mismatch: {suffix}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def check_native_implementation(entry: dict, root: Path = REPO) -> None:
    """Check maintained rebuild code against the migration of its historical pin."""
    pins = entry.get("native_recipe", {}).get("implementation", [])
    if not pins:
        return
    manifest = json.loads(
        (root / "synrxn/reconstruction/provenance/migration.json").read_text(encoding="utf8")
    )
    migrated = {item["historical_path"]: item
                for item in (*manifest["entries"], *manifest.get("relocated_modules", []))}
    for pin in pins:
        record = migrated.get(pin["path"])
        if record is not None:
            require(pin["sha256"] == record["original_sha256"],
                    f"Historical implementation pin changed: {pin['path']}")
            pin = {"path": record["module_path"], "sha256": record["module_sha256"]}
        checked_file(pin, root)


def native_command(entry: dict, output: Path, args) -> list[str]:
    """Keep native source processing separate from the common export schema."""
    python = args.processor_python
    name = entry["name"]
    if name == "sihumix":
        command = [python, "-m", "synrxn.reconstruction.families.representation.atom_mapping._sources.sihumix", "--output", str(output)]
        if args.fetch:
            command.append("--fetch")
        return command
    if name == "flower":
        command = [python, "-m", "synrxn.reconstruction.families.representation.atom_mapping._sources.flower_representatives",
                   "--output", str(output)]
        if getattr(args, "rebuild_sources", False):
            command.append("--rebuild")
        if args.flower_work_root:
            command += ["--work-root", str(args.flower_work_root)]
        return command
    require(name in {"metamdb", "metambd_cyclic"}, f"Unknown native recipe: {name}")
    command = [python, "-m", "synrxn.reconstruction.families.representation.atom_mapping._sources.build_metamdb_aam",
               "--datasets", name, "--output", str(output)]
    if args.fetch:
        command.append("--fetch")
    return command


def verify_native_manifest(folder: Path) -> None:
    manifest = json.loads((folder / "manifest.json").read_text())
    files = manifest["files"]
    pins = (
        files if isinstance(files, list) else [{"file": name, "sha256": checksum} for name, checksum in files.items()]
    )
    for pin in pins:
        path = (folder / pin["file"]).resolve()
        require(path.is_relative_to(folder.resolve()), "Unsafe native manifest path")
        require(digest(path) == pin["sha256"], f"Native bundle checksum mismatch: {path.name}")


def prepare_source(entry: dict, registry: dict, args, root: Path = REPO) -> Path:
    """Resolve a pinned product; never trust an existing common benchmark output."""
    checked_file(entry["source_lock"], root)
    for evidence in entry["evidence"]:
        checked_file(evidence, root)
    if entry["mode"] == "published_product":
        return published_product(entry, registry, args.fetch, root)
    source = physical_path(entry["source_product"], root)
    cache = (root / ".cache/flower/native-product" if entry["name"] == "flower"
             else physical_path(Path("Data/raw/aam", entry["name"], "native-product"), root))
    cached = cache / "aam.csv.gz"
    if cached.is_file() and not args.rebuild_sources:
        source = cached
    elif args.rebuild_sources or not source.is_file():
        check_native_implementation(entry, root)
        if "native_archive" in entry:
            checked_file(entry["native_archive"], root)
        subprocess.run(native_command(entry, cache, args), cwd=root, check=True)
        source = cached
    require(digest(source) == entry["product_sha256"], f"Native product changed: {entry['name']}")
    if entry.get("native_bundle"):
        verify_native_manifest(source.parent)
    return source


def source_metadata(entry: dict, source: Path, registry: dict, root: Path = REPO) -> dict:
    """Describe source scope and rights without promoting a legacy licence tag."""
    result = {
        "schema": "synrxn.aam-provenance.v1",
        "dataset_key": "aam/" + entry["name"],
        "source_mode": entry["mode"],
        "source_product_sha256": digest(source),
        "source_lock_sha256": entry["source_lock"]["sha256"],
        "source_columns": entry["product_columns"],
        "source_lock": json.loads(checked_file(entry["source_lock"], root).read_text()),
        "rights": entry["rights"],
        "qualification": entry["qualification"],
        "license_evidence": entry["evidence"],
        "rights_evidence_records": entry.get("rights_evidence_records", []),
    }
    if entry["mode"] == "published_product":
        result["published_release"] = {**registry["published_archive"], "member": entry["archive_member"]}
    if "native_recipe" in entry:
        result["native_recipe"] = entry["native_recipe"]
    if 'center_annotations' in entry:
        result['center_annotations'] = entry['center_annotations']
    return result


def processor_default() -> str:
    return sys.executable
