"""Build and verify the family/task data tree and its source metadata."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import tempfile

from synrxn.reconstruction.paths import REPO, physical_path
from synrxn.reconstruction.catalogue.hierarchy import NAVIGATION_KEYS

SCHEMA = "synrxn.family-layout.v1"
MANIFEST = ".synrxn-layout.json"
MANAGER = "synrxn.reconstruction.catalogue.layout"
PREVIOUS_MANAGER = "synrxn.reconstruction.layout"
INPUTS = ("kdd/task_registry.json", "kdd/hierarchy/hierarchy.json",
          "kdd/coverage/release_manifest.json")
FORMATS = {"csv", "csv.gz", "parquet", "jsonl", "jsonl.gz"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _relative(value):
    """Require portable, nonescaping logical identities and file paths."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(
            part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part)
            for part in path.parts):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path


def _component(value):
    result = _relative(value)
    if len(result.parts) != 1:
        raise ValueError(f"Expected one identity component: {value}")
    return result.as_posix()


def _file(repo, relative):
    logical = physical_path(_relative(relative), repo)
    resolved = logical.resolve(strict=True)
    if not resolved.is_relative_to(repo) or not resolved.is_file():
        raise ValueError(f"Missing or escaping source file: {relative}")
    return logical


def _identity(entry):
    namespace, key = _component(entry["namespace"]), _relative(entry["key"]).as_posix()
    return namespace, key


def _unique(items, key, label):
    result = {}
    for item in items:
        identity = key(item)
        if identity in result:
            raise ValueError(f"Duplicate {label}: {identity}")
        result[identity] = item
    return result


def _output_path(repo, output):
    path = Path(output)
    if not path.is_absolute():
        path = repo / path
    if path.is_symlink():
        raise ValueError("Layout output must not itself be a symbolic link")
    path = path.resolve()
    if not path.is_relative_to(repo) or path == repo:
        raise ValueError("Layout output must be a subdirectory of the repository")
    if path.exists() and not path.is_dir():
        raise ValueError("Layout output must be a directory")
    return path


def _load(repo):
    documents = []
    inputs = {}
    for name in INPUTS:
        path = _file(repo, name)
        documents.append(json.loads(path.read_text()))
        inputs[name] = digest(path)
    registry, hierarchy, release = documents
    if release.get("status") != "passed":
        raise ValueError("The source release audit must have passed")
    families = _unique(registry["families"], lambda x: _component(x["id"]), "family")
    tasks = _unique(registry["tasks"], lambda x: _component(x["id"]), "task")
    h_families = _unique(hierarchy["families"], lambda x: _component(x["id"]),
                         "hierarchy family")
    if set(families) != set(h_families):
        raise ValueError("Registry and hierarchy families disagree")
    admitted = _unique(release["release_tables"], _identity, "admitted table")
    held = _unique(release["held_tables"], _identity, "held table")
    if set(admitted) & set(held):
        raise ValueError("A table cannot be both admitted and held")
    all_bindings, tables, seen_tasks = [], {}, set()
    for family in hierarchy["families"]:
        for task in family["tasks"]:
            task_id = _component(task["id"])
            if task_id in seen_tasks or task_id not in tasks:
                raise ValueError(f"Unexpected or duplicated hierarchy task: {task_id}")
            seen_tasks.add(task_id)
            native = tasks[task_id]
            if native["family_id"] != family["id"]:
                raise ValueError(f"Task family disagreement: {task_id}")
            registry_bindings = _unique(native["bindings"], _identity, "task binding")
            hierarchy_bindings = _unique(task["tables"], _identity, "hierarchy binding")
            if set(registry_bindings) != set(hierarchy_bindings):
                raise ValueError(f"Registry/hierarchy bindings disagree: {task_id}")
            for identity, table in hierarchy_bindings.items():
                binding = registry_bindings[identity]
                rights = admitted.get(identity) or held.get(identity)
                if rights is None:
                    raise ValueError(f"Missing rights decision: {identity}")
                for field in ("rows", "role", "counting_unit"):
                    if table[field] != binding[field]:
                        raise ValueError(f"Registry/hierarchy {field} disagreement: {identity}")
                for field, pin in (("path", "path"), ("rows", "expected_rows"),
                                   ("sha256", "expected_sha256")):
                    if table[field] != rights[pin]:
                        raise ValueError(f"Hierarchy/release {field} disagreement: {identity}")
                _relative(table["path"])
                if rights["format"] not in FORMATS:
                    raise ValueError(f"Unsupported data format: {rights['format']}")
                if binding.get("release_eligible", identity in admitted) != (identity in admitted):
                    raise ValueError(f"Binding release eligibility disagreement: {identity}")
                if identity in tables:
                    for field in ("path", "rows", "sha256", "role"):
                        if tables[identity][field] != table[field]:
                            raise ValueError(f"Reused table identity changed {field}: {identity}")
                else:
                    tables[identity] = table
                if identity in admitted and os.path.lexists(physical_path(table["path"], repo)):
                    source = _file(repo, table["path"])
                    expected_size = rights.get("expected_bytes")
                    if expected_size is not None and source.stat().st_size != expected_size:
                        raise ValueError(f"Source byte count disagrees: {identity}")
                all_bindings.append({"family_id": family["id"], "task_id": task_id,
                                     "table": table, "binding": binding,
                                     "rights": rights, "admitted": identity in admitted})
    if seen_tasks != set(tasks):
        raise ValueError("Registry and hierarchy tasks disagree")
    # Quarantined source views outside the registry are deliberately not included.
    if set(tables) - (set(admitted) | set(held)):
        raise ValueError("Inventory contains a table without a release decision")
    return {"registry": registry, "hierarchy": hierarchy, "release": release,
            "families": families, "tasks": tasks, "tables": tables,
            "bindings": all_bindings, "admitted": admitted, "held": held,
            "inputs": inputs}


def _summary(inventory, bindings=None):
    entries = inventory["bindings"] if bindings is None else bindings
    unique = {_identity(b["table"]): b for b in entries}
    return {
        "families": len({b["family_id"] for b in entries}),
        "tasks": len({b["task_id"] for b in entries}),
        "task_table_bindings": len(entries),
        "unique_table_views": len(unique),
        "primary_table_views": sum(b["table"]["role"] != "overlapping_view" for b in unique.values()),
        "overlapping_table_views": sum(b["table"]["role"] == "overlapping_view" for b in unique.values()),
        "admitted_table_views": sum(b["admitted"] for b in unique.values()),
        "held_table_views": sum(not b["admitted"] for b in unique.values()),
        "data_links": sum(b["admitted"] for b in entries),
        "held_bindings": sum(not b["admitted"] for b in entries),
        "stored_rows_across_unique_views": sum(b["table"]["rows"] for b in unique.values()),
    }


def _plan(repo, output, inventory):
    files, links = {}, {}
    owners = {}
    summary = _summary(inventory)
    counting = ("Stored rows retain their source-specific units. Shared tables appear under "
                "multiple tasks; do not add task or family row totals to count independent "
                "experiments. Explicit overlapping views are separate catalogue views.")

    def document(path, value):
        if str(path) in files:
            raise ValueError(f"Generated metadata paths collide: {path}")
        files[str(path)] = _json_bytes(value)

    index = {"schema": SCHEMA, "summary": summary, "counting_policy": counting,
             "families": [], "tables": []}
    for family in inventory["registry"]["families"]:
        family_id = family["id"]
        family_bindings = [b for b in inventory["bindings"] if b["family_id"] == family_id]
        family_summary = _summary(inventory, family_bindings)
        task_nodes = []
        for task in inventory["registry"]["tasks"]:
            if task["family_id"] != family_id:
                continue
            task_id = task["id"]
            task_bindings = [b for b in family_bindings if b["task_id"] == task_id]
            base = PurePosixPath(family_id, task_id)
            table_nodes = []
            for entry in task_bindings:
                table, rights = entry["table"], entry["rights"]
                namespace, key = _identity(table)
                display_key = _relative(NAVIGATION_KEYS.get((namespace, key), key)).as_posix()
                table_dir = base / namespace / display_key
                metadata_path = table_dir / "table.json"
                data_name = "data." + rights["format"] if entry["admitted"] else None
                if data_name:
                    link_path = table_dir / data_name
                    owner = owners.setdefault((namespace, key), str(link_path))
                    if str(link_path) != owner:
                        if str(link_path) in links:
                            raise ValueError(f"Generated data-link paths collide: {link_path}")
                        links[str(link_path)] = os.path.relpath(output / owner, output / table_dir)
                stored = output / owner if data_name else physical_path(table["path"], repo)
                source_fields = ("path", "sha256", "source_catalogue", "source_id",
                                 "source_urls", "citations")
                metadata = {"schema": SCHEMA, "family_id": family_id, "task_id": task_id,
                            "namespace": namespace, "key": key, "title": table["title"],
                            **({"display_key": display_key} if display_key != key else {}),
                            "rows": table["rows"], "counting_unit": table["counting_unit"],
                            "role": table["role"], "format": rights["format"],
                            "release_admitted": entry["admitted"], "data_link": data_name,
                            "source": {k: table[k] for k in source_fields if k in table},
                            "stored_path": stored.relative_to(repo).as_posix(),
                            "source_table_metadata": table, "registry_binding": entry["binding"],
                            "rights": rights}
                document(metadata_path, metadata)
                table_nodes.append({"namespace": namespace, "key": key,
                                    **({"display_key": display_key} if display_key != key else {}),
                                    "metadata": str(metadata_path.relative_to(base)),
                                    "data_link": str((table_dir / data_name).relative_to(base))
                                    if data_name else None})
            task_metadata = {"schema": SCHEMA, "task": task, "tables": table_nodes,
                             "summary": _summary(inventory, task_bindings),
                             "counting_policy": counting}
            document(base / "task.json", task_metadata)
            task_nodes.append({"id": task_id, "title": task["title"],
                               "metadata": f"{task_id}/task.json"})
        document(PurePosixPath(family_id, "family.json"),
                 {"schema": SCHEMA, "family": family, "tasks": task_nodes,
                  "summary": family_summary, "counting_policy": counting})
        index["families"].append({"id": family_id, "title": family["title"],
                                  "metadata": f"{family_id}/family.json",
                                  "summary": family_summary})
    for identity, table in inventory["tables"].items():
        bindings = [b for b in inventory["bindings"] if _identity(b["table"]) == identity]
        index["tables"].append({"namespace": identity[0], "key": identity[1],
                                "source_path": table["path"], "sha256": table["sha256"],
                                "storage_path": owners.get(identity),
                                "rows": table["rows"], "release_admitted": bindings[0]["admitted"],
                                "bindings": [{"family_id": b["family_id"], "task_id": b["task_id"]}
                                             for b in bindings]})
    document("index.json", index)
    payloads = {name: {"source_path": inventory["tables"][identity]["path"],
                       "sha256": inventory["tables"][identity]["sha256"],
                       "bytes": inventory["admitted"][identity].get("expected_bytes")}
                for identity, name in owners.items()}
    paths = set(files) | set(links) | set(payloads)
    if len(paths) != len(files) + len(links) + len(owners):
        raise ValueError("Generated metadata and data paths collide")
    for value in paths:
        _relative(value)
        if any(str(parent) in paths for parent in PurePosixPath(value).parents if str(parent) != "."):
            raise ValueError(f"Generated paths collide: {value}")
    directories = sorted({str(parent) for name in paths for parent in PurePosixPath(name).parents
                          if str(parent) != "."})
    generator = Path(__file__).resolve()
    try:
        generator_name = generator.relative_to(repo).as_posix()
    except ValueError:
        generator_name = generator.as_posix()
    manifest = {"schema": SCHEMA, "managed_by": MANAGER,
                "generator": {"path": generator_name, "sha256": digest(generator),
                              "naming_sha256": digest(generator.with_name("hierarchy.py"))},
                "python_version": platform.python_version(),
                "input_sha256": inventory["inputs"], "summary": summary,
                "counting_policy": counting, "directories": directories,
                "files": {name: {"sha256": hashlib.sha256(value).hexdigest(),
                                 "bytes": len(value)} for name, value in sorted(files.items())},
                "symlinks": dict(sorted(links.items())), "payloads": payloads}
    return files, links, manifest


def _physical_inventory(output):
    files, links, directories = set(), set(), set()
    for current, dirs, names in os.walk(output, followlinks=False):
        for name in list(dirs):
            path = Path(current) / name
            relative = path.relative_to(output).as_posix()
            if path.is_symlink():
                links.add(relative)
                dirs.remove(name)
            else:
                directories.add(relative)
        for name in names:
            path = Path(current) / name
            relative = path.relative_to(output).as_posix()
            (links if path.is_symlink() else files).add(relative)
    return files, links, directories


def _managed_manifest(output, *, allow_missing_links=False):
    """Check the previous generation before replacing any managed output."""
    marker = output / MANIFEST
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("Existing layout is unmanaged; refusing to replace it")
    manifest = json.loads(marker.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("managed_by") not in {MANAGER, PREVIOUS_MANAGER}:
        raise ValueError("Existing layout has an unknown management marker")
    files, links, directories = _physical_inventory(output)
    payloads = manifest.get("payloads", {})
    materialized = files & set(manifest["symlinks"]) if "payloads" not in manifest else set()
    expected_links = set(manifest["symlinks"])
    present_links = links | materialized
    if (files - materialized - set(payloads) != set(manifest["files"]) | {MANIFEST}
            or present_links - expected_links
            or (not allow_missing_links and present_links != expected_links)):
        raise ValueError("Layout contains missing, extra or unmanaged files/links")
    if directories != set(manifest["directories"]):
        raise ValueError("Layout contains missing, extra or unmanaged directories")
    for name, pin in manifest["files"].items():
        _relative(name)
        path = output / name
        if path.stat().st_size != pin["bytes"] or digest(path) != pin["sha256"]:
            raise ValueError(f"Managed metadata was modified: {name}")
    for name, target in manifest["symlinks"].items():
        _relative(name)
        path = output / name
        if name in materialized:
            metadata = path.with_name("table.json")
            if metadata.relative_to(output).as_posix() not in manifest["files"]:
                raise ValueError(f"Materialized data lacks managed metadata: {name}")
            if digest(path) != json.loads(metadata.read_text())["source"]["sha256"]:
                raise ValueError(f"Materialized data was modified: {name}")
        elif name in links and os.readlink(path) != target:
            raise ValueError(f"Managed data link was modified: {name}")
    return manifest


def _check_hashes(repo, inventory, output, materialized, allow_missing):
    checked = []
    prepared = {}
    for name in sorted(materialized):
        path = output / name
        metadata = json.loads(path.with_name("table.json").read_text())
        prepared[_identity(metadata)] = path
    for identity, table in sorted(inventory["tables"].items()):
        if identity not in inventory["admitted"]:
            continue
        path = prepared.get(identity)
        if path is None:
            if allow_missing and not os.path.lexists(physical_path(table["path"], repo)):
                continue
            path = _file(repo, table["path"])
        if digest(path) != table["sha256"]:
            raise ValueError(f"Source SHA256 mismatch: {identity[0]}/{identity[1]}")
        checked.append({"namespace": identity[0], "key": identity[1], "path": table["path"]})
    return checked


def _missing_outputs(repo, output, links, materialized):
    missing = []
    for name, target in links.items():
        if name in materialized:
            continue
        resolved = (output / name).parent.joinpath(target).resolve()
        if os.path.isabs(target) or not resolved.is_relative_to(repo):
            raise ValueError(f"Data link escapes repository: {name}")
        if not resolved.is_file():
            missing.append(name)
    return missing


def _materialized_outputs(output, links):
    return {name for name in links if (output / name).is_file() and not (output / name).is_symlink()}


def verify_layout(repo=REPO, output="Data/families", check_hashes=False, *, allow_missing=False):
    """Check exact metadata/link coverage; optionally hash each admitted view once."""
    repo = Path(repo).resolve()
    output = _output_path(repo, output)
    inventory = _load(repo)
    files, links, expected = _plan(repo, output, inventory)
    actual = _managed_manifest(output, allow_missing_links=allow_missing)
    # Python patch versions do not change the scientific navigation contract.
    comparable = set(expected) - {"python_version"}
    if any(actual.get(key) != expected[key] for key in comparable):
        raise ValueError("Layout generation differs from current source metadata or generator; rebuild it")
    materialized = {name for name in expected["payloads"] if (output / name).is_file()}
    missing_owners = set(expected["payloads"]) - materialized
    missing = sorted(missing_owners)
    _missing_outputs(repo, output, links, set())
    if missing and not allow_missing:
        raise ValueError(f"Missing table payload: {missing[0]}")
    checked = _check_hashes(repo, inventory, output, materialized, allow_missing) if check_hashes else []
    return {"schema": SCHEMA, "status": "passed", "output": output.relative_to(repo).as_posix(),
            "summary": expected["summary"], "input_sha256": inventory["inputs"],
            "source_hashes_checked": len(checked), "source_hash_checks": checked,
            "materialized_tables": len(materialized), "pending_payloads": len(missing),
            "source_row_counts": "Pinned metadata checked for agreement; payload rows were not recounted.",
            "held_payloads": "Metadata only; payloads are not linked or hashed."}


@contextmanager
def _layout_lock(repo, output):
    key = hashlib.sha256(os.fsencode(output)).hexdigest()
    directory = repo / ".cache/synrxn/layout-locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / key).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def build_layout(repo=REPO, output="Data/families", check_hashes=False, *, allow_missing=False):
    """Atomically publish metadata and retain prepared table files."""
    repo = Path(repo).resolve()
    output = _output_path(repo, output)
    with _layout_lock(repo, output):
        return _build_layout(repo, output, check_hashes, allow_missing)


def _build_layout(repo, output, check_hashes, allow_missing):
    inventory = _load(repo)
    files, links, manifest = _plan(repo, output, inventory)
    for entry in inventory["tables"].values():
        source = physical_path(entry["path"], repo).resolve()
        if source.is_relative_to(output) and output != repo / "Data/families":
            raise ValueError("Layout output contains a pinned source file")
    sources = {}
    if output.exists() and any(output.iterdir()):
        previous = _managed_manifest(output, allow_missing_links=True)
        if "payloads" in previous:
            for name, pin in previous["payloads"].items():
                if (output / name).is_file():
                    sources[pin["source_path"]] = output / name
        else:
            for name in _materialized_outputs(output, previous["symlinks"]):
                path = output / name
                metadata = json.loads(path.with_name("table.json").read_text())
                sources.setdefault(metadata["source"]["path"], path)
    current_keys = {pin["source_path"] for pin in manifest["payloads"].values()}
    if set(sources) - current_keys:
        raise ValueError("Layout update would remove materialized data")
    prepared, checked = {}, []
    for name, pin in manifest["payloads"].items():
        logical = pin["source_path"]
        path = sources.get(logical, physical_path(logical, repo))
        if not path.is_file():
            if not allow_missing:
                raise ValueError(f"Missing table payload: {name}")
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(repo):
            raise ValueError(f"Missing or escaping source file: {logical}")
        if digest(path) != pin["sha256"]:
            raise ValueError(f"Source SHA256 mismatch: {logical}")
        prepared[name] = path
        if check_hashes:
            metadata = json.loads(files[str(PurePosixPath(name).with_name("table.json"))])
            checked.append({"namespace": metadata["namespace"], "key": metadata["key"], "path": logical})
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + "-stage-", dir=output.parent))
    backup = None
    try:
        for name, payload in files.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        for name, source in prepared.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if source.is_relative_to(output):
                os.link(source, path)
            else:
                shutil.copyfile(source, path)
        for name, target in links.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)
        (stage / MANIFEST).write_bytes(_json_bytes(manifest))
        if output.exists():
            # Recheck immediately before replacement to preserve intervening edits.
            if any(output.iterdir()):
                _managed_manifest(output, allow_missing_links=True)
            backup = Path(tempfile.mkdtemp(prefix="." + output.name + "-previous-", dir=output.parent))
            backup.rmdir()
            output.rename(backup)
        try:
            stage.rename(output)
        except BaseException:
            if backup is not None:
                backup.rename(output)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    result = verify_layout(repo, output, check_hashes=False, allow_missing=allow_missing)
    result["source_hashes_checked"] = len(checked)
    result["source_hash_checks"] = checked
    return result


def place_table(source, family, task, namespace, key, repo=REPO, output="Data/families"):
    """Place a checksum-verified table in a managed family/task directory."""
    repo = Path(repo).resolve()
    output = _output_path(repo, output)
    inventory = _load(repo)
    files, links, expected = _plan(repo, output, inventory)
    display_key = NAVIGATION_KEYS.get((namespace, key), key)
    directory = PurePosixPath(_component(family), _component(task), _component(namespace), _relative(display_key))
    metadata = json.loads(files[str(directory / "table.json")])
    if not metadata["release_admitted"]:
        raise ValueError(f"Table is held by the release manifest: {namespace}/{key}")
    if digest(source) != metadata["source"]["sha256"]:
        raise ValueError(f"Prepared table differs from the pinned release: {namespace}/{key}")
    name = str(directory / metadata["data_link"])
    owner = name if name in expected["payloads"] else str(
        PurePosixPath(os.path.normpath(str(PurePosixPath(name).parent / links[name])))
    )
    with _layout_lock(repo, output):
        manifest = (_managed_manifest(output, allow_missing_links=True)
                    if (output / MANIFEST).exists() else {})
        if any(manifest.get(field) != expected[field]
               for field in ("managed_by", "generator", "input_sha256", "payloads")) or any(
                   not (output / link).is_symlink() for link in links):
            _build_layout(repo, output, check_hashes=False, allow_missing=True)
        destination = output / owner
        if destination.is_file():
            if digest(destination) != metadata["source"]["sha256"]:
                raise ValueError("Existing family payload differs from the pinned release")
            return output / name
        with tempfile.TemporaryDirectory(prefix=".synrxn-table-", dir=output.parent) as temporary:
            staged = Path(temporary) / destination.name
            shutil.copyfile(source, staged)
            if digest(staged) != metadata["source"]["sha256"]:
                raise ValueError("Prepared table changed during placement")
            staged.replace(destination)
        return output / name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--output", type=Path, default=Path("Data/families"))
    parser.add_argument("--check-hashes", action="store_true",
                        help="Hash each admitted source view once; never recount payload rows")
    parser.add_argument("--allow-missing", action="store_true", help="Allow pending tables in a partial checkout")
    args = parser.parse_args(argv)
    run = build_layout if args.command == "build" else verify_layout
    print(json.dumps(run(args.repo, args.output, args.check_hashes, allow_missing=args.allow_missing), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
