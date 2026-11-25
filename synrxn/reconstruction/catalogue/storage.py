#!/usr/bin/env python3
"""Build and verify DuckDB catalogues or self-contained dataset snapshots."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zlib
from datetime import datetime, timezone

from synrxn.reconstruction.catalogue import compact, evaluation
from synrxn.reconstruction.catalogue.release_scope import MAIN, SCOPES, select_scope
from synrxn.reconstruction.paths import REPO as ROOT, source_path

SCHEMA = "synrxn.duckdb-catalogue.v1"
ASSET_CHUNK_BYTES = 256 * 1024
MATERIALIZED_ROW_GROUP_SIZE = 2048
TRANSITION1X_ASSET = "Data/raw/coverage/physics/Transition1x.h5"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def local_file(root: Path, relative: str) -> Path:
    """Require an existing regular file inside the repository, including symlinks."""
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"Unsafe repository path: {relative}")
    path = source_path(value, repo=root).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or escaping repository file: {relative}")
    return path


def check_file(root: Path, relative: str, expected: str) -> dict:
    path = local_file(root, relative)
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {relative}: expected {expected}, got {actual}")
    return {"path": relative, "resolved_source_path": path.relative_to(root.resolve()).as_posix(),
            "sha256": actual, "bytes": path.stat().st_size}


def check_table_source(root: Path, key: tuple, table: dict, rights: dict, admitted: bool) -> dict:
    for field, right_field in (("path", "path"), ("sha256", "expected_sha256"),
                               ("rows", "expected_rows")):
        if table[field] != rights[right_field]:
            raise ValueError(f"Hierarchy/release disagreement for {key}: {field}")
    if not admitted:
        return {}
    checked = check_file(root, table["path"], table["sha256"])
    if rights.get("expected_bytes") is not None and checked["bytes"] != rights["expected_bytes"]:
        raise ValueError(f"Byte count disagrees for {table['path']}")
    return {table["path"]: checked}


def merge_input_hashes(expected: dict, additions) -> None:
    for path, sha256 in additions:
        previous = expected.get(path)
        if previous and previous != sha256:
            raise ValueError(f"Conflicting source hashes: {path}")
        expected[path] = sha256


def load_inventory(root: Path, scope=MAIN) -> dict:
    """Cross-check generated hierarchy, registry, rights decisions and source files."""
    names = ("kdd/task_registry.json", "kdd/hierarchy/hierarchy.json",
             "kdd/coverage/release_manifest.json")
    registry, hierarchy, release = [json.loads(local_file(root, p).read_text()) for p in names]
    if release.get("status") != "passed":
        raise ValueError("Release manifest must have passed its audit")
    checks = {}
    expected_inputs = dict(registry["source_sha256"])
    merge_input_hashes(expected_inputs, hierarchy["input_sha256"].items())
    merge_input_hashes(expected_inputs, ((entry["path"], entry["sha256"]) for entry in release["catalogs"]))
    for name, expected in sorted(expected_inputs.items()):
        checks[name] = check_file(root, name, expected)
    for name in names:
        path = local_file(root, name)
        checks[name] = {"path": name, "sha256": digest(path), "bytes": path.stat().st_size}

    admitted = {(r["namespace"], r["key"]): r for r in release["release_tables"]}
    held = {(r["namespace"], r["key"]): r for r in release["held_tables"]}
    tables, bindings = {}, []
    for family in hierarchy["families"]:
        for task in family["tasks"]:
            for entry in task["tables"]:
                key = (entry["namespace"], entry["key"])
                if key in tables and tables[key] != entry:
                    # Titles and counting units can be task-specific; physical identity cannot.
                    for field in ("path", "sha256", "rows", "source_catalogue"):
                        if tables[key][field] != entry[field]:
                            raise ValueError(f"Inconsistent table {key}: {field}")
                tables.setdefault(key, entry)
                bindings.append({"family_id": family["id"], "task_id": task["id"], **entry})
    registry_bindings = {(t["id"], b["namespace"], b["key"], b["role"])
                         for t in registry["tasks"] for b in t["bindings"]}
    hierarchy_bindings = {(b["task_id"], b["namespace"], b["key"], b["role"])
                          for b in bindings}
    if registry_bindings != hierarchy_bindings or len(hierarchy_bindings) != len(bindings):
        raise ValueError("Registry and hierarchy task bindings disagree or contain duplicates")
    registry, hierarchy = select_scope(registry, hierarchy, scope)
    task_ids = {task["id"] for task in registry["tasks"]}
    bindings = [binding for binding in bindings if binding["task_id"] in task_ids]
    selected_keys = {(binding["namespace"], binding["key"]) for binding in bindings}
    tables = {key: table for key, table in tables.items() if key in selected_keys}
    for key, table in sorted(tables.items()):
        rights = admitted.get(key) or held.get(key)
        if not rights:
            raise ValueError(f"No release decision for catalogue table {key}")
        checks.update(check_table_source(root, key, table, rights, key in admitted))
    return {"registry": registry, "hierarchy": hierarchy, "release": release,
            "tables": tables, "bindings": bindings, "admitted": admitted,
            "held": held, "checks": checks}


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def view_name(namespace: str, key: str) -> str:
    identity = namespace + ":" + key
    label = re.sub(r"[^a-z0-9_]", "_", identity.lower())
    return "v_" + label + "_" + hashlib.sha256(identity.encode()).hexdigest()[:10]


def view_query(path: Path, file_format: str) -> str:
    value = sql_string(str(path.resolve()))
    if file_format in ("csv", "csv.gz"):
        # Preserve leading zeros, identifiers, source labels and numeric source text.
        return f"SELECT * FROM read_csv({value}, header=true, all_varchar=true, delim=',', quote='\"', escape='\"')"
    if file_format == "parquet":
        return f"SELECT * FROM read_parquet({value})"
    if file_format in ("jsonl", "jsonl.gz"):
        # Raw JSON avoids schema inference that coerces heterogeneous native arrays.
        return f"SELECT json AS payload FROM read_json_objects({value}, format='newline_delimited')"
    raise ValueError(f"Unsupported catalogue format: {file_format}")


def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def populate(connection, root: Path, inventory: dict, metadata_only: bool,
             materialize: bool = False, progress=None) -> list[dict]:
    registry = inventory["registry"]
    connection.execute("CREATE SCHEMA catalog; CREATE SCHEMA data")
    connection.execute("""CREATE TABLE catalog.families (
        family_id VARCHAR PRIMARY KEY, title VARCHAR, metadata_json JSON)""")
    connection.execute("""CREATE TABLE catalog.tasks (
        task_id VARCHAR PRIMARY KEY, family_id VARCHAR, title VARCHAR, metadata_json JSON)""")
    connection.execute("""CREATE TABLE catalog.dataset_tables (
        namespace VARCHAR, table_key VARCHAR, title VARCHAR, source_path VARCHAR,
        sha256 VARCHAR, bytes BIGINT, stored_rows BIGINT, format VARCHAR,
        source_catalogue VARCHAR, source_id VARCHAR, source_urls_json JSON,
        release_admitted BOOLEAN, effective_license_expression VARCHAR,
        release_partition VARCHAR, rights_status VARCHAR, rights_json JSON,
        view_name VARCHAR, view_status VARCHAR, view_schema_json JSON,
        PRIMARY KEY (namespace, table_key))""")
    connection.execute("""CREATE TABLE catalog.task_table_bindings (
        family_id VARCHAR, task_id VARCHAR, namespace VARCHAR, table_key VARCHAR,
        role VARCHAR, counting_unit VARCHAR, qualification VARCHAR,
        binding_json JSON, PRIMARY KEY (task_id, namespace, table_key))""")
    connection.execute("CREATE TABLE catalog.verified_inputs (path VARCHAR PRIMARY KEY, sha256 VARCHAR, bytes BIGINT)")
    connection.executemany("INSERT INTO catalog.families VALUES (?, ?, ?)",
                           [(f["id"], f["title"], json_text(f)) for f in registry["families"]])
    connection.executemany("INSERT INTO catalog.tasks VALUES (?, ?, ?, ?)",
                           [(t["id"], t["family_id"], t["title"], json_text(t)) for t in registry["tasks"]])
    connection.executemany("INSERT INTO catalog.verified_inputs VALUES (?, ?, ?)",
                           [(v["path"], v["sha256"], v["bytes"]) for v in inventory["checks"].values()])
    registry_binding_map = {(t["id"], b["namespace"], b["key"]): b
                            for t in registry["tasks"] for b in t["bindings"]}
    mapping = []
    for index, (key, table) in enumerate(sorted(inventory["tables"].items()), 1):
        admitted = key in inventory["admitted"]
        rights = (inventory["admitted"] if admitted else inventory["held"])[key]
        name = view_name(*key) if admitted and not metadata_only else None
        status = "held_no_view" if not admitted else "metadata_only" if metadata_only else "file_view"
        columns, query = None, None
        if name:
            if progress:
                progress(f"Table {index}/{len(inventory['tables'])}: {key[0]}/{key[1]}")
            try:
                query = view_query(local_file(root, table["path"]), rights["format"])
                relation = "TABLE" if materialize else "VIEW"
                connection.execute(f'CREATE {relation} data."{name}" AS {query}')
                if materialize:
                    actual_rows = connection.execute(f'SELECT count(*) FROM data."{name}"').fetchone()[0]
                    if actual_rows != table["rows"]:
                        raise ValueError(f"Expected {table['rows']} rows, imported {actual_rows}")
                    status = "materialized_table"
                columns = [{"name": row[0], "type": row[1]} for row in
                           connection.execute(f'DESCRIBE SELECT * FROM data."{name}"').fetchall()]
            except Exception as exc:
                raise ValueError(f"Cannot bind data.{name} ({table['path']}): {exc}") from exc
        connection.execute("INSERT INTO catalog.dataset_tables VALUES (" + ",".join(["?"] * 19) + ")", [
            *key, table["title"], table["path"], table["sha256"],
            inventory["checks"].get(table["path"], {}).get("bytes", rights.get("expected_bytes")),
            table["rows"], rights["format"],
            table["source_catalogue"], table.get("source_id"), json_text(table.get("source_urls", [])),
            admitted, rights.get("license_expression"), rights.get("partition"),
            rights.get("rights_status", "admitted_by_release_manifest" if admitted else "held_by_release_manifest"),
            json_text(rights), name, status,
            json_text(columns) if columns is not None else None])
        mapping.append({"namespace": key[0], "table_key": key[1], "source_path": table["path"],
                        "release_admitted": admitted, "view": f"data.{name}" if name else None,
                        "status": status, "columns": columns, "sql": query})
    connection.executemany("INSERT INTO catalog.task_table_bindings VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
        (b["family_id"], b["task_id"], b["namespace"], b["key"], b["role"],
         b["counting_unit"], b.get("qualification"),
         json_text(registry_binding_map[b["task_id"], b["namespace"], b["key"]]))
        for b in inventory["bindings"]])
    return mapping


def asset_records(root: Path, inventory: dict) -> list[dict]:
    """Select only admitted native assets, notices and explicit metadata inputs."""
    records = {}

    def add(record, kind):
        path = record["path"]
        if path in records:
            if records[path]["sha256"] != record["sha256"]:
                raise ValueError(f"Conflicting embedded asset hashes: {path}")
            return
        records[path] = {**record, "kind": kind}

    for record in inventory["release"].get("release_native_assets", []):
        add(record, "native_source")
    for record in inventory["release"].get("release_notices", []):
        add(record, "license_notice")
    payloads = {table["path"] for table in inventory["tables"].values()}
    for record in inventory["checks"].values():
        if record["path"] not in payloads:
            add(record, "metadata")
    for relative in ("Data/metadata.yaml", "Data/LICENSE", "LICENSE", "CITATION.cff",
                     "manifest.json", "Data/metadata/sharealike-manifest.json", "manifest.schema.json",
                     "synrxn/data/task_protocols.json",
                     "synrxn/data/expansion_tasks.json", "synrxn/data/expansion_sources.json"):
        path = source_path(relative, repo=root)
        if path.is_file():
            add({"path": relative, "sha256": digest(path), "bytes": path.stat().st_size}, "metadata")
    for task in inventory["registry"]["tasks"]:
        for binding in task["bindings"]:
            key = (binding["namespace"], binding["key"])
            if key not in inventory["admitted"]:
                continue
            for fields in binding.get("task_fields", {}).values():
                for field in ("target_path", "split_metadata"):
                    relative = fields.get(field)
                    if relative:
                        path = local_file(root, relative)
                        add({"path": relative, "sha256": digest(path), "bytes": path.stat().st_size,
                             "task_id": task["id"], "source_table": list(key),
                             "license_expression": inventory["admitted"][key].get("license_expression")},
                            "task_auxiliary")
    return sorted(records.values(), key=lambda item: item["path"])


def embed_assets(connection, root: Path, records: list[dict], progress=None) -> None:
    """Store exact asset bytes in bounded, compressed chunks, deduplicated by hash."""
    connection.execute("CREATE SCHEMA assets")
    connection.execute("""CREATE TABLE catalog.assets (
        path VARCHAR PRIMARY KEY, sha256 VARCHAR, bytes BIGINT, kind VARCHAR,
        license_expression VARCHAR, codec VARCHAR, chunk_count INTEGER,
        stored_bytes BIGINT, metadata_json JSON)""")
    connection.execute("""CREATE TABLE assets.chunks (
        sha256 VARCHAR, chunk_index INTEGER, payload BLOB,
        PRIMARY KEY (sha256, chunk_index))""")
    stored = {}
    for index, record in enumerate(records, 1):
        path = local_file(root, record["path"])
        expected = record["sha256"]
        size = path.stat().st_size
        if record.get("bytes") is not None and size != record["bytes"]:
            raise ValueError(f"Asset byte count differs: {record['path']}")
        if progress:
            progress(f"Asset {index}/{len(records)}: {record['path']} ({size:,} bytes)")
        observed = hashlib.sha256()
        chunks, packed_bytes = 0, 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(ASSET_CHUNK_BYTES), b""):
                observed.update(chunk)
                if expected not in stored:
                    packed = zlib.compress(chunk, level=1)
                    connection.execute("INSERT INTO assets.chunks VALUES (?, ?, ?)", [expected, chunks, packed])
                    packed_bytes += len(packed)
                chunks += 1
        if observed.hexdigest() != expected:
            raise ValueError(f"SHA256 mismatch for embedded asset: {record['path']}")
        if expected in stored:
            previous_size, previous_chunks, packed_bytes = stored[expected]
            if (size, chunks) != (previous_size, previous_chunks):
                raise ValueError(f"Inconsistent duplicate asset: {record['path']}")
        else:
            stored[expected] = (size, chunks, packed_bytes)
        connection.execute("INSERT INTO catalog.assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            record["path"], expected, size, record["kind"], record.get("license_expression"),
            "zlib", chunks, packed_bytes, json_text(record)])


def _asset_chunks(connection, sha256: str, codec: str):
    if codec != "zlib":
        raise ValueError(f"Unsupported embedded asset codec: {codec}")
    indices = connection.execute(
        "SELECT chunk_index FROM assets.chunks WHERE sha256=? ORDER BY chunk_index", [sha256]).fetchall()
    for index, row in enumerate(indices):
        if row[0] != index:
            raise ValueError(f"Missing or unordered embedded chunk for {sha256}")
        # Fetch one bounded BLOB rather than buffering a multi-gigabyte query result.
        packed = connection.execute(
            "SELECT payload FROM assets.chunks WHERE sha256=? AND chunk_index=?", [sha256, index]).fetchone()[0]
        try:
            yield zlib.decompress(packed)
        except zlib.error as error:
            raise ValueError(f"Corrupt compressed asset chunk for {sha256}") from error


def extract_asset(database: Path, path: str, output: Path) -> dict:
    """Recover an embedded source file, checking its original hash before publication.

    :param database: Self-contained DuckDB snapshot.
    :param path: Original logical asset path from ``catalog.assets``.
    :param output: Destination file; existing files are never replaced.
    :return: Original byte count and SHA256 of the extracted asset.
    """
    import duckdb
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    connection = duckdb.connect(str(Path(database).resolve()), read_only=True,
                                config={"enable_external_access": "false"})
    try:
        record = connection.execute(
            "SELECT sha256, bytes, codec, chunk_count FROM catalog.assets WHERE path=?", [path]).fetchone()
        if record is None:
            raise KeyError(f"Asset is not embedded: {path}")
        expected, size, codec, count = record
        observed = hashlib.sha256()
        total, chunks = 0, 0
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for chunk in _asset_chunks(connection, expected, codec):
                stream.write(chunk)
                observed.update(chunk)
                total += len(chunk)
                chunks += 1
        if (total, chunks, observed.hexdigest()) != (size, count, expected):
            raise ValueError(f"Embedded asset size, chunks or SHA256 mismatch: {path}")
        # Linking publishes atomically and fails rather than replacing a destination.
        os.link(temporary, output)
        return {"path": path, "output": str(output), "bytes": total, "sha256": expected}
    finally:
        connection.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_database(database: Path, verify_assets: bool = True) -> dict:
    """Verify a portable snapshot using only the database, with external reads disabled."""
    import duckdb
    connection = duckdb.connect(str(Path(database).resolve()), read_only=True,
                                config={"enable_external_access": "false"})
    try:
        info = dict(connection.execute("SELECT key, value_json FROM catalog.build_info").fetchall())
        if json.loads(info["storage_mode"]) != "materialized":
            raise ValueError("Database is not a self-contained materialized snapshot")
        actual_views = {row[0] for row in connection.execute(
            "SELECT view_name FROM duckdb_views() WHERE schema_name='data'").fetchall()}
        compact_snapshot = json.loads(info.get("compact", "false"))
        if actual_views and not compact_snapshot:
            raise ValueError("Portable snapshots cannot contain data views")
        tables = connection.execute("""SELECT namespace, table_key, stored_rows, release_admitted,
            view_name, view_status FROM catalog.dataset_tables ORDER BY namespace, table_key""").fetchall()
        expected_relations, expected_views = set(), set()
        rows, admitted = 0, 0
        statuses = {"materialized_table", "internal_view"} if compact_snapshot else {"materialized_table"}
        for namespace, key, expected_rows, allowed, name, status in tables:
            if not allowed:
                if name is not None or status != "held_no_view":
                    raise ValueError(f"Held table exposes a data relation: {namespace}/{key}")
                continue
            if name != view_name(namespace, key) or status not in statuses:
                raise ValueError(f"Unexpected materialized relation: {namespace}/{key}")
            actual = connection.execute(f'SELECT count(*) FROM data."{name}"').fetchone()[0]
            if actual != expected_rows:
                raise ValueError(f"Row count mismatch for {namespace}/{key}: {actual} != {expected_rows}")
            (expected_views if status == "internal_view" else expected_relations).add(name)
            rows += actual
            admitted += 1
        actual_relations = {r[0] for r in connection.execute(
            "SELECT table_name FROM duckdb_tables() WHERE schema_name='data'").fetchall()}
        if actual_relations != expected_relations or actual_views != expected_views:
            raise ValueError("Database contains missing or unregistered data relations")
        assets = connection.execute(
            "SELECT DISTINCT sha256, bytes, codec, chunk_count FROM catalog.assets").fetchall()
        if verify_assets:
            for expected, size, codec, count in assets:
                observed = hashlib.sha256()
                total, chunks = 0, 0
                for chunk in _asset_chunks(connection, expected, codec):
                    observed.update(chunk)
                    total += len(chunk)
                    chunks += 1
                if (total, chunks, observed.hexdigest()) != (size, count, expected):
                    raise ValueError(f"Embedded asset size, chunks or SHA256 mismatch: {expected}")
        result = {
            "status": "passed", "external_access": False, "materialized_tables": admitted,
            "materialized_rows": rows, "catalogue_tables": len(tables),
            "embedded_asset_paths": connection.execute("SELECT count(*) FROM catalog.assets").fetchone()[0],
            "verified_unique_assets": len(assets) if verify_assets else 0,
            "asset_hashes_checked": verify_assets, "software_version": json.loads(info["software_version"]),
        }
        if compact_snapshot:
            result.update(materialized_tables=len(expected_relations), internal_views=len(expected_views),
                          accessible_data_relations=admitted)
        return result
    finally:
        connection.close()


def validate_output(root: Path, output: Path) -> Path:
    output = output.resolve()
    allowed = [root.resolve() / ".cache", root.resolve() / "reports"]
    if not any(output != base and output.is_relative_to(base) for base in allowed):
        raise ValueError("Output must be a new or empty directory below repository .cache/ or reports/")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Refusing to replace nonempty output: {output}")
    return output


def write_assets(connection, root, inventory, external_transition1x, progress):
    """Embed admitted assets and record the optional trajectory-file pin."""
    assets = asset_records(root, inventory)
    external_assets = []
    if external_transition1x:
        external_assets = [asset for asset in assets if asset["path"] == TRANSITION1X_ASSET]
        assets = [asset for asset in assets if asset["path"] != TRANSITION1X_ASSET]
        connection.execute("""CREATE TABLE catalog.external_assets (
            path VARCHAR PRIMARY KEY, sha256 VARCHAR, bytes BIGINT, metadata_json JSON)""")
        for asset in external_assets:
            connection.execute("INSERT INTO catalog.external_assets VALUES (?, ?, ?, ?)",
                               [asset["path"], asset["sha256"], asset["bytes"], json_text(asset)])
    embed_assets(connection, root, assets, progress)
    return external_assets


def describe_portable_build(manifest, verification, compact_snapshot, external_assets):
    """Report the portable build's available relations and native asset scope."""
    manifest["portability_verification"] = verification
    fields = ["materialized_tables", "materialized_rows", "embedded_asset_paths", "verified_unique_assets"]
    manifest["limits"] = [
        "Admitted data relations, source metadata, notices and the selected native assets are in the database.",
        "CSV values retain source text; check units and target qualification before numeric conversion.",
        "Embedded native files can be recovered using their original logical asset paths.",
        "Stored rows overlap across views; no predictive benchmark was executed.",
        "Per-source noncommercial, ShareAlike and other obligations remain in catalog.dataset_tables.",
    ]
    if compact_snapshot:
        fields.extend(("internal_views", "accessible_data_relations"))
        manifest["compact"] = True
        manifest["limits"].append("Internal views reproduce the pinned SQL rows from stored relations.")
    if external_assets:
        manifest["external_assets"] = external_assets
        manifest["limits"].append(
            "The Transition1x HDF5 is optional and is listed with its pin in catalog.external_assets.")
    manifest["summary"].update({key: verification[key] for key in fields})


def build(root: Path, output: Path, metadata_only: bool = False, *, materialize: bool = False,
          memory_limit: str = "4GB", threads: int = 2, progress=None, scope=MAIN,
          compact_snapshot: bool = False, external_transition1x: bool = False) -> dict:
    """Build a new verified catalogue; materialized mode embeds admitted data and assets."""
    import duckdb
    from synrxn import __version__
    if metadata_only and materialize:
        raise ValueError("Metadata-only and materialized modes are mutually exclusive")
    if (compact_snapshot or external_transition1x) and not materialize:
        raise ValueError("Compact and optional-asset builds require materialized mode")
    if threads < 1:
        raise ValueError("Thread count must be positive")
    root = root.resolve()
    output = validate_output(root, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / ("." + output.name + ".build.lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    staging = None
    try:
        os.close(descriptor)
        inventory = load_inventory(root, scope)
        staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
        database = staging / "synrxn.duckdb"
        connection = duckdb.connect(":memory:" if materialize else str(database), config={
            "memory_limit": memory_limit, "threads": str(threads),
            "temp_directory": str(staging / "spill")})
        try:
            if materialize:
                connection.execute(f"ATTACH {sql_string(str(database))} AS snapshot "
                                   f"(ROW_GROUP_SIZE {MATERIALIZED_ROW_GROUP_SIZE})")
                connection.execute("USE snapshot")
            connection.execute("BEGIN TRANSACTION")
            mapping = populate(connection, root, inventory, metadata_only, materialize, progress)
            if compact_snapshot:
                compact.install(connection, mapping)
            evaluation.install(connection, mapping)
            connection.execute("CREATE TABLE catalog.build_info (key VARCHAR PRIMARY KEY, value_json JSON)")
            build_info = {
                "schema": SCHEMA, "software_version": __version__,
                "release_scope": scope,
                "compact": compact_snapshot,
                "storage_mode": "materialized" if materialize else "metadata_only" if metadata_only else "file_views",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "storage_builder_sha256": digest(Path(__file__)),
                "view_code_sha256": {module.__name__: digest(Path(module.__file__))
                                     for module in (compact, evaluation)},
                "duckdb_version": importlib.metadata.version("duckdb"),
                "asset_chunk_bytes": ASSET_CHUNK_BYTES if materialize else None,
                "row_group_size": MATERIALIZED_ROW_GROUP_SIZE if materialize else None,
                "memory_limit": memory_limit, "threads": threads,
                "counting_policy": "Stored rows overlap across source views; they are not independent experiments.",
                "license_policy": inventory["release"].get("policy"),
            }
            connection.executemany("INSERT INTO catalog.build_info VALUES (?, ?)",
                                   [(key, json_text(value)) for key, value in build_info.items()])
            external_assets = (
                write_assets(connection, root, inventory, external_transition1x, progress) if materialize else []
            )
            connection.execute("COMMIT")
            connection.execute("CHECKPOINT")
        finally:
            connection.close()
        verification = None
        if materialize:
            if progress:
                progress("Verifying materialized tables and embedded asset hashes with external access disabled")
            verification = verify_database(database)
            for record in inventory["checks"].values():
                check_file(root, record["path"], record["sha256"])
        dependency_versions = {}
        for package in ("duckdb", "pyarrow", "pandas"):
            try:
                dependency_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                dependency_versions[package] = None
        manifest = {"schema": SCHEMA, "status": "passed", "created_at": datetime.now(timezone.utc).isoformat(),
                    "repository_root": str(root), "output": str(output), "metadata_only": metadata_only,
                    "release_scope": scope,
                    "storage_mode": build_info["storage_mode"], "software_version": __version__,
                    "python_version": sys.version,
                    "code": {"path": "synrxn/reconstruction/catalogue/storage.py", "sha256": digest(Path(__file__))},
                    "dependency_versions": dependency_versions,
                    "inputs": sorted(inventory["checks"].values(), key=lambda v: v["path"]),
                    "summary": {"families": len(inventory["registry"]["families"]),
                                "tasks": len(inventory["registry"]["tasks"]),
                                "dataset_tables": len(inventory["tables"]),
                                "task_table_bindings": len(inventory["bindings"]),
                                "stored_rows": sum(t["rows"] for t in inventory["tables"].values()),
                                "file_views": sum(m["status"] == "file_view" for m in mapping)},
                    "database": {"path": "synrxn.duckdb", "sha256": digest(database),
                                 "bytes": database.stat().st_size},
                    "views": mapping,
                    "limits": ["File views use absolute source paths; rebuild after relocating the repository.",
                               "Source hashes were checked at build time; later changes require rebuilding.",
                               "Row counts come from audited catalogues; file-view mode does not recount rows.",
                               "No source rows are materialized or copied; no predictive benchmark is executed.",
                               "Read-only SQL access is not a sandbox for untrusted SQL or filesystem operations."]}
        if materialize:
            describe_portable_build(manifest, verification, compact_snapshot, external_assets)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        validate_output(root, output)
        os.rename(staging, output)
        staging = None
        return manifest
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build", help="verify inputs and build a new local catalogue")
    builder.add_argument("--output", required=True, type=Path, help="new/empty directory below .cache/ or reports/")
    modes = builder.add_mutually_exclusive_group()
    modes.add_argument("--metadata-only", action="store_true")
    modes.add_argument("--materialize", action="store_true",
                       help="embed admitted tables, native assets, metadata and notices in one portable database")
    builder.add_argument("--memory-limit", default="4GB")
    builder.add_argument("--threads", type=int, default=2)
    builder.add_argument("--scope", choices=SCOPES, default=MAIN,
                         help="main download scope or the full local catalogue")
    builder.add_argument("--compact", action="store_true",
                         help="use verified internal views for overlapping source relations")
    builder.add_argument("--external-transition1x", action="store_true",
                         help="list the Transition1x HDF5 as an external asset instead of embedding it")
    verifier = commands.add_parser("verify", help="verify a portable database without reading source files")
    verifier.add_argument("--database", required=True, type=Path)
    verifier.add_argument("--skip-asset-hashes", action="store_true",
                          help="check table counts without scanning asset bytes")
    extractor = commands.add_parser("extract-asset", help="recover one embedded native asset or license notice")
    extractor.add_argument("--database", required=True, type=Path)
    extractor.add_argument("--path", required=True, help="logical path from catalog.assets")
    extractor.add_argument("--output", required=True, type=Path)
    query = commands.add_parser("query", help="open a catalogue database read-only and execute SQL")
    query.add_argument("--database", required=True, type=Path)
    query.add_argument("--sql", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        manifest = build(ROOT, args.output, args.metadata_only, materialize=args.materialize,
                         memory_limit=args.memory_limit, threads=args.threads,
                         scope=args.scope,
                         compact_snapshot=args.compact, external_transition1x=args.external_transition1x,
                         progress=lambda message: print(message, file=sys.stderr, flush=True))
        print(json.dumps({"output": manifest["output"], **manifest["summary"]}, indent=2))
    elif args.command == "verify":
        print(json.dumps(verify_database(args.database, not args.skip_asset_hashes), indent=2))
    elif args.command == "extract-asset":
        print(json.dumps(extract_asset(args.database, args.path, args.output), indent=2))
    else:
        import duckdb
        connection = duckdb.connect(str(args.database.resolve()), read_only=True)
        try:
            cursor = connection.execute(args.sql)
            print(json.dumps({"columns": [c[0] for c in cursor.description], "rows": cursor.fetchall()}, default=str))
        finally:
            connection.close()


if __name__ == "__main__":
    main()
