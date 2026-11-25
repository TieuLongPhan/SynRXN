"""Storage contracts: provenance, rights, identity and heterogeneous native files."""
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import duckdb

from synrxn.reconstruction.catalogue.storage import build, digest, extract_asset, local_file, main, verify_database, view_name


class StorageTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "kdd/hierarchy").mkdir(parents=True)
        (self.root / "kdd/coverage").mkdir()
        (self.root / "Data").mkdir()
        (self.root / "Data/catalog.json").write_text('{"sources": []}\n')
        with gzip.open(self.root / "Data/a.csv.gz", "wt") as stream:
            stream.write('native_id,label,license_spdx\n001,0,Apache-2.0\n002,1,Apache-2.0\n')
        (self.root / "Data/b.csv").write_text('native_id,value\n0001,10.20\n')
        with gzip.open(self.root / "Data/c.jsonl.gz", "wt") as stream:
            stream.write('{"id":"007","items":[1,"x",{"a":true}]}\n{"id":"008","items":null}\n')
        con = duckdb.connect()
        con.execute("COPY (SELECT 1 AS native_integer, '009' AS native_id) TO ? (FORMAT PARQUET)",
                    [str(self.root / "Data/d.parquet")])
        con.close()
        (self.root / "Data/held.csv").write_text('native_id,value\n0001,9\n')
        self.tables = []
        release_tables, held_tables = [], []
        for name, fmt, count in (("a.csv.gz", "csv.gz", 2), ("b.csv", "csv", 1),
                                 ("c.jsonl.gz", "jsonl.gz", 2), ("d.parquet", "parquet", 1),
                                 ("held.csv", "csv", 1)):
            path = "Data/" + name
            table = {"namespace": "fixture", "key": name, "title": name, "rows": count,
                     "role": "primary", "counting_unit": "observation", "path": path,
                     "sha256": digest(self.root / path), "source_catalogue": "Data/catalog.json",
                     "source_id": "original", "source_urls": ["https://example.org/original"]}
            self.tables.append(table)
            rights = {"namespace": "fixture", "key": name, "path": path,
                      "expected_sha256": table["sha256"], "expected_rows": count,
                      "expected_bytes": (self.root / path).stat().st_size, "format": fmt,
                      "license_expression": None if name == "held.csv" else "CC-BY-4.0",
                      "partition": "main", "rights_status": "unresolved" if name == "held.csv" else "verified"}
            (held_tables if name == "held.csv" else release_tables).append(rights)
        registry = {"families": [{"id": "family", "title": "Family"}],
                    "tasks": [{"id": "task", "title": "Task", "family_id": "family",
                               "bindings": [{"namespace": t["namespace"], "key": t["key"], "role": t["role"]}
                                            for t in self.tables]}],
                    "source_sha256": {"Data/catalog.json": digest(self.root / "Data/catalog.json")}}
        self.write_json("kdd/task_registry.json", registry)
        release = {"status": "passed",
                   "catalogs": [{"path": "Data/catalog.json", "sha256": digest(self.root / "Data/catalog.json")}],
                   "release_tables": release_tables, "held_tables": held_tables}
        self.write_json("kdd/coverage/release_manifest.json", release)
        hierarchy = {"input_sha256": {p: digest(self.root / p) for p in (
            "kdd/task_registry.json", "kdd/coverage/release_manifest.json")},
                     "families": [{"id": "family", "title": "Family",
                                   "tasks": [{"id": "task", "title": "Task", "tables": self.tables}]}]}
        self.write_json("kdd/hierarchy/hierarchy.json", hierarchy)

    def tearDown(self):
        self.temporary.cleanup()

    def write_json(self, path, value):
        (self.root / path).write_text(json.dumps(value))

    def register_native_assets(self):
        payload = b"\x00\xffnative-spectrum\n" * 13
        for name in ("native.bin", "native-alias.bin"):
            (self.root / "Data" / name).write_bytes(payload)
        (self.root / "Data/NOTICE.txt").write_text("Fixture attribution; CC BY 4.0.\n")
        (self.root / "Data/unapproved.bin").write_bytes(b"not admitted to the release")
        (self.root / "Data/targets.csv").write_text("native_id,target\n001,4.25\n")
        registry = json.loads((self.root / "kdd/task_registry.json").read_text())
        registry["tasks"][0]["bindings"][0]["task_fields"] = {
            "task": {"target_path": "Data/targets.csv", "split_metadata": "Data/targets.csv"}}
        registry["tasks"][0]["bindings"][-1]["task_fields"] = {
            "task": {"target_path": "Data/unapproved.bin"}}
        self.write_json("kdd/task_registry.json", registry)
        release = json.loads((self.root / "kdd/coverage/release_manifest.json").read_text())
        release["release_native_assets"] = [
            {"path": "Data/" + name, "sha256": digest(self.root / "Data" / name),
             "bytes": len(payload), "license_expression": "CC-BY-4.0"}
            for name in ("native.bin", "native-alias.bin")]
        release["release_notices"] = [{"path": "Data/NOTICE.txt",
                                       "sha256": digest(self.root / "Data/NOTICE.txt")}]
        self.write_json("kdd/coverage/release_manifest.json", release)
        self.refresh_release_pin()
        return payload

    def refresh_release_pin(self):
        hierarchy = json.loads((self.root / "kdd/hierarchy/hierarchy.json").read_text())
        hierarchy["input_sha256"]["kdd/coverage/release_manifest.json"] = digest(
            self.root / "kdd/coverage/release_manifest.json")
        hierarchy["input_sha256"]["kdd/task_registry.json"] = digest(self.root / "kdd/task_registry.json")
        self.write_json("kdd/hierarchy/hierarchy.json", hierarchy)

    def test_full_catalogue_preserves_identity_formats_and_effective_rights(self):
        manifest = build(self.root, self.root / ".cache/catalog")
        self.assertEqual(manifest["summary"], {
            "families": 1, "tasks": 1, "dataset_tables": 5,
            "task_table_bindings": 5, "stored_rows": 7, "file_views": 4})
        connection = duckdb.connect(str(self.root / ".cache/catalog/synrxn.duckdb"), read_only=True)
        try:
            self.assertEqual(connection.execute("SELECT count(*) FROM catalog.task_table_bindings").fetchone()[0], 5)
            self.assertEqual(connection.execute("""SELECT effective_license_expression FROM catalog.dataset_tables
                WHERE table_key='a.csv.gz'""").fetchone()[0], "CC-BY-4.0")
            self.assertEqual(connection.execute("""SELECT view_name, view_status FROM catalog.dataset_tables
                WHERE table_key='held.csv'""").fetchone(), (None, "held_no_view"))
            a = view_name("fixture", "a.csv.gz")
            self.assertEqual(connection.execute(
                f'SELECT native_id, label, license_spdx FROM data."{a}" ORDER BY native_id').fetchall(),
                [("001", "0", "Apache-2.0"), ("002", "1", "Apache-2.0")])
            b = view_name("fixture", "b.csv")
            self.assertEqual(connection.execute(f'SELECT * FROM data."{b}"').fetchall(), [("0001", "10.20")])
            c = view_name("fixture", "c.jsonl.gz")
            native = [json.loads(row[0]) for row in connection.execute(f'SELECT payload FROM data."{c}"').fetchall()]
            self.assertEqual(native[0]["items"], [1, "x", {"a": True}])
            self.assertEqual(native[0]["id"], "007")
            d = view_name("fixture", "d.parquet")
            self.assertEqual(connection.execute(f'SELECT * FROM data."{d}"').fetchone(), (1, "009"))
            self.assertEqual(manifest["database"]["sha256"], digest(self.root / ".cache/catalog/synrxn.duckdb"))
        finally:
            connection.close()

    def test_metadata_only_does_not_bind_data_views(self):
        output = self.root / "reports/metadata"
        output.mkdir(parents=True)  # Existing empty destinations are allowed.
        manifest = build(self.root, output, metadata_only=True)
        self.assertEqual(manifest["summary"]["file_views"], 0)
        self.assertTrue(all(row["columns"] is None for row in manifest["views"]))

    def test_main_scope_does_not_require_the_optional_feasibility_payload(self):
        registry = json.loads((self.root / "kdd/task_registry.json").read_text())
        task = registry["tasks"][0]
        counts = {table["key"]: table["rows"] for table in self.tables}
        for binding in task["bindings"]:
            binding["rows"] = counts[binding["key"]]
        optional = task["bindings"].pop(1)
        registry["tasks"].append({**task, "id": "reaction_feasibility", "bindings": [optional]})
        self.write_json("kdd/task_registry.json", registry)
        hierarchy = json.loads((self.root / "kdd/hierarchy/hierarchy.json").read_text())
        entry = hierarchy["families"][0]["tasks"][0]
        table = entry["tables"].pop(1)
        hierarchy["families"][0]["tasks"].append({"id": "reaction_feasibility", "tables": [table]})
        self.write_json("kdd/hierarchy/hierarchy.json", hierarchy)
        self.refresh_release_pin()
        (self.root / table["path"]).unlink()
        manifest = build(self.root, self.root / ".cache/main", materialize=True)
        self.assertEqual(manifest["release_scope"], "main")
        self.assertEqual(manifest["summary"]["tasks"], 1)
        self.assertEqual(manifest["summary"]["materialized_tables"], 3)
        with self.assertRaisesRegex(ValueError, "Missing or escaping repository file"):
            build(self.root, self.root / ".cache/full", scope="catalogue")

    def test_materialized_snapshot_survives_removal_of_sources_and_deduplicates_assets(self):
        payload = self.register_native_assets()
        # An unresolved payload is not required, read or embedded.
        (self.root / "Data/held.csv").unlink()
        with mock.patch("synrxn.reconstruction.catalogue.storage.ASSET_CHUNK_BYTES", 32):
            manifest = build(self.root, self.root / ".cache/portable", materialize=True)
        self.assertEqual(manifest["summary"]["materialized_rows"], 6)
        self.assertEqual(manifest["summary"]["materialized_tables"], 4)
        self.assertEqual(manifest["summary"]["file_views"], 0)
        with tempfile.TemporaryDirectory() as directory:
            moved = Path(directory) / "standalone.duckdb"
            shutil.copyfile(self.root / ".cache/portable/synrxn.duckdb", moved)
            shutil.rmtree(self.root / "Data")
            shutil.rmtree(self.root / "kdd")
            self.assertEqual(verify_database(moved)["materialized_rows"], 6)
            with duckdb.connect(str(moved), read_only=True, config={"enable_external_access": "false"}) as con:
                a = view_name("fixture", "a.csv.gz")
                self.assertEqual(con.execute(f'SELECT native_id FROM data."{a}" ORDER BY native_id').fetchall(),
                                 [("001",), ("002",)])
                b = view_name("fixture", "b.csv")
                self.assertEqual(con.execute(f'SELECT value FROM data."{b}"').fetchone()[0], "10.20")
                c = view_name("fixture", "c.jsonl.gz")
                row = json.loads(con.execute(f'SELECT payload FROM data."{c}" LIMIT 1').fetchone()[0])
                self.assertEqual(row["items"], [1, "x", {"a": True}])
                d = view_name("fixture", "d.parquet")
                self.assertEqual(con.execute(f'SELECT * FROM data."{d}"').fetchone(), (1, "009"))
                assets = dict(con.execute("SELECT path, sha256 FROM catalog.assets").fetchall())
                self.assertEqual(assets["Data/native.bin"], assets["Data/native-alias.bin"])
                self.assertNotIn("Data/unapproved.bin", assets)
                self.assertNotIn("Data/held.csv", assets)
                self.assertIn("Data/targets.csv", assets)
                self.assertEqual(con.execute("SELECT count(*) FROM assets.chunks WHERE sha256=?",
                                             [assets["Data/native.bin"]]).fetchone()[0], 8)
            extracted = Path(directory) / "recovered.bin"
            extract_asset(moved, "Data/native.bin", extracted)
            self.assertEqual(extracted.read_bytes(), payload)
            with self.assertRaises(FileExistsError):
                extract_asset(moved, "Data/native.bin", extracted)
            self.assertEqual(extracted.read_bytes(), payload)
            with self.assertRaises(KeyError):
                extract_asset(moved, "Data/unapproved.bin", Path(directory) / "unexpected")
            self.assertFalse((Path(directory) / "unexpected").exists())
            extract_asset(moved, "Data/NOTICE.txt", Path(directory) / "NOTICE.txt")
            self.assertEqual((Path(directory) / "NOTICE.txt").read_text(), "Fixture attribution; CC BY 4.0.\n")
            extract_asset(moved, "Data/targets.csv", Path(directory) / "targets.csv")
            self.assertEqual((Path(directory) / "targets.csv").read_text(), "native_id,target\n001,4.25\n")

    def test_asset_row_groups_are_bounded_across_a_checkpoint(self):
        self.register_native_assets()
        path = self.root / "Data/native.bin"
        path.write_bytes(bytes(range(256)) * 9)
        release = json.loads((self.root / "kdd/coverage/release_manifest.json").read_text())
        release["release_native_assets"][0].update(sha256=digest(path), bytes=path.stat().st_size)
        self.write_json("kdd/coverage/release_manifest.json", release)
        self.refresh_release_pin()
        with mock.patch("synrxn.reconstruction.catalogue.storage.ASSET_CHUNK_BYTES", 1):
            build(self.root, self.root / ".cache/row-groups", materialize=True)
        database = self.root / ".cache/row-groups/synrxn.duckdb"
        with duckdb.connect(str(database), read_only=True) as con:
            count = con.execute("""SELECT count(DISTINCT row_group_id)
                FROM pragma_storage_info('assets.chunks')""").fetchone()[0]
            self.assertGreater(count, 1)
        self.assertEqual(verify_database(database)["status"], "passed")

    def test_materialized_row_count_mismatch_is_not_published(self):
        release = json.loads((self.root / "kdd/coverage/release_manifest.json").read_text())
        release["release_tables"][0]["expected_rows"] = 3
        self.write_json("kdd/coverage/release_manifest.json", release)
        self.refresh_release_pin()
        hierarchy = json.loads((self.root / "kdd/hierarchy/hierarchy.json").read_text())
        hierarchy["families"][0]["tasks"][0]["tables"][0]["rows"] = 3
        self.write_json("kdd/hierarchy/hierarchy.json", hierarchy)
        with self.assertRaisesRegex(ValueError, "Expected 3 rows, imported 2"):
            build(self.root, self.root / ".cache/bad-count", materialize=True)
        self.assertFalse((self.root / ".cache/bad-count").exists())

    def test_changed_native_asset_is_not_published(self):
        self.register_native_assets()
        path = self.root / "Data/native.bin"
        path.write_bytes(b"x" * path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch for embedded asset"):
            build(self.root, self.root / ".cache/bad-asset", materialize=True)
        self.assertFalse((self.root / ".cache/bad-asset").exists())
        self.assertFalse(list((self.root / ".cache").glob(".bad-asset.*")))

    def test_database_verification_detects_deleted_rows(self):
        build(self.root, self.root / ".cache/portable", materialize=True)
        database = self.root / ".cache/portable/synrxn.duckdb"
        with duckdb.connect(str(database)) as con:
            con.execute(f'DELETE FROM data."{view_name("fixture", "b.csv")}"')
        with self.assertRaisesRegex(ValueError, "Row count mismatch"):
            verify_database(database)

    def test_corrupt_asset_is_detected_and_never_extracted(self):
        self.register_native_assets()
        build(self.root, self.root / ".cache/portable", materialize=True)
        database = self.root / ".cache/portable/synrxn.duckdb"
        with duckdb.connect(str(database)) as con:
            sha256 = con.execute("SELECT sha256 FROM catalog.assets WHERE path='Data/native.bin'").fetchone()[0]
            con.execute("UPDATE assets.chunks SET payload=? WHERE sha256=?", [b"corrupt", sha256])
        with self.assertRaisesRegex(ValueError, "Corrupt compressed asset chunk"):
            verify_database(database)
        output = self.root / "recovered.bin"
        with self.assertRaisesRegex(ValueError, "Corrupt compressed asset chunk"):
            extract_asset(database, "Data/native.bin", output)
        self.assertFalse(output.exists())

    def test_file_view_catalogue_is_not_misidentified_as_a_portable_snapshot(self):
        build(self.root, self.root / ".cache/linked")
        with self.assertRaisesRegex(ValueError, "not a self-contained"):
            verify_database(self.root / ".cache/linked/synrxn.duckdb")

    def test_storage_modes_are_mutually_exclusive_before_creating_output(self):
        output = self.root / ".cache/invalid-mode"
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            build(self.root, output, metadata_only=True, materialize=True)
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            main(["build", "--output", str(output), "--metadata-only", "--materialize"])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(output.exists())

    def test_modified_data_rejected_without_publishing_output(self):
        (self.root / "Data/b.csv").write_text('native_id,value\n0001,changed\n')
        output = self.root / "reports/rejected"
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch for Data/b.csv"):
            build(self.root, output)
        self.assertFalse(output.exists())

    def test_modified_locked_input_rejected(self):
        (self.root / "Data/catalog.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            build(self.root, self.root / ".cache/rejected")

    def test_paths_and_existing_outputs_are_protected(self):
        with self.assertRaisesRegex(ValueError, "Unsafe repository path"):
            local_file(self.root, "../outside")
        with self.assertRaisesRegex(ValueError, "Output must"):
            build(self.root, self.root / "Data/new")
        target = self.root / ".cache/occupied"
        target.mkdir(parents=True)
        (target / "keep").write_text("existing work")
        with self.assertRaisesRegex(ValueError, "Refusing to replace"):
            build(self.root, target)
        self.assertEqual((target / "keep").read_text(), "existing work")

    def test_relocated_symlink_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory() as outside:
            payload = Path(outside) / "payload"
            payload.write_text("external")
            (self.root / "Data/link").symlink_to(payload)
            with self.assertRaisesRegex(ValueError, "escaping repository file"):
                local_file(self.root, "Data/link")

    def test_invalid_native_format_fails_with_named_view(self):
        path = "Data/d.parquet"
        (self.root / path).write_bytes(b"not a parquet file")
        release = json.loads((self.root / "kdd/coverage/release_manifest.json").read_text())
        for table in release["release_tables"]:
            if table["path"] == path:
                table["expected_sha256"] = digest(self.root / path)
                table["expected_bytes"] = (self.root / path).stat().st_size
        self.write_json("kdd/coverage/release_manifest.json", release)
        hierarchy = json.loads((self.root / "kdd/hierarchy/hierarchy.json").read_text())
        hierarchy["input_sha256"]["kdd/coverage/release_manifest.json"] = digest(
            self.root / "kdd/coverage/release_manifest.json")
        for table in hierarchy["families"][0]["tasks"][0]["tables"]:
            if table["path"] == path:
                table["sha256"] = digest(self.root / path)
        self.write_json("kdd/hierarchy/hierarchy.json", hierarchy)
        with self.assertRaisesRegex(ValueError, "Cannot bind data.v_fixture_d_parquet"):
            build(self.root, self.root / ".cache/invalid-native")
        self.assertFalse((self.root / ".cache/invalid-native").exists())


if __name__ == "__main__":
    unittest.main()
