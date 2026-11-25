"""Behavioral checks for isolated, offline coverage rebuilding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from synrxn.reconstruction.coverage import rebuild
from synrxn.reconstruction.families.selectivity.impurity_prediction import _source_tables as impurity


class IsolatedRebuild(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.repo = self.base / "repo"
        for name in ["kdd/coverage/adapters", "kdd/integration/sources", "script",
                     "Data/raw/coverage", "Data/raw/synrxn2", "Data/expansion",
                     "Data/coverage_expansion/fixture"]:
            (self.repo / name).mkdir(parents=True, exist_ok=True)
        (self.repo / "kdd/scope_policy.json").write_text("{}\n")
        (self.repo / "script/fetch_expansion_data.py").write_text("# Unused offline fixture.\n")
        for name in rebuild.ADAPTERS:
            (self.repo / "kdd/coverage/adapters" / (name + ".py")).write_text("pass\n")
        self.native = self.repo / "Data/raw/coverage/native.txt"
        self.native.write_bytes(b"immutable native reference\n")
        (self.repo / "Data/raw/synrxn2/native.txt").write_bytes(b"second cache\n")
        self.relative_output = Path("Data/coverage_expansion/fixture/table.txt")
        (self.repo / self.relative_output).write_bytes(b"derived reference\n")
        self.commands = []

    def adapter(self, command, **kwargs):
        self.commands.append(command)
        workspace = Path(kwargs["cwd"])
        if Path(command[1]).stem == "knowledge":
            destination = workspace / self.relative_output
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"derived reference\n")
        return SimpleNamespace(returncode=0)

    def test_offline_rebuild_preserves_inputs_and_removes_native_bundle(self):
        before = self.native.stat()
        with patch.object(rebuild.subprocess, "run", side_effect=self.adapter):
            manifest = rebuild.rebuild(self.base / "result", repo=self.repo)
        after = self.native.stat()
        self.assertEqual(manifest["status"], "passed")
        self.assertEqual(manifest["verified_files"], 1)
        self.assertEqual(len(self.commands), len(rebuild.ADAPTERS))
        self.assertEqual({Path(command[1]).stem for command in self.commands}, set(rebuild.ADAPTERS))
        self.assertTrue(all("--fetch" not in command for command in self.commands))
        self.assertEqual((before.st_size, before.st_mtime_ns, before.st_ino),
                         (after.st_size, after.st_mtime_ns, after.st_ino))
        self.assertEqual(self.native.read_bytes(), b"immutable native reference\n")
        workspace = self.base / "result/workspace"
        self.assertFalse((workspace / "Data/raw").exists())
        self.assertFalse((workspace / "Data/expansion").exists())
        self.assertTrue((workspace / self.relative_output).is_file())
        comparison = manifest["comparisons"][0]
        self.assertEqual(comparison["observed_sha256"],
                         hashlib.sha256(b"derived reference\n").hexdigest())

    def test_output_cannot_be_inside_source_or_expected_output_tree(self):
        protected = ["Data/raw/coverage", "Data/raw/synrxn2",
                     "Data/coverage_expansion", "Data/expansion", "kdd"]
        for relative in protected:
            with self.subTest(relative=relative):
                output = self.repo / relative / "new-result"
                # Stop before recursive copying if a path is incorrectly admitted.
                with patch.object(rebuild.shutil, "copytree", side_effect=AssertionError("unsafe path admitted")):
                    with self.assertRaises(ValueError):
                        rebuild.rebuild(output, repo=self.repo)

    def test_selected_adapter_scope_uses_catalogue_ownership_and_is_explicit(self):
        catalog = self.repo / "Data/coverage_expansion/fixture/catalog.json"
        catalog.write_text(json.dumps({"adapter": {"path": "kdd/coverage/adapters/knowledge.py"}}))
        other = self.repo / "Data/coverage_expansion/other/table.txt"
        other.parent.mkdir()
        other.write_bytes(b"unselected reference\n")

        def selected_output(command, **kwargs):
            result = self.adapter(command, **kwargs)
            destination = Path(kwargs["cwd"]) / catalog.relative_to(self.repo)
            destination.write_bytes(catalog.read_bytes())
            return result

        with patch.object(rebuild.subprocess, "run", side_effect=selected_output):
            manifest = rebuild.rebuild(self.base / "selected-result", repo=self.repo, adapters=["knowledge"])
        self.assertEqual(manifest["status"], "passed")
        self.assertEqual(manifest["scope"], "selected_adapters_only")
        self.assertEqual(manifest["selected_adapters"], ["knowledge"])
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(manifest["verified_files"], 2)
        self.assertTrue(all("/fixture/" in comparison["path"] for comparison in manifest["comparisons"]))

    def test_unexpected_generated_file_fails_exact_output_comparison(self):
        def extra_output(command, **kwargs):
            result = self.adapter(command, **kwargs)
            if Path(command[1]).stem == "knowledge":
                extra = Path(kwargs["cwd"]) / "Data/coverage_expansion/fixture/unexpected.txt"
                extra.write_bytes(b"not in the captured output inventory\n")
            return result
        with patch.object(rebuild.subprocess, "run", side_effect=extra_output):
            manifest = rebuild.rebuild(self.base / "extra-result", repo=self.repo)
        self.assertEqual(manifest["status"], "failed")

    def test_selected_process_compares_its_impurity_outputs_too(self):
        for branch, adapter in [('fixture', 'process'), ('impurity', 'impurity')]:
            directory = self.repo / 'Data/coverage_expansion' / branch
            directory.mkdir(exist_ok=True)
            (directory / 'catalog.json').write_text(json.dumps({'adapter': f'kdd/coverage/adapters/{adapter}.py'}))
            (directory / 'table.txt').write_bytes(b'derived reference\n')

        def process_output(command, **kwargs):
            workspace = Path(kwargs['cwd'])
            for branch in ['fixture', 'impurity']:
                shutil_source = self.repo / 'Data/coverage_expansion' / branch
                destination = workspace / 'Data/coverage_expansion' / branch
                destination.mkdir(parents=True, exist_ok=True)
                for source in shutil_source.iterdir():
                    (destination / source.name).write_bytes(source.read_bytes())
            return SimpleNamespace(returncode=0)

        with patch.object(rebuild.subprocess, 'run', side_effect=process_output):
            manifest = rebuild.rebuild(self.base / 'process-result', repo=self.repo, adapters=['process'])
        self.assertEqual(manifest['status'], 'passed')
        self.assertEqual(manifest['output_adapter_closure'], ['impurity', 'process'])
        self.assertEqual(manifest['verified_files'], 4)

    def test_selected_rebuild_does_not_require_an_unrelated_source_lock(self):
        catalog = self.repo / 'Data/coverage_expansion/fixture/catalog.json'
        catalog.write_text(json.dumps({'adapter': 'kdd/coverage/adapters/knowledge.py'}))
        lock = self.repo / 'kdd/coverage/adapters/bio_sources.json'
        lock.write_text(json.dumps({'assets': [{'path': 'ncs/sources/not-acquired.tar.gz'}]}))

        def selected_output(command, **kwargs):
            result = self.adapter(command, **kwargs)
            target = Path(kwargs['cwd']) / catalog.relative_to(self.repo)
            target.write_bytes(catalog.read_bytes())
            return result

        with patch.object(rebuild.subprocess, 'run', side_effect=selected_output):
            manifest = rebuild.rebuild(self.base / 'narrow-result', repo=self.repo, adapters=['knowledge'])
        self.assertEqual(manifest['status'], 'passed')

    def test_pinned_external_native_input_is_available_then_removed(self):
        relative = Path("ncs/sources/native-reference.tar.gz")
        source = self.repo / relative
        source.parent.mkdir(parents=True)
        source.write_bytes(b"native archive fixture\n")
        lock = self.repo / "kdd/coverage/adapters/fixture_sources.json"
        lock.write_text(json.dumps({"assets": [{"path": relative.as_posix(),
                                               "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]}))
        before = source.stat()

        def read_external_input(command, **kwargs):
            self.assertEqual((Path(kwargs["cwd"]) / relative).read_bytes(), b"native archive fixture\n")
            return self.adapter(command, **kwargs)

        with patch.object(rebuild.subprocess, "run", side_effect=read_external_input):
            manifest = rebuild.rebuild(self.base / "external-result", repo=self.repo)
        self.assertEqual(manifest["status"], "passed")
        self.assertFalse((self.base / "external-result/workspace" / relative).exists())
        after = source.stat()
        self.assertEqual((before.st_mtime_ns, before.st_ino), (after.st_mtime_ns, after.st_ino))

    def test_failed_source_setup_does_not_retain_native_carriers(self):
        lock = self.repo / "kdd/coverage/adapters/fixture_sources.json"
        lock.write_text(json.dumps({"assets": [{"path": "ncs/sources/missing.tar.gz", "sha256": "a" * 64}]}))
        output = self.base / "failed-result"
        try:
            manifest = rebuild.rebuild(output, repo=self.repo)
        except (OSError, ValueError):
            pass
        else:
            self.assertEqual(manifest["status"], "failed")
        self.assertFalse((output / "workspace/Data/raw").exists(),
                         "Failed setup must not leave native source carriers in the report")
        self.assertFalse((output / "workspace/Data/expansion").exists())


class OfflineNativeExtraction(unittest.TestCase):
    def test_existing_identical_archive_members_are_not_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            members = [impurity.PBA_MEMBER, impurity.PBA_README, impurity.PBA_LICENSE]
            with zipfile.ZipFile(raw / "github-code-repository.zip", "w") as archive:
                for index, member in enumerate(members):
                    archive.writestr(member, f"native member {index}\n")
            with patch.object(impurity, "RAW", raw):
                impurity.extract()
                paths = [raw / "extracted" / member for member in members]
                before = [(path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino) for path in paths]
                impurity.extract()
                after = [(path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino) for path in paths]
            self.assertEqual(before, after, "Offline parsing must preserve already verified native cache files")



def test_native_outputs_compare_against_relocated_sources_and_detect_changes(tmp_path):
    repo, workspace = tmp_path / 'repo', tmp_path / 'workspace'
    stored = repo / 'Data/tables/coverage/knowledge/documents.csv'
    native = workspace / 'Data/coverage_expansion/knowledge/documents.csv'
    for path in (stored, native):
        path.parent.mkdir(parents=True)
        path.write_text('id,label\n1,reference\n')
    records = rebuild.compare_workspace(workspace, repo, {'knowledge'})
    assert [(r['path'], r['status']) for r in records] == [
        ('Data/coverage_expansion/knowledge/documents.csv', 'verified')]
    native.write_text('id,label\n1,changed\n')
    assert rebuild.compare_workspace(workspace, repo, {'knowledge'})[0]['status'] == 'failed'


def test_native_comparison_rejects_unexpected_outputs(tmp_path):
    repo, workspace = tmp_path / 'repo', tmp_path / 'workspace'
    stored = repo / 'Data/tables/coverage/knowledge/catalog.json'
    native = workspace / 'Data/coverage_expansion/knowledge/catalog.json'
    for path in (stored, native):
        path.parent.mkdir(parents=True)
        path.write_text('{}\n')
    native.with_name('undeclared.csv').write_text('id\n1\n')
    records = rebuild.compare_workspace(workspace, repo)
    assert sum(r['status'] == 'verified' for r in records) == 1
    assert any(r.get('error') == 'Unexpected generated file' for r in records)

if __name__ == "__main__":
    unittest.main()
