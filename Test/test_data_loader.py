import json
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path


import synrxn.data_loader as dlmod
from synrxn.data_loader import DataLoader


def _write_bytes(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


class FakeApi:
    """Simple fake HfApi-like object returning a preconfigured file list."""

    def __init__(self, file_list):
        # file_list is a list of repo paths, e.g. ["Data/syntemp.csv.gz", "README.md"]
        self._files = list(file_list)

    def list_repo_files(self, *args, **kwargs):
        # keep signature tolerant (ignores args/kwargs)
        return list(self._files)


class DataLoaderLocalTests(unittest.TestCase):
    def setUp(self):
        # create a temporary directory to act as HF cache and output
        self.tmpdir = Path(tempfile.mkdtemp(prefix="dltest_"))
        self.hf_cache = self.tmpdir / "hf_cache"
        self.out_dir = self.tmpdir / "out"
        self.hf_cache.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # keep original hf_hub_download so we can restore later
        self._orig_hf_hub_download = dlmod.hf_hub_download

    def tearDown(self):
        # restore module-level hf_hub_download
        dlmod.hf_hub_download = self._orig_hf_hub_download
        # remove temp directory
        shutil.rmtree(self.tmpdir)

    def test_normalize(self):
        assert DataLoader._normalize("SynTemp") == "syntemp"
        assert DataLoader._normalize("SYNTEMP") == "syntemp"
        assert DataLoader._normalize("syn temp!!") == "syntemp"
        assert DataLoader._normalize(" UsPtO-50K ") == "uspto50k"

    def test_available_lists_files(self):
        # prepare fake file list
        fake_files = [
            "Data/syntemp.csv.gz",
            "Data/uspto50k.csv.gz",
            "README.md",
            "Data/notes.txt",
        ]
        api = FakeApi(fake_files)
        dl = DataLoader(out_dir=self.out_dir)
        # inject fake api
        dl.api = api

        avail = dl.available()
        self.assertIn("syntemp", avail)
        self.assertIn("uspto50k", avail)
        self.assertNotIn("notes", avail)  # wrong extension not included

    def test_download_without_manifest(self):
        # prepare fake remote file list
        file_name = "syntemp.csv.gz"
        remote_path = f"Data/{file_name}"
        fake_files = [remote_path]
        api = FakeApi(fake_files)

        # create fake hf cache file
        cache_file = self.hf_cache / file_name
        _write_bytes(cache_file, b"hello-syntemp")

        # fake hf_hub_download returns the cache_file path
        def fake_hf_download(repo_id, filename, repo_type="dataset", revision=None):
            assert filename == remote_path
            return str(cache_file)

        dlmod.hf_hub_download = fake_hf_download

        dl = DataLoader(out_dir=self.out_dir)
        dl.api = api

        local = dl.download("SynTemp")
        self.assertTrue(local.exists())
        self.assertEqual(local.read_bytes(), b"hello-syntemp")

    def test_download_with_manifest_checksum_ok(self):
        # setup remote paths (data + manifest)
        file_name = "syntemp.csv.gz"
        remote_path = f"Data/{file_name}"
        manifest_remote = "Data/manifest.json"
        fake_files = [remote_path, manifest_remote]
        api = FakeApi(fake_files)

        # create data bytes and its sha
        data_bytes = b"content-for-check"
        cache_data = self.hf_cache / file_name
        _write_bytes(cache_data, data_bytes)
        sha = _sha256_bytes(data_bytes)

        # create manifest in cache
        manifest = {"files": [{"path": remote_path, "sha256": sha}]}
        cache_manifest = self.hf_cache / "manifest.json"
        _write_bytes(cache_manifest, json.dumps(manifest).encode("utf-8"))

        # fake hf_hub_download returns appropriate cache path
        def fake_hf_download(repo_id, filename, repo_type="dataset", revision=None):
            if filename.endswith("manifest.json"):
                return str(cache_manifest)
            if filename == remote_path:
                return str(cache_data)
            raise FileNotFoundError(filename)

        dlmod.hf_hub_download = fake_hf_download

        dl = DataLoader(out_dir=self.out_dir)
        dl.api = api

        local = dl.download("SYNtemp")
        self.assertTrue(local.exists())
        self.assertEqual(local.read_bytes(), data_bytes)
        self.assertEqual(_sha256_bytes(local.read_bytes()), sha)

    def test_download_with_manifest_checksum_mismatch_raises(self):
        # manifest says one checksum, actual data differs
        file_name = "syntemp.csv.gz"
        remote_path = f"Data/{file_name}"
        manifest_remote = "Data/manifest.json"
        fake_files = [remote_path, manifest_remote]
        api = FakeApi(fake_files)

        data_bytes = b"actual-content"
        cache_data = self.hf_cache / file_name
        _write_bytes(cache_data, data_bytes)

        wrong_sha = _sha256_bytes(b"different-content")
        manifest = {"files": [{"path": remote_path, "sha256": wrong_sha}]}
        cache_manifest = self.hf_cache / "manifest.json"
        _write_bytes(cache_manifest, json.dumps(manifest).encode("utf-8"))

        def fake_hf_download(repo_id, filename, repo_type="dataset", revision=None):
            if filename.endswith("manifest.json"):
                return str(cache_manifest)
            if filename == remote_path:
                return str(cache_data)
            raise FileNotFoundError(filename)

        dlmod.hf_hub_download = fake_hf_download

        dl = DataLoader(out_dir=self.out_dir)
        dl.api = api

        with self.assertRaises(IOError):
            dl.download("syntemp")

    def test_download_file_not_found_raises(self):
        api = FakeApi([])
        dl = DataLoader(out_dir=self.out_dir)
        dl.api = api
        with self.assertRaises(FileNotFoundError):
            dl.download("doesnotexist")

    def test_overwrite_flag_behavior(self):
        # no manifest: existing file should be returned if overwrite=False
        file_name = "syntemp.csv.gz"
        remote_path = f"Data/{file_name}"
        fake_files = [remote_path]
        api = FakeApi(fake_files)

        cache_data = self.hf_cache / file_name
        _write_bytes(cache_data, b"new-content")

        # create existing local file with old content
        existing = self.out_dir / file_name
        _write_bytes(existing, b"old-content")

        def fake_hf_download(repo_id, filename, repo_type="dataset", revision=None):
            assert filename == remote_path
            return str(cache_data)

        dlmod.hf_hub_download = fake_hf_download

        dl = DataLoader(out_dir=self.out_dir)
        dl.api = api

        # overwrite=False => returns existing file unchanged
        path_returned = dl.download("syntemp", overwrite=False)
        self.assertEqual(path_returned, existing)
        self.assertEqual(existing.read_bytes(), b"old-content")

        # overwrite=True => file replaced by cache content
        path_returned2 = dl.download("syntemp", overwrite=True)
        self.assertEqual(path_returned2, existing)
        self.assertEqual(existing.read_bytes(), b"new-content")


if __name__ == "__main__":
    unittest.main(verbosity=2)
