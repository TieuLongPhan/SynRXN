import unittest
import io
from synrxn.data.zenodo_client import ZenodoClient
from types import SimpleNamespace
import json
import zipfile
import tarfile
from pathlib import Path
import tempfile
import hashlib


class FakeResponse:
    def __init__(self, status_code=200, headers=None, content=b"", json_data=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._content = content
        self._json = json_data

    def json(self):
        return self._json

    def iter_content(self, chunk_size=8192):
        # yield from content in chunks
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    @property
    def content(self):
        return self._content

    def close(self):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        """
        responses: mapping from url (or 'search') to FakeResponse objects
        """
        self._responses = responses
        self.requests = []

    def get(self, url, params=None, timeout=None, stream=False):
        self.requests.append((url, params, timeout, stream))
        # simple dispatch
        if url.startswith("https://zenodo.org/api/records") and params:
            return self._responses.get("search")
        if url.startswith("https://zenodo.org/api/records/"):
            return self._responses.get("record")
        # fallback: raw file URL or others:
        return self._responses.get("file")


class TestZenodoClient(unittest.TestCase):
    def setUp(self):
        # Prepare fake search hits with versions
        hits = {
            "hits": {
                "hits": [
                    {
                        "id": 1,
                        "metadata": {"version": "0.0.1"},
                        "updated": "2020-01-01",
                    },
                    {
                        "id": 17297723,
                        "metadata": {"version": "0.0.5"},
                        "updated": "2024-01-01",
                    },
                ]
            }
        }
        # Fake record json: files list
        files = [
            {
                "key": "TieuLongPhan/SynRXN-v0.0.5.zip",
                "links": {"self": "https://zenodo/self"},
                "checksum": "",
            },
            {
                "key": "Data/rbl/mis.csv.gz",
                "links": {"self": "https://zenodo/file"},
                "checksum": "",
            },
        ]
        record_json = {"files": files}

        # create a small zip archive in memory that contains Data/rbl/mis.csv
        b = io.BytesIO()
        with zipfile.ZipFile(b, mode="w") as zf:
            zf.writestr(
                "TieuLongPhan/SynRXN-v0.0.5/Data/rbl/mis.csv", "a,b\n1,2\n3,4\n"
            )
        zip_bytes = b.getvalue()

        # Create a FakeResponse mapping
        responses = {
            "search": FakeResponse(status_code=200, json_data=hits),
            "record": FakeResponse(status_code=200, json_data=record_json),
            # file response for archive download
            "file": FakeResponse(
                status_code=200,
                headers={"Content-Type": "application/zip"},
                content=zip_bytes,
            ),
        }
        self.session = FakeSession(responses)
        self.client = ZenodoClient(
            session=self.session, cache_dir=None, cache_record_index=False, timeout=5
        )
        # No network; we'll call methods that use session

    def test_resolve_record_id_by_version(self):
        rid = self.client.resolve_record_id("10.5281/zenodo.17297258", "0.0.5")
        self.assertEqual(rid, 17297723)

    def test_build_file_index(self):
        idx = self.client.build_file_index(17297723)
        self.assertIn("TieuLongPhan/SynRXN-v0.0.5.zip", idx)
        self.assertIn("Data/rbl/mis.csv.gz", idx)

    def test_list_archive_members_cached_reads_zip_members(self):
        # Use build_file_index output
        idx = self.client.build_file_index(17297723)
        members = self.client.list_archive_members_cached(
            17297723,
            "TieuLongPhan/SynRXN-v0.0.5.zip",
            idx["TieuLongPhan/SynRXN-v0.0.5.zip"],
        )
        # The zip we created contains a file path ending with Data/rbl/mis.csv
        found = [m for m in members if "Data/rbl/mis.csv" in m]
        self.assertTrue(len(found) >= 1)

    def test_get_download_response_prefers_non_json(self):
        # meta for the archive
        idx = self.client.build_file_index(17297723)
        meta = idx["TieuLongPhan/SynRXN-v0.0.5.zip"]
        resp = self.client.get_download_response(meta, 17297723)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Data/rbl/mis.csv", resp.content)

    def test_stream_to_temp_and_verify_with_checksum(self):
        # Build content and an explicit sha256 checksum meta
        content = b"hello world"
        sha = hashlib.sha256(content).hexdigest()
        # make response providing content
        resp = FakeResponse(status_code=200, content=content)
        # meta with checksum
        meta = {"checksum": f"sha256:{sha}"}
        # stream_to_temp_and_verify should return a temp path and not raise
        tmp = self.client.stream_to_temp_and_verify(resp, meta, suffix=".txt")
        self.assertTrue(Path(tmp).exists())
        data = Path(tmp).read_bytes()
        self.assertEqual(data, content)
        Path(tmp).unlink()

    def test_extract_member_bytes_from_real_zip(self):
        # write a real temp zip and then extract a member via client.extract_member_bytes
        td = tempfile.TemporaryDirectory()
        zip_path = Path(td.name) / "x.zip"
        with zipfile.ZipFile(zip_path, mode="w") as zf:
            zf.writestr("Data/rbl/test.csv", "x,y\n1,2\n")
        b = self.client.extract_member_bytes(zip_path, "Data/rbl/test.csv")
        self.assertIsNotNone(b)
        self.assertIn(b"x,y", b)
        td.cleanup()


if __name__ == "__main__":
    unittest.main()
