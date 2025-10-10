import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import io
import gzip
import hashlib
import json

from synrxn.data_loader import DataLoader, _ZENODO_SEARCH_API, _ZENODO_RECORD_API


class MockResponse:
    def __init__(self, status_code: int, content):
        self.status_code = status_code
        if isinstance(content, bytes):
            self.content = content
            self._json = None
        else:
            # JSON-like content
            self._json = content
            self.content = json.dumps(content).encode("utf-8")

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class TestDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        # Use a temporary directory for caching to avoid polluting project files.
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.tmpdir.name)

        # Prepare a tiny CSV and gzipped bytes for the "ecoli" dataset
        csv_text = "col1,col2\n1,2\n3,4\n"
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            gz.write(csv_text.encode("utf-8"))
        self.gz_bytes = buf.getvalue()
        # md5 checksum for gz bytes (Zenodo returns checksums like "md5:abcdef")
        h = hashlib.new("md5")
        h.update(self.gz_bytes)
        self.gz_md5 = h.hexdigest()

        # Zenodo fake record id & endpoints used in DataLoader resolution
        self.record_id = 17297723
        self.search_url = _ZENODO_SEARCH_API
        self.record_url = _ZENODO_RECORD_API.format(record_id=self.record_id)

        # Minimal search response returning a single matching hit for the version
        self.search_json = {
            "hits": {
                "hits": [
                    {
                        "id": self.record_id,
                        "metadata": {"version": "0.0.5"},
                        "updated": "2025-01-01T00:00:00Z",
                    }
                ]
            }
        }

        # Minimal record response containing a direct file entry for Data/aam/ecoli.csv.gz
        self.record_json = {
            "files": [
                {
                    "key": "Data/aam/ecoli.csv.gz",
                    "links": {"download": "https://zenodo.fake/download/ecoli.gz"},
                    "checksum": f"md5:{self.gz_md5}",
                }
            ]
        }

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _mocked_requests_get_success(self, url, *args, **kwargs):
        """
        Side effect function that returns MockResponse objects depending on URL.
        - Zenodo search API -> self.search_json
        - Zenodo record API -> self.record_json
        - zenodo.fake/download/ecoli.gz -> gz bytes content
        - GitHub API -> empty listing (safe fallback)
        - anything else -> 404 MockResponse
        """
        # Zenodo search endpoint
        if url.startswith(self.search_url):
            return MockResponse(200, self.search_json)

        # Zenodo record endpoint
        if url.startswith(self.record_url):
            return MockResponse(200, self.record_json)

        # the actual file download link returned in record_json
        if url == "https://zenodo.fake/download/ecoli.gz":
            return MockResponse(200, self.gz_bytes)

        # GitHub API (contents listing) - return an empty list payload (safe)
        if url.startswith("https://api.github.com"):
            return MockResponse(200, [])

        # Any other URL: return 404
        return MockResponse(404, {"message": "not found"})

    @patch("synrxn.data_loader.requests.get")
    def test_load_ecoli_success(self, mock_get):
        """Test that aam/ecoli loads successfully and returns a non-empty DataFrame.

        This test mocks Zenodo APIs and the file download, and verifies:
          - DataFrame type & non-empty
          - cache file created when use_cache=True
        """
        mock_get.side_effect = self._mocked_requests_get_success

        dl = DataLoader(
            task="aam",
            version="0.0.5",
            cache_dir=self.cache_dir,
            timeout=30,
            fallback=False,
        )
        df = dl.load("ecoli", use_cache=True)

        import pandas as pd

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(
            df.shape[0], 1, "Expected at least one row in aam/ecoli"
        )
        self.assertGreaterEqual(
            df.shape[1], 1, "Expected at least one column in aam/ecoli"
        )

        # Cache should exist
        cache_path = self.cache_dir / f"{dl.task}__ecoli.csv.gz"
        self.assertTrue(cache_path.exists(), "Expected gz cache file to be written")

        # Ensure that the gz on disk matches the mocked gz bytes
        on_disk = cache_path.read_bytes()
        self.assertEqual(on_disk, self.gz_bytes)

        # Ensure requests.get was called for search, record and download (at least once each)
        called_urls = [call_args[0][0] for call_args in mock_get.call_args_list]
        self.assertTrue(
            any(self.search_url in u for u in called_urls),
            "Zenodo search API should be called",
        )
        self.assertTrue(
            any(self.record_url in u for u in called_urls),
            "Zenodo record API should be called",
        )
        self.assertTrue(
            any("zenodo.fake/download" in u for u in called_urls),
            "Zenodo download link should be called",
        )

    @patch("synrxn.data_loader.requests.get")
    def test_bad_name_shows_helpful_message(self, mock_get):
        """Requesting a non-existent name should raise FileNotFoundError with useful info.

        Simulate a Zenodo record that contains no matching data files; loader must raise.
        """
        # Mock search -> returns record, but record has no files
        search_json = {
            "hits": {
                "hits": [
                    {
                        "id": self.record_id,
                        "metadata": {"version": "0.0.5"},
                        "updated": "2025-01-01T00:00:00Z",
                    }
                ]
            }
        }
        record_json_empty = {"files": []}

        def side_effect(url, *args, **kwargs):
            if url.startswith(self.search_url):
                return MockResponse(200, search_json)
            if url.startswith(self.record_url):
                return MockResponse(200, record_json_empty)
            if url.startswith("https://api.github.com"):
                return MockResponse(200, [])  # github listing safe
            return MockResponse(404, {"message": "not found"})

        mock_get.side_effect = side_effect

        dl = DataLoader(
            task="aam",
            version="0.0.5",
            cache_dir=self.cache_dir,
            timeout=30,
            fallback=False,
        )

        bad_name = "this_name_definitely_does_not_exist_12345"
        with self.assertRaises(FileNotFoundError) as cm:
            dl.load(bad_name, use_cache=False)

        msg = str(cm.exception)
        # Should include the concept DOI & version info so user knows which archive was consulted
        self.assertIn("Concept DOI", msg)
        self.assertIn("Version", msg)

        # Must contain either "Tried URLs" header or explicit note that no candidate URLs/archives found
        self.assertTrue(
            "Tried URLs" in msg or "(no candidate URLs/archives found)" in msg
        )

        # If the loader included available dataset names, they must be listed in the message (optional)
        if "Available dataset names" in msg:
            # There are none in this mocked record, but the message should be well-formed;
            # just assert the phrase exists.
            self.assertIn("Available dataset names", msg)


if __name__ == "__main__":
    unittest.main()
