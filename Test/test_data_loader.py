import unittest
import tempfile
from pathlib import Path

from synrxn.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self) -> None:
        # Use a temporary directory for caching to avoid polluting project files.
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_load_ecoli_success(self):
        """Test that aam/ecoli loads successfully and returns a non-empty DataFrame."""
        dl = DataLoader(task="aam", cache_dir=self.cache_dir, timeout=30)
        df = dl.load("ecoli", use_cache=True)
        # Should be a pandas DataFrame with at least one row/column
        import pandas as pd  # local import to keep top-level minimal

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(
            df.shape[0], 1, "Expected at least one row in aam/ecoli"
        )
        self.assertGreaterEqual(
            df.shape[1], 1, "Expected at least one column in aam/ecoli"
        )

    def test_bad_name_shows_helpful_message(self):
        """Requesting a non-existent name should raise FileNotFoundError with useful info."""
        dl = DataLoader(task="aam", cache_dir=self.cache_dir, timeout=30)
        bad_name = "this_name_definitely_does_not_exist_12345"
        with self.assertRaises(FileNotFoundError) as cm:
            dl.load(bad_name, use_cache=False)

        msg = str(cm.exception)
        # Must contain either the tried raw URLs (always present) or an "Available dataset names"
        self.assertIn("Tried URLs", msg)
        # If the API returned an available list, we expect 'Available dataset names' to appear.
        if "Available dataset names" in msg:
            # the repo should include 'ecoli' under aam, so assert that appears in the listing
            self.assertRegex(msg, r"\becoli\b", msg)
        else:
            # otherwise ensure the tried URLs are present and look like raw github urls
            self.assertRegex(
                msg, r"https://github\.com/.+/.+/raw/refs/heads/.+/Data/.+/.+\.csv", msg
            )


if __name__ == "__main__":
    unittest.main()
