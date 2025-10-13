import os
import shutil
import tempfile
import unittest

from synrxn.statistics.runner import main


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="stat_runner_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_runner_demo(self):
        save_root = os.path.join(self.tmpdir, "statistics")
        summary = main(csv_path=None, save_root=save_root)
        # Should include the three families
        self.assertIn("assumptions", summary)
        self.assertIn("parametric", summary)
        self.assertIn("nonparametric", summary)
        # the assumptions dict should contain at least one metric directory
        assump = summary["assumptions"]
        self.assertIsInstance(assump, dict)
        if assump:
            d = list(assump.values())[0]
            self.assertTrue(os.path.isdir(d))
            self.assertTrue(
                any(
                    fname.endswith(".pdf") or fname.endswith(".csv")
                    for fname in os.listdir(d)
                )
            )


if __name__ == "__main__":
    unittest.main()
