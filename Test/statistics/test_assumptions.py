import os
import shutil
import tempfile
import unittest
import pandas as pd
import numpy as np

from synrxn.statistics.assumptions import run_assumptions


def make_demo_df():
    rng = np.random.default_rng(1)
    metrics = ["acc"]
    methods = ["A", "B"]
    rows = []
    for m in metrics:
        for subj in range(1, 7):
            row = {"scoring": m, "cv_cycle": subj}
            for meth in methods:
                row[meth] = (
                    0.8 + 0.05 * (1 if meth == "A" else 0) + float(rng.normal(0, 0.01))
                )
            rows.append(row)
    return pd.DataFrame(rows)


class TestAssumptions(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="stat_assump_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_run_assumptions_creates_files(self):
        df = make_demo_df()
        save_root = os.path.join(self.tmpdir, "statistics")
        out = run_assumptions(df, save_root=save_root)
        self.assertIn("acc", out)
        acc_dir = out["acc"]
        self.assertTrue(os.path.isdir(acc_dir))
        # check for files
        self.assertTrue(os.path.exists(os.path.join(acc_dir, "normality.pdf")))
        self.assertTrue(
            os.path.exists(os.path.join(acc_dir, "variance_homogeneity.csv"))
        )
        # validate CSV has expected columns
        csv = pd.read_csv(os.path.join(acc_dir, "variance_homogeneity.csv"))
        self.assertTrue(
            {"variance_fold_difference", "p_value"}.issubset(set(csv.columns))
        )


if __name__ == "__main__":
    unittest.main()
