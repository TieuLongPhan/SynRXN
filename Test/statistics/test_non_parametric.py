import os
import shutil
import tempfile
import unittest
import pandas as pd
import numpy as np

from synrxn.statistics.nonparametric import run_nonparametric


def make_demo_df():
    rng = np.random.default_rng(3)
    metrics = ["f1"]
    methods = ["M1", "M2", "M3"]
    rows = []
    for met in metrics:
        for subj in range(1, 9):
            row = {"scoring": met, "cv_cycle": subj}
            for m in methods:
                row[m] = 0.6 + (0.02 if m == "M1" else 0.0) + float(rng.normal(0, 0.02))
            rows.append(row)
    return pd.DataFrame(rows)


class TestNonparametric(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="stat_npar_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_run_nonparametric_writes_files(self):
        df = make_demo_df()
        save_root = os.path.join(self.tmpdir, "statistics")
        out = run_nonparametric(df, save_root=save_root)
        self.assertIn("f1", out)
        dirs = out["f1"]
        self.assertTrue(os.path.isdir(dirs["friedman_dir"]))
        self.assertTrue(os.path.isdir(dirs["conover_dir"]))
        # check conover outputs
        conover_files = os.listdir(dirs["conover_dir"])
        self.assertTrue(any("cofried_pc" in f for f in conover_files))
        self.assertTrue(any(f.endswith(".pdf") for f in conover_files))
        # verify CSV loads
        pc_csvs = [f for f in conover_files if f.startswith("cofried_pc_")]
        self.assertTrue(len(pc_csvs) >= 1)
        pc = pd.read_csv(os.path.join(dirs["conover_dir"], pc_csvs[0]), index_col=0)
        self.assertGreaterEqual(pc.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
