import os
import shutil
import tempfile
import unittest
import pandas as pd
import numpy as np

from synrxn.statistics.parametric import run_parametric


def make_demo_df():
    rng = np.random.default_rng(2)
    metrics = ["accuracy", "mcc"]
    methods = ["A", "B", "C"]
    rows = []
    for met in metrics:
        for subj in range(1, 11):
            row = {"scoring": met, "cv_cycle": subj}
            for m in methods:
                base = 0.9 if met == "accuracy" else 0.8
                row[m] = base + (0.01 if m == "A" else 0.0) + float(rng.normal(0, 0.01))
            rows.append(row)
    return pd.DataFrame(rows)


class TestParametric(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="stat_param_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_run_parametric_writes_files(self):
        df = make_demo_df()
        save_root = os.path.join(self.tmpdir, "statistics")
        out = run_parametric(
            df,
            save_root=save_root,
            direction_dict={"accuracy": "maximize", "mcc": "maximize"},
            effect_dict={"accuracy": 0.05, "mcc": 0.05},
        )
        # expect both metrics present
        self.assertIn("accuracy", out)
        self.assertIn("mcc", out)
        for metric, dirs in out.items():
            self.assertTrue(os.path.isdir(dirs["anova_dir"]))
            self.assertTrue(os.path.isdir(dirs["tukey_dir"]))
            files = os.listdir(dirs["tukey_dir"])
            # expect at least CSVs and PDFs
            self.assertTrue(any(f.endswith(".csv") for f in files))
            self.assertTrue(any(f.endswith(".pdf") for f in files))
            # load one csv to ensure content (tukey_pc_*.csv)
            pc_csvs = [f for f in files if f.startswith("tukey_pc_")]
            self.assertTrue(len(pc_csvs) >= 1)
            pc_path = os.path.join(dirs["tukey_dir"], pc_csvs[0])
            pc = pd.read_csv(pc_path, index_col=0)
            self.assertGreaterEqual(pc.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
