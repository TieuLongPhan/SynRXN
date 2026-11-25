"""Regression controls use aligned numeric targets and regression estimators."""
import gzip
from io import BytesIO
import json

import numpy as np
import pandas as pd
import pytest

from synrxn.baseline import property as baseline


def write_inputs(root, targets, *, rxnfp_rows=None):
    source = root / "Data/property/example.csv.gz"
    source.parent.mkdir(parents=True)
    pd.DataFrame({"barrier": targets}).to_csv(source, index=False, compression="gzip")
    features = np.arange(len(targets) * 3).reshape(-1, 3)
    for fingerprint in ("drfp", "rxnfp"):
        directory = root / "Data/Benchmark" / fingerprint / "property"
        directory.mkdir(parents=True)
        values = features if fingerprint == "drfp" else features + 100
        if fingerprint == "rxnfp" and rxnfp_rows is not None:
            values = values[:rxnfp_rows]
        if fingerprint == "drfp":
            np.savez(directory / "drfp_example.npz", fps=values)
        else:
            buffer = BytesIO()
            np.savez(buffer, fps=values)
            (directory / "rxnfp_example.npz.gz").write_bytes(gzip.compress(buffer.getvalue()))
    return features


def test_property_targets_keep_both_feature_matrices_aligned(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    features = write_inputs(tmp_path, [1.5, "", "bad", float("inf"), -2.5, 3])

    drfp, rxnfp, target = baseline.get_data("example", target_col="barrier")

    np.testing.assert_array_equal(target, [1.5, -2.5, 3])
    np.testing.assert_array_equal(drfp, features[[0, 4, 5]])
    np.testing.assert_array_equal(rxnfp, features[[0, 4, 5]] + 100)


def test_property_feature_alignment_is_checked_before_target_filtering(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_inputs(tmp_path, [1, "", 2], rxnfp_rows=2)
    with pytest.raises(ValueError, match="row-count mismatch"):
        baseline.get_data("example", "barrier")


@pytest.mark.parametrize("targets,column,message", [
    ([1, 2], "missing", "not found"),
    (["bad", float("inf")], "barrier", "No finite numeric targets"),
])
def test_property_requires_an_eligible_target(tmp_path, monkeypatch, targets, column, message):
    monkeypatch.chdir(tmp_path)
    write_inputs(tmp_path, targets)
    with pytest.raises((KeyError, ValueError), match=message):
        baseline.get_data("example", column)


def test_property_benchmark_fits_regression_to_continuous_synthetic_targets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    features = np.arange(60, dtype=float).reshape(20, 3)
    target = 0.25 * features[:, 0] + 0.3
    calls = []

    def get_data(name, target_col):
        calls.append((name, target_col))
        return features, features[:, ::-1], target

    monkeypatch.setattr(baseline, "get_data", get_data)
    results = baseline.Benchmark("synthetic", target_col="barrier",
                                 n_splits=2, n_repeats=1, n_jobs=1)

    assert calls == [("synthetic", "barrier")]
    assert set(results) == {"drfp_random", "rxnfp_random"}
    for values in results.values():
        assert set(values) == {"fit_time", "score_time", "test_neg_mae", "test_neg_rmse", "test_r2"}
        assert all(len(scores) == 2 and np.isfinite(scores).all() for scores in values.values())
        assert (values["test_neg_mae"] <= 0).all()
        assert (values["test_neg_rmse"] <= 0).all()
    output = tmp_path / ".cache/synrxn/runs/baselines/property/synthetic_barrier.json"
    assert set(json.loads(output.read_text())) == set(results)
