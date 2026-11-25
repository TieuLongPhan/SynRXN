"""Random-forest regression control for legacy property fingerprint datasets.

Random folds measure an interpolation control. Family-specific split protocols
must be used separately for claims about chemical or source generalization.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, cross_validate

from synrxn.io.io import load_df_gz, save_results_json
from synrxn.reconstruction.paths import physical_path
from .classification import _make_cv_results_jsonable, summarize_cv_results

SCORING = {
    "neg_mae": "neg_mean_absolute_error",
    "neg_rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def _try_load_npz_fps(path: str | Path) -> np.ndarray:
    """Read fingerprint arrays, closing the file after materializing the data."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle, np.load(handle, allow_pickle=False) as arrays:
        if "fps" not in arrays:
            raise KeyError(f"'fps' not found in {path}")
        features = np.asarray(arrays["fps"])
    if features.ndim != 2:
        raise ValueError(f"Fingerprint array must be two-dimensional: {path}")
    return features


def get_data(name: str, target_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load property fingerprints and apply one finite-target mask to all rows.

    Feature matrices must follow the complete source CSV row order. Missing,
    nonnumeric and infinite targets are excluded from both feature matrices.
    A row-count mismatch fails before filtering to prevent silent misalignment.
    """
    source = physical_path(Path(f"Data/property/{name}.csv.gz").resolve())
    frame = load_df_gz(source)
    if target_col not in frame.columns:
        raise KeyError(f"Target {target_col!r} not found in {source}")
    features = []
    for fingerprint in ("drfp", "rxnfp"):
        directory = physical_path(Path(f"Data/Benchmark/{fingerprint}/property").resolve())
        candidates = [directory / f"{fingerprint}_{name}{suffix}"
                      for suffix in (".npz", ".npz.gz")]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError("Missing property fingerprints; tried: " +
                                    ", ".join(str(candidate) for candidate in candidates))
        matrix = _try_load_npz_fps(path)
        if len(matrix) != len(frame):
            raise ValueError(f"Feature/target row-count mismatch: {path} has {len(matrix)}, "
                             f"{source} has {len(frame)}")
        features.append(matrix)
    target = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)
    eligible = np.isfinite(target)
    if not eligible.any():
        raise ValueError(f"No finite numeric targets in {target_col!r}")
    return features[0][eligible], features[1][eligible], target[eligible]


def Benchmark(
    name: str,
    target_col: str,
    n_splits: int = 5,
    n_repeats: int = 2,
    random_state: int = 42,
    n_jobs: int = 4,
    scoring: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate DRFP and RXNFP with identical repeated random regression folds.

    MAE and RMSE scores are negative sklearn losses; larger scores are better.
    Predictions are evaluated in the target column's original numeric units.
    Result JSON is written under .cache/synrxn/runs/baselines/property/.
    """
    drfp, rxnfp, target = get_data(name=name, target_col=target_col)
    folds = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                          random_state=random_state)
    results = {}
    for fingerprint, matrix in (("drfp", drfp), ("rxnfp", rxnfp)):
        estimator = RandomForestRegressor(n_estimators=100, random_state=random_state,
                                          n_jobs=n_jobs)
        result = cross_validate(estimator, matrix, y=target, cv=folds,
                                scoring=SCORING if scoring is None else scoring,
                                return_train_score=False, n_jobs=1, error_score="raise")
        summarize_cv_results(result, tag=f"{fingerprint.upper()} - random regression")
        results[f"{fingerprint}_random"] = result
    output = Path(".cache/synrxn/runs/baselines/property") / f"{name}_{target_col}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_results_json(str(output), {key: _make_cv_results_jsonable(value)
                                    for key, value in results.items()})
    return results
