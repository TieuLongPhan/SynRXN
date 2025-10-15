# benchmark_property.py
from __future__ import annotations

import os
from typing import Dict, Any, Tuple
import numpy as np

from synrxn.io.io import load_df_gz, save_results_json

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error


SCORING = {
    "r2": "r2",
    "rmse": make_scorer(mean_squared_error, greater_is_better=False, squared=False),
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
}


def summarize_cv_results(cv_res: Dict[str, Any], tag: str) -> None:
    """Print mean/std and per-split values for test_* metrics from cross_validate output."""
    import numpy as _np

    print(f"\n--- {tag} ---")
    keys = [k for k in cv_res.keys() if k.startswith("test_")]
    for k in keys:
        arr = _np.asarray(cv_res[k])
        print(f"{k}: mean={arr.mean():.4f}, std={arr.std(ddof=0):.4f}, values={arr}")


def get_data(name: str, target_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load features and the target column for a property dataset.

    Expects:
      ./Data/Benchmark/drfp/property/drfp_{name}.npz   (npz with 'fps')
      ./Data/Benchmark/rxnfp/property/rxnfp_{name}.npz (npz with 'fps')
      Data/property/{name}.csv.gz  (contains target_col)

    :param name: dataset id (filename base)
    :param target_col: column name in CSV to use as target
    :returns: X_drfp, X_rxnfp, y (numpy arrays)
    :raises FileNotFoundError / KeyError
    """
    drfp_path = f"./Data/Benchmark/drfp/property/drfp_{name}.npz"
    rxnfp_path = f"./Data/Benchmark/rxnfp/property/rxnfp_{name}.npz"
    csv_path = f"Data/property/{name}.csv.gz"

    if not os.path.exists(drfp_path):
        raise FileNotFoundError(drfp_path)
    if not os.path.exists(rxnfp_path):
        raise FileNotFoundError(rxnfp_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    X_drfp = np.load(drfp_path)["fps"]
    X_rxnfp = np.load(rxnfp_path)["fps"]
    data = load_df_gz(csv_path)

    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not found in {csv_path}")

    y = np.asarray(data[target_col].values).ravel()
    return X_drfp, X_rxnfp, y


def Benchmark(
    name: str,
    target_col: str,
    n_splits: int = 5,
    n_repeats: int = 2,
    random_state: int = 42,
    n_jobs: int = 4,
    scoring: Dict[str, Any] = SCORING,
) -> Dict[str, Dict[str, Any]]:
    """
    Run RANDOM (unstratified) cross-validation for a property dataset.

    :param name: dataset id (matches filenames under Data/)
    :param target_col: the CSV column to use as regression target
    :param n_splits: number of outer folds
    :param n_repeats: number of repeats
    :param random_state: RNG seed
    :param n_jobs: n_jobs passed to RandomForestRegressor (internal parallelism)
    :param scoring: scoring dict for cross_validate
    :returns: dict with keys: 'drfp_random','rxnfp_random' each mapping to cross_validate output
    """
    X_drfp, X_rxnfp, y = get_data(name=name, target_col=target_col)
    results: Dict[str, Dict[str, Any]] = {}

    print(
        f"\nBenchmark (property RANDOM only): {name}  target={target_col}  (n_samples={len(y)})\n"
    )

    # RANDOM mode (unstratified) using RepeatedKFold
    print("=== RANDOM mode (unstratified, RepeatedKFold) ===")
    reg = RandomForestRegressor(
        n_estimators=200, random_state=random_state, n_jobs=n_jobs
    )
    rkf = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    # run on drfp
    cv_drfp_random = cross_validate(
        reg, X_drfp, y=y, cv=rkf, scoring=scoring, return_train_score=False, n_jobs=1
    )
    summarize_cv_results(cv_drfp_random, tag=f"{name}:{target_col} - DRFP - random")
    results["drfp_random"] = cv_drfp_random

    # run on rxnfp
    cv_rxnfp_random = cross_validate(
        reg, X_rxnfp, y=y, cv=rkf, scoring=scoring, return_train_score=False, n_jobs=1
    )
    summarize_cv_results(cv_rxnfp_random, tag=f"{name}:{target_col} - RXNFP - random")
    results["rxnfp_random"] = cv_rxnfp_random

    # save JSON results (create directory if needed)
    out_dir = os.path.dirname(
        f"Data/Benchmark/result/property/{name}_{target_col}.json"
    )
    os.makedirs(out_dir, exist_ok=True)
    save_results_json(
        f"Data/Benchmark/result/property/{name}_{target_col}.json", results
    )

    return results
