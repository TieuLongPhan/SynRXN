# benchmark_property.py  (fixed - handles string scorers correctly)
from __future__ import annotations

import os
from typing import Dict, Any, Tuple
import numpy as np
import warnings

from synrxn.io.io import load_df_gz, save_results_json

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, RepeatedKFold


SCORING = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}


def _scorer_is_negated(scorer_spec: Any) -> bool:
    """
    Return True if the scorer specification represents a negated scorer.
    Accepts either a scorer object (has greater_is_better attr) or a string
    like 'neg_mean_absolute_error'.
    """
    # string spec: starts with 'neg_' means sklearn returns negative of metric
    if isinstance(scorer_spec, str):
        return scorer_spec.startswith("neg_")
    # scorer object
    try:
        return getattr(scorer_spec, "greater_is_better", True) is False
    except Exception:
        return False


def summarize_cv_results(
    cv_res: Dict[str, Any], tag: str, scoring: Dict[str, Any] | None = None
) -> None:
    """
    Print mean/std and per-split values for test_* metrics from cross_validate output.
    If scoring is provided, metrics that are negated (e.g. 'neg_mean_absolute_error')
    are shown with sign inverted for human-readable display (positive MAE / RMSE).
    """
    import numpy as _np

    print(f"\n--- {tag} ---")
    keys = [k for k in cv_res.keys() if k.startswith("test_")]

    # build set of metric names that are negated (i.e., should be sign-flipped)
    negated_metrics = set()
    if scoring is not None:
        for metric_name, spec in scoring.items():
            if _scorer_is_negated(spec):
                negated_metrics.add(metric_name)

    for k in keys:
        arr = _np.asarray(cv_res[k], dtype=float)
        metric_name = k.split("_", 1)[1]
        if metric_name in negated_metrics:
            arr = -arr
        print(f"{k}: mean={arr.mean():.4f}, std={arr.std(ddof=0):.4f}, values={arr}")


def normalize_cv_results(
    cv_res: Dict[str, Any], scoring: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Return a copy of cv_res where arrays produced by scorers that are negated
    (greater_is_better==False or strings starting with 'neg_') are multiplied
    by -1 so the values become the natural (positive) metric values for MAE/RMSE.
    """
    out: Dict[str, Any] = {}
    # copy arrays/lists to numpy arrays for manipulation
    for k, v in cv_res.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            out[k] = np.asarray(v, dtype=float).copy()
        else:
            out[k] = v

    negated = {name for name, spec in scoring.items() if _scorer_is_negated(spec)}

    # flip sign for 'test_X' and 'train_X' where X in negated
    for key in list(out.keys()):
        if not (key.startswith("test_") or key.startswith("train_")):
            continue
        metric_name = key.split("_", 1)[1]
        if metric_name in negated and isinstance(out[key], np.ndarray):
            out[key] = (-out[key]).tolist()

    return out


def get_data(name: str, target_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load features and the target column for a property dataset.

    Expects:
      ./Data/Benchmark/drfp/property/drfp_{name}.npz   (npz with 'fps')
      ./Data/Benchmark/rxnfp/property/rxnfp_{name}.npz (npz with 'fps')
      Data/property/{name}.csv.gz  (contains target_col)
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

    Returns a dict with keys 'drfp_random' and 'rxnfp_random' mapping to
    the cross_validate output, but where MAE/RMSE entries have been normalized
    to positive values (for human readability and downstream reporting).
    """
    # quick check: ensure RMSE scorer is available in this sklearn version
    if scoring.get("rmse", "").startswith("neg_root_mean_squared_error"):
        # sklearn >= 0.24 defines neg_root_mean_squared_error
        pass
    else:
        # if user environment doesn't support neg_root_mean_squared_error, warn
        # (we still accept the string, cross_validate will error if invalid)
        warnings.warn(
            "SCORING['rmse'] is set to 'neg_root_mean_squared_error'. "
            "Ensure your scikit-learn version supports this (>=0.24)."
        )

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
    # normalize negated scorers to positive reporting values
    cv_drfp_random_pos = normalize_cv_results(cv_drfp_random, scoring)
    summarize_cv_results(
        cv_drfp_random_pos, tag=f"{name}:{target_col} - DRFP - random", scoring=scoring
    )
    results["drfp_random"] = cv_drfp_random_pos

    # run on rxnfp
    cv_rxnfp_random = cross_validate(
        reg, X_rxnfp, y=y, cv=rkf, scoring=scoring, return_train_score=False, n_jobs=1
    )
    cv_rxnfp_random_pos = normalize_cv_results(cv_rxnfp_random, scoring)
    summarize_cv_results(
        cv_rxnfp_random_pos,
        tag=f"{name}:{target_col} - RXNFP - random",
        scoring=scoring,
    )
    results["rxnfp_random"] = cv_rxnfp_random_pos

    # save JSON results (create directory if needed)
    out_dir = os.path.dirname(
        f"Data/Benchmark/result/property/{name}_{target_col}.json"
    )
    os.makedirs(out_dir, exist_ok=True)
    save_results_json(
        f"Data/Benchmark/result/property/{name}_{target_col}.json", results
    )

    return results
