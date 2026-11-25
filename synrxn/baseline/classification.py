from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

from synrxn.io.io import load_df_gz, save_results_json
from synrxn.split.repeated_kfold import RepeatedKFoldsSplitter
from synrxn.reconstruction.paths import physical_path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.metrics import make_scorer, f1_score, matthews_corrcoef

SCORING = {
    "accuracy": "accuracy",
    "f1_weighted": make_scorer(f1_score, average="weighted"),
    "mcc": make_scorer(matthews_corrcoef),
}


def summarize_cv_results(cv_res: Dict[str, Any], tag: str) -> None:
    """Print mean/std and per-split values for test_* metrics from cross_validate output.

    :param cv_res: dict returned by sklearn.model_selection.cross_validate
    :param tag: short string used as header when printing
    """
    import numpy as _np

    print(f"\n--- {tag} ---")
    keys = [k for k in cv_res.keys() if k.startswith("test_")]
    for k in keys:
        arr = _np.asarray(cv_res[k])
        print(f"{k}: mean={arr.mean():.4f}, std={arr.std(ddof=0):.4f}, values={arr}")


def _make_cv_results_jsonable(cv_res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert numpy arrays/lists in sklearn cross_validate output to plain Python lists
    so the structure is JSON serializable (safe to pass to save_results_json).
    """
    out: Dict[str, Any] = {}
    for k, v in cv_res.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            # convert list of numpy scalars/arrays -> list
            try:
                out[k] = [
                    (
                        np.asarray(x).tolist()
                        if isinstance(x, (np.ndarray, np.generic))
                        else x
                    )
                    for x in v
                ]
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def get_data(name: str, level: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load features and labels for a benchmark dataset.

    Expects feature files:
      ./Data/Benchmark/drfp/classification/drfp_{name}.npz  (array 'fps')
      ./Data/Benchmark/rxnfp/classification/rxnfp_{name}.npz (array 'fps')

    and label CSV:
      Data/classification/{name}.csv.gz

    :param name: dataset name (e.g., 'syntemp' or other benchmark id)
    :param level: integer label level for syntemp (default 0)
    :return: (X_drfp, X_rxnfp, y) where X_* are 2D numpy arrays and y is 1D labels array
    :raises FileNotFoundError: if expected files are missing
    """
    drfp_path = physical_path(Path(f"Data/Benchmark/drfp/classification/drfp_{name}.npz").resolve())
    rxnfp_path = physical_path(Path(f"Data/Benchmark/rxnfp/classification/rxnfp_{name}.npz").resolve())
    csv_path = physical_path(Path(f"Data/classification/{name}.csv.gz").resolve())

    if not os.path.exists(drfp_path):
        raise FileNotFoundError(drfp_path)
    if not os.path.exists(rxnfp_path):
        raise FileNotFoundError(rxnfp_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    X_drfp = np.load(drfp_path)["fps"]
    X_rxnfp = np.load(rxnfp_path)["fps"]
    data = load_df_gz(csv_path)

    if name == "syntemp":
        if f"label_{level}" not in data.columns:
            raise KeyError(f"label_{level} not found in {csv_path}")
        y = data[f"label_{level}"].values
    elif name == "ecreact":
        if f"ec{level}" not in data.columns:
            raise KeyError(f"label_{level} not found in {csv_path}")
        y = data[f"ec{level}"].values
    else:
        if "label" not in data.columns:
            raise KeyError(f"'label' column not found in {csv_path}")
        y = data["label"].values

    # flatten/ensure 1D labels, and convert to numpy
    y = np.asarray(y).ravel()
    return X_drfp, X_rxnfp, y


def Benchmark(
    name: str,
    level: int = 0,
    n_splits: int = 5,
    n_repeats: int = 2,
    random_state: int = 42,
    n_jobs: int = 4,
    scoring: Dict[str, Any] = SCORING,
) -> Dict[str, Dict[str, Any]]:
    """Run random and stratified cross-validation for DRFP and RXNFP features.

    Random cross-validation uses scikit-learn's RepeatedKFold. Stratified
    cross-validation uses RepeatedKFoldsSplitter with label-stratified outer
    folds. Cross-validation runs with one worker to avoid nested parallelism;
    ``n_jobs`` controls the classifier's workers.

    :param name: Dataset name used to locate stored features and labels.
    :param level: Label level for hierarchical classification datasets.
    :param n_splits: Number of outer folds.
    :param n_repeats: Number of repetitions.
    :param random_state: Random seed for reproducible assignments.
    :param n_jobs: Number of workers passed to RandomForestClassifier.
    :param scoring: Scoring dictionary passed to cross_validate.
    :return: Cross-validation results under ``drfp_random``, ``drfp_strat``,
        ``rxnfp_random`` and ``rxnfp_strat``.
    :rtype: Dict[str, Dict[str, Any]]
    """
    X_drfp, X_rxnfp, y = get_data(name=name, level=level)
    results: Dict[str, Dict[str, Any]] = {}

    print(f"\nBenchmark: {name}  (n_samples={len(y)})\n")

    # ----------------------
    # 1) RANDOM mode (unstratified) using RepeatedKFold
    # ----------------------
    print("=== RANDOM mode (unstratified, RepeatedKFold) ===")
    clf = RandomForestClassifier(
        n_estimators=100, random_state=random_state, n_jobs=n_jobs
    )
    rkf = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    # run on drfp
    cv_drfp_random = cross_validate(
        clf, X_drfp, y=y, cv=rkf, scoring=scoring, return_train_score=False, n_jobs=1
    )
    summarize_cv_results(cv_drfp_random, tag="DRFP - random")
    results["drfp_random"] = cv_drfp_random

    # run on rxnfp
    cv_rxnfp_random = cross_validate(
        clf, X_rxnfp, y=y, cv=rkf, scoring=scoring, return_train_score=False, n_jobs=1
    )
    summarize_cv_results(cv_rxnfp_random, tag="RXNFP - random")
    results["rxnfp_random"] = cv_rxnfp_random

    # ----------------------
    # 2) STRATIFIED mode (use RepeatedKFoldsSplitter)
    # ----------------------
    print("\n=== STRATIFIED mode (RepeatedKFoldsSplitter) ===")
    splitter = RepeatedKFoldsSplitter(
        n_splits=n_splits,
        n_repeats=n_repeats,
        ratio=(8, 1, 1),
        shuffle=True,
        random_state=random_state,
    )
    clf = RandomForestClassifier(
        n_estimators=100, random_state=random_state, n_jobs=n_jobs
    )

    # cross_validate will call splitter.split(X, y), so pass y (used for stratification)
    cv_drfp_strat = cross_validate(
        clf,
        X_drfp,
        y=y,
        cv=splitter,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,
    )
    summarize_cv_results(cv_drfp_strat, tag="DRFP - stratified")
    results["drfp_strat"] = cv_drfp_strat

    cv_rxnfp_strat = cross_validate(
        clf,
        X_rxnfp,
        y=y,
        cv=splitter,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,
    )
    summarize_cv_results(cv_rxnfp_strat, tag="RXNFP - stratified")
    results["rxnfp_strat"] = cv_rxnfp_strat

    print("\nBenchmark completed.\n")

    # --- NEW: ensure result directory exists and make cv results jsonable ---
    out_path = f".cache/synrxn/runs/baselines/classification/{name}_{level}.json"
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    # convert numpy arrays/lists into python lists for safe JSON serialization
    jsonable_results: Dict[str, Any] = {}
    for k, v in results.items():
        jsonable_results[k] = _make_cv_results_jsonable(v)

    save_results_json(out_path, jsonable_results)
    return results
