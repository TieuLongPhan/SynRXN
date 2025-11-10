from __future__ import annotations

import os
import gzip
from io import BytesIO
from typing import Dict, Any, Tuple
import numpy as np

from synrxn.io.io import load_df_gz, save_results_json
from synrxn.split.repeated_kfold import RepeatedKFoldsSplitter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, RepeatedKFold
from sklearn.metrics import make_scorer, f1_score, matthews_corrcoef

# ----------------------------
# scoring
# ----------------------------
SCORING = {
    "accuracy": "accuracy",
    "f1_weighted": make_scorer(f1_score, average="weighted"),
    "mcc": make_scorer(matthews_corrcoef),
}


# ----------------------------
# helpers: printing + jsonability
# ----------------------------
def summarize_cv_results(cv_res: Dict[str, Any], tag: str) -> None:
    """Print mean/std and per-split values for test_* metrics from cross_validate output.

    :param cv_res: dict returned by sklearn.model_selection.cross_validate
    :param tag: short string used as header when printing
    """
    import numpy as _np

    print(f"\n--- {tag} ---")
    keys = [k for k in cv_res.keys() if k.startswith("test_")]
    for k in keys:
        arr = _np.asarray(cv_res[k], dtype=float)
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
            converted = []
            for x in v:
                if isinstance(x, (np.ndarray, np.generic)):
                    try:
                        converted.append(np.asarray(x).tolist())
                    except Exception:
                        converted.append(x)
                else:
                    converted.append(x)
            out[k] = converted
        else:
            out[k] = v
    return out


# ----------------------------
# robust .npz / .npz.gz loader
# ----------------------------
def _try_load_npz_fps(path: str) -> np.ndarray:
    """
    Load 'fps' array from .npz or .npz.gz file.
    Returns numpy array on success or raises Exception with descriptive message.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".gz"):
        # read gz bytes and pass to numpy load via BytesIO
        with gzip.open(path, "rb") as fh:
            data = fh.read()
            buf = BytesIO(data)
            arrs = np.load(buf, allow_pickle=False)
    else:
        arrs = np.load(path, allow_pickle=False)

    if "fps" not in arrs:
        raise KeyError(f"'fps' not found in {path}")

    fps = arrs["fps"]
    # ensure we return a real ndarray (not an open NpzFile)
    fps = np.asarray(fps)
    print(f"Loaded 'fps' from: {path} (shape={fps.shape})")
    return fps


def get_data(name: str, level: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load features and labels for a benchmark dataset.

    This loader will try:
      ./Data/Benchmark/drfp/classification/drfp_{name}.npz
      ./Data/Benchmark/drfp/classification/drfp_{name}.npz.gz
      ./Data/Benchmark/drfp/classification/drfp_{base}.npz
      ./Data/Benchmark/drfp/classification/drfp_{base}.npz.gz

    and similarly for rxnfp. 'base' is name.split('_', 1)[0], so datasets
    like 'ecreact_1' will fall back to 'ecreact'.

    :param name: dataset name (e.g., 'ecreact_1' or 'ecreact')
    :param level: integer label level for datasets that use levels (syntemp, claire)
    :returns: (X_drfp, X_rxnfp, y)
    :raises FileNotFoundError / KeyError if files or expected columns are missing
    """
    drfp_dir = "./Data/Benchmark/drfp/classification"
    rxnfp_dir = "./Data/Benchmark/rxnfp/classification"
    csv_path = f"Data/classification/{name}.csv.gz"

    # candidate base names: name itself, then prefix before first '_'
    candidates = [name]
    if "_" in name:
        base = name.split("_", 1)[0]
        if base != name:
            candidates.append(base)

    tried = []
    X_drfp = X_rxnfp = None

    for cand in candidates:
        for ext in (".npz", ".npz.gz"):
            drfp_path = os.path.join(drfp_dir, f"drfp_{cand}{ext}")
            rxnfp_path = os.path.join(rxnfp_dir, f"rxnfp_{cand}{ext}")
            tried.append(drfp_path)
            tried.append(rxnfp_path)
            if os.path.exists(drfp_path) and os.path.exists(rxnfp_path):
                # load them and return
                X_drfp = _try_load_npz_fps(drfp_path)
                X_rxnfp = _try_load_npz_fps(rxnfp_path)
                break
        if X_drfp is not None and X_rxnfp is not None:
            break

    if X_drfp is None or X_rxnfp is None:
        tried_unique = sorted(set(tried))
        raise FileNotFoundError(
            "Could not find DRFP/RXNFP feature files for dataset. "
            "Tried the following paths:\n  " + "\n  ".join(tried_unique)
        )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    data = load_df_gz(csv_path)
    print(name)
    # label logic (dataset-specific)
    if name.startswith("syntemp"):
        # allow calling with 'syntemp' or 'syntemp_0' etc.
        if f"label_{level}" not in data.columns:
            raise KeyError(f"label_{level} not found in {csv_path}")
        y = data[f"label_{level}"].values
    elif name.startswith("ecreact"):
        # claire and ecreact use ec{level}
        if f"ec{level}" not in data.columns:
            raise KeyError(f"ec{level} not found in {csv_path}")
        y = data[f"ec{level}"].values
    else:
        if "label" not in data.columns:
            raise KeyError(f"'label' column not found in {csv_path}")
        y = data["label"].values

    y = np.asarray(y).ravel()
    return X_drfp, X_rxnfp, y


# ----------------------------
# Benchmark runner
# ----------------------------
def Benchmark(
    name: str,
    level: int = 0,
    n_splits: int = 5,
    n_repeats: int = 2,
    random_state: int = 42,
    n_jobs: int = 4,
    scoring: Dict[str, Any] = SCORING,
) -> Dict[str, Dict[str, Any]]:
    """
    Run random and stratified cross-validation for DRFP and RXNFP feature sets.

    Returns a dict with the 4 cv result dicts. Also saves a JSON file into:
      Data/Benchmark/result/classification/{name}_{level}.json
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

    # --- ensure result directory exists and make cv results jsonable ---
    out_path = f"Data/Benchmark/result/classification/{name}_{level}.json"
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    jsonable_results: Dict[str, Any] = {}
    for k, v in results.items():
        jsonable_results[k] = _make_cv_results_jsonable(v)

    save_results_json(out_path, jsonable_results)
    print(f"Saved results to: {out_path}")
    return results
