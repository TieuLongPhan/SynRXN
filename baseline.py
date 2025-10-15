#!/usr/bin/env python3
# main.py - unified runner for classification and property benchmarks

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse
import traceback
import sys
import joblib
import numpy as np
import pandas as pd

try:
    from synrxn.baseline.classification import Benchmark as classification_Benchmark
except Exception:
    classification_Benchmark = None

try:
    from synrxn.baseline.property import Benchmark as property_Benchmark
except Exception:
    property_Benchmark = None


# ------------------------------
# Configuration: datasets, levels, targets
# ------------------------------
CLASS_NAMES = [
    "claire",
    "schneider_b",
    "schneider_u",
    "syntemp",
    "tpl_b",
    "tpl_u",
    "uspto_50k_b",
    "uspto_50k_u",
]
CLASS_BASE_LEVELS = [0, 1, 2, 3]
CLASS_SPECIAL_LEVELS = {
    "syntemp": [0, 1, 2],
    "claire": [1, 2, 3],
}

PROPERTY_PAIRS: List[Tuple[str, str]] = [
    ("b97xd3", "ea"),
    ("b97xd3", "dh"),
    ("cycloadd", "G_act"),
    ("cycloadd", "G_r"),
    ("e2", "ea"),
    ("e2sn2", "ea"),
    ("lograte", "lograte"),
    ("phosphatase", "Conversion"),
    ("rad6re", "dh"),
    ("rdb7", "ea"),
    ("rgd1", "ea"),
    ("sn2", "ea"),
    ("snar", "ea"),
    ("suzuki_miyaura", "yield"),
    ("uspto_yield", "yield"),
]


# ------------------------------
# Utilities: checkpointing, flattening, summary
# ------------------------------
def levels_for(name: str) -> List[int]:
    return CLASS_SPECIAL_LEVELS.get(name, CLASS_BASE_LEVELS)


def checkpoint_path(out_dir: Path, task: str) -> Path:
    if task == "classification":
        return out_dir / "all_results_classification.joblib"
    else:
        return out_dir / "all_results_property.joblib"


def summary_csv_path(out_dir: Path, task: str) -> Path:
    if task == "classification":
        return out_dir / "summary_metrics_classification.csv"
    else:
        return out_dir / "summary_metrics_property.csv"


def checkpoint_save(all_results: Dict[Any, Any], path: Path) -> None:
    joblib.dump(all_results, path, compress=("gzip", 3))


def flatten_cv_to_rows_class(
    name: str, level: int, cvdict: Dict[str, Any]
) -> List[Dict[str, Any]]:
    rows = []
    for feat_key, metrics in cvdict.items():
        for metric_name, arr in metrics.items():
            arr = np.asarray(arr)
            for fold_idx, v in enumerate(arr):
                rows.append(
                    {
                        "name": name,
                        "level": level,
                        "feature_mode": feat_key,
                        "metric": metric_name,
                        "fold": int(fold_idx),
                        "value": float(v),
                    }
                )
    return rows


def flatten_cv_to_rows_property(
    name: str, target: str, cvdict: Dict[str, Any]
) -> List[Dict[str, Any]]:
    rows = []
    for feat_key, metrics in cvdict.items():
        for metric_name, arr in metrics.items():
            arr = np.asarray(arr)
            for fold_idx, v in enumerate(arr):
                rows.append(
                    {
                        "name": name,
                        "target": target,
                        "feature_mode": feat_key,
                        "metric": metric_name,
                        "fold": int(fold_idx),
                        "value": float(v),
                    }
                )
    return rows


def produce_summary_csv_class(
    all_results: Dict[str, Dict[int, Dict[str, Any]]], outpath: Path
) -> None:
    rows = []
    for name, levels in all_results.items():
        for level, cvdict in levels.items():
            if cvdict is None:
                continue
            for feat_key, metrics in cvdict.items():
                for metric_name, arr in metrics.items():
                    arr = np.asarray(arr)
                    rows.append(
                        {
                            "name": name,
                            "level": level,
                            "feature_mode": feat_key,
                            "metric": metric_name,
                            "mean": float(np.mean(arr)),
                            "std": float(np.std(arr, ddof=0)),
                            "n_splits": int(arr.size),
                        }
                    )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["name", "level", "feature_mode", "metric"]).reset_index(
            drop=True
        )
    df.to_csv(outpath, index=False)
    print(f"Saved summary CSV to: {outpath}")


def produce_summary_csv_property(
    all_results: Dict[Tuple[str, str], Dict[str, Any]], outpath: Path
) -> None:
    rows = []
    for (name, target), cvdict in all_results.items():
        if cvdict is None:
            continue
        for feat_key, metrics in cvdict.items():
            for metric_name, arr in metrics.items():
                arr = np.asarray(arr)
                rows.append(
                    {
                        "name": name,
                        "target": target,
                        "feature_mode": feat_key,
                        "metric": metric_name,
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr, ddof=0)),
                        "n_splits": int(arr.size),
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["name", "target", "feature_mode", "metric"]).reset_index(
            drop=True
        )
    df.to_csv(outpath, index=False)
    print(f"Saved summary CSV to: {outpath}")


# ------------------------------
# Main runner
# ------------------------------
def run_classification(
    Benchmark_fn,
    names: List[str],
    out_dir: Path,
    dry_run: bool,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    n_jobs: int,
):
    OUT_DIR = out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT = checkpoint_path(OUT_DIR, "classification")
    SUMMARY = summary_csv_path(OUT_DIR, "classification")

    all_results: Dict[str, Dict[int, Any]] = {}
    if CHECKPOINT.exists():
        try:
            all_results = joblib.load(CHECKPOINT)
            print("Loaded classification checkpoint", CHECKPOINT)
        except Exception as e:
            print("Failed to load checkpoint, starting fresh:", e)
            all_results = {}

    for name in names:
        levs = levels_for(name)
        print(f"\nDataset: {name} -> levels: {levs}")
        all_results.setdefault(name, {})
        for level in levs:
            if level in all_results[name] and all_results[name][level] is not None:
                print(f"  - {name} level {level}: already done, skipping")
                continue

            print(f"  -> Running classification Benchmark(name={name}, level={level})")
            if dry_run:
                all_results[name][level] = None
                checkpoint_save(all_results, CHECKPOINT)
                continue

            start = time.time()
            try:
                res = Benchmark_fn(
                    name=name,
                    level=level,
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )
            except Exception as e:
                print(f"  !!! Error running Benchmark for {name} level {level}: {e}")
                traceback.print_exc()
                all_results[name][level] = None
                checkpoint_save(all_results, CHECKPOINT)
                continue

            elapsed = time.time() - start
            print(f"  -> Completed in {elapsed:.1f}s; storing results")
            all_results[name][level] = res
            checkpoint_save(all_results, CHECKPOINT)

    produce_summary_csv_class(all_results, SUMMARY)
    joblib.dump(all_results, CHECKPOINT, compress=("gzip", 3))
    print(f"Classification results checkpoint saved to {CHECKPOINT}")


def run_property(
    Benchmark_fn,
    pairs: List[Tuple[str, str]],
    out_dir: Path,
    dry_run: bool,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    n_jobs: int,
):
    OUT_DIR = out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT = checkpoint_path(OUT_DIR, "property")
    SUMMARY = summary_csv_path(OUT_DIR, "property")

    all_results: Dict[Tuple[str, str], Any] = {}
    if CHECKPOINT.exists():
        try:
            all_results = joblib.load(CHECKPOINT)
            print("Loaded property checkpoint", CHECKPOINT)
        except Exception as e:
            print("Failed to load checkpoint, starting fresh:", e)
            all_results = {}

    for name, target in pairs:
        key = (name, target)
        if key in all_results and all_results[key] is not None:
            print(f"  - {name}:{target} already done, skipping")
            continue

        print(f"  -> Running property Benchmark(name={name}, target={target})")
        if dry_run:
            all_results[key] = None
            checkpoint_save(all_results, CHECKPOINT)
            continue

        start = time.time()
        try:
            res = Benchmark_fn(
                name=name,
                target_col=target,
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        except Exception as e:
            print(f"  !!! Error running property Benchmark for {name}:{target} -> {e}")
            traceback.print_exc()
            all_results[key] = None
            checkpoint_save(all_results, CHECKPOINT)
            continue

        elapsed = time.time() - start
        print(f"  -> Completed in {elapsed:.1f}s; storing results")
        all_results[key] = res
        checkpoint_save(all_results, CHECKPOINT)

    produce_summary_csv_property(all_results, SUMMARY)
    joblib.dump(all_results, CHECKPOINT, compress=("gzip", 3))
    print(f"Property results checkpoint saved to {CHECKPOINT}")


def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Run classification or property benchmarks")
    p.add_argument(
        "--task",
        choices=["classification", "property"],
        required=True,
        help="Which task to run",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Do not execute Benchmarks, just iterate"
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--out-dir", type=str, default="results_benchmark")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "classification":
        if classification_Benchmark is None:
            print(
                "Error: classification Benchmark not importable. Ensure synrxn.baseline.classification is available."
            )
            sys.exit(1)
        run_classification(
            classification_Benchmark,
            CLASS_NAMES,
            out_dir,
            args.dry_run,
            args.n_splits,
            args.n_repeats,
            args.random_state,
            args.n_jobs,
        )
    else:
        if property_Benchmark is None:
            print(
                "Error: property Benchmark not importable. Ensure property.py (or proper package) is available."
            )
            sys.exit(1)
        run_property(
            property_Benchmark,
            PROPERTY_PAIRS,
            out_dir,
            args.dry_run,
            args.n_splits,
            args.n_repeats,
            args.random_state,
            args.n_jobs,
        )


if __name__ == "__main__":
    main()

# python baseline.py --task classification --n-splits 5 --n-repeats 5 --n-jobs 4
# python baseline.py --task property --n-splits 5 --n-repeats 5 --n-jobs 4
