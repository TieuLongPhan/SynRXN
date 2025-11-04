from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import sys
from pathlib import Path

parents = Path(__file__).resolve().parents
sys.path.append(str(parents[0]))

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
    # ("rgd1", "ea"),
    ("sn2", "ea"),
    ("snar", "ea"),
    ("suzuki_miyaura", "yield"),
    ("uspto_yield", "yield"),
]


# ------------------------------
# Logging setup
# ------------------------------
def setup_logging(log_dir: Path, task: str, verbose: bool = True) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task}.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if verbose:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    logging.info("==== Logger initialized ====")
    logging.info("Log file: %s", log_path)


# ------------------------------
# Utilities: levels, per-run files, master, summaries
# ------------------------------
def levels_for(name: str) -> List[int]:
    return CLASS_SPECIAL_LEVELS.get(name, CLASS_BASE_LEVELS)


def master_rows_path(out_dir: Path, task: str) -> Path:
    return out_dir / (f"{task}_rows.csv")


def runs_dir(out_dir: Path, task: str) -> Path:
    d = out_dir / "runs" / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def summary_csv_path(out_dir: Path, task: str) -> Path:
    return out_dir / (f"summary_metrics_{task}.csv")


def sanitize_for_fname(s: str) -> str:
    # keep alphanumerics, dash and underscore
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in str(s))


def write_run_file(run_path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write a per-run CSV (overwrite if exists)."""
    rows = list(rows)
    if not rows:
        return
    df = pd.DataFrame(rows)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_path, index=False)
    logging.info("Wrote run file: %s (rows=%d)", run_path, len(df))


def write_run_jsonl(run_jsonl: Path, rows: Iterable[Dict[str, Any]]) -> None:
    run_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with run_jsonl.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    logging.info("Wrote run jsonl: %s", run_jsonl)


def find_existing_run_file(
    run_dir: Path,
    task: str,
    name: str,
    level: int | None = None,
    target: str | None = None,
) -> Path | None:
    """Search run_dir for an existing run file for (task,name,level/target)."""
    name_s = sanitize_for_fname(name)
    if level is not None:
        pattern = f"{task}__{name_s}__level{level}__*.csv"
    else:
        target_s = sanitize_for_fname(target if target is not None else "")
        pattern = f"{task}__{name_s}__target_{target_s}__*.csv"
    for p in run_dir.glob(pattern):
        return p
    return None


def flatten_cv_to_rows_class(
    name: str, level: int, cvdict: Dict[str, Any]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for feat_key, metrics in cvdict.items():
        for metric_name, arr in metrics.items():
            arr = np.asarray(arr)
            for fold_idx, v in enumerate(arr):
                rows.append(
                    {
                        "task": "classification",
                        "name": name,
                        "level": int(level),
                        "feature_mode": str(feat_key),
                        "metric": str(metric_name),
                        "fold": int(fold_idx),
                        "value": float(v),
                    }
                )
    return rows


def flatten_cv_to_rows_property(
    name: str, target: str, cvdict: Dict[str, Any]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for feat_key, metrics in cvdict.items():
        for metric_name, arr in metrics.items():
            arr = np.asarray(arr)
            for fold_idx, v in enumerate(arr):
                rows.append(
                    {
                        "task": "property",
                        "name": name,
                        "target": target,
                        "feature_mode": str(feat_key),
                        "metric": str(metric_name),
                        "fold": int(fold_idx),
                        "value": float(v),
                    }
                )
    return rows


def append_runs_to_master(run_files: List[Path], master_csv: Path) -> None:
    """Concatenate run_files and append to master_csv (create if missing)."""
    if not run_files:
        logging.warning("No run files to append to master: %s", master_csv)
        return

    # Read and concat in chunks/one by one to avoid huge memory spikes
    # first = True
    for run_file in run_files:
        try:
            df = pd.read_csv(run_file)
        except Exception as e:
            logging.warning("Skipping unreadable run file %s: %s", run_file, e)
            continue
        if df.empty:
            continue
        # append to master
        if not master_csv.exists():
            df.to_csv(master_csv, index=False)
            logging.info(
                "Created master CSV %s with %d rows (from %s)",
                master_csv,
                len(df),
                run_file,
            )
        else:
            # ensure consistent columns: reorder df columns to master if possible
            try:
                master_cols = pd.read_csv(master_csv, nrows=0).columns.tolist()
                df = df.reindex(columns=master_cols)
            except Exception:
                pass
            df.to_csv(master_csv, mode="a", header=False, index=False)
            logging.info(
                "Appended %d rows from %s to master %s", len(df), run_file, master_csv
            )


def produce_summary_csv_from_master(task: str, master_csv: Path, outpath: Path) -> None:
    """Aggregate the master CSV into a summary CSV."""
    if not master_csv.exists():
        logging.warning("Master CSV not found at %s; summary not created.", master_csv)
        return

    df = pd.read_csv(master_csv)
    if df.empty:
        logging.warning("Master CSV %s is empty; summary not created.", master_csv)
        return

    if task == "classification":
        group_cols = ["name", "level", "feature_mode", "metric"]
    else:
        group_cols = ["name", "target", "feature_mode", "metric"]

    agg = (
        df.groupby(group_cols)["value"]
        .agg(mean="mean", std=lambda x: float(np.std(x, ddof=0)), n_splits="count")
        .reset_index()
    )
    agg = agg.sort_values(group_cols).reset_index(drop=True)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(outpath, index=False)
    logging.info("Saved summary CSV to: %s", outpath)


# ------------------------------
# Main runner: write per-run files immediately; append to master only at end
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
    write_jsonl: bool,
):
    task = "classification"
    run_folder = runs_dir(out_dir, task)
    master_csv = master_rows_path(out_dir, task)
    summary_csv = summary_csv_path(out_dir, task)

    run_files: List[Path] = []

    for name in names:
        levs = levels_for(name)
        logging.info("Dataset: %s -> levels: %s", name, levs)
        for level in levs:
            # duplicate protection based on per-run files
            existing = find_existing_run_file(run_folder, task, name, level=level)
            if existing is not None:
                logging.info(
                    "  - %s level %d: run file exists (%s), skipping",
                    name,
                    level,
                    existing.name,
                )
                run_files.append(existing)
                continue

            if dry_run:
                logging.info(
                    "  -> [dry-run] would run Benchmark(name=%s, level=%d)", name, level
                )
                # write a stub run file
                ts = int(time.time())
                run_path = (
                    run_folder
                    / f"{task}__{sanitize_for_fname(name)}__level{level}__{ts}.csv"
                )
                stub = [
                    {
                        "task": task,
                        "name": name,
                        "level": level,
                        "feature_mode": "NA",
                        "metric": "NA",
                        "fold": -1,
                        "value": float("nan"),
                    }
                ]
                write_run_file(run_path, stub)
                if write_jsonl:
                    write_run_jsonl(run_path.with_suffix(".jsonl"), stub)
                run_files.append(run_path)
                continue

            logging.info("  -> Running Benchmark(name=%s, level=%d)", name, level)
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
                elapsed = time.time() - start
                logging.info("  -> Completed in %.1fs", elapsed)
            except Exception as e:
                logging.error(
                    "  !!! Error running Benchmark for %s level %d: %s", name, level, e
                )
                logging.debug("Traceback:\n%s", traceback.format_exc())
                ts = int(time.time())
                run_path = (
                    run_folder
                    / f"{task}__{sanitize_for_fname(name)}__level{level}__{ts}.csv"
                )
                stub = [
                    {
                        "task": task,
                        "name": name,
                        "level": level,
                        "feature_mode": "ERROR",
                        "metric": "exception",
                        "fold": -1,
                        "value": float("nan"),
                    }
                ]
                write_run_file(run_path, stub)
                if write_jsonl:
                    write_run_jsonl(run_path.with_suffix(".jsonl"), stub)
                run_files.append(run_path)
                continue

            # write per-run CSV immediately
            rows = flatten_cv_to_rows_class(name, level, res)
            ts = int(time.time())
            run_path = (
                run_folder
                / f"{task}__{sanitize_for_fname(name)}__level{level}__{ts}.csv"
            )
            write_run_file(run_path, rows)
            if write_jsonl:
                write_run_jsonl(run_path.with_suffix(".jsonl"), rows)
            run_files.append(run_path)

    # AFTER ALL runs: append per-run CSVs to single master CSV
    append_runs_to_master(run_files, master_csv)

    # produce summary from master
    produce_summary_csv_from_master(task, master_csv, summary_csv)
    logging.info("Classification run completed.")


def run_property(
    Benchmark_fn,
    pairs: List[Tuple[str, str]],
    out_dir: Path,
    dry_run: bool,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    n_jobs: int,
    write_jsonl: bool,
):
    task = "property"
    run_folder = runs_dir(out_dir, task)
    master_csv = master_rows_path(out_dir, task)
    summary_csv = summary_csv_path(out_dir, task)

    run_files: List[Path] = []

    for name, target in pairs:
        existing = find_existing_run_file(
            run_folder, task, name, level=None, target=target
        )
        if existing is not None:
            logging.info(
                "  - %s:%s run file exists (%s), skipping", name, target, existing.name
            )
            run_files.append(existing)
            continue

        if dry_run:
            logging.info(
                "  -> [dry-run] would run Benchmark(name=%s, target=%s)", name, target
            )
            ts = int(time.time())
            run_path = (
                run_folder
                / f"{task}__{sanitize_for_fname(name)}__target_{sanitize_for_fname(target)}__{ts}.csv"
            )
            stub = [
                {
                    "task": task,
                    "name": name,
                    "target": target,
                    "feature_mode": "NA",
                    "metric": "NA",
                    "fold": -1,
                    "value": float("nan"),
                }
            ]
            write_run_file(run_path, stub)
            if write_jsonl:
                write_run_jsonl(run_path.with_suffix(".jsonl"), stub)
            run_files.append(run_path)
            continue

        logging.info("  -> Running Benchmark(name=%s, target=%s)", name, target)
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
            elapsed = time.time() - start
            logging.info("  -> Completed in %.1fs", elapsed)
        except Exception as e:
            logging.error(
                "  !!! Error running property Benchmark for %s:%s -> %s",
                name,
                target,
                e,
            )
            logging.debug("Traceback:\n%s", traceback.format_exc())
            ts = int(time.time())
            run_path = (
                run_folder
                / f"{task}__{sanitize_for_fname(name)}__target_{sanitize_for_fname(target)}__{ts}.csv"
            )
            stub = [
                {
                    "task": task,
                    "name": name,
                    "target": target,
                    "feature_mode": "ERROR",
                    "metric": "exception",
                    "fold": -1,
                    "value": float("nan"),
                }
            ]
            write_run_file(run_path, stub)
            if write_jsonl:
                write_run_jsonl(run_path.with_suffix(".jsonl"), stub)
            run_files.append(run_path)
            continue

        rows = flatten_cv_to_rows_property(name, target, res)
        ts = int(time.time())
        run_path = (
            run_folder
            / f"{task}__{sanitize_for_fname(name)}__target_{sanitize_for_fname(target)}__{ts}.csv"
        )
        write_run_file(run_path, rows)
        if write_jsonl:
            write_run_jsonl(run_path.with_suffix(".jsonl"), rows)
        run_files.append(run_path)

    append_runs_to_master(run_files, master_csv)
    produce_summary_csv_from_master(task, master_csv, summary_csv)
    logging.info("Property run completed.")


# ------------------------------
# CLI
# ------------------------------
def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(
        description="Run classification or property benchmarks (per-run files + master appended at end)"
    )
    p.add_argument(
        "--task",
        choices=["classification", "property"],
        required=True,
        help="Which task to run",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not execute Benchmarks, just iterate and write stub run files",
    )
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--out-dir", type=str, default="results_benchmark")
    p.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Disable JSONL run logging (CSV per-run files are always written)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Disable console logging (file logging remains)",
    )
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, args.task, verbose=not args.quiet)

    logging.info("Args: %s", vars(args))

    if args.task == "classification":
        if classification_Benchmark is None:
            logging.error(
                "classification Benchmark not importable. Ensure synrxn.baseline.classification is available."
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
            write_jsonl=not args.no_jsonl,
        )
    else:
        if property_Benchmark is None:
            logging.error(
                "property Benchmark not importable. Ensure synrxn.baseline.property is available."
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
            write_jsonl=not args.no_jsonl,
        )

    logging.info("All done.")


if __name__ == "__main__":
    main()
