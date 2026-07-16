#!/usr/bin/env python3
"""Reproducible isolated-process benchmarks for CSV and Parquet access paths."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _run_case(queue, case, task, name, data_dir, parquet_dir):
    import pandas as pd

    from synrxn.data import DatasetCatalog
    from synrxn.query import QueryEngine

    dataset = DatasetCatalog().get(task, name)
    csv_path = Path(data_dir) / task / f"{name}.csv.gz"
    started = time.perf_counter()
    if case == "csv-full":
        result = pd.read_csv(csv_path)
        rows = len(result)
    elif case == "csv-projected":
        columns = [dataset.row_identifier, *dataset.targets][:2]
        result = pd.read_csv(csv_path, usecols=columns)
        rows = len(result)
    else:
        engine = QueryEngine(Path(parquet_dir))
        columns = [dataset.row_identifier, *dataset.targets][:2]
        filters = {"split": dataset.split_values[0]} if dataset.has_split else None
        result = engine.query(
            task,
            name,
            columns=columns,
            filters=filters if case == "parquet-filtered" else None,
            source_order=True,
            limit=100 if case == "parquet-first-page" else 10_000,
        )
        rows = len(result)
        engine.close()
    elapsed = time.perf_counter() - started
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    queue.put({"case": case, "seconds": elapsed, "peak_kib": peak_kib, "rows": rows})


def benchmark_case(case, task, name, data_dir, parquet_dir):
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_run_case,
        args=(queue, case, task, name, str(data_dir), str(parquet_dir)),
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"benchmark child failed with exit code {process.exitcode}")
    return queue.get()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["aam/ecoli", "classification/schneider_b", "synthesis/uspto_mit"],
    )
    parser.add_argument("--output", type=Path, default=Path("query-benchmark.json"))
    args = parser.parse_args()
    cases = ("csv-full", "csv-projected", "parquet-filtered", "parquet-first-page")
    results = []
    for dataset_id in args.datasets:
        task, name = dataset_id.split("/", 1)
        for case in cases:
            result = benchmark_case(
                case, task, name, args.data_dir.resolve(), args.parquet_dir.resolve()
            )
            results.append({"dataset": dataset_id, **result})
    report = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": version("pandas"),
            "pyarrow": version("pyarrow"),
            "duckdb": version("duckdb"),
        },
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
