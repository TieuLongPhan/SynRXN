#!/usr/bin/env python3
"""Compare SynKit 1.5 and legacy SynRXN AAM validation row by row."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synkit.Chem.Reaction.Mapper import AAMValidator as SynKitValidator  # noqa: E402
from synrxn.aam.aam_validator import AAMValidator as SynRXNValidator  # noqa: E402

MAPPER_COLUMNS = ("rxn_mapper", "graphormer", "local_mapper", "rdt")


def compare_dataset(
    path: Path,
    methods: list[str],
    limit: int | None,
    n_jobs: int,
) -> list[dict]:
    frame = pd.read_csv(path, nrows=limit)
    mapped_columns = [column for column in MAPPER_COLUMNS if column in frame.columns]
    if not mapped_columns:
        return []
    legacy = SynRXNValidator()
    current = SynKitValidator(strip_unbalanced_maps=True)
    comparisons = []
    for method in methods:
        legacy_results = legacy.validate_smiles(
            frame,
            ground_truth_col="ground_truth",
            mapped_cols=mapped_columns,
            check_method=method,
            ignore_tautomers=True,
            n_jobs=n_jobs,
        )[0]
        synkit_results = current.validate_smiles(
            frame,
            ground_truth_col="ground_truth",
            mapped_cols=mapped_columns,
            check_method=method,
            ignore_tautomers=True,
            n_jobs=n_jobs,
        )
        for old, new in zip(legacy_results, synkit_results):
            mismatches = [
                str(frame.iloc[index].get("r_id", index))
                for index, (old_value, new_value) in enumerate(
                    zip(old["results"], new["results"])
                )
                if bool(old_value) != bool(new_value)
            ]
            comparisons.append(
                {
                    "dataset": path.stem.removesuffix(".csv"),
                    "method": method,
                    "mapper": old["mapper"],
                    "rows": len(frame),
                    "synrxn_accuracy": old["accuracy"],
                    "synkit_accuracy": new["accuracy"],
                    "mismatch_count": len(mismatches),
                    "mismatch_ids": mismatches[:50],
                }
            )
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Data/aam"))
    parser.add_argument("--methods", nargs="+", choices=["RC", "ITS"], default=["RC"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # RDKit 2026 emits a valence deprecation warning for nearly every graph
    # conversion; keep the comparison report machine-readable by suppressing it.
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.warning")

    comparisons = []
    for path in sorted(args.data_dir.glob("*.csv.gz")):
        comparisons.extend(compare_dataset(path, args.methods, args.limit, args.n_jobs))
    report = {
        "synkit_version": version("synkit"),
        "settings": {
            "methods": args.methods,
            "limit": args.limit,
            "n_jobs": args.n_jobs,
            "ignore_tautomers": True,
            "strip_unbalanced_maps": True,
        },
        "comparisons": comparisons,
        "summary": {
            "datasets": len({item["dataset"] for item in comparisons}),
            "result_vectors": len(comparisons),
            "rows_checked": sum(
                item["rows"]
                for item in comparisons
                if item["mapper"] == MAPPER_COLUMNS[0]
            ),
            "mismatches": sum(item["mismatch_count"] for item in comparisons),
        },
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf8")
    print(rendered, end="")
    return 1 if report["summary"]["mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
