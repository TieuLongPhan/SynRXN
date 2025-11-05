#!/usr/bin/env python3
"""
report_rbl_datasets.py

Read the RBL raw data (local path or URL), produce the four datasets (mos, mbs, mnc, complex)
in memory (do NOT save when --dry-run) and print a summary table and small samples.

Usage (example):
  python report_rbl_datasets.py --input-url /path/to/USPTO_50K.csv --dry-run
  python report_rbl_datasets.py --input-url "https://raw.githubusercontent.com/..." --repo-root /home/me/synrxn --dry-run

Behavior:
 - If repo helpers (clean_synrbl, curate_records) are available under `--repo-root` it will prefer them.
 - Otherwise it uses a fallback normalizer + heuristic splitting.
 - Default: --dry-run (no files written).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import importlib
import pandas as pd
import re
import json
import os
from typing import Optional, Any, Dict


def add_repo_root(repo_root: Optional[str]):
    if not repo_root:
        return
    root = Path(repo_root).resolve()
    sys.path.insert(0, str(root))


def try_import_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def normalize_rebalancer_output(result: Any) -> pd.DataFrame:
    """Fallback normalizer to convert various result shapes to DataFrame with R-id, rxn, ground_truth."""
    if isinstance(result, pd.DataFrame):
        df = result.copy()
    elif isinstance(result, list):
        df = pd.DataFrame(result)
    elif isinstance(result, dict):
        recs = result.get("records") or result.get("data") or result.get("results")
        if recs:
            df = pd.DataFrame(recs)
        else:
            df = pd.DataFrame([result])
    else:
        df = pd.DataFrame([result])

    # Ensure R-id
    if "R-id" not in df.columns:
        for c in ("id", "rid", "R_id", "r_id"):
            if c in df.columns:
                df["R-id"] = df[c].astype(str)
                break
        else:
            df = df.reset_index(drop=True)
            df["R-id"] = df.index.map(lambda i: f"R_{i}")

    # Drop rows with truthy 'error' if present
    if "error" in df.columns:
        try:
            df = df[~df["error"].astype(bool)]
        except Exception:
            df = df[df["error"].isna()]

    # pick rxn/ground truth columns if present
    def _pick(row, keys):
        for k in keys:
            if k in row and row[k] is not None:
                return row[k]
        return None

    df["rxn"] = df.apply(
        lambda r: _pick(
            r,
            ["input_reaction", "reactions", "rxn", "reaction", "standardized_reaction"],
        ),
        axis=1,
    )
    df["ground_truth"] = df.apply(
        lambda r: _pick(r, ["ground_truth", "groundtruth", "gt", "new_products"]),
        axis=1,
    )

    # If rxn all empty, try alt columns
    if df["rxn"].isnull().all():
        for alt in ["reactions", "reaction", "input_reaction", "rxn"]:
            if alt in df.columns:
                df["rxn"] = df[alt]
                break

    # reorder columns
    cols = list(df.columns)
    for c in reversed(["R-id", "rxn", "ground_truth"]):
        if c in cols:
            cols.insert(0, cols.pop(cols.index(c)))
    df = df[cols]
    return df.reset_index(drop=True)


def heuristic_split_by_dataset_column(df: pd.DataFrame):
    """Try to produce mos, mbs, mnc, complex by guessing a dataset column."""
    candidates = [
        c
        for c in df.columns
        if re.search(r"dataset|set|split|type|source|label", c, flags=re.I)
    ]
    mos = pd.DataFrame(columns=df.columns)
    mbs = pd.DataFrame(columns=df.columns)
    mnc = pd.DataFrame(columns=df.columns)
    complex_df = pd.DataFrame(columns=df.columns)

    if "dataset" in df.columns:
        for name, group in df.groupby("dataset"):
            key = str(name).lower()
            if "mos" in key or "major" in key or "golden" in key:
                mos = pd.concat([mos, group], ignore_index=True)
            elif "mbs" in key or "minor" in key or "jaworski" in key:
                mbs = pd.concat([mbs, group], ignore_index=True)
            elif "mnc" in key or "none" in key:
                mnc = pd.concat([mnc, group], ignore_index=True)
            else:
                complex_df = pd.concat([complex_df, group], ignore_index=True)
    else:
        # No dataset column: use heuristics (e.g., look for label columns or split by index ranges)
        complex_df = df.copy()

    return mos, mbs, mnc, complex_df


def prepare_report(dsets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for k in ("mos", "mbs", "mnc", "complex"):
        dfk = dsets.get(k, pd.DataFrame())
        cnt = len(dfk)
        sample_rxn = ""
        sample_id = ""
        if cnt > 0:
            for col in ("reactions", "reaction", "rxn", "input_reaction"):
                if col in dfk.columns:
                    sample_rxn = str(dfk.iloc[0][col])
                    break
            for idc in ("R-id", "id", "rid", "record_id"):
                if idc in dfk.columns:
                    sample_id = str(dfk.iloc[0][idc])
                    break
        rows.append(
            {
                "dataset": k,
                "rows": cnt,
                "sample_id": sample_id,
                "sample_rxn": sample_rxn,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-url", required=True, help="Local path or URL to the raw CSV input."
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional repo root to import project helpers (clean_synrbl/curate_records).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="If set, do not write files (default True).",
    )
    parser.add_argument(
        "--print-samples",
        action="store_true",
        default=True,
        help="Print first 3 rows of each dataset to stdout.",
    )
    parser.add_argument(
        "--report-format",
        choices=("text", "markdown", "json"),
        default="markdown",
        help="Report output format.",
    )
    args = parser.parse_args()

    # add repo root if provided
    add_repo_root(args.repo_root)

    # Try to import repo helpers (best-effort)
    clean_synrbl = None
    curate_records = None
    try:
        mod_names = ["synrbl", "synrxn", "synkit"]
        for mn in mod_names:
            m = try_import_module(mn)
            if m:
                if hasattr(m, "clean_synrbl"):
                    clean_synrbl = getattr(m, "clean_synrbl")
                if hasattr(m, "curate_records"):
                    curate_records = getattr(m, "curate_records")
    except Exception:
        pass

    # Load input (pandas supports local path or http(s) URL)
    try:
        df = pd.read_csv(args.input_url)
    except Exception as e:
        print("Failed to read input file/URL:", e, file=sys.stderr)
        sys.exit(2)

    # Try curated pipeline
    mos = mbs = mnc = complex_df = None

    if curate_records:
        try:
            curated = curate_records(df.to_dict("records"))
            # try to unpack
            if isinstance(curated, dict):
                mos = pd.DataFrame(curated.get("mos") or [])
                mbs = pd.DataFrame(curated.get("mbs") or [])
                mnc = pd.DataFrame(curated.get("mnc") or [])
                complex_df = pd.DataFrame(curated.get("complex") or [])
            elif isinstance(curated, (list, tuple)):
                # map by position (best-effort)
                items = list(curated)
                mos = pd.DataFrame(items[0]) if len(items) > 0 else pd.DataFrame()
                mbs = pd.DataFrame(items[1]) if len(items) > 1 else pd.DataFrame()
                mnc = pd.DataFrame(items[2]) if len(items) > 2 else pd.DataFrame()
                complex_df = (
                    pd.DataFrame(items[3]) if len(items) > 3 else pd.DataFrame()
                )
        except Exception as e:
            print("curate_records failed (falling back):", e)

    if (
        mos is None or mbs is None or mnc is None or complex_df is None
    ) and clean_synrbl:
        try:
            cleaned = clean_synrbl(df)
            complex_df = pd.DataFrame(cleaned)
        except Exception as e:
            print("clean_synrbl failed (falling back):", e)

    # If still not produced, heuristic split
    if complex_df is None:
        mos, mbs, mnc, complex_df = heuristic_split_by_dataset_column(df)

    # Ensure DataFrames
    mos = pd.DataFrame(mos) if mos is not None else pd.DataFrame(columns=df.columns)
    mbs = pd.DataFrame(mbs) if mbs is not None else pd.DataFrame(columns=df.columns)
    mnc = pd.DataFrame(mnc) if mnc is not None else pd.DataFrame(columns=df.columns)
    complex_df = (
        pd.DataFrame(complex_df)
        if complex_df is not None
        else pd.DataFrame(columns=df.columns)
    )

    dsets = {"mos": mos, "mbs": mbs, "mnc": mnc, "complex": complex_df}
    report_df = prepare_report(dsets)

    # Print report in requested format
    if args.report_format == "markdown":
        print(report_df.to_markdown(index=False))
    elif args.report_format == "json":
        print(report_df.to_json(orient="records", indent=2))
    else:
        print(report_df.to_string(index=False))

    # Print samples if requested
    if args.print_samples:
        for k, dfk in dsets.items():
            print(f"\n--- {k} sample (rows={len(dfk)}) ---")
            if len(dfk) > 0:
                print(dfk.head(3).to_string(index=False))
            else:
                print("(empty)")


if __name__ == "__main__":
    main()
