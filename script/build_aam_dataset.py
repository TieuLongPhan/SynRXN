#!/usr/bin/env python3
"""
build_aam_dataset.py

Automate AAM data construction:
 - load JSON (local path or raw GitHub .json.gz URL)
 - run process_aam(...) from your project
 - convert to pandas.DataFrame
 - save as compressed CSV (.csv.gz)

Now also collects a per-dataset result table and writes reports/aam_summary.csv.gz.

Usage:
  PYTHONPATH=. python script/build_aam_dataset.py --dry-run
  PYTHONPATH=. python script/build_aam_dataset.py --entries ecoli,golden
  python script/build_aam_dataset.py --write-default aam_default.json
"""
from __future__ import annotations
import argparse
import gzip
import json
import logging
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ---------------------------
# DEFAULT CONFIG (built-in)
# ---------------------------
DEFAULT_CONFIG: Dict[str, Dict[str, str]] = {
    "ecoli": {
        "src": "https://raw.githubusercontent.com/TieuLongPhan/SynTemp/main/Data/AAM/results_benchmark/ecoli/ecoli_aam_reactions.json.gz",
        "out": "Data/aam/ecoli.csv.gz",
    },
    "recon3d": {
        "src": "https://raw.githubusercontent.com/TieuLongPhan/SynTemp/main/Data/AAM/results_benchmark/recon3d/recon3d_aam_reactions.json.gz",
        "out": "Data/aam/recon3d.csv.gz",
    },
    "golden": {
        "src": "https://raw.githubusercontent.com/TieuLongPhan/SynTemp/main/Data/AAM/results_benchmark/golden/golden_aam_reactions.json.gz",
        "out": "Data/aam/golden.csv.gz",
    },
    "natcomm": {
        "src": "https://raw.githubusercontent.com/TieuLongPhan/SynTemp/main/Data/AAM/results_benchmark/natcomm/natcomm_aam_reactions.json.gz",
        "out": "Data/aam/natcomm.csv.gz",
    },
    "uspto3k": {
        "src": "https://raw.githubusercontent.com/TieuLongPhan/SynTemp/main/Data/AAM/results_benchmark/uspto_3k/uspto_3k_aam_reactions.json.gz",
        "out": "Data/aam/uspto_3k.csv.gz",
    },
}

# ---------------------------
# Try imports (project utilities, optional fast savers/loaders)
# ---------------------------
try:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from process.aam import process_aam  # type: ignore
except Exception as e:  # pragma: no cover - import guard
    process_aam = None
    logger.debug("process.aam.process_aam not importable: %s", e)

try:
    from synrxn.io.io import load_json_from_raw_github, save_df_gz  # type: ignore
except Exception:  # pragma: no cover - import guard
    load_json_from_raw_github = None
    save_df_gz = None


# ---------------------------
# Fallback helpers
# ---------------------------
def fallback_load_json_from_raw_github(url_or_path: str, timeout: int = 60) -> Any:
    import requests

    p = Path(url_or_path)
    if p.exists():
        if str(p).endswith(".gz"):
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        else:
            with open(p, "rt", encoding="utf-8") as fh:
                return json.load(fh)

    resp = requests.get(url_or_path, timeout=timeout)
    resp.raise_for_status()
    content = resp.content
    try:
        buf = gzip.decompress(content)
        return json.loads(buf.decode("utf-8"))
    except (OSError, gzip.BadGzipFile):
        try:
            return resp.json()
        except Exception:
            return json.loads(resp.text)


def fallback_save_df_gz(df, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # pandas supports compression="gzip"
    df.to_csv(out, index=False, compression="gzip")


def ensure_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("pandas is required but not importable") from e
    return pd


# ---------------------------
# Normalization + processing (original logic refactored into entry processor)
# ---------------------------
def _normalize_loaded_to_entries(loaded: Any) -> list:
    """
    Convert loader output to list-of-entries (list[dict]) that process_aam expects.
    Accepts DataFrame, dict-with-records, objects with to_dict, iterables, etc.
    """
    # defensive normalization, similar to previous code
    pd = ensure_pandas()
    data = loaded

    if isinstance(data, pd.DataFrame):
        logger.info(
            "Converting pandas.DataFrame -> list[dict] via to_dict(orient='records')"
        )
        return data.to_dict(orient="records")

    if isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            logger.info("Using top-level 'records' list")
            return data["records"]
        if "data" in data and isinstance(data["data"], list):
            logger.info("Using top-level 'data' list")
            return data["data"]
        # if dict-of-lists -> convert to records
        if all(isinstance(v, (list, tuple)) for v in data.values()):
            lengths = {len(v) for v in data.values()}
            if len(lengths) == 1:
                n = next(iter(lengths))
                records = [{k: data[k][i] for k in data.keys()} for i in range(n)]
                logger.info("Converted dict-of-lists -> %d records", n)
                return records
    # objects exposing to_dict(orient='records')
    if hasattr(data, "to_dict") and callable(getattr(data, "to_dict")):
        try:
            cand = data.to_dict(orient="records")
            if isinstance(cand, list):
                logger.info("Converted object via to_dict(orient='records')")
                return cand
        except Exception:
            try:
                cand = data.to_dict()
                if isinstance(cand, dict) and all(
                    isinstance(v, (list, tuple)) for v in cand.values()
                ):
                    n = len(next(iter(cand.values())))
                    records = [{k: cand[k][i] for k in cand.keys()} for i in range(n)]
                    logger.info("Converted to_dict() dict-of-lists -> %d records", n)
                    return records
            except Exception:
                pass

    # final fallback: try to iterate
    try:
        return list(data)
    except Exception:
        raise RuntimeError(
            f"Unable to coerce loaded data into list of entries; got {type(data).__name__}"
        )


def process_single_entry(
    name: str, src: str, out: str, *, dry_run: bool = False, retries: int = 2
) -> Dict[str, Any]:
    """
    Load -> normalize -> process_aam -> save and return a result dict:
      { name, src, out, status, message, input_size, processed_items, saved, time_s }
    """
    pd = ensure_pandas()
    loader = load_json_from_raw_github or fallback_load_json_from_raw_github
    saver = save_df_gz or fallback_save_df_gz

    start = time.time()
    input_size = None
    processed_count = None
    saved = False
    status = "failed"
    message = ""
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            logger.info(
                "[%s] Loading source %s (attempt %d/%d)...", name, src, attempt, retries
            )
            loaded = loader(src)
            # record input size: if DataFrame, rows; if list/dict -> len
            try:
                if hasattr(loaded, "shape"):
                    input_size = int(getattr(loaded, "shape")[0])
                elif isinstance(loaded, (list, tuple)):
                    input_size = len(loaded)
                elif isinstance(loaded, dict):
                    input_size = len(loaded)
                else:
                    # unknown; attempt len()
                    input_size = len(loaded) if hasattr(loaded, "__len__") else None
            except Exception:
                input_size = None

            entries = _normalize_loaded_to_entries(loaded)
            logger.info(
                "[%s] Prepared entries count: %d",
                name,
                len(entries) if hasattr(entries, "__len__") else -1,
            )

            if process_aam is None:
                raise RuntimeError(
                    "process_aam not importable. Run script from repository root or ensure package is on PYTHONPATH."
                )
            logger.info("[%s] Running process_aam(...)", name)
            processed = process_aam(entries)

            # coerce processed to list
            try:
                processed_list = list(processed)
            except TypeError:
                processed_list = [processed]

            processed_count = len(processed_list)
            logger.info("[%s] process_aam produced %d items", name, processed_count)

            for key, value in enumerate(processed_list):
                value["original_id"] = value["R-id"]
                value["r_id"] = f"{name}_{key+1}"

            # make DataFrame
            df = pd.DataFrame(processed_list)
            df.drop(columns=["R-id"], inplace=True)
            # df.drop(['R-id'], inplace=True)
            df = df[["r_id"] + [c for c in df.columns if c != "r_id"]]

            logger.info(
                "[%s] DataFrame created with shape %s", name, getattr(df, "shape", None)
            )

            if dry_run:
                logger.info("[%s] dry-run: skipping save to %s", name, out)
                saved = False
            else:
                # df.rename(columns={"ground_truth": "aam"}, inplace=True)

                saver(df, out)
                saved = True
                logger.info("[%s] Saved DataFrame to %s", name, out)

            status = "success"
            message = "Processed OK"
            break
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception("[%s] attempt %d failed: %s", name, attempt, exc)
            last_exc = exc
            message = str(exc)
            # continue retry loop

    end = time.time()
    time_s = round(end - start, 3)

    # truncate message
    if message is None:
        message = ""
    if len(message) > 400:
        message = message[:400] + "...(truncated)"

    result = {
        "name": name,
        "src": src,
        "out": out,
        "status": status,
        "message": message,
        "input_size": (
            int(input_size)
            if input_size is not None
            and not (isinstance(input_size, float) and math.isnan(input_size))
            else None
        ),
        "processed_items": (
            int(processed_count) if processed_count is not None else None
        ),
        "saved": bool(saved),
        "time_s": time_s,
    }
    return result


# ---------------------------
# CLI + summary printing + main
# ---------------------------
def load_config_file(path: Path) -> Dict[str, Dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except Exception:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text)
        except Exception as e:
            raise RuntimeError("Config must be valid JSON or YAML") from e


def _print_summary_table(results):
    try:
        from tabulate import tabulate  # type: ignore

        cols = [
            "name",
            "status",
            "input_size",
            "processed_items",
            "saved",
            "time_s",
            "message",
        ]
        table = [[r.get(c) for c in cols] for r in results]
        print(tabulate(table, headers=cols, tablefmt="github"))
    except Exception:
        pd = ensure_pandas()
        df = pd.DataFrame(results)
        display_cols = [
            "name",
            "status",
            "input_size",
            "processed_items",
            "saved",
            "time_s",
            "message",
        ]
        print(df[display_cols].to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(
        description="Build AAM datasets (load -> process_aam -> save) with summary."
    )
    p.add_argument(
        "--config",
        help="JSON/YAML config file (optional). If omitted DEFAULT_CONFIG is used.",
    )
    p.add_argument(
        "--src", help="Process a single source URL or local path (overrides config)."
    )
    p.add_argument("--out", help="Output path for --src (required when --src used).")
    p.add_argument(
        "--dry-run", action="store_true", help="Do everything except saving files."
    )
    p.add_argument(
        "--retries", type=int, default=2, help="Retries for load/process (default: 2)."
    )
    p.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR)."
    )
    p.add_argument(
        "--write-default",
        help="Write the builtin DEFAULT_CONFIG JSON to this path and exit.",
    )
    p.add_argument(
        "--entries",
        help="Comma-separated subset of config keys to process (default: all).",
    )
    p.add_argument(
        "--summary-out",
        default="reports/aam_summary.csv.gz",
        help="Path to write summary CSV (gzipped).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s - %(message)s",
    )
    logger.info("build_aam_dataset starting")

    if args.write_default:
        outp = Path(args.write_default)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        logger.info("Wrote default config to %s", outp)
        return

    # Single-source quick path
    results = []
    if args.src:
        if not args.out:
            raise SystemExit("When --src is provided you must also provide --out")
        res = process_single_entry(
            "single", args.src, args.out, dry_run=args.dry_run, retries=args.retries
        )
        results.append(res)
    else:
        if args.config:
            cfg_path = Path(args.config)
            if not cfg_path.exists():
                raise SystemExit(f"Config file not found: {cfg_path}")
            cfg = load_config_file(cfg_path)
        else:
            cfg = DEFAULT_CONFIG
            logger.info(
                "No config provided: using DEFAULT_CONFIG with %d entries", len(cfg)
            )

        entries = None
        if args.entries:
            entries = [e.strip() for e in args.entries.split(",") if e.strip()]
            logger.info("Processing subset entries: %s", entries)

        for name, info in cfg.items():
            if entries and name not in entries:
                logger.debug("Skipping %s (not requested)", name)
                continue
            if not isinstance(info, dict) or "src" not in info:
                logger.warning(
                    "Skipping invalid config entry %s (expected dict with 'src')", name
                )
                continue
            src = info["src"]
            out = info.get("out") or f"Data/aam/{name}.csv.gz"
            logger.info("Entry '%s': %s -> %s", name, src, out)
            try:
                res = process_single_entry(
                    name, src, out, dry_run=args.dry_run, retries=args.retries
                )
            except Exception as exc:
                logger.exception("Failed to process entry %s: %s", name, exc)
                res = {
                    "name": name,
                    "src": src,
                    "out": out,
                    "status": "failed",
                    "message": str(exc)[:400]
                    + ("...(truncated)" if len(str(exc)) > 400 else ""),
                    "input_size": None,
                    "processed_items": None,
                    "saved": False,
                    "time_s": None,
                }
            results.append(res)

    # Print and save summary
    _print_summary_table(results)

    # pd = ensure_pandas()
    # summary_df = pd.DataFrame(results)
    # summary_out = Path(args.summary_out)
    # summary_out.parent.mkdir(parents=True, exist_ok=True)
    # summary_df.to_csv(summary_out, index=False, compression="gzip")
    # logger.info("Wrote summary to %s", summary_out)

    logger.info("build_aam_dataset finished")


if __name__ == "__main__":
    main()