#!/usr/bin/env python3
"""
build_classification_dataset.py

Same behavior as before, but now collects a processing summary table that records
success/failure and metrics per dataset and writes a summary CSV.

Usage examples:
  PYTHONPATH=. python script/build_classification_dataset.py --dry-run
  PYTHONPATH=. python script/build_classification_dataset.py --entries syntemp,ecreact
  PYTHONPATH=. python script/build_classification_dataset.py --summary-out reports/classification_summary.csv.gz
"""
from __future__ import annotations
import argparse
import gzip
import json
import logging
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------
# DEFAULT CONFIG (built-in)
# ---------------------------
DEFAULT_CONFIG: Dict[str, Dict[str, str]] = {
    "ecreact": {
        "src": "https://raw.githubusercontent.com/phuocchung123/SynCat/main/Data/raw/claire_full.csv.gz",
        "out": "Data/classification/ecreact.csv.gz",
    },
    "uspto_50k_b": {
        "src": "https://raw.githubusercontent.com/phuocchung123/SynCat/main/Data/raw/USPTO_50k_balanced.csv.gz",
        "out": "Data/classification/uspto_50k_b.csv.gz",
    },
    "uspto_50k_u": {
        "src": "https://raw.githubusercontent.com/phuocchung123/SynCat/main/Data/raw/USPTO_50k_unbalanced.csv.gz",
        "out": "Data/classification/uspto_50k_u.csv.gz",
    },
    "tpl_b": {
        "src": "https://raw.githubusercontent.com/phuocchung123/SynCat/main/Data/raw/USPTO_TPL_balanced.csv.gz",
        "out": "Data/classification/tpl_b.csv.gz",
    },
    "tpl_u": {
        "src": "https://raw.githubusercontent.com/phuocchung123/SynCat/main/Data/raw/USPTO_TPL_unbalanced.csv.gz",
        "out": "Data/classification/tpl_u.csv.gz",
    },
    "schneider_b": {
        "src": "https://raw.githubusercontent.com/phuocchung123/SynCat/main/Data/raw/schneider50k_balanced.csv.gz",
        "out": "Data/classification/schneider_b.csv.gz",
    },
    "schneider_u": {
        "src": "https://raw.githubusercontent.com/phuocchung123/SynCat/main/Data/raw/schneider50k_unbalanced.csv.gz",
        "out": "Data/classification/schneider_u.csv.gz",
    },
    "syntemp": {
        "src": "https://raw.githubusercontent.com/phuocchung123/SynCat/main/Data/raw/Syntemp_cluster.csv.gz",
        "out": "Data/classification/syntemp.csv.gz",
    },
}


# ---------------------------
# Helpers (unchanged from previous version, with small additions)
# ---------------------------
def ensure_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("pandas is required but not importable") from e
    return pd


def download_or_open_csv(
    path_or_url: str, timeout: int = 60, **pd_read_csv_kwargs
) -> "pandas.DataFrame":
    pd = ensure_pandas()
    p = Path(path_or_url)
    if p.exists():
        logger.debug("Reading local file %s", p)
        return pd.read_csv(
            p, compression="infer", low_memory=False, **pd_read_csv_kwargs
        )

    import requests, io

    logger.debug("Fetching remote URL %s", path_or_url)
    resp = requests.get(path_or_url, timeout=timeout)
    resp.raise_for_status()
    raw = resp.content
    try:
        buf = gzip.decompress(raw)
        return pd.read_csv(
            io.BytesIO(buf), compression="infer", low_memory=False, **pd_read_csv_kwargs
        )
    except (OSError, gzip.BadGzipFile):
        return pd.read_csv(
            io.BytesIO(raw), compression="infer", low_memory=False, **pd_read_csv_kwargs
        )


def _extract_first_str(value: Any) -> Optional[str]:
    pd = ensure_pandas()
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, tuple)):
        for el in value:
            if el is None:
                continue
            s = str(el).strip()
            if s:
                return s
        return None
    if isinstance(value, str):
        s = value.strip()
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("(") and s.endswith(")")
        ):
            import ast

            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)) and parsed:
                    return _extract_first_str(parsed)
            except Exception:
                pass
        return s or None
    try:
        s = str(value).strip()
        return s or None
    except Exception:
        return None


def _extract_nth_from_listish(value: Any, n: int) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) > n:
            return _extract_first_str(value[n])
        return None
    if isinstance(value, str):
        s = value.strip()
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("(") and s.endswith(")")
        ):
            import ast

            try:
                parsed = ast.literal_eval(s)
                return _extract_nth_from_listish(parsed, n)
            except Exception:
                pass
        for sep in [",", ";", "|"]:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if len(parts) > n:
                return parts[n]
        return s if n == 0 else None
    try:
        seq = list(value)
        if len(seq) > n:
            return _extract_first_str(seq[n])
    except Exception:
        pass
    return None


def curate_to_ec123_split(
    df_in,
    *,
    rxn_col_candidates: Sequence[str] = (
        "rxn",
        "reaction",
        "reaction_smiles",
        "smiles",
        "reactants>products",
        "reaction_smiles",
    ),
    ec_cols_candidates: Sequence[str] = (
        "ec1",
        "ec2",
        "ec3",
        "ec1_encode",
        "ec2_encode",
        "ec3_encode",
        "ec",
        "ec_number",
        "ec_numbers",
        "label",
        "class",
    ),
    split_col_candidates: Sequence[str] = ("split", "set", "split_col"),
    id_col_candidates: Sequence[str] = ("id", "R-id", "R_ID", "R-id", "rid"),
    id_prefix: str = "rx",
    zero_pad: Optional[int] = None,
    default_split: str = "train",
    keep_orig_index: bool = False,
):
    """
    Convert raw DataFrame to canonical columns ["R-id", "rxn", "ec1","ec2","ec3","split"].
    Enhanced rxn detection: handles RSMI and New_R* patterns.
    """
    pd = ensure_pandas()
    df = df_in.copy()

    def _first_non_null_of_list(values):
        for v in values:
            s = _extract_first_str(v)
            if s is not None:
                return s
        return None

    # --- rxn detection ---
    rxn_series = None
    # direct candidates (include RSMI)
    for cand in (
        "RSMI",
        "RsmI",
        "rsmI",
        "RSMI".lower(),
    ) + tuple(rxn_col_candidates):
        if cand in df.columns:
            logger.info("Using rxn column candidate '%s'", cand)
            rxn_series = df[cand].apply(_extract_first_str)
            break

    # New_R* pattern
    if rxn_series is None:
        new_r_cols = [c for c in df.columns if c.lower().startswith("new_r")]
        if new_r_cols:
            try:
                new_r_cols_sorted = sorted(
                    new_r_cols,
                    key=lambda c: int("".join(ch for ch in c if ch.isdigit()) or 0),
                )
            except Exception:
                new_r_cols_sorted = new_r_cols
            logger.info("Detected New_R* columns: %s", new_r_cols_sorted)
            rxn_series = df[new_r_cols_sorted].apply(
                lambda row: _first_non_null_of_list(row.values), axis=1
            )

    # fallback keyword search
    if rxn_series is None:
        for c in df.columns:
            if any(k in c.lower() for k in ("rxn", "reaction", "smiles", "rsm")):
                logger.info("Falling back to column '%s' for rxn detection", c)
                rxn_series = df[c].apply(_extract_first_str)
                break

    if rxn_series is None:
        raise RuntimeError(
            "Could not find reaction column among candidates; available columns: {}".format(
                list(df.columns)
            )
        )

    # --- ID generation ---
    id_col = None
    for cand in id_col_candidates:
        if cand in df.columns:
            id_col = cand
            logger.info("Using ID column '%s' for R-id", cand)
            break
    if id_col:
        R_ids = df[id_col].apply(lambda x: str(x).strip() if x is not None else "")
        if zero_pad:

            def try_pad(val):
                try:
                    n = int(val)
                    return str(n).zfill(zero_pad)
                except Exception:
                    return val

            R_ids = R_ids.map(try_pad)
    else:
        if zero_pad:
            R_ids = df.index.to_series().apply(
                lambda i: f"{id_prefix}{str(int(i)).zfill(zero_pad)}"
            )
        else:
            R_ids = df.index.to_series().apply(lambda i: f"{id_prefix}{i}")

    # --- split detection ---
    split_col = None
    for cand in split_col_candidates:
        if cand in df.columns:
            split_col = cand
            break
    if split_col is None:
        for c in df.columns:
            if c.lower() in ("set", "split"):
                split_col = c
                break
    if split_col is not None:
        split_series = df[split_col].apply(_extract_first_str).fillna(default_split)
    else:
        split_series = pd.Series([default_split] * len(df), index=df.index)

    # --- ec extraction ---
    ec1_series = pd.Series([None] * len(df), index=df.index)
    ec2_series = pd.Series([None] * len(df), index=df.index)
    ec3_series = pd.Series([None] * len(df), index=df.index)

    if "ec1" in df.columns:
        ec1_series = df["ec1"].apply(_extract_first_str)
    if "ec2" in df.columns:
        ec2_series = df["ec2"].apply(_extract_first_str)
    if "ec3" in df.columns:
        ec3_series = df["ec3"].apply(_extract_first_str)

    if ec1_series.isna().all() or ec1_series.eq(None).all():
        for cand in ec_cols_candidates:
            if cand in ("ec1", "ec2", "ec3"):
                continue
            if cand in df.columns:
                ec1_series = df[cand].apply(lambda v: _extract_nth_from_listish(v, 0))
                ec2_series = df[cand].apply(lambda v: _extract_nth_from_listish(v, 1))
                ec3_series = df[cand].apply(lambda v: _extract_nth_from_listish(v, 2))
                logger.info("Extracted ECs from column '%s'", cand)
                break

    if (ec1_series.isna().all() or ec1_series.eq(None).all()) and "label" in df.columns:
        ec1_series = df["label"].apply(_extract_first_str)

    ec1_series = ec1_series.apply(
        lambda x: x if (x is None or (isinstance(x, str) and x.strip())) else None
    )
    ec2_series = ec2_series.apply(
        lambda x: x if (x is None or (isinstance(x, str) and x.strip())) else None
    )
    ec3_series = ec3_series.apply(
        lambda x: x if (x is None or (isinstance(x, str) and x.strip())) else None
    )

    out = pd.DataFrame(
        {
            "R-id": R_ids,
            "rxn": rxn_series,
            "ec1": ec1_series,
            "ec2": ec2_series,
            "ec3": ec3_series,
            "split": split_series,
        },
        index=df.index,
    )

    if keep_orig_index:
        out.insert(0, "orig_index", df.index)

    final_cols = ["R-id", "rxn", "ec1", "ec2", "ec3", "split"]
    out = out[final_cols].reset_index(drop=True)
    return out


def fallback_save_df_gz(df, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")


# ---------------------------
# Processing with summary collection
# ---------------------------
def process_single_entry(
    name: str, src: str, out: str, *, dry_run: bool = False, retries: int = 2
):
    """
    Returns a result dict with keys:
      name, src, out, status ('success'/'failed'), message, input_rows, output_rows, saved (bool), time_s
    """
    pd = ensure_pandas()
    loader = download_or_open_csv
    saver = fallback_save_df_gz

    last_exc = None
    start_time = time.time()
    input_rows = None
    output_rows = None
    saved = False
    message = ""
    status = "failed"

    for attempt in range(1, retries + 1):
        try:
            logger.info("[%s] Loading %s (attempt %d/%d)", name, src, attempt, retries)
            df = loader(src)
            input_rows = getattr(df, "shape", (None, None))[0]
            logger.info(
                "[%s] Loaded DataFrame shape=%s", name, getattr(df, "shape", None)
            )

            curated = curate_to_ec123_split(df, id_prefix=name, default_split="train")
            output_rows = getattr(curated, "shape", (None, None))[0]

            # count missing rxn rows
            missing_rxn = (
                curated["rxn"].isna().sum()
                if "rxn" in curated.columns
                else (output_rows if output_rows is not None else None)
            )
            if missing_rxn and missing_rxn > 0:
                logger.warning(
                    "[%s] %d rows have missing 'rxn' after curation",
                    name,
                    int(missing_rxn),
                )

            if dry_run:
                logger.info("[%s] dry-run: skipping save to %s", name, out)
                saved = False
            else:
                saver(curated, out)
                saved = True
                logger.info("[%s] saved curated file to %s", name, out)

            status = "success"
            message = f"Processed OK; missing_rxn={int(missing_rxn)}"
            break
        except Exception as exc:
            tb = traceback.format_exc()
            logger.exception("[%s] attempt %d failed: %s", name, attempt, exc)
            last_exc = exc
            message = str(exc)
            # keep trying until retries exhausted
    end_time = time.time()
    time_s = end_time - start_time
    # truncate long messages
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
        "input_rows": (
            int(input_rows)
            if input_rows is not None
            and not (isinstance(input_rows, float) and math.isnan(input_rows))
            else None
        ),
        "output_rows": (
            int(output_rows)
            if output_rows is not None
            and not (isinstance(output_rows, float) and math.isnan(output_rows))
            else None
        ),
        "saved": bool(saved),
        "time_s": round(time_s, 3),
    }
    return result


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


def parse_args():
    p = argparse.ArgumentParser(
        description="Build classification datasets (curate -> save) with summary."
    )
    p.add_argument(
        "--config", help="JSON/YAML config file; if omitted DEFAULT_CONFIG is used."
    )
    p.add_argument(
        "--entries",
        help="Comma-separated subset of config keys to process (default: all).",
    )
    p.add_argument(
        "--src", help="Process a single source URL/local_path (overrides config)."
    )
    p.add_argument("--out", help="Output path when using --src (required).")
    p.add_argument(
        "--dry-run", action="store_true", help="Do everything except saving files."
    )
    p.add_argument("--retries", type=int, default=2, help="Load/process retries.")
    p.add_argument("--write-default", help="Write DEFAULT_CONFIG to file and exit.")
    p.add_argument("--log-level", default="INFO", help="Logging level.")
    p.add_argument(
        "--summary-out",
        default="reports/classification_summary.csv.gz",
        help="Path to write summary CSV (gzipped).",
    )
    return p.parse_args()


def _print_summary_table(results):
    # pretty print with tabulate if available
    try:
        from tabulate import tabulate  # type: ignore

        # choose columns and order
        cols = [
            "name",
            "status",
            "input_rows",
            "output_rows",
            "saved",
            "time_s",
            "message",
        ]
        table = [[r.get(c) for c in cols] for r in results]
        print(tabulate(table, headers=cols, tablefmt="github"))
    except Exception:
        pd = ensure_pandas()
        df = pd.DataFrame(results)
        # show a compact view
        display_cols = [
            "name",
            "status",
            "input_rows",
            "output_rows",
            "saved",
            "time_s",
            "message",
        ]
        print(df[display_cols].to_string(index=False))


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s - %(message)s",
    )
    logger.info("build_classification_dataset starting")

    if args.write_default:
        outp = Path(args.write_default)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        logger.info("Wrote default config to %s", outp)
        return

    results = []

    # single src shortcut
    if args.src:
        if not args.out:
            raise SystemExit("When --src is provided you must also provide --out")
        res = process_single_entry(
            "single", args.src, args.out, dry_run=args.dry_run, retries=args.retries
        )
        results.append(res)
    else:
        # load config or use default
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
            out = info.get("out") or f"Data/classification/{name}.csv.gz"
            logger.info("Processing '%s': %s -> %s", name, src, out)
            try:
                res = process_single_entry(
                    name, src, out, dry_run=args.dry_run, retries=args.retries
                )
            except Exception as exc:
                logger.exception("Failed to process %s: %s", name, exc)
                res = {
                    "name": name,
                    "src": src,
                    "out": out,
                    "status": "failed",
                    "message": str(exc)[:400]
                    + ("...(truncated)" if len(str(exc)) > 400 else ""),
                    "input_rows": None,
                    "output_rows": None,
                    "saved": False,
                    "time_s": None,
                }
            results.append(res)

    # Print and save summary
    _print_summary_table(results)

    # save summary as gzipped csv
    pd = ensure_pandas()
    summary_df = pd.DataFrame(results)
    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_out, index=False, compression="gzip")
    logger.info("Wrote summary to %s", summary_out)

    logger.info("build_classification_dataset finished")


if __name__ == "__main__":
    main()
