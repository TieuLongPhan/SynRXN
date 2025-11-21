#!/usr/bin/env python3
"""
build_property_dataset.py

End-to-end pipeline to build reaction–property datasets for SynRXN.

Datasets built
--------------
1. b97xd3
   - Source: https://doi.org/10.5281/zenodo.3715478  (b97d3.csv)
   - Columns: R-id, r_id, aam, ea, dh

2. snar
   - Source: https://www.rsc.org/suppdata/d0/sc/d0sc04896h/d0sc04896h2.zip
   - Columns: R-id, r_id, rxn, ea

3. e2sn2
   - Source: https://github.com/hesther/reactiondatabase (e2sn2.csv)
   - Columns: R-id, r_id, aam, ea

4. rad6re
   - Source: https://github.com/hesther/reactiondatabase (rad6re.csv)
   - Columns: R-id, r_id, aam, dh

5. lograte
   - Source: https://github.com/hesther/reactiondatabase (lograte.csv)
   - Columns: R-id, r_id, aam, lograte

6. phosphatase
   - Source: reactiondatabase phosphatase.csv + phosphatase_onehotenzyme.csv
   - Columns: R-id, r_id, aam, Conversion, onehot

7. e2, 8. sn2, 9. rdb7, 10. cycloadd, 11. rgd1
   - Source: Zenodo chemprop barriers tarball (data.tar.gz)
   - Combined per-barrier splits with 'split' column, then curated.
   - Columns: R-id, r_id, aam, <targets>, split

Output (defaults)
-----------------
Data/property/b97xd3.csv.gz
Data/property/snar.csv.gz
Data/property/e2sn2.csv.gz
Data/property/rad6re.csv.gz
Data/property/lograte.csv.gz
Data/property/phosphatase.csv.gz
Data/property/e2.csv.gz
Data/property/sn2.csv.gz
Data/property/rdb7.csv.gz
Data/property/cycloadd.csv.gz
Data/property/rgd1.csv.gz

Usage
-----
From repo root:

  PYTHONPATH=. python script/build_property_dataset.py

Subset:

  PYTHONPATH=. python script/build_property_dataset.py --entries b97xd3,snar

"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import re
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project imports (save_df_gz) with fallback
# ---------------------------------------------------------------------------

try:
    # ensure repo root is importable when running from script/
    sys.path.append(str(Path(__file__).resolve().parents[1]))
except Exception:
    pass

try:
    from synrxn.io.io import save_df_gz  # type: ignore
except Exception:
    save_df_gz = None


def _fallback_save_df_gz(df: pd.DataFrame, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")


def _get_saver():
    return save_df_gz or _fallback_save_df_gz


# ---------------------------------------------------------------------------
# Core curate_reactions helper (from notebook)
# ---------------------------------------------------------------------------

from typing import Union, Iterable


def curate_reactions(
    data: Union[str, Path, pd.DataFrame],
    *,
    data_name: str,
    rxn_col: str,
    target_cols: Union[str, Iterable[str]],
    split_col: Optional[Union[str, Iterable[str]]] = None,
    r_id_col: str = "R-id",
    index_base: int = 0,
    index_zero_pad: Optional[int] = None,
    keep_other_columns: bool = False,
    inplace: bool = False,
    out_csv: Optional[Union[str, Path]] = None,
    encoding: str = "utf-8",
    check_unique_rid: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    if isinstance(data, (str, Path)):
        df_in = pd.read_csv(str(data), encoding=encoding)
        orig_is_df = False
    elif isinstance(data, pd.DataFrame):
        df_in = data if inplace else data.copy(deep=True)
        orig_is_df = True
    else:
        raise TypeError("`data` must be a file path or a pandas DataFrame")

    if isinstance(target_cols, str):
        target_list: List[str] = [target_cols]
    else:
        target_list = list(target_cols)

    if split_col is None:
        split_list: List[str] = []
    elif isinstance(split_col, str):
        split_list = [split_col]
    else:
        split_list = list(split_col)

    if not isinstance(index_base, int) or index_base < 0:
        raise ValueError("index_base must be a non-negative integer")

    expected_cols = [rxn_col] + target_list + split_list
    missing = [c for c in expected_cols if c not in df_in.columns]
    if missing:
        raise KeyError(f"Missing expected column(s) in input data: {missing}")

    df_in.reset_index(drop=True, inplace=True)

    idx_vals = (df_in.index + index_base).astype(int).astype(str)
    if index_zero_pad is not None:
        if not isinstance(index_zero_pad, int) or index_zero_pad <= 0:
            raise ValueError("index_zero_pad must be a positive integer or None")
        idx_vals = idx_vals.str.zfill(index_zero_pad)

    rids = data_name + "_" + idx_vals
    df_in[r_id_col] = rids

    if rxn_col != "rxn":
        if "rxn" in df_in.columns and rxn_col != "rxn":
            df_in.rename(columns={"rxn": "rxn_orig"}, inplace=True)
            if verbose:
                print("Renamed existing 'rxn' column to 'rxn_orig' to avoid collision.")
        df_in = df_in.rename(columns={rxn_col: "rxn"})

    df_in["rxn"] = df_in["rxn"].astype(str).str.strip()

    keep_cols = [r_id_col, "rxn"] + target_list + split_list

    if keep_other_columns:
        other_cols = [c for c in df_in.columns if c not in keep_cols]
        ordered_cols = keep_cols + other_cols
        result = df_in.loc[:, ordered_cols]
    else:
        result = df_in.loc[:, [c for c in keep_cols if c in df_in.columns]]

    if check_unique_rid:
        if result[r_id_col].duplicated().any():
            dupes = (
                result[result[r_id_col].duplicated(keep=False)][r_id_col]
                .unique()
                .tolist()
            )
            raise ValueError(f"Non-unique R-id values produced (sample): {dupes[:10]}")

    if out_csv:
        result.to_csv(str(out_csv), index=False, encoding=encoding)
        if verbose:
            print(f"Wrote curated DataFrame to {out_csv}")

    if inplace and orig_is_df:
        orig_df = data  # type: ignore[assignment]
        for col in list(orig_df.columns):
            orig_df.drop(columns=col, inplace=True)
        for col in result.columns:
            orig_df[col] = result[col].values
        return orig_df

    return result


def _add_r_id_column(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Add 'r_id' column of the form '{prefix}_{index+1}' if not present.
    Returns a copy.
    """
    if "r_id" in df.columns:
        return df
    out = df.copy()
    out.insert(0, "r_id", [f"{prefix}_{i+1}" for i in range(len(out))])
    out.drop(columns=["R-id"], inplace=True)
    return out


# ---------------------------------------------------------------------------
# 1. b97xd3 helpers
# ---------------------------------------------------------------------------

B97XD3_URL = "https://zenodo.org/records/3715478/files/b97d3.csv?download=1"


def build_b97xd3(
    name: str, src: str, out_path: str | Path, *, dry_run: bool
) -> Dict[str, Any]:
    saver = _get_saver()
    out_path = Path(out_path)
    status = "failed"
    message = ""
    processed_count: Optional[int] = None

    start = time.time()
    try:
        logger.info("[%s] Loading b97xd3 CSV from %s", name, src)
        df = pd.read_csv(src)

        if not {"rsmi", "psmi"}.issubset(df.columns):
            missing = {"rsmi", "psmi"} - set(df.columns)
            raise KeyError(f"Missing required columns for b97xd3: {missing}")

        def combine(r, p):
            return f"{r}>>{p}"

        df["rxn"] = df.apply(lambda row: combine(row["rsmi"], row["psmi"]), axis=1)

        logger.info("[%s] Curating reactions", name)
        curated = curate_reactions(
            df,
            data_name="b97xd3",
            rxn_col="rxn",
            target_cols=["ea", "dh"],
            split_col=None,
            index_base=1,
            keep_other_columns=False,
        )

        # Drop rows with missing rxn or targets
        curated = curated.dropna(subset=["rxn", "ea", "dh"])
        # Rename output rxn -> aam
        curated = curated.rename(columns={"rxn": "aam"})
        curated = _add_r_id_column(curated, prefix=name)

        processed_count = len(curated)

        if dry_run:
            logger.info(
                "[%s] dry-run: skipping save to %s (n=%d)",
                name,
                out_path,
                processed_count,
            )
        else:
            saver(curated, out_path)
            logger.info(
                "[%s] Saved b97xd3 dataset to %s (n=%d)",
                name,
                out_path,
                processed_count,
            )

        status = "success"
        message = "Processed OK"
    except Exception as exc:
        message = str(exc)
        logger.exception("[%s] Failed: %s", name, exc)

    end = time.time()
    time_s = round(end - start, 3)
    if len(message) > 400:
        message = message[:400] + "...(truncated)"

    return {
        "name": name,
        "src": src,
        "out": str(out_path),
        "status": status,
        "message": message,
        "input_size": None,
        "processed_items": processed_count,
        "saved": (status == "success" and not dry_run),
        "time_s": time_s,
    }


# ---------------------------------------------------------------------------
# 2. SnAr helpers (keep 'rxn' here)
# ---------------------------------------------------------------------------

SNAR_ZIP_URL = "https://www.rsc.org/suppdata/d0/sc/d0sc04896h/d0sc04896h2.zip"
SNAR_TARGET_BASENAME = "SNAR_reaction_dataset_SI.csv"


def fetch_snar_df(
    url: str = SNAR_ZIP_URL,
    target_basename: str = SNAR_TARGET_BASENAME,
    timeout: int = 30,
) -> pd.DataFrame:
    encodings_default = ["cp1252", "latin1", "iso-8859-1", "utf-8", "utf-16"]

    with tempfile.TemporaryDirectory() as td:
        zip_path = os.path.join(td, "archive.zip")

        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as z:
            names = z.namelist()
            candidate: Optional[str] = None
            for n in names:
                if target_basename.lower() in os.path.basename(n).lower():
                    candidate = n
                    break
            if candidate is None:
                for n in names:
                    if target_basename.lower() in n.lower():
                        candidate = n
                        break
            if candidate is None:
                sample = names[:40]
                raise FileNotFoundError(
                    f"Could not find a file matching {target_basename!r} inside the ZIP. Sample entries: {sample}"
                )

            raw_bytes = z.read(candidate)
            sample_bytes = raw_bytes[:200_000]

            detected_enc = None
            try:
                import chardet  # type: ignore

                det = chardet.detect(sample_bytes)
                detected_enc = det.get("encoding")
            except Exception:
                detected_enc = None

            encodings = []
            if detected_enc:
                encodings.append(detected_enc)
            for e in encodings_default:
                if e not in encodings:
                    encodings.append(e)

            delim = ","
            for enc in encodings:
                try:
                    sample_text = sample_bytes.decode(enc, errors="replace")
                    sniff = csv.Sniffer()
                    dialect = sniff.sniff(sample_text)
                    delim = dialect.delimiter
                    break
                except Exception:
                    continue

            last_exc: Optional[Exception] = None
            for enc in encodings:
                try:
                    df = pd.read_csv(
                        io.BytesIO(raw_bytes),
                        encoding=enc,
                        delimiter=delim,
                        engine="c",
                        low_memory=False,
                    )
                    return df
                except Exception as e_c:
                    last_exc = e_c
                    try:
                        df = pd.read_csv(
                            io.BytesIO(raw_bytes),
                            encoding=enc,
                            delimiter=delim,
                            engine="python",
                        )
                        return df
                    except Exception as e_py:
                        last_exc = e_py
                        continue

            text = raw_bytes.decode("utf-8", errors="replace")
            try:
                df = pd.read_csv(io.StringIO(text), delimiter=delim)
                return df
            except Exception as final_e:
                raise RuntimeError(
                    "Failed to parse CSV inside ZIP with multiple encodings and "
                    f"fallbacks. Last parsing error: {last_exc!r}. Final error: {final_e!r}"
                ) from final_e


def build_snar(
    name: str, src: str, out_path: str | Path, *, dry_run: bool
) -> Dict[str, Any]:
    """SnAr keeps output column name 'rxn' (non-atom-mapped)."""
    saver = _get_saver()
    out_path = Path(out_path)
    status = "failed"
    message = ""
    processed_count: Optional[int] = None

    start = time.time()
    try:
        logger.info("[%s] Fetching SnAr dataset from %s", name, src)
        df = fetch_snar_df(url=src)

        if "Activation Free Energy (kcalmol-1)" not in df.columns:
            raise KeyError(
                "Expected column 'Activation Free Energy (kcalmol-1)' in SnAr data."
            )

        df["ea"] = df["Activation Free Energy (kcalmol-1)"]

        logger.info("[%s] Curating SnAr reactions", name)
        curated = curate_reactions(
            df,
            data_name="snar",
            rxn_col="Reaction SMILES",
            target_cols="ea",
            split_col=None,
            index_base=1,
            keep_other_columns=False,
        )

        # Here we deliberately keep 'rxn' as is (non-AAM)
        curated = curated.dropna(subset=["rxn", "ea"])
        curated = _add_r_id_column(curated, prefix=name)
        processed_count = len(curated)

        if dry_run:
            logger.info(
                "[%s] dry-run: skipping save to %s (n=%d)",
                name,
                out_path,
                processed_count,
            )
        else:
            saver(curated, out_path)
            logger.info(
                "[%s] Saved SnAr dataset to %s (n=%d)",
                name,
                out_path,
                processed_count,
            )

        status = "success"
        message = "Processed OK"
    except Exception as exc:
        message = str(exc)
        logger.exception("[%s] Failed: %s", name, exc)

    end = time.time()
    time_s = round(end - start, 3)
    if len(message) > 400:
        message = message[:400] + "...(truncated)"
    return {
        "name": name,
        "src": src,
        "out": str(out_path),
        "status": status,
        "message": message,
        "input_size": None,
        "processed_items": processed_count,
        "saved": (status == "success" and not dry_run),
        "time_s": time_s,
    }


# ---------------------------------------------------------------------------
# 3–5: simple reactiondatabase CSVs (e2sn2, rad6re, lograte) -> output 'aam'
# ---------------------------------------------------------------------------

E2SN2_URL = "https://raw.githubusercontent.com/hesther/reactiondatabase/refs/heads/main/data/e2sn2.csv"
RAD6RE_URL = (
    "https://github.com/hesther/reactiondatabase/raw/refs/heads/main/data/rad6re.csv"
)
LOGRATE_URL = "https://raw.githubusercontent.com/hesther/reactiondatabase/refs/heads/main/data/lograte.csv"


def build_simple_property_csv(
    name: str,
    src: str,
    out_path: str | Path,
    *,
    rxn_col: str,
    target_cols: Union[str, List[str]],
    dry_run: bool,
) -> Dict[str, Any]:
    saver = _get_saver()
    out_path = Path(out_path)
    status = "failed"
    message = ""
    processed_count: Optional[int] = None

    start = time.time()
    try:
        logger.info("[%s] Loading CSV from %s", name, src)
        df = pd.read_csv(src)

        logger.info("[%s] Curating reactions", name)
        curated = curate_reactions(
            df,
            data_name=name,
            rxn_col=rxn_col,
            target_cols=target_cols,
            split_col=None,
            index_base=1,
            keep_other_columns=False,
        )

        # Drop rows with missing rxn or all targets
        tcols = [target_cols] if isinstance(target_cols, str) else list(target_cols)
        curated = curated.dropna(subset=["rxn"] + tcols)
        # Rename output rxn -> aam (these are atom-mapped)
        curated = curated.rename(columns={"rxn": "aam"})
        curated = _add_r_id_column(curated, prefix=name)
        processed_count = len(curated)

        if dry_run:
            logger.info(
                "[%s] dry-run: skipping save to %s (n=%d)",
                name,
                out_path,
                processed_count,
            )
        else:
            saver(curated, out_path)
            logger.info(
                "[%s] Saved dataset to %s (n=%d)", name, out_path, processed_count
            )

        status = "success"
        message = "Processed OK"
    except Exception as exc:
        message = str(exc)
        logger.exception("[%s] Failed: %s", name, exc)

    end = time.time()
    time_s = round(end - start, 3)
    if len(message) > 400:
        message = message[:400] + "...(truncated)"
    return {
        "name": name,
        "src": src,
        "out": str(out_path),
        "status": status,
        "message": message,
        "input_size": None,
        "processed_items": processed_count,
        "saved": (status == "success" and not dry_run),
        "time_s": time_s,
    }


# ---------------------------------------------------------------------------
# 6. Phosphatase helpers -> output 'aam'
# ---------------------------------------------------------------------------

PHOSPHATASE_CSV = "https://github.com/hesther/reactiondatabase/raw/refs/heads/main/data/phosphatase.csv"
PHOSPHATASE_ONEHOT = "https://github.com/hesther/reactiondatabase/raw/refs/heads/main/data/phosphatase_onehotenzyme.csv"


def add_array_column(df: pd.DataFrame, arr: Any, col_name: str) -> pd.DataFrame:
    arr_np = np.asarray(arr)
    if arr_np.ndim == 0:
        values = [arr_np.item()] * len(df)
    elif arr_np.ndim == 1:
        values = arr_np.tolist()
    elif arr_np.ndim == 2 and arr_np.shape[1] == 1:
        values = arr_np.ravel().tolist()
    else:
        values = [list(row) for row in arr_np]
    if len(df) != len(values):
        raise ValueError(
            f"Length mismatch: df has {len(df)} rows but array has {len(values)} rows"
        )
    new_df = df.copy()
    new_df[col_name] = values
    return new_df


def build_phosphatase(
    name: str,
    src_main: str,
    src_onehot: str,
    out_path: str | Path,
    *,
    dry_run: bool,
) -> Dict[str, Any]:
    saver = _get_saver()
    out_path = Path(out_path)
    status = "failed"
    message = ""
    processed_count: Optional[int] = None

    start = time.time()
    try:
        logger.info("[%s] Loading phosphatase main CSV from %s", name, src_main)
        phosph = pd.read_csv(src_main)
        logger.info("[%s] Loading onehot enzyme CSV from %s", name, src_onehot)
        po = pd.read_csv(src_onehot)

        logger.info("[%s] Adding onehot column", name)
        phosph = add_array_column(phosph, po.values, col_name="onehot")

        logger.info("[%s] Curating phosphatase reactions", name)
        curated = curate_reactions(
            phosph,
            data_name="phosphatase",
            rxn_col="AAM",
            target_cols="Conversion",
            split_col="onehot",
            index_base=1,
            keep_other_columns=False,
        )

        curated = curated.dropna(subset=["rxn", "Conversion"])
        # Rename output rxn -> aam
        curated = curated.rename(columns={"rxn": "aam"})
        curated = _add_r_id_column(curated, prefix=name)
        processed_count = len(curated)

        if dry_run:
            logger.info(
                "[%s] dry-run: skipping save to %s (n=%d)",
                name,
                out_path,
                processed_count,
            )
        else:
            saver(curated, out_path)
            logger.info(
                "[%s] Saved phosphatase dataset to %s (n=%d)",
                name,
                out_path,
                processed_count,
            )

        status = "success"
        message = "Processed OK"
    except Exception as exc:
        message = str(exc)
        logger.exception("[%s] Failed: %s", name, exc)

    end = time.time()
    time_s = round(end - start, 3)
    if len(message) > 400:
        message = message[:400] + "...(truncated)"
    return {
        "name": name,
        "src": f"{src_main} + {src_onehot}",
        "out": str(out_path),
        "status": status,
        "message": message,
        "input_size": None,
        "processed_items": processed_count,
        "saved": (status == "success" and not dry_run),
        "time_s": time_s,
    }


# ---------------------------------------------------------------------------
# 7–11: chemprop barrier datasets (E2, SN2, RDB7, Cycloadd, RGD1) -> 'aam'
# ---------------------------------------------------------------------------

CHEMPROP_ZENODO_URL = "https://zenodo.org/records/10078142/files/data.tar.gz?download=1"
CHEMPROP_TARGETS = [
    "barriers_e2",
    "barriers_sn2",
    "barriers_cycloadd",
    "barriers_rdb7",
    "barriers_rgd1",
]


def _normalize_token(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "_", s.lower())


def stream_extract_selected_from_targz_safe(
    url: str,
    targets: List[str],
    dest_dir: str = "chemprop_zenodo",
    timeout: int = 1000,
    max_no_progress: int = 100000,
    min_matches_per_target: int = 1,
) -> Dict[str, List[Path]]:
    """
    Stream-download a .tar.gz and extract only members matching tokens in `targets`.
    """
    dest_dir_path = Path(dest_dir)
    dest_dir_path.mkdir(parents=True, exist_ok=True)

    norm_targets = [_normalize_token(t) for t in targets]
    target_map = {nt: t for nt, t in zip(norm_targets, targets)}

    extracted_by_target: Dict[str, List[Path]] = {nt: [] for nt in norm_targets}
    tokens_found: Set[str] = set()
    seen_members: Set[str] = set()
    no_progress = 0
    total_processed = 0

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            print("Opened remote tar.gz in streaming mode; iterating members...")
            for member in tar:
                total_processed += 1
                if member is None or not getattr(member, "name", None):
                    continue
                name = member.name
                if name in seen_members:
                    no_progress += 1
                    if no_progress >= max_no_progress:
                        print(
                            f"No progress for {max_no_progress} members; stopping early."
                        )
                        break
                    continue
                seen_members.add(name)
                no_progress += 1

                name_norm = _normalize_token(name)
                basename_norm = _normalize_token(Path(name).name)

                matched_tokens = [
                    tok
                    for tok in norm_targets
                    if (tok in name_norm or tok in basename_norm)
                ]
                if not matched_tokens:
                    if total_processed % 1000 == 0:
                        print(
                            f"Processed {total_processed} members so far; still searching..."
                        )
                    if no_progress >= max_no_progress:
                        print(
                            f"No progress for {max_no_progress} members; stopping early."
                        )
                        break
                    continue

                no_progress = 0
                for tok in matched_tokens:
                    tokens_found.add(tok)

                target_path = dest_dir_path / name
                target_resolved = target_path.resolve(strict=False)
                dest_resolved = dest_dir_path.resolve()
                if not str(target_resolved).startswith(str(dest_resolved)):
                    print(f"Skipping unsafe member path: {name}")
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)

                if member.isdir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    for tok in matched_tokens:
                        extracted_by_target[tok].append(target_path.resolve())
                    print(f"Matched directory: {name}")
                elif member.isreg():
                    f = tar.extractfile(member)
                    if f is None:
                        print(f"Warning: could not extract file member {name}")
                        continue
                    with open(target_path, "wb") as outfh:
                        while True:
                            chunk = f.read(1024 * 64)
                            if not chunk:
                                break
                            outfh.write(chunk)
                    try:
                        os.chmod(target_path, member.mode)
                    except Exception:
                        pass
                    for tok in matched_tokens:
                        extracted_by_target[tok].append(target_path.resolve())
                    print(f"Extracted -> {target_path}")
                else:
                    print(f"Skipping non-regular member: {name}")
                    with open(target_path, "w", encoding="utf-8") as outfh:
                        outfh.write(f"# skipped non-regular archive member: {name}\n")
                    for tok in matched_tokens:
                        extracted_by_target[tok].append(target_path.resolve())

                if all(
                    len(extracted_by_target[tok]) >= min_matches_per_target
                    for tok in norm_targets
                ):
                    print(
                        "All requested targets have at least",
                        min_matches_per_target,
                        "matches. Stopping early.",
                    )
                    break

    print("Streaming extraction finished. Processed members:", total_processed)
    for nt in norm_targets:
        print(
            f"Target '{target_map[nt]}' -> extracted {len(extracted_by_target[nt])} item(s)."
        )
    return extracted_by_target


def ensure_chemprop_extracted(root: str = "chemprop_zenodo") -> None:
    root_path = Path(root)
    needed_dirs = [root_path / "data" / t for t in CHEMPROP_TARGETS]
    if all(d.exists() for d in needed_dirs):
        logger.info("Chemprop barriers already extracted under %s", root_path)
        return

    logger.info("Extracting chemprop barriers from Zenodo into %s", root_path)
    stream_extract_selected_from_targz_safe(
        CHEMPROP_ZENODO_URL,
        CHEMPROP_TARGETS,
        dest_dir=root,
        timeout=1000,
        min_matches_per_target=1,
    )


def _try_read_csv(
    path: Path, encodings: Optional[List[str]] = None, **pd_kwargs
) -> pd.DataFrame:
    if encodings is None:
        encodings = ["utf-8", "cp1252", "latin1", "iso-8859-1", "utf-16"]
    raw = path.read_bytes()
    last_exc = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, **pd_kwargs)
        except Exception as e:
            last_exc = e
    try:
        txt = raw.decode("utf-8", errors="replace")
        return pd.read_csv(io.StringIO(txt), **pd_kwargs)
    except Exception as final_e:
        raise RuntimeError(
            f"Failed to read CSV {path!s}. Last error: {last_exc!r}; final attempt: {final_e!r}"
        )


def combine_barriers_split(
    base_dir: str,
    patterns_split: Optional[List[tuple]] = None,
    encodings: Optional[List[str]] = None,
    verbose: bool = True,
    save: bool = False,
    out_csv: str = "combined.csv",
    return_splits: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Combine train/val/test CSVs under base_dir into one DataFrame with a 'split' column.
    """
    base = Path(base_dir)
    if patterns_split is None:
        patterns_split = [
            ("train", "train"),
            ("val", "val"),
            ("validation", "val"),
            ("test", "test"),
        ]
    if encodings is None:
        encodings = ["utf-8", "cp1252", "latin1", "iso-8859-1", "utf-16"]

    if not base.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base.resolve()}")

    csv_files = sorted(list(base.rglob("*.csv")))
    if verbose:
        print(f"Found {len(csv_files)} CSV file(s) under {base}")

    rows = []
    files_used = 0
    split_frames: Dict[str, List[pd.DataFrame]] = {"train": [], "val": [], "test": []}

    for p in csv_files:
        name_l = p.name.lower()
        assigned_split = None
        for token, label in patterns_split:
            if token in name_l:
                assigned_split = label
                break

        if assigned_split is None:
            parents_to_check = list(p.parents)[:3]
            for parent in parents_to_check:
                part_l = parent.name.lower()
                for token, label in patterns_split:
                    if token in part_l:
                        assigned_split = label
                        break
                if assigned_split is not None:
                    break

        if assigned_split is None:
            if verbose:
                print(f"Skipping (no split token found): {p}")
            continue

        if verbose:
            print(f"Loading {p}  -> split='{assigned_split}'")
        try:
            df = _try_read_csv(p, encodings=encodings)
        except Exception as e:
            print(f"Failed to read {p}: {type(e).__name__}: {e}")
            continue

        df = df.copy()
        df["split"] = assigned_split
        try:
            df["_source_file"] = str(p.relative_to(base.parent))
        except Exception:
            df["_source_file"] = str(p)

        rows.append(df)
        split_frames.setdefault(assigned_split, []).append(df)
        files_used += 1

    if files_used == 0:
        raise RuntimeError(
            "No CSV files were loaded/labelled. Check filenames and patterns_split."
        )

    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined["split"] = pd.Categorical(
        combined["split"], categories=["train", "val", "test"], ordered=True
    )

    if save:
        combined.to_csv(out_csv, index=False, encoding="utf-8")
        if verbose:
            print(f"Saved combined CSV to: {Path(out_csv).resolve()}")

    if verbose:
        print(
            f"Combined {files_used} file(s) -> {combined.shape[0]} rows, {combined.shape[1]} cols"
        )

    if return_splits:
        per_split = {
            k: (pd.concat(v, ignore_index=True, sort=False) if v else pd.DataFrame())
            for k, v in split_frames.items()
        }
        return combined, per_split

    return combined


def build_barrier_dataset(
    name: str,
    barrier_key: str,
    out_path: str | Path,
    *,
    rxn_col: str,
    target_cols: Union[str, List[str]],
    dry_run: bool,
) -> Dict[str, Any]:
    saver = _get_saver()
    out_path = Path(out_path)
    status = "failed"
    message = ""
    processed_count: Optional[int] = None

    start = time.time()
    try:
        ensure_chemprop_extracted(root="chemprop_zenodo")
        base_dir = f"chemprop_zenodo/data/{barrier_key}"

        logger.info("[%s] Combining barrier CSVs from %s", name, base_dir)
        df = combine_barriers_split(base_dir, verbose=False)

        logger.info("[%s] Curating barrier reactions", name)
        curated = curate_reactions(
            df,
            data_name=name,
            rxn_col=rxn_col,
            target_cols=target_cols,
            split_col="split",
            index_base=1,
            keep_other_columns=False,
        )

        tcols = [target_cols] if isinstance(target_cols, str) else list(target_cols)
        curated = curated.dropna(subset=["rxn"] + tcols)
        # Rename output rxn -> aam
        curated = curated.rename(columns={"rxn": "aam"})
        curated = _add_r_id_column(curated, prefix=name)
        processed_count = len(curated)

        if dry_run:
            logger.info(
                "[%s] dry-run: skipping save to %s (n=%d)",
                name,
                out_path,
                processed_count,
            )
        else:
            saver(curated, out_path)
            logger.info(
                "[%s] Saved barrier dataset to %s (n=%d)",
                name,
                out_path,
                processed_count,
            )

        status = "success"
        message = "Processed OK"
    except Exception as exc:
        message = str(exc)
        logger.exception("[%s] Failed: %s", name, exc)

    end = time.time()
    time_s = round(end - start, 3)
    if len(message) > 400:
        message = message[:400] + "...(truncated)"
    return {
        "name": name,
        "src": f"chemprop barriers ({barrier_key})",
        "out": str(out_path),
        "status": status,
        "message": message,
        "input_size": None,
        "processed_items": processed_count,
        "saved": (status == "success" and not dry_run),
        "time_s": time_s,
    }


# ---------------------------------------------------------------------------
# DEFAULT CONFIG
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "b97xd3": {
        "kind": "b97xd3",
        "src": B97XD3_URL,
        "out": "Data/property/b97xd3.csv.gz",
    },
    "snar": {
        "kind": "snar",
        "src": SNAR_ZIP_URL,
        "out": "Data/property/snar.csv.gz",
    },
    "e2sn2": {
        "kind": "simple",
        "src": E2SN2_URL,
        "out": "Data/property/e2sn2.csv.gz",
        "rxn_col": "AAM",
        "target_cols": "ea",
    },
    "rad6re": {
        "kind": "simple",
        "src": RAD6RE_URL,
        "out": "Data/property/rad6re.csv.gz",
        "rxn_col": "AAM",
        "target_cols": "dh",
    },
    "lograte": {
        "kind": "simple",
        "src": LOGRATE_URL,
        "out": "Data/property/lograte.csv.gz",
        "rxn_col": "AAM",
        "target_cols": "lograte",
    },
    "phosphatase": {
        "kind": "phosphatase",
        "src_main": PHOSPHATASE_CSV,
        "src_onehot": PHOSPHATASE_ONEHOT,
        "out": "Data/property/phosphatase.csv.gz",
    },
    "e2": {
        "kind": "barrier",
        "barrier_key": "barriers_e2",
        "out": "Data/property/e2.csv.gz",
        "rxn_col": "AAM",
        "target_cols": "ea",
    },
    "sn2": {
        "kind": "barrier",
        "barrier_key": "barriers_sn2",
        "out": "Data/property/sn2.csv.gz",
        "rxn_col": "AAM",
        "target_cols": "ea",
    },
    "rdb7": {
        "kind": "barrier",
        "barrier_key": "barriers_rdb7",
        "out": "Data/property/rdb7.csv.gz",
        "rxn_col": "smiles",
        "target_cols": "ea",
    },
    "cycloadd": {
        "kind": "barrier",
        "barrier_key": "barriers_cycloadd",
        "out": "Data/property/cycloadd.csv.gz",
        "rxn_col": "rxn_smiles",
        "target_cols": ["G_act", "G_r"],
    },
    "rgd1": {
        "kind": "barrier",
        "barrier_key": "barriers_rgd1",
        "out": "Data/property/rgd1.csv.gz",
        "rxn_col": "smiles",
        "target_cols": "ea",
    },
}


# ---------------------------------------------------------------------------
# CLI + summary
# ---------------------------------------------------------------------------


def _print_summary_table(results: List[Dict[str, Any]]) -> None:
    cols = [
        "name",
        "status",
        "processed_items",
        "saved",
        "out",
        "time_s",
        "message",
    ]
    df = pd.DataFrame(results)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    try:
        from tabulate import tabulate  # type: ignore

        table = [[r.get(c) for c in cols] for r in results]
        print(tabulate(table, headers=cols, tablefmt="github"))
    except Exception:
        print(df[cols].to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(
        description="Build property datasets (b97xd3, SnAr, E2SN2, etc.)"
    )
    p.add_argument(
        "--entries",
        help=(
            "Comma-separated subset of datasets to build "
            "(default: all). "
            "Choices: " + ",".join(DEFAULT_CONFIG.keys())
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do everything except saving files.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG/INFO/WARNING/ERROR).",
    )
    p.add_argument(
        "--summary-out",
        default="reports/property_summary.csv.gz",
        help="Path to write summary CSV (gzipped).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(args, "log_level", "INFO").upper(),
        format="%(asctime)s %(levelname)s - %(message)s",
    )
    logger.info("build_property_dataset starting")

    cfg = DEFAULT_CONFIG.copy()

    entries: Optional[List[str]] = None
    if args.entries:
        entries = [e.strip() for e in args.entries.split(",") if e.strip()]
        logger.info("Processing subset entries: %s", entries)

    results: List[Dict[str, Any]] = []

    for name, info in cfg.items():
        if entries and name not in entries:
            logger.debug("Skipping %s (not requested)", name)
            continue

        kind = info.get("kind")
        logger.info("Building property dataset '%s' (kind=%s)", name, kind)

        try:
            if kind == "b97xd3":
                row = build_b97xd3(
                    name=name,
                    src=info["src"],
                    out_path=info["out"],
                    dry_run=args.dry_run,
                )
            elif kind == "snar":
                row = build_snar(
                    name=name,
                    src=info["src"],
                    out_path=info["out"],
                    dry_run=args.dry_run,
                )
            elif kind == "simple":
                row = build_simple_property_csv(
                    name=name,
                    src=info["src"],
                    out_path=info["out"],
                    rxn_col=info["rxn_col"],
                    target_cols=info["target_cols"],
                    dry_run=args.dry_run,
                )
            elif kind == "phosphatase":
                row = build_phosphatase(
                    name=name,
                    src_main=info["src_main"],
                    src_onehot=info["src_onehot"],
                    out_path=info["out"],
                    dry_run=args.dry_run,
                )
            elif kind == "barrier":
                row = build_barrier_dataset(
                    name=name,
                    barrier_key=info["barrier_key"],
                    out_path=info["out"],
                    rxn_col=info["rxn_col"],
                    target_cols=info["target_cols"],
                    dry_run=args.dry_run,
                )
            else:
                raise ValueError(f"Unknown dataset kind: {kind!r}")
        except Exception as exc:
            msg = str(exc)
            if len(msg) > 400:
                msg = msg[:400] + "...(truncated)"
            logger.exception("Failed to build dataset %s: %s", name, exc)
            row = {
                "name": name,
                "src": info.get("src", ""),
                "out": info.get("out", ""),
                "status": "failed",
                "message": msg,
                "processed_items": None,
                "saved": False,
                "time_s": None,
            }

        results.append(row)

    _print_summary_table(results)

    # Save summary
    try:
        summary_df = pd.DataFrame(results)
        summary_out = Path(args.summary_out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_out, index=False, compression="gzip")
        logger.info("Wrote summary to %s", summary_out)
    except Exception:
        logger.exception("Failed to write summary CSV; printing only.")

    logger.info("build_property_dataset finished")


if __name__ == "__main__":
    main()