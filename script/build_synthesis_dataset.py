#!/usr/bin/env python3
"""
build_synthesis_dataset.py

End-to-end pipeline for synthesis datasets used in SynRXN:

1. USPTO 50k (AAM-style reactions with splits)
   - Download & combine raw train/valid/test from Dropbox ZIP.
   - Curate to minimal format: R-id, aam, split, source.
   - Canonicalise AAM with Standardize + CanonRSMI.
   - Add r_id = uspto_50k_{i+1}.
   - Save to Data/synthesis/uspto_50k.csv.gz.

2. USPTO MIT (NIPS17 RexGen)
   - Download ZIP of train/valid/test txt.
   - Extract reaction + optional reaction-centre (rc) tokens.
   - Canonicalise AAM with Standardize + CanonRSMI.
   - Add R-id & r_id = uspto_mit_{i+1}.
   - Save to Data/synthesis/uspto_mit.csv.gz.

3. USPTO 500 MT
   - Download tar.bz2 of USPTO_500_MT reagents (source/target).
   - Standardize sources with Standardize.
   - Build rxn, reagent, split table.
   - Add R-id & r_id = uspto_500_{i+1}.
   - Save to Data/synthesis/uspto_500.csv.gz.

Usage
-----
From repo root:

  PYTHONPATH=. python script/build_synthesis_dataset.py

Options
-------
  --entries uspto_50k,uspto_mit,uspto_500  # subset
  --n-jobs 4                               # for canonicalisation
  --dry-run                                # do everything except saving
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import re
import sys
import tarfile
import traceback
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
    Tuple,
    Union,
)

import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional project imports (saver)
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
# COMMON HELPERS
# ---------------------------------------------------------------------------


def _add_r_id_column(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Add 'r_id' column with values prefix_{index+1} if not already present.
    Operates on a copy and returns it.
    """
    if "r_id" in df.columns:
        return df
    df = df.copy()
    df.insert(0, "r_id", [f"{prefix}_{i+1}" for i in range(len(df))])
    df.drop(columns=["R-id"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# 1. USPTO 50k helpers
# ---------------------------------------------------------------------------

DEFAULT_PATTERNS = {
    "raw_train": "train",
    "raw_val": "valid",
    "raw_valid": "valid",
    "raw_test": "test",
}


def _normalize_dropbox_url(url: str) -> str:
    if "dropbox.com" not in url:
        return url
    if "dl=0" in url:
        return url.replace("dl=0", "dl=1")
    if "dl=1" in url:
        return url
    return url + ("&dl=1" if "?" in url else "?dl=1")


def _find_member_for_pattern(namelist: Iterable[str], token: str) -> Optional[str]:
    token_l = token.lower()
    for nm in namelist:
        if (
            nm.lower().endswith(token_l)
            or nm.lower().endswith(token_l + ".csv")
            or nm.lower().endswith(token_l + ".txt")
        ):
            return nm
    for nm in namelist:
        if token_l in nm.lower():
            return nm
    return None


def download_and_combine_raw_splits(
    url: str,
    patterns: Optional[Dict[str, str]] = None,
    *,
    encoding: str = "utf-8",
    treat_txt_as_lines: bool = True,
    save_csv: Optional[str] = None,
    timeout: int = 60,
) -> pd.DataFrame:
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    url = _normalize_dropbox_url(url)
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    raw = resp.content
    bio = io.BytesIO(raw)

    # ZIP path
    try:
        with zipfile.ZipFile(bio) as zf:
            namelist = zf.namelist()
            dfs = []
            for token, split_label in patterns.items():
                member = _find_member_for_pattern(namelist, token)
                if not member:
                    continue
                with zf.open(member) as fh:
                    if member.lower().endswith(".csv"):
                        df = pd.read_csv(io.TextIOWrapper(fh, encoding=encoding))
                    else:
                        text = fh.read().decode(encoding, errors="replace")
                        if treat_txt_as_lines:
                            lines = [
                                ln.strip() for ln in text.splitlines() if ln.strip()
                            ]
                            df = pd.DataFrame({"rxn": lines})
                        else:
                            fh.seek(0)
                            df = pd.read_csv(io.TextIOWrapper(fh, encoding=encoding))
                    df["split"] = split_label
                    dfs.append(df)
            if not dfs:
                raise RuntimeError(
                    "No files matching raw_train/raw_val/raw_test found inside ZIP."
                )
            combined = pd.concat(dfs, ignore_index=True, sort=False)
            if save_csv:
                combined.to_csv(save_csv, index=False)
            return combined
    except zipfile.BadZipFile:
        pass

    # Fallback: CSV
    try:
        bio.seek(0)
        df_all = pd.read_csv(io.BytesIO(raw))
        if "split" in df_all.columns:
            if save_csv:
                df_all.to_csv(save_csv, index=False)
            return df_all
        # infer split from URL
        inferred_split = None
        lower_url = url.lower()
        for token, split_label in patterns.items():
            if token in lower_url:
                inferred_split = split_label
                break
        if inferred_split is None:
            inferred_split = "train"
        df_all["split"] = inferred_split
        if save_csv:
            df_all.to_csv(save_csv, index=False)
        return df_all
    except Exception:
        # Fallback: plain text lines
        text = raw.decode(encoding, errors="replace")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        lower_url = url.lower()
        inferred_split = None
        for token, split_label in patterns.items():
            if token in lower_url:
                inferred_split = split_label
                break
        if inferred_split is None:
            inferred_split = "train"
        df = pd.DataFrame({"rxn": lines})
        df["split"] = inferred_split
        if save_csv:
            df.to_csv(save_csv, index=False)
        return df


def curate_uspto_minimal(
    df_or_path: Union[pd.DataFrame, str],
    id_col: str = "id",
    rxn_col: str = "reactants>reagents>production",
    split_col: str = "split",
    zero_based_rid: bool = True,
    drop_original_rxn: bool = True,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path, encoding=encoding)
    else:
        df = df_or_path.copy()

    # map id -> source
    if id_col in df.columns:
        df = df.rename(columns={id_col: "source"})
    else:
        df["source"] = df.index.astype(str)

    if rxn_col not in df.columns:
        raise KeyError(f"Reaction column '{rxn_col}' not found.")

    start = 0 if zero_based_rid else 1
    df = df.reset_index(drop=True)
    df["R-id"] = [f"uspto_{i + start}" for i in df.index]

    def _split_make_rxn(cell):
        if pd.isna(cell):
            return None
        s = str(cell).strip()
        parts = [p.strip() for p in s.split(">")]
        if len(parts) >= 3:
            reactants = parts[0] or ""
            production = ">".join(p for p in parts[2:] if p != "") or ""
        elif len(parts) == 2:
            reactants = parts[0] or ""
            production = parts[1] or ""
        else:
            reactants = parts[0] or ""
            production = ""
        if reactants == "" and production == "":
            return None
        return f"{reactants}>>{production}"

    df["aam"] = df[rxn_col].apply(_split_make_rxn)

    if split_col in df.columns:
        out_split = df[split_col].astype(object)
    else:
        out_split = pd.Series([None] * len(df), name="split")

    out = pd.DataFrame(
        {
            "R-id": df["R-id"],
            "aam": df["aam"],
            "split": out_split,
            "source": df["source"],
        }
    )

    return out


# ---------------------------------------------------------------------------
# Canonicalisation (fix_aam) – shared for USPTO 50k + MIT
# ---------------------------------------------------------------------------

_WORKER_STD = None
_WORKER_CANON = None


def _create_worker_instances(std_factory=None, canon_factory=None):
    """Create or return cached Standardize/CanonRSMI instances inside a worker."""
    global _WORKER_STD, _WORKER_CANON
    if _WORKER_STD is None or _WORKER_CANON is None:
        if std_factory is not None and canon_factory is not None:
            _WORKER_STD = std_factory()
            _WORKER_CANON = canon_factory()
        else:
            # default lazy import/construct
            try:
                from synkit.Chem.Reaction.standardize import Standardize
                from synkit.Chem.Reaction.canon_rsmi import CanonRSMI
            except Exception as e:
                raise RuntimeError(
                    "Failed to import Standardize/CanonRSMI in worker: " + str(e)
                )
            _WORKER_STD = Standardize()
            _WORKER_CANON = CanonRSMI()
    return _WORKER_STD, _WORKER_CANON


def _canonicalise_value_worker_v2(
    idx,
    original_value,
    aam_col,
    std_factory,
    canon_factory,
):
    """
    Worker function: apply std.fit(..., remove_aam=False) then canon.canonicalise(...).canonical_rsmi
    Returns: (idx, canonical_string_or_None, error_or_None)
    """
    try:
        std, canon = _create_worker_instances(
            std_factory=std_factory, canon_factory=canon_factory
        )
    except Exception as e:
        tb = traceback.format_exc(limit=6)
        return idx, None, f"worker init/import error: {e}\n{tb}"

    def _to_str_or_none(x: Any) -> Optional[str]:
        if x is None:
            return None
        if isinstance(x, float) and pd.isna(x):
            return None
        s = str(x).strip()
        return s if s != "" else None

    try:
        s = _to_str_or_none(original_value)
        if s is None:
            return idx, None, "empty_or_nan"

        # IMPORTANT: use remove_aam=False
        try:
            fitted = std.fit(s, remove_aam=False)
        except TypeError:
            try:
                fitted = std.fit(s)
            except Exception as e:
                tb = traceback.format_exc(limit=6)
                return idx, None, f"std.fit error: {e}\n{tb}"
        except Exception as e:
            tb = traceback.format_exc(limit=6)
            return idx, None, f"std.fit error: {e}\n{tb}"

        # pick candidate string
        cand = None
        if isinstance(fitted, str):
            cand = fitted
        else:
            for attr in ("canonical_rsmi", "rsmi", "rxn", "reaction", "smiles"):
                if hasattr(fitted, attr):
                    val = getattr(fitted, attr)
                    if isinstance(val, str) and val.strip():
                        cand = val.strip()
                        break
            if cand is None:
                cand = str(fitted).strip() if str(fitted).strip() else None

        if not cand:
            return idx, None, "empty_after_std"

        try:
            canon_out = canon.canonicalise(cand)
        except Exception as e:
            tb = traceback.format_exc(limit=6)
            return idx, None, f"canon.canonicalise error: {e}\n{tb}"

        canonical_string = None
        if isinstance(canon_out, str):
            canonical_string = canon_out.strip()
        else:
            if hasattr(canon_out, "canonical_rsmi"):
                val = getattr(canon_out, "canonical_rsmi")
                if isinstance(val, str) and val.strip():
                    canonical_string = val.strip()
            if canonical_string is None:
                for attr in ("canonical", "canonical_smiles", "rsmi", "r_smiles"):
                    if hasattr(canon_out, attr):
                        val = getattr(canon_out, attr)
                        if isinstance(val, str) and val.strip():
                            canonical_string = val.strip()
                            break
            if canonical_string is None:
                srep = str(canon_out).strip()
                canonical_string = srep if srep else None

        if not canonical_string:
            return idx, None, "empty_after_canonicalise"

        return idx, canonical_string, None

    except Exception as e:
        tb = traceback.format_exc(limit=6)
        return idx, None, f"unexpected error: {e}\n{tb}"


def fix_aam(
    df: pd.DataFrame,
    std=None,
    canon=None,
    aam_col: str = "aam",
    out_col: Optional[str] = None,
    overwrite: bool = True,
    show_progress: bool = False,
    stop_on_error: bool = False,
    n_jobs: int = 1,
    backend: str = "loky",
    std_factory=None,
    canon_factory=None,
    batch_size: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parallel canonicalisation using std.fit(..., remove_aam=False) and canon.canonicalise(...).canonical_rsmi.

    - If n_jobs==1, uses provided std/canon instances (or imports locally).
    - If n_jobs!=1, workers instantiate std/canon themselves via std_factory/canon_factory or default import.
    - Returns (out_df, errors_df).
    """
    from joblib import Parallel, delayed

    if out_col is None:
        out_col = aam_col if overwrite else f"{aam_col}_fixed"

    if aam_col not in df.columns:
        raise KeyError(f"Column '{aam_col}' not found in DataFrame")

    # Serial path
    if n_jobs == 1:
        if std is None or canon is None:
            try:
                from synkit.Chem.Reaction.standardize import Standardize
                from synkit.Chem.Reaction.canon_rsmi import CanonRSMI
            except Exception as e:
                raise RuntimeError(
                    "Failed to import Standardize/CanonRSMI locally: " + str(e)
                )
            std = std or Standardize()
            canon = canon or CanonRSMI()

        def _canonicalise_one_serial(original_value):
            try:
                if original_value is None or (
                    isinstance(original_value, float) and pd.isna(original_value)
                ):
                    return None, "empty_or_nan"
                s = str(original_value).strip()
                if s == "":
                    return None, "empty_or_nan"

                try:
                    fitted = std.fit(s, remove_aam=False)
                except TypeError:
                    fitted = std.fit(s)
                except Exception as e:
                    tb = traceback.format_exc(limit=6)
                    return None, f"std.fit error: {e}\n{tb}"

                cand = None
                if isinstance(fitted, str):
                    cand = fitted
                else:
                    for attr in (
                        "canonical_rsmi",
                        "rsmi",
                        "rxn",
                        "reaction",
                        "smiles",
                    ):
                        if hasattr(fitted, attr):
                            val = getattr(fitted, attr)
                            if isinstance(val, str) and val.strip():
                                cand = val.strip()
                                break
                    if cand is None:
                        cand = str(fitted).strip() if str(fitted).strip() else None

                if not cand:
                    return None, "empty_after_std"

                try:
                    canon_out = canon.canonicalise(cand)
                except Exception as e:
                    tb = traceback.format_exc(limit=6)
                    return None, f"canon.canonicalise error: {e}\n{tb}"

                canonical_string = None
                if hasattr(canon_out, "canonical_rsmi"):
                    val = getattr(canon_out, "canonical_rsmi")
                    if isinstance(val, str) and val.strip():
                        canonical_string = val.strip()
                if canonical_string is None:
                    for attr in (
                        "canonical",
                        "canonical_smiles",
                        "rsmi",
                        "r_smiles",
                    ):
                        if hasattr(canon_out, attr):
                            val = getattr(canon_out, attr)
                            if isinstance(val, str) and val.strip():
                                canonical_string = val.strip()
                                break
                if canonical_string is None:
                    sval = str(canon_out).strip()
                    canonical_string = sval if sval else None

                if not canonical_string:
                    return None, "empty_after_canonicalise"
                return canonical_string, None
            except Exception as e:
                tb = traceback.format_exc(limit=6)
                return None, f"unexpected error: {e}\n{tb}"

        out_df = df.copy()
        errors: List[Dict[str, Any]] = []
        for idx in df.index:
            original = out_df.at[idx, aam_col]
            fixed, err = _canonicalise_one_serial(original)
            out_df.at[idx, out_col] = fixed
            if err is not None:
                errors.append(
                    {
                        "index": idx,
                        "R-id": (
                            out_df.at[idx, "R-id"] if "R-id" in out_df.columns else None
                        ),
                        "source": (
                            out_df.at[idx, "source"]
                            if "source" in out_df.columns
                            else None
                        ),
                        "original": original,
                        "error": err,
                    }
                )
                if stop_on_error:
                    raise RuntimeError(f"Row {idx} failed: {err}; original={original}")
        errors_df = pd.DataFrame(errors)
        if overwrite and out_col != aam_col:
            out_df[aam_col] = out_df[out_col]
            out_df.drop(columns=[out_col], inplace=True)
        return out_df, errors_df

    # Parallel path
    tasks = [(int(idx), df.at[idx, aam_col]) for idx in df.index]

    results = Parallel(n_jobs=n_jobs, backend=backend, batch_size=batch_size)(
        delayed(_canonicalise_value_worker_v2)(
            idx, original, aam_col, std_factory, canon_factory
        )
        for idx, original in tasks
    )

    out_df = df.copy()
    errors: List[Dict[str, Any]] = []
    for idx, canonical_string, err in results:
        out_df.at[idx, out_col] = canonical_string
        if err is not None:
            errors.append(
                {
                    "index": idx,
                    "R-id": (
                        out_df.at[idx, "R-id"] if "R-id" in out_df.columns else None
                    ),
                    "source": (
                        out_df.at[idx, "source"] if "source" in out_df.columns else None
                    ),
                    "original": df.at[idx, aam_col],
                    "error": err,
                }
            )

    errors_df = pd.DataFrame(errors)

    if stop_on_error and not errors_df.empty:
        sample = errors_df.iloc[0].to_dict()
        raise RuntimeError(
            f"Errors occurred during parallel processing; sample error: {sample}"
        )

    if overwrite and out_col != aam_col:
        out_df[aam_col] = out_df[out_col]
        out_df.drop(columns=[out_col], inplace=True)

    return out_df, errors_df


# ---------------------------------------------------------------------------
# 2. USPTO MIT helpers
# ---------------------------------------------------------------------------

RC_PATTERN = re.compile(r"^\d+-\d+(?:;\d+-\d+)*$")


def _parse_rc_string_to_tuples(
    rc: str, zero_index: bool = False
) -> Optional[List[Tuple[int, int]]]:
    """'15-19;6-15;6-8' -> [(15,19),(6,15),(6,8)] (or zero-indexed)."""
    if not rc:
        return None
    parts = rc.split(";")
    parsed = []
    for p in parts:
        if "-" not in p:
            return None
        a, b = p.split("-", 1)
        try:
            ai, bi = int(a), int(b)
        except ValueError:
            return None
        if zero_index:
            ai -= 1
            bi -= 1
        parsed.append((ai, bi))
    return parsed


def process_uspto_mt(
    url: str,
    files_map: Optional[Dict[str, str]] = None,
    *,
    encoding: str = "utf-8",
    strip_lines: bool = True,
    zero_index: bool = False,
    save_csv: Optional[str] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Download ZIP at `url`, extract split files and return DataFrame with columns:
      - aam   : reaction string (line content without trailing rc token)
      - split : 'train'|'valid'|'test'
      - rc    : parsed list of (int,int) tuples or None
    """
    if files_map is None:
        files_map = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}

    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    zbytes = io.BytesIO(resp.content)

    def _find_member(zf: zipfile.ZipFile, target: str) -> Optional[str]:
        nm_list = zf.namelist()
        if target in nm_list:
            return target
        low = target.lower()
        for nm in nm_list:
            if nm.lower().endswith(low):
                return nm
        for nm in nm_list:
            if low in nm.lower():
                return nm
        return None

    rows = []
    with zipfile.ZipFile(zbytes) as zf:
        for split, desired_name in files_map.items():
            member = _find_member(zf, desired_name)
            if member is None:
                logger.warning(
                    "Warning: '%s' not found in archive; skipping split '%s'.",
                    desired_name,
                    split,
                )
                continue
            with zf.open(member) as fh:
                text = fh.read().decode(encoding, errors="replace")

            for line in text.splitlines():
                if strip_lines:
                    line = line.strip()
                if not line:
                    continue

                parts = line.rsplit(None, 1)
                rc_parsed = None
                if len(parts) == 2 and RC_PATTERN.fullmatch(parts[1]):
                    rc_parsed = _parse_rc_string_to_tuples(
                        parts[1], zero_index=zero_index
                    )
                    rxn = parts[0]
                else:
                    rxn = line
                    rc_parsed = None

                rows.append({"aam": rxn, "split": split, "rc": rc_parsed})

    if not rows:
        raise RuntimeError("No data found in ZIP for any requested split files.")

    df = pd.DataFrame(rows)
    if save_csv:
        df.to_csv(save_csv, index=False)
    return df


# ---------------------------------------------------------------------------
# 3. USPTO_500_MT reagents helpers
# ---------------------------------------------------------------------------


def load_reagents_from_tar_url(
    url: str,
    sep: Optional[str] = None,
    names: Optional[list] = None,
    encoding: str = "utf-8",
    timeout: int = 60,
    max_in_memory_bytes: int = 300 * 1024 * 1024,
) -> Dict[str, pd.DataFrame]:
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    results: Dict[str, pd.DataFrame] = {}
    with tarfile.open(fileobj=resp.raw, mode="r|bz2") as tf:
        for member in tf:
            if not member.isreg():
                continue
            member_name = member.name
            if not member_name.startswith("data/USPTO_500_MT/Reagents/"):
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            try:
                b = fh.read()
            except Exception:
                continue
            if len(b) > max_in_memory_bytes:
                # here we just ignore very large files; could stream to disk if needed
                continue
            try:
                text = b.decode(encoding)
            except Exception:
                text = b.decode(encoding, errors="replace")
            stream = io.StringIO(text)
            try:
                if names is not None:
                    df = pd.read_csv(stream, header=None, names=names, sep=sep)
                else:
                    if sep is None:
                        df = pd.read_csv(
                            stream, sep=None, engine="python", low_memory=False
                        )
                    else:
                        df = pd.read_csv(stream, sep=sep, low_memory=False)
            except Exception:
                try:
                    stream.seek(0)
                    df = pd.read_csv(
                        stream,
                        header=None,
                        names=names or ["col0"],
                        sep=sep or r"\s+",
                        engine="python",
                    )
                except Exception:
                    continue
            results[member_name] = df
    return results


def combine_reagents_dict(
    dfs: Dict[str, pd.DataFrame],
    std,
    prefix: str = "data/USPTO_500_MT/Reagents/",
) -> pd.DataFrame:
    mapping: Dict[str, Dict[str, pd.DataFrame]] = {}
    for k, df in dfs.items():
        name = k[len(prefix) :] if k.startswith(prefix) else k.split("/")[-1]
        parts = name.split(".")
        if len(parts) < 2:
            continue
        split, role = parts[0], parts[1]
        mapping.setdefault(split, {})[role] = df.reset_index(drop=True)

    rows = []
    for split, grp in mapping.items():
        src_df = grp.get("source")
        tgt_df = grp.get("target")

        def col_series(df):
            if df is None:
                return None
            return df.iloc[:, 0].astype(str).str.strip().reset_index(drop=True)

        src_s = col_series(src_df)
        tgt_s = col_series(tgt_df)
        n = max(
            (len(src_s) if src_s is not None else 0),
            (len(tgt_s) if tgt_s is not None else 0),
        )
        for i in range(n):
            rxn = src_s.iloc[i] if (src_s is not None and i < len(src_s)) else np.nan
            reagent = (
                tgt_s.iloc[i] if (tgt_s is not None and i < len(tgt_s)) else np.nan
            )

            std_rxn = rxn
            if isinstance(rxn, str) and rxn.strip():
                try:
                    std_rxn = std.fit(rxn)
                except Exception:
                    try:
                        std_rxn = std.fit(rxn.strip())
                    except Exception:
                        std_rxn = rxn
            rows.append({"rxn": std_rxn, "reagent": reagent, "split": split})
    return pd.DataFrame(rows)[["rxn", "reagent", "split"]]


# ---------------------------------------------------------------------------
# PER-DATASET BUILDERS (wrappers)
# ---------------------------------------------------------------------------


def build_uspto_50k(
    name: str,
    url: str,
    out_path: str | Path,
    *,
    n_jobs: int = 4,
    dry_run: bool = False,
) -> Dict[str, Any]:
    saver = _get_saver()
    out_path = Path(out_path)
    status = "failed"
    message = ""
    processed_count: Optional[int] = None

    from synkit.Chem.Reaction.standardize import Standardize
    from synkit.Chem.Reaction.canon_rsmi import CanonRSMI

    std = Standardize()
    canon = CanonRSMI()

    start_t = pd.Timestamp.now().timestamp()

    try:
        logger.info("[%s] Downloading & combining raw USPTO 50k splits", name)
        df_raw = download_and_combine_raw_splits(url, save_csv=None)

        logger.info("[%s] Curating minimal format", name)
        df_min = curate_uspto_minimal(df_raw)

        logger.info("[%s] Canonicalising AAM", name)
        df_fixed, errors_df = fix_aam(
            df_min,
            std=std,
            canon=canon,
            n_jobs=n_jobs,
            show_progress=False,
        )

        # add r_id based on dataset name
        df_fixed = _add_r_id_column(df_fixed, prefix=name)

        processed_count = len(df_fixed)
        if dry_run:
            logger.info(
                "[%s] dry-run: skipping save to %s (n=%d)",
                name,
                out_path,
                processed_count,
            )
        else:
            saver(df_fixed, out_path)
            logger.info(
                "[%s] Saved dataset to %s (n=%d)", name, out_path, processed_count
            )
            # optional: save errors
            if not errors_df.empty:
                err_out = out_path.with_suffix(".errors.csv.gz")
                saver(errors_df, err_out)
                logger.info(
                    "[%s] Saved canonicalisation errors to %s (n=%d)",
                    name,
                    err_out,
                    len(errors_df),
                )

        status = "success"
        message = "Processed OK"
    except Exception as exc:
        message = str(exc)
        logger.exception("[%s] Failed: %s", name, exc)

    end_t = pd.Timestamp.now().timestamp()
    time_s = round(end_t - start_t, 3)
    if len(message) > 400:
        message = message[:400] + "...(truncated)"

    return {
        "name": name,
        "src": url,
        "out": str(out_path),
        "status": status,
        "message": message,
        "input_size": None,
        "processed_items": (
            int(processed_count) if processed_count is not None else None
        ),
        "saved": (status == "success" and not dry_run),
        "time_s": time_s,
    }


def build_uspto_mit(
    name: str,
    url: str,
    out_path: str | Path,
    *,
    n_jobs: int = 4,
    dry_run: bool = False,
) -> Dict[str, Any]:
    saver = _get_saver()
    out_path = Path(out_path)
    status = "failed"
    message = ""
    processed_count: Optional[int] = None

    from synkit.Chem.Reaction.standardize import Standardize
    from synkit.Chem.Reaction.canon_rsmi import CanonRSMI

    std = Standardize()
    canon = CanonRSMI()

    start_t = pd.Timestamp.now().timestamp()

    try:
        logger.info("[%s] Downloading & processing USPTO MIT", name)
        df_raw = process_uspto_mt(url)

        # Add R-id before canonicalisation
        df_raw = df_raw.reset_index(drop=True)
        df_raw["R-id"] = [f"{name}_{i+1}" for i in df_raw.index]

        logger.info("[%s] Canonicalising AAM", name)
        df_fixed, errors_df = fix_aam(
            df_raw,
            std=std,
            canon=canon,
            n_jobs=n_jobs,
            show_progress=False,
        )

        # add r_id
        df_fixed = _add_r_id_column(df_fixed, prefix=name)

        processed_count = len(df_fixed)
        if dry_run:
            logger.info(
                "[%s] dry-run: skipping save to %s (n=%d)",
                name,
                out_path,
                processed_count,
            )
        else:
            saver(df_fixed, out_path)
            logger.info(
                "[%s] Saved dataset to %s (n=%d)", name, out_path, processed_count
            )
            if not errors_df.empty:
                err_out = out_path.with_suffix(".errors.csv.gz")
                saver(errors_df, err_out)
                logger.info(
                    "[%s] Saved canonicalisation errors to %s (n=%d)",
                    name,
                    err_out,
                    len(errors_df),
                )

        status = "success"
        message = "Processed OK"
    except Exception as exc:
        message = str(exc)
        logger.exception("[%s] Failed: %s", name, exc)

    end_t = pd.Timestamp.now().timestamp()
    time_s = round(end_t - start_t, 3)
    if len(message) > 400:
        message = message[:400] + "...(truncated)"

    return {
        "name": name,
        "src": url,
        "out": str(out_path),
        "status": status,
        "message": message,
        "input_size": None,
        "processed_items": (
            int(processed_count) if processed_count is not None else None
        ),
        "saved": (status == "success" and not dry_run),
        "time_s": time_s,
    }


def build_uspto_500(
    name: str,
    url: str,
    out_path: str | Path,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    saver = _get_saver()
    out_path = Path(out_path)
    status = "failed"
    message = ""
    processed_count: Optional[int] = None

    from synkit.Chem.Reaction.standardize import Standardize

    std = Standardize()
    start_t = pd.Timestamp.now().timestamp()

    try:
        logger.info("[%s] Downloading USPTO_500_MT reagents tarball", name)
        data = load_reagents_from_tar_url(url)

        logger.info("[%s] Combining reagents into table", name)
        results = combine_reagents_dict(data, std=std)

        # Add R-id and r_id
        results = results.reset_index(drop=True)
        results["R-id"] = [f"{name}_{i+1}" for i in results.index]
        results = _add_r_id_column(results, prefix=name)

        processed_count = len(results)
        if dry_run:
            logger.info(
                "[%s] dry-run: skipping save to %s (n=%d)",
                name,
                out_path,
                processed_count,
            )
        else:
            saver(results, out_path)
            logger.info(
                "[%s] Saved dataset to %s (n=%d)", name, out_path, processed_count
            )

        status = "success"
        message = "Processed OK"
    except Exception as exc:
        message = str(exc)
        logger.exception("[%s] Failed: %s", name, exc)

    end_t = pd.Timestamp.now().timestamp()
    time_s = round(end_t - start_t, 3)
    if len(message) > 400:
        message = message[:400] + "...(truncated)"

    return {
        "name": name,
        "src": url,
        "out": str(out_path),
        "status": status,
        "message": message,
        "input_size": None,
        "processed_items": (
            int(processed_count) if processed_count is not None else None
        ),
        "saved": (status == "success" and not dry_run),
        "time_s": time_s,
    }


# ---------------------------------------------------------------------------
# DEFAULT CONFIG
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "uspto_50k": {
        "kind": "uspto_50k",
        "src": (
            "https://www.dropbox.com/scl/fo/df10x2546d7a0483tousa/"
            "AGhjiD7hSUY4AmQJd3DrUQE/USPTO_50K_data"
            "?dl=0&rlkey=n2s3kn34bnfkzkmii4jeb9woy&subfolder_nav_tracking=1"
        ),
        "out": "Data/synthesis/uspto_50k.csv.gz",
    },
    "uspto_mit": {
        "kind": "uspto_mit",
        "src": "https://github.com/wengong-jin/nips17-rexgen/raw/refs/heads/master/USPTO/data.zip",
        "out": "Data/synthesis/uspto_mit.csv.gz",
    },
    "uspto_500": {
        "kind": "uspto_500",
        "src": "https://yzhang.hpc.nyu.edu/T5Chem/data/USPTO_500_MT.tar.bz2",
        "out": "Data/synthesis/uspto_500.csv.gz",
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
        description="Build synthesis datasets (USPTO 50k, MIT, USPTO 500)."
    )
    p.add_argument(
        "--entries",
        help="Comma-separated subset of datasets to build "
        "(default: all). Choices: uspto_50k,uspto_mit,uspto_500",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of jobs for canonicalisation (USPTO 50k / MIT).",
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
        default="reports/synthesis_summary.csv.gz",
        help="Path to write summary CSV (gzipped).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s - %(message)s",
    )
    logger.info("build_synthesis_dataset starting")

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
        src = info["src"]
        out = info["out"]

        logger.info("Building dataset '%s' (kind=%s)", name, kind)

        try:
            if kind == "uspto_50k":
                row = build_uspto_50k(
                    name=name,
                    url=src,
                    out_path=out,
                    n_jobs=args.n_jobs,
                    dry_run=args.dry_run,
                )
            elif kind == "uspto_mit":
                row = build_uspto_mit(
                    name=name,
                    url=src,
                    out_path=out,
                    n_jobs=args.n_jobs,
                    dry_run=args.dry_run,
                )
            elif kind == "uspto_500":
                row = build_uspto_500(
                    name=name,
                    url=src,
                    out_path=out,
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
                "src": src,
                "out": out,
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

    logger.info("build_synthesis_dataset finished")


if __name__ == "__main__":
    main()
