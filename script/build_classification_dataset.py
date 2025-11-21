#!/usr/bin/env python3
"""
build_classification_dataset.py

Build classification datasets from raw reaction CSVs, mirroring the logic
originally implemented in a Jupyter notebook.

Datasets covered (from DEFAULT_CONFIG):
 - USPTO_50k (balanced / unbalanced)
 - Schneider_50k (balanced / unbalanced)
 - USPTO_TPL (balanced / unbalanced)
 - SynTemp cluster dataset
 - ECREACT (Claire)

Each curated dataset is written as a gzipped CSV with a common schema:

  - For single-label datasets:
      columns: ["r_id", "rxn", "label", "split"]

  - For ECREACT:
      columns: ["r_id", "rxn", "ec1", "ec2", "ec3", "split"]

Usage examples
--------------
From repo root:

  PYTHONPATH=. python script/build_classification_dataset.py --dry-run
  PYTHONPATH=. python script/build_classification_dataset.py --entries uspto_50k_u,uspto_50k_b
  PYTHONPATH=. python script/build_classification_dataset.py --entries ecreact
  python script/build_classification_dataset.py --write-default classification_default.json

Ad-hoc example (non-default source):

  python script/build_classification_dataset.py \\
      --src path_or_url.csv.gz \\
      --out Data/classification/custom.csv.gz \\
      --curate uspto

"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional import of synrxn.io.io helpers
# ---------------------------------------------------------------------------

try:
    # When called from script/, this makes project root importable.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from synrxn.io.io import save_df_gz  # type: ignore
except Exception:
    save_df_gz = None  # fallback implemented below


def _fallback_save_df_gz(df, out_path: str) -> None:
    """Simple fallback if synrxn.io.io.save_df_gz is not available."""
    import pandas as pd  # noqa: F401  # only for type checking

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")


def _ensure_pandas():
    """Lazy import of pandas with a clearer error."""
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("pandas is required but could not be imported") from e
    return pd


# ---------------------------------------------------------------------------
# Shared small helpers (ported and slightly cleaned from the notebook)
# ---------------------------------------------------------------------------


def _extract_first_str(value: Any) -> Optional[str]:
    """
    Return the first non-empty string from value which may be:
      - a string
      - a list/tuple of strings
      - a stringified python list "['a','b']"
      - None / NaN -> returns None
    """
    pd = _ensure_pandas()

    # Missing / NaN
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    # List / tuple: first non-empty element
    if isinstance(value, (list, tuple)):
        for el in value:
            if el is None:
                continue
            s = str(el).strip()
            if s:
                return s
        return None

    # Strings: maybe literal list
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Try to parse python literal list/tuple e.g. "['a','b']"
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("(") and s.endswith(")")
        ):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    for el in parsed:
                        if el is None:
                            continue
                        ss = str(el).strip()
                        if ss:
                            return ss
                    return None
            except Exception:
                # fall back to raw string
                pass
        return s

    # Fallback: coerce to string
    s = str(value).strip()
    return s or None


def _first_present_column(df, candidates: Iterable[str]) -> Optional[str]:
    """
    Return the first candidate column name that exists in df.columns.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _try_int(v: Any) -> Optional[int]:
    """
    Attempt to cast v to int; return None if NaN or casting fails.
    """
    pd = _ensure_pandas()
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return int(v)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dataset-specific curation functions
# ---------------------------------------------------------------------------


def uspto_curate(
    df,
    *,
    id_col: str = "uspto_index",
    id_prefix: str = "USPTO",
    class_col: str = "new_class",
    reactions_col: str = "reactions",
    split_col: str = "split",
    class_map: Optional[Mapping[int, str]] = None,
    default_split: str = "train",
):
    """
    Curate a USPTO-style dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe (must contain `reactions` and `new_class` in typical usage).
    id_col : str
        Column to use for a numeric/index identifier; if missing, df.index is used.
    id_prefix : str
        Prefix for r_id (final id is f"{id_prefix}_{id_value}").
    class_col : str
        Column containing class labels (integers or strings).
    reactions_col : str
        Column containing reaction SMILES (renamed to `rxn`).
    split_col : str
        Optional column specifying split label.
    class_map : Mapping[int,str] or None
        Optional mapping from class int -> human-readable label.
    default_split : str
        Used when split_col is missing or NaN.

    Returns
    -------
    pandas.DataFrame
        Columns: ["r_id", "rxn", "label", "split"].
    """
    pd = _ensure_pandas()
    df_in = df.copy()

    # Determine id values
    if id_col in df_in.columns:
        id_values = df_in[id_col].astype(str)
    else:
        id_values = df_in.index.astype(str)

    R_ids = [f"{id_prefix}_{int(v)+1}" for v in id_values]

    # Reaction SMILES
    if reactions_col not in df_in.columns:
        raise ValueError(f"Input dataframe must contain a '{reactions_col}' column.")
    rxn_series = df_in[reactions_col].apply(_extract_first_str)

    # Class labels
    if class_col in df_in.columns:
        raw_labels = df_in[class_col]

        if class_map is not None:

            def _map_label(x: Any) -> Any:
                try:
                    # try integer key first
                    ix = int(x)
                    return class_map.get(ix, class_map.get(x, x))
                except Exception:
                    return class_map.get(x, x)

            label_series = raw_labels.map(_map_label)
        else:
            # Try to convert to int when possible, else keep as-is
            def _to_int_or_pass(x: Any) -> Any:
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return None
                try:
                    return int(x)
                except Exception:
                    return x

            label_series = raw_labels.map(_to_int_or_pass)
    else:
        label_series = pd.Series([None] * len(df_in), index=df_in.index)

    # Split handling
    if split_col in df_in.columns:
        split_series = df_in[split_col].fillna(default_split).astype(str)
    else:
        split_series = pd.Series([default_split] * len(df_in), index=df_in.index)

    out = pd.DataFrame(
        {
            "r_id": R_ids,
            "rxn": rxn_series,
            "label": label_series,
            "split": split_series,
        },
        index=df_in.index,
    )

    out = out[["r_id", "rxn", "label", "split"]].reset_index(drop=True)
    return out


def schneider_curate(
    df,
    *,
    id_col: str = "schneider_index",
    id_prefix: str = "sch",
    rxn_col: str = "rxn",
    split_col: str = "split",
    y_col: str = "y",
    class_map: Optional[Mapping[int, str]] = None,
    default_split: str = "train",
):
    """
    Curate Schneider-style classification data (balanced or unbalanced).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe; typically has columns such as 'rxn' or 'reactions' and 'y'.
    id_col : str
        Column providing numeric id; if missing, df.index is used.
    id_prefix : str
        Prefix for r_id.
    rxn_col : str
        Column name containing reaction SMILES.
    split_col : str
        Optional column specifying split.
    y_col : str
        Column name for labels.
    class_map : Mapping[int,str] or None
        Optional mapping from y -> human-readable label.
    default_split : str
        Split when split_col is missing/NaN.

    Returns
    -------
    pandas.DataFrame
        Columns: ["r_id", "rxn", "label", "split"].
    """
    pd = _ensure_pandas()
    df_in = df.copy()

    # ID / r_id
    if id_col in df_in.columns:
        id_values = df_in[id_col].astype(str)
    else:
        id_values = df_in.index.astype(str)

    R_ids = [f"{id_prefix}_{int(v)+1}" for v in id_values]

    # Reaction SMILES
    if rxn_col not in df_in.columns:
        raise ValueError(f"Input dataframe must contain a '{rxn_col}' column.")
    rxn_series = df_in[rxn_col].apply(_extract_first_str)

    # Labels
    if y_col in df_in.columns:
        raw_labels = df_in[y_col]

        if class_map is not None:

            def _map_label(x: Any) -> Any:
                try:
                    ix = int(x)
                    return class_map.get(ix, class_map.get(x, x))
                except Exception:
                    return class_map.get(x, x)

            label_series = raw_labels.map(_map_label)
        else:
            # try to cast to int when possible
            def _to_int_or_pass(x: Any) -> Any:
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return None
                try:
                    return int(x)
                except Exception:
                    return x

            label_series = raw_labels.map(_to_int_or_pass)
    else:
        label_series = pd.Series([None] * len(df_in), index=df_in.index)

    # Split
    if split_col in df_in.columns:
        split_series = df_in[split_col].fillna(default_split).astype(str)
    else:
        split_series = pd.Series([default_split] * len(df_in), index=df_in.index)

    out = pd.DataFrame(
        {
            "r_id": R_ids,
            "rxn": rxn_series,
            "label": label_series,
            "split": split_series,
        },
        index=df_in.index,
    )

    out = out[["r_id", "rxn", "label", "split"]].reset_index(drop=True)
    return out


def rxnclass_curate(
    df,
    *,
    id_col: Optional[str] = None,
    id_prefix: str = "RXN",
    rxn_col: str = "rxn",
    class_col: str = "rxn_class",
    split_col: str = "split",
    class_map: Optional[Mapping[int, str]] = None,
    default_split: str = "train",
    zero_pad: Optional[int] = None,
    keep_orig_index: bool = False,
):
    """
    Curate a reaction-classification dataframe (e.g. USPTO_TPL).

    Parameters
    ----------
    df : pandas.DataFrame
    id_col : str or None
        Optional explicit id column; if None or missing, df.index is used.
    id_prefix : str
        Prefix for generated r_id (default 'RXN').
    rxn_col : str
        Column containing reaction SMILES.
    class_col : str
        Column containing integer or encoded class labels.
    split_col : str
        Column for dataset split (train/valid/test); optional.
    class_map : Mapping[int,str] or None
        Optional mapping from integer label -> human-readable string.
    default_split : str
        Default split if split_col is missing.
    zero_pad : int or None
        If provided and id is numeric, zero-pad to this width.
    keep_orig_index : bool
        If True, keep original index in an 'orig_index' column.

    Returns
    -------
    pandas.DataFrame
        Columns: ["r_id", "rxn", "label", "split"] plus optional 'orig_index'.
    """
    pd = _ensure_pandas()
    df_in = df.copy()

    # ID / r_id
    if id_col is not None and id_col in df_in.columns:
        id_values = df_in[id_col]
    else:
        id_values = df_in.index

    def _fmt_id(v: Any) -> str:
        ni = _try_int(v)
        if ni is not None:
            if zero_pad is not None:
                return f"{ni:0{zero_pad}d}"
            return str(ni)
        return str(v)

    R_ids = [f"{id_prefix}_{int(_fmt_id(v))+1}" for v in id_values]

    # Reaction SMILES
    if rxn_col not in df_in.columns:
        raise ValueError(f"Input dataframe must contain a '{rxn_col}' column.")
    rxn_series = df_in[rxn_col].apply(_extract_first_str)

    # Labels
    if class_col in df_in.columns:
        raw_labels = df_in[class_col]

        if class_map is not None:

            def _map_label(x: Any) -> Any:
                try:
                    ix = int(x)
                    return class_map.get(ix, class_map.get(x, x))
                except Exception:
                    return class_map.get(x, x)

            label_series = raw_labels.map(_map_label)
        else:

            def _to_int_or_pass(x: Any) -> Any:
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return None
                try:
                    return int(x)
                except Exception:
                    return x

            label_series = raw_labels.map(_to_int_or_pass)
    else:
        label_series = pd.Series([None] * len(df_in), index=df_in.index)

    # Split
    if split_col in df_in.columns:
        split_series = df_in[split_col].fillna(default_split).astype(str)
    else:
        split_series = pd.Series([default_split] * len(df_in), index=df_in.index)

    out_dict = {
        "r_id": R_ids,
        "rxn": rxn_series,
        "label": label_series,
        "split": split_series,
    }

    if keep_orig_index:
        out_dict["orig_index"] = df_in.index

    out = pd.DataFrame(out_dict, index=df_in.index)
    cols = ["r_id", "rxn", "label", "split"]
    if keep_orig_index:
        cols.append("orig_index")
    out = out[cols].reset_index(drop=True)
    return out


def syntemp_curate(
    df,
    *,
    id_col_candidates: Optional[List[str]] = None,
    rsmicol_candidates: Optional[List[str]] = None,
    newr0_candidates: Optional[List[str]] = None,
    newr1_candidates: Optional[List[str]] = None,
    newr2_candidates: Optional[List[str]] = None,
    id_prefix: str = "syntemp",
    zero_pad: Optional[int] = None,
    keep_orig_index: bool = False,
):
    """
    Curate a 'syntemp' DataFrame into columns:
        ["r_id", "rxn", "label_0", "label_1", "label_2"]

    Expected columns in raw SynTemp:
        - RSMI / RsmI / R_smI / rxn / reaction
        - New_R0 / New_R1 / New_R2  (or variants)
        - R_ID / r_id / R_ID / index
    """
    pd = _ensure_pandas()
    df_in = df.copy()

    # --- Default candidate lists ------------------------------
    if id_col_candidates is None:
        id_col_candidates = ["R_ID", "r_id", "r_id", "R_ID", "id", "index"]
    if rsmicol_candidates is None:
        rsmicol_candidates = [
            "RSMI",
            "RsmI",
            "R_smI",
            "rsmi",
            "rxn",
            "reaction",
            "RSmiles",
        ]
    if newr0_candidates is None:
        newr0_candidates = ["New_R0", "NewR0", "New_R_0", "New0"]
    if newr1_candidates is None:
        newr1_candidates = ["New_R1", "NewR1", "New_R_1", "New1"]
    if newr2_candidates is None:
        newr2_candidates = ["New_R2", "NewR2", "New_R_2", "New2"]

    # --- Select columns -----------------------------------------
    id_col = _first_present_column(df_in, id_col_candidates)
    rsmicol = _first_present_column(df_in, rsmicol_candidates)
    c_newr0 = _first_present_column(df_in, newr0_candidates)
    c_newr1 = _first_present_column(df_in, newr1_candidates)
    c_newr2 = _first_present_column(df_in, newr2_candidates)

    # ---- Prepare ID values -------------------------------------
    if id_col is not None:
        id_vals = df_in[id_col].astype(str)
    else:
        id_vals = df_in.index.astype(str)

    def _maybe_pad(v: str) -> str:
        if zero_pad is None:
            return v
        try:
            return str(int(v)).zfill(zero_pad)
        except Exception:
            return v

    R_ids = [f"{id_prefix}_{_maybe_pad(v)}" for v in id_vals]

    # ---- Reaction SMILES ---------------------------------------
    if rsmicol is None:
        # Try a lower-case fallback
        fallback = [c for c in df_in.columns if c.lower() == "rsmi"]
        rsmicol = fallback[0] if fallback else None

    if rsmicol is None:
        raise ValueError(f"No reaction SMILES column found among: {rsmicol_candidates}")

    rxn_series = df_in[rsmicol].apply(_extract_first_str)

    # ---- Extract labels -----------------------------------------
    def _get_label(col):
        if col is None or col not in df_in.columns:
            return pd.Series([None] * len(df_in))
        extracted = df_in[col].map(_extract_first_str)
        return extracted.map(_try_int)

    label_0 = _get_label(c_newr0)
    label_1 = _get_label(c_newr1)
    label_2 = _get_label(c_newr2)

    # ---- Build output -------------------------------------------
    out = pd.DataFrame(
        {
            "r_id": R_ids,
            "rxn": rxn_series,
            "label_0": label_0,
            "label_1": label_1,
            "label_2": label_2,
        }
    )

    if keep_orig_index:
        out.insert(0, "orig_index", df_in.index)

    return out.reset_index(drop=True)


def claire_curate(
    df,
    *,
    id_col_candidates: Optional[Sequence[str]] = None,
    rxn_col_candidates: Optional[Sequence[str]] = None,
    ec1_col_candidates: Optional[Sequence[str]] = None,
    ec2_col_candidates: Optional[Sequence[str]] = None,
    ec3_col_candidates: Optional[Sequence[str]] = None,
    id_prefix: str = "ecreact",
    zero_pad: Optional[int] = None,
    default_split: str = "train",
    keep_orig_index: bool = False,
):
    """
    Curate the ECREACT (Claire) dataset into multi-task EC classification.

    This follows the structure in the notebook:
    - find 'rxn_smiles'/'rxn' as reaction column
    - find EC encoding columns (ec1encode/ec2_encode/ec3encode or similar)
    - produce columns: ["r_id","rxn","ec1","ec2","ec3","split"]

    Parameters
    ----------
    df : pandas.DataFrame
    id_col_candidates : sequence[str] or None
        Candidate ID columns; if None, defaults to ["id", "index", "orig_index"].
    rxn_col_candidates : sequence[str] or None
        Candidate reaction SMILES columns; defaults to ["rxn_smiles","rxn","reaction"].
    ec1_col_candidates/ec2_col_candidates/ec3_col_candidates : sequence[str] or None
        Candidate names for EC encodings. Defaults prefer "*encode" then plain "ec1".
    id_prefix : str
        Prefix for r_id (default 'ecreact').
    zero_pad : int or None
        If provided and id is numeric, zero-pad to this width.
    default_split : str
        Default split when no split column present.
    keep_orig_index : bool
        If True, keep 'orig_index' column.

    Returns
    -------
    pandas.DataFrame
        Columns: ["r_id","rxn","ec1","ec2","ec3","split"] (+ optional 'orig_index').
    """
    pd = _ensure_pandas()
    df_in = df.copy()

    if id_col_candidates is None:
        id_col_candidates = ("id", "index", "orig_index")
    if rxn_col_candidates is None:
        rxn_col_candidates = ("rxn_smiles", "rxn", "reaction")
    if ec1_col_candidates is None:
        ec1_col_candidates = ("ec1encode", "ec1_encode", "ec1")
    if ec2_col_candidates is None:
        ec2_col_candidates = ("ec2encode", "ec2_encode", "ec2")
    if ec3_col_candidates is None:
        ec3_col_candidates = ("ec3encode", "ec3_encode", "ec3")

    id_col = _first_present_column(df_in, id_col_candidates)
    rxn_col = _first_present_column(df_in, rxn_col_candidates)

    if rxn_col is None:
        raise ValueError(f"No reaction SMILES column found among {rxn_col_candidates}")

    # IDs
    if id_col is None:
        id_values = df_in.index
    else:
        id_values = df_in[id_col]

    def _fmt_id(v: Any) -> str:
        ni = _try_int(v)
        if ni is not None:
            if zero_pad is not None:
                return f"{ni:0{zero_pad}d}"
            return str(ni)
        return str(v)

    R_ids = [f"{id_prefix}_{int(_fmt_id(v))+1}" for v in id_values]

    rxn_series = df_in[rxn_col].apply(_extract_first_str)

    # EC label columns
    ec1_col = _first_present_column(df_in, ec1_col_candidates)
    ec2_col = _first_present_column(df_in, ec2_col_candidates)
    ec3_col = _first_present_column(df_in, ec3_col_candidates)

    if ec1_col is not None:
        ec1 = df_in[ec1_col].map(_try_int)
    else:
        ec1 = pd.Series([None] * len(df_in), index=df_in.index)

    if ec2_col is not None:
        ec2 = df_in[ec2_col].map(_try_int)
    else:
        ec2 = pd.Series([None] * len(df_in), index=df_in.index)

    if ec3_col is not None:
        ec3 = df_in[ec3_col].map(_try_int)
    else:
        ec3 = pd.Series([None] * len(df_in), index=df_in.index)

    # Split (if present)
    if "split" in df_in.columns:
        split_series = df_in["split"].fillna(default_split).astype(str)
    else:
        split_series = pd.Series([default_split] * len(df_in), index=df_in.index)

    out_dict = {
        "r_id": R_ids,
        "rxn": rxn_series,
        "ec1": ec1,
        "ec2": ec2,
        "ec3": ec3,
        "split": split_series,
    }
    if keep_orig_index:
        out_dict["orig_index"] = df_in.index

    final_cols = ["r_id", "rxn", "ec1", "ec2", "ec3", "split"]
    if keep_orig_index:
        final_cols.append("orig_index")

    out = pd.DataFrame(out_dict, index=df_in.index)
    out = out[final_cols].reset_index(drop=True)
    return out


# Registry to look up curator functions by short name
CURATORS = {
    "uspto": uspto_curate,
    "schneider": schneider_curate,
    "tpl": rxnclass_curate,
    "rxnclass": rxnclass_curate,
    "syntemp": syntemp_curate,
    "ecreact": claire_curate,
    "claire": claire_curate,
}


# ---------------------------------------------------------------------------
# DEFAULT CONFIG (built-in)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    # 1. USPTO 50k (balanced / unbalanced)
    "uspto_50k_u": {
        "src": "https://raw.githubusercontent.com/phuocchung123/SynCat/main/Data/raw/USPTO_50k_unbalanced.csv.gz",
        "out": "Data/classification/uspto_50k_u.csv.gz",
        "curate": "uspto",
        "curate_kwargs": {},
    },
    "uspto_50k_b": {
        "src": "https://raw.githubusercontent.com/phuocchung123/SynCat/main/Data/raw/USPTO_50k_balanced.csv.gz",
        "out": "Data/classification/uspto_50k_b.csv.gz",
        "curate": "uspto",
        "curate_kwargs": {},
    },
    # 2. Schneider 50k
    "schneider_u": {
        "src": "https://raw.githubusercontent.com/phuocchung123/SynCat/main/Data/raw/schneider50k_unbalanced.csv.gz",
        "out": "Data/classification/schneider_u.csv.gz",
        "curate": "schneider",
        "curate_kwargs": {},  # uses default rxn_col="rxn"
    },
    "schneider_b": {
        "src": "https://raw.githubusercontent.com/phuocchung123/SynCat/main/Data/raw/schneider50k_balanced.csv.gz",
        "out": "Data/classification/schneider_b.csv.gz",
        "curate": "schneider",
        "curate_kwargs": {"rxn_col": "reactions"},
    },
    # 3. USPTO_TPL (balanced / unbalanced)
    "tpl_u": {
        "src": "https://raw.githubusercontent.com/phuocchung123/SynCat/main/Data/raw/USPTO_TPL_unbalanced.csv.gz",
        "out": "Data/classification/tpl_u.csv.gz",
        "curate": "tpl",
        "curate_kwargs": {"id_prefix": "tpl", "zero_pad": 6},
    },
    "tpl_b": {
        "src": "https://raw.githubusercontent.com/phuocchung123/SynCat/main/Data/raw/USPTO_TPL_balanced.csv.gz",
        "out": "Data/classification/tpl_b.csv.gz",
        "curate": "tpl",
        "curate_kwargs": {"id_prefix": "tpl", "zero_pad": 6},
    },
    # 4. SynTemp
    "syntemp": {
        "src": "https://raw.githubusercontent.com/phuocchung123/SynCat/main/Data/raw/Syntemp_cluster.csv.gz",
        "out": "Data/classification/syntemp.csv.gz",
        "curate": "syntemp",
        "curate_kwargs": {"keep_orig_index": True},
    },
    # 5. ECREACT (Claire)
    "ecreact": {
        "src": "https://raw.githubusercontent.com/phuocchung123/SynCat/main/Data/raw/claire_full.csv.gz",
        "out": "Data/classification/ecreact.csv.gz",
        "curate": "ecreact",
        "curate_kwargs": {"keep_orig_index": True},
    },
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_csv_gz(url_or_path: str, *, low_memory: bool = False):
    """
    Load a CSV (optionally gzipped) from a local path or URL using pandas.
    """
    pd = _ensure_pandas()
    return pd.read_csv(url_or_path, compression="gzip", low_memory=low_memory)


def load_config_file(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load a JSON or YAML configuration file describing datasets.

    Expected structure:
      {
        "key": {
          "src": "...",
          "out": "...",
          "curate": "uspto",
          "curate_kwargs": {...}
        },
        ...
      }
    """
    text = path.read_text(encoding="utf-8")
    try:
        cfg = json.loads(text)
        if not isinstance(cfg, dict):
            raise ValueError("Top-level config must be an object/dict.")
        return cfg
    except Exception:
        try:
            import yaml  # type: ignore

            cfg = yaml.safe_load(text)
            if not isinstance(cfg, dict):
                raise ValueError("Top-level config must be an object/dict.")
            return cfg
        except Exception as e:
            raise RuntimeError("Config must be valid JSON or YAML") from e


def _print_summary_table(results):
    """
    Pretty print a summary table using tabulate if available, otherwise pandas.
    """
    cols = [
        "name",
        "status",
        "input_size",
        "processed_items",
        "saved",
        "time_s",
        "message",
    ]

    try:
        from tabulate import tabulate  # type: ignore

        table = [[r.get(c) for c in cols] for r in results]
        print(tabulate(table, headers=cols, tablefmt="github"))
    except Exception:
        pd = _ensure_pandas()
        df = pd.DataFrame(results)
        print(df[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_single_entry(
    name: str,
    src: str,
    out: str,
    curate_name: str,
    curate_kwargs: Optional[Dict[str, Any]] = None,
    *,
    dry_run: bool = False,
    retries: int = 2,
) -> Dict[str, Any]:
    """
    Load -> curate -> save and return a result dict:

    {
        "name", "src", "out", "status", "message",
        "input_size", "processed_items", "saved", "time_s"
    }
    """
    pd = _ensure_pandas()
    saver = save_df_gz or _fallback_save_df_gz
    curate_fn = CURATORS.get(curate_name)
    if curate_fn is None:
        raise ValueError(f"Unknown curator name: {curate_name}")

    if curate_kwargs is None:
        curate_kwargs = {}

    start = time.time()
    input_size: Optional[int] = None
    processed_count: Optional[int] = None
    saved = False
    status = "failed"
    message = ""
    last_exc: Optional[BaseException] = None

    for attempt in range(1, retries + 1):
        try:
            logger.info(
                "[%s] Loading source %s (attempt %d/%d)...",
                name,
                src,
                attempt,
                retries,
            )
            df_in = _load_csv_gz(src, low_memory=False)

            try:
                input_size = int(len(df_in))
            except Exception:
                input_size = None

            logger.info("[%s] Running %s(...)", name, curate_name)
            df_out = curate_fn(df_in, **curate_kwargs)

            processed_count = len(df_out)
            logger.info("[%s] Curator produced %d rows", name, processed_count)

            if dry_run:
                logger.info("[%s] dry-run: skipping save to %s", name, out)
                saved = False
            else:
                saver(df_out, out)
                saved = True
                logger.info("[%s] Saved curated dataset to %s", name, out)

            status = "success"
            message = "Processed OK"
            break

        except Exception as exc:
            last_exc = exc
            message = str(exc)
            logger.exception("[%s] attempt %d failed: %s", name, attempt, exc)

    end = time.time()
    time_s = round(end - start, 3)

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
            and not isinstance(input_size, float)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Build classification datasets (load CSV -> curate -> save)."
    )
    p.add_argument(
        "--config",
        help="JSON/YAML config file overriding DEFAULT_CONFIG.",
    )
    p.add_argument(
        "--src",
        help="Process a single source URL or local path (overrides config).",
    )
    p.add_argument("--out", help="Output path for --src (required with --src).")
    p.add_argument(
        "--curate",
        choices=sorted(CURATORS.keys()),
        help="Curator name to use for --src (e.g. uspto, schneider, tpl, syntemp, ecreact).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do everything except saving files.",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries for load/curate (default: 2).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG/INFO/WARNING/ERROR).",
    )
    p.add_argument(
        "--write-default",
        help="Write the builtin DEFAULT_CONFIG JSON (without functions) to this path and exit.",
    )
    p.add_argument(
        "--entries",
        help="Comma-separated subset of config keys to process (default: all).",
    )
    p.add_argument(
        "--summary-out",
        default="reports/classification_summary.csv.gz",
        help="Path to write summary CSV (gzipped).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s - %(message)s",
    )
    logger.info("build_classification_dataset starting")

    # Write default config (only serializable bits)
    if args.write_default:
        outp = Path(args.write_default)
        outp.parent.mkdir(parents=True, exist_ok=True)

        serializable_cfg: Dict[str, Dict[str, Any]] = {}
        for name, info in DEFAULT_CONFIG.items():
            serializable_cfg[name] = {
                "src": info.get("src"),
                "out": info.get("out"),
                "curate": info.get("curate"),
                "curate_kwargs": info.get("curate_kwargs", {}),
            }

        outp.write_text(json.dumps(serializable_cfg, indent=2), encoding="utf-8")
        logger.info("Wrote default config to %s", outp)
        return

    results = []

    # Single-source quick path
    if args.src:
        if not args.out:
            raise SystemExit("When --src is provided you must also provide --out")
        if not args.curate:
            raise SystemExit(
                "When --src is provided you must also specify --curate "
                f"from: {sorted(CURATORS.keys())}"
            )

        res = process_single_entry(
            "single",
            args.src,
            args.out,
            args.curate,
            curate_kwargs={},
            dry_run=args.dry_run,
            retries=args.retries,
        )
        results.append(res)

    else:
        # Use config (either external or built-in)
        if args.config:
            cfg_path = Path(args.config)
            if not cfg_path.exists():
                raise SystemExit(f"Config file not found: {cfg_path}")
            cfg = load_config_file(cfg_path)
        else:
            cfg = DEFAULT_CONFIG
            logger.info(
                "No config provided: using DEFAULT_CONFIG with %d entries",
                len(cfg),
            )

        entries: Optional[Sequence[str]] = None
        if args.entries:
            entries = [e.strip() for e in args.entries.split(",") if e.strip()]
            logger.info("Processing subset entries: %s", entries)

        for name, info in cfg.items():
            if entries and name not in entries:
                logger.debug("Skipping %s (not requested)", name)
                continue

            if not isinstance(info, dict) or "src" not in info:
                logger.warning(
                    "Skipping invalid config entry %s (expected dict with 'src')",
                    name,
                )
                continue

            src = info["src"]
            out = info.get("out") or f"Data/classification/{name}.csv.gz"
            curate_name = info.get("curate")
            if not curate_name:
                logger.warning("Skipping %s: no 'curate' field in config entry", name)
                continue
            curate_kwargs = info.get("curate_kwargs") or {}

            logger.info(
                "Entry '%s': %s -> %s (curate=%s)",
                name,
                src,
                out,
                curate_name,
            )

            try:
                res = process_single_entry(
                    name,
                    src,
                    out,
                    curate_name,
                    curate_kwargs=curate_kwargs,
                    dry_run=args.dry_run,
                    retries=args.retries,
                )
            except Exception as exc:
                logger.exception("Failed to process entry %s: %s", name, exc)
                msg = str(exc)
                if len(msg) > 400:
                    msg = msg[:400] + "...(truncated)"
                res = {
                    "name": name,
                    "src": src,
                    "out": out,
                    "status": "failed",
                    "message": msg,
                    "input_size": None,
                    "processed_items": None,
                    "saved": False,
                    "time_s": None,
                }

            results.append(res)

    # Print and save summary
    _print_summary_table(results)

    try:
        pd = _ensure_pandas()
        summary_df = pd.DataFrame(results)
        summary_out = Path(args.summary_out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_out, index=False, compression="gzip")
        logger.info("Wrote summary to %s", summary_out)
    except Exception:
        logger.exception("Failed to write summary CSV; printing only.")

    logger.info("build_classification_dataset finished")


if __name__ == "__main__":
    main()