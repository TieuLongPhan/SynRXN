#!/usr/bin/env python3
"""
build_rbl_dataset.py

End-to-end pipeline for SynRBL-based reaction balancing:

1. Load a raw reaction CSV (local path or URL).
2. Run `ReactionRebalancer` from `synrbl` on all reactions.
3. Clean the resulting records into tidy DataFrames:
      - main:      R-id, rxn, ground_truth, dataset, error
      - subsets:   MOS, MBS, MNC (classification of successful rebalancings)
4. Build a "complex" benchmark dataset from the SynRBL validation set
   (golden_dataset + Jaworski).

Outputs (default, from repo root):
  - Data/rbl/uspto_50k_clean.csv.gz
  - Data/rbl/mos.csv.gz
  - Data/rbl/mbs.csv.gz
  - Data/rbl/mnc.csv.gz
  - Data/rbl/complex.csv.gz

Usage
-----
From repo root:

  PYTHONPATH=. python script/build_rbl_dataset.py

You can still use:
  --src / --out / --reaction-col      for ad-hoc data
  --config / --entries                for multi-entry configs
  --dry-run                           to skip saving
"""

from __future__ import annotations

import argparse
import ast
import logging
import math
import sys
import time
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional project imports
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

try:
    from synkit.IO import configure_warnings_and_logs  # type: ignore
except Exception:
    configure_warnings_and_logs = None

try:
    from synrbl import ReactionRebalancer, RebalanceConfig  # type: ignore
except Exception:
    ReactionRebalancer = None
    RebalanceConfig = None


VALIDATION_SET_URL = (
    "https://raw.githubusercontent.com/TieuLongPhan/SynRBL/refs/heads/main/"
    "Data/Validation_set/validation_set.csv"
)


def _ensure_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("pandas is required but could not be imported") from e
    return pd


def _ensure_rdkit():
    try:
        from rdkit import Chem  # type: ignore
    except Exception as e:
        raise RuntimeError("rdkit is required but could not be imported") from e
    return Chem


# ---------------------------------------------------------------------------
# Helpers (normalization, reaction analysis)
# ---------------------------------------------------------------------------


def _normalize_to_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        # Try Python literal list/tuple
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("(") and s.endswith(")")
        ):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set)):
                    return [str(v) for v in parsed]
            except Exception:
                pass
        return [s]
    return [str(value)]


def _get_first_present(rec: Mapping[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in rec:
            v = rec[k]
            if v is None:
                continue
            if isinstance(v, str):
                s = v.strip()
                if s:
                    return s
            else:
                return v
    return None


def _split_reaction(reaction: str) -> Tuple[List[str], List[str]]:
    if ">>" in reaction:
        left, right = reaction.split(">>", 1)
    elif ">" in reaction:
        left, right = reaction.split(">", 1)
    else:
        raise ValueError("Reaction string must contain '>' or '>>' as separator.")
    left_mols = [s.strip() for s in left.split(".") if s.strip()]
    right_mols = [s.strip() for s in right.split(".") if s.strip()]
    return left_mols, right_mols


def _mol_element_counts(smiles: str) -> Dict[str, int]:
    Chem = _ensure_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    mol_h = Chem.AddHs(mol)
    counts: Dict[str, int] = defaultdict(int)
    for a in mol_h.GetAtoms():
        counts[a.GetSymbol()] += 1
    return dict(counts)


def _sum_counts(smiles_list: List[str]) -> Dict[str, int]:
    total: Dict[str, int] = defaultdict(int)
    for s in smiles_list:
        c = _mol_element_counts(s)
        for el, n in c.items():
            total[el] += n
    return dict(total)


def reaction_missing_side(reaction: str) -> str:
    """
    Analyze reaction SMILES and return one of:
      - "one-side"  : all non-zero element differences have the same sign
      - "both"      : some elements more on left and some more on right
      - "balanced"  : no element differences
    """
    left_mols, right_mols = _split_reaction(reaction)
    left_counts = _sum_counts(left_mols)
    right_counts = _sum_counts(right_mols)

    elems = set(left_counts) | set(right_counts)
    diff = {el: left_counts.get(el, 0) - right_counts.get(el, 0) for el in elems}

    pos = any(v > 0 for v in diff.values())
    neg = any(v < 0 for v in diff.values())

    if not pos and not neg:
        return "balanced"
    if pos and neg:
        return "both"
    return "one-side"


def _reaction_element_delta(reaction: str) -> Dict[str, int]:
    left, right = _split_reaction(reaction)
    total_l: Dict[str, int] = defaultdict(int)
    total_r: Dict[str, int] = defaultdict(int)

    for smi in left:
        c = _mol_element_counts(smi)
        for e, v in c.items():
            total_l[e] += v

    for smi in right:
        c = _mol_element_counts(smi)
        for e, v in c.items():
            total_r[e] += v

    all_elems = set(total_l.keys()) | set(total_r.keys())
    delta: Dict[str, int] = {}
    for e in all_elems:
        delta[e] = total_r.get(e, 0) - total_l.get(e, 0)
    return delta


def _is_balanced_stoichiometry(reaction: str) -> bool:
    delta = _reaction_element_delta(reaction)
    return all(v == 0 for v in delta.values())


# ---------------------------------------------------------------------------
# Curating SynRBL records
# ---------------------------------------------------------------------------


def curate_records(records: List[Dict[str, Any]]):
    """
    Curate a list of SynRBL reaction-record dicts into a DataFrame with
    columns ["R-id","rxn","ground_truth"].
    """
    pd = _ensure_pandas()
    curated: List[Dict[str, Any]] = []
    for rec in records:
        rid = _get_first_present(rec, ["R-id", "R_id", "rid", "id"])
        if rid is None:
            continue
        rxn = _get_first_present(rec, ["input_reaction", "reactions", "rxn"])
        ground_truth = _get_first_present(
            rec,
            [
                "standardized_reactions",
                "standardized_reaction",
                "new_products",
                "ground_truth",
            ],
        )
        curated.append({"R-id": rid, "rxn": rxn, "ground_truth": ground_truth})

    df = pd.DataFrame(curated)
    # Ensure columns even if empty
    cols = ["R-id", "rxn", "ground_truth"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    return df


def _add_r_id_column(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Add 'r_id' column with values prefix_{index+1} if not already present.
    Operates on a copy and returns it.
    """
    if "r_id" in df.columns:
        return df
    df = df.copy()
    # df["r_id"] = [f"{prefix}_{i+1}" for i in range(len(df))]
    df.insert(0, "r_id", [f"{prefix}_{i+1}" for i in range(len(df))])
    df.drop(columns=["R-id"], inplace=True)
    df.drop_duplicates(subset=["ground_truth"], inplace=True)
    return df


def clean_synrbl(
    data: Any,
    *,
    dataset_include: Optional[Sequence[str]] = None,
    dataset_exclude: Optional[Sequence[str]] = None,
    dataset_col_candidates: Sequence[str] = (
        "dataset",
        "dataset_name",
        "source",
        "origin",
        "subset",
    ),
):
    """
    Clean SynRBL rebalance results into a tidy DataFrame.

    `data` may be:
      - pandas.DataFrame
      - list[dict]
      - dict with "records" / "data"

    Returns a DataFrame with columns:
      ["R-id", "rxn", "ground_truth", "dataset", "error"]
    """
    pd = _ensure_pandas()

    # Normalize to list-of-dicts
    if isinstance(data, pd.DataFrame):
        records = data.to_dict("records")
    elif isinstance(data, list):
        records = list(data)
    elif isinstance(data, dict):
        if "records" in data and isinstance(data["records"], list):
            records = list(data["records"])
        elif "data" in data and isinstance(data["data"], list):
            records = list(data["data"])
        else:
            records = [data]
    else:
        raise TypeError(f"Unsupported data type for clean_synrbl: {type(data)}")

    dataset_include_norm = (
        set(_normalize_to_list(dataset_include))
        if dataset_include is not None
        else None
    )
    dataset_exclude_norm = (
        set(_normalize_to_list(dataset_exclude))
        if dataset_exclude is not None
        else None
    )

    curated: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        # Dataset label (value, not key)
        dataset_val: Optional[str] = None
        for k in dataset_col_candidates:
            if k in rec and rec[k] not in (None, ""):
                dataset_val = str(rec[k]).strip()
                break

        if dataset_include_norm is not None and dataset_val not in dataset_include_norm:
            continue
        if dataset_exclude_norm is not None and dataset_val in dataset_exclude_norm:
            continue

        # ID
        rid = _get_first_present(
            rec,
            ["R-id", "R_ID", "R_id", "r_id", "id"],
        )
        if rid is None:
            rid = f"R_{idx}"

        # Input reaction
        rxn = _get_first_present(
            rec,
            ["input_reaction", "reactions", "rxn", "reaction_smiles", "reaction"],
        )

        # Ground-truth / balanced reaction
        ground_truth = _get_first_present(
            rec,
            [
                "standardized_reactions",
                "standardized_reaction",
                "balanced_reaction",
                "new_products",
                "ground_truth",
                "expected_reaction",
            ],
        )

        # Build an error message (if any)
        error_msg_parts: List[str] = []

        if rxn is None:
            error_msg_parts.append("missing input_reaction")
        else:
            try:
                if not _is_balanced_stoichiometry(rxn):
                    error_msg_parts.append("input_reaction not element-balanced")
            except Exception as e:
                error_msg_parts.append(f"input_reaction parse error: {e}")

        if ground_truth is not None:
            try:
                if not _is_balanced_stoichiometry(ground_truth):
                    error_msg_parts.append("ground_truth not element-balanced")
            except Exception as e:
                error_msg_parts.append(f"ground_truth parse error: {e}")
        else:
            error_msg_parts.append("missing ground_truth")

        error_msg = "; ".join(error_msg_parts) if error_msg_parts else None

        curated.append(
            {
                "R-id": rid,
                "rxn": rxn,
                "ground_truth": ground_truth,
                "dataset": dataset_val,
                "error": error_msg,
            }
        )

    result_df = pd.DataFrame(curated)
    cols_order = ["R-id", "rxn", "ground_truth", "dataset", "error"]
    for c in cols_order:
        if c not in result_df.columns:
            result_df[c] = None
    result_df = result_df[cols_order].reset_index(drop=True)
    return result_df


# ---------------------------------------------------------------------------
# DEFAULT CONFIG
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "uspto_50k": {
        "src": (
            "https://raw.githubusercontent.com/TieuLongPhan/SynRBL/refs/heads/main/"
            "Data/Raw_data/USPTO/USPTO_50K.csv"
        ),
        "out": "Data/rbl/uspto_50k_clean.csv.gz",
        "reaction_col": "reactions",
        "make_subsets": True,
        "dataset_include": None,
    },
}


# ---------------------------------------------------------------------------
# Core pipeline: run SynRBL + clean + subsets
# ---------------------------------------------------------------------------


def _load_raw_csv(url_or_path: str):
    pd = _ensure_pandas()
    return pd.read_csv(url_or_path)


def _run_synrbl_on_df(
    df,
    *,
    reaction_col: str,
    id_col: str = "R-id",
    id_prefix: str = "R",
    n_jobs: int = 2,
    batch_size: int = 500,
    enable_logging: bool = False,
    use_default_reduction: bool = True,
) -> List[Dict[str, Any]]:
    if ReactionRebalancer is None or RebalanceConfig is None:
        raise RuntimeError(
            "synrbl is not importable. Please install it (e.g. `pip install synrbl`)."
        )

    pd = _ensure_pandas()

    df_in = df.copy()
    if id_col not in df_in.columns:
        df_in[id_col] = [f"{id_prefix}_{i}" for i in range(len(df_in))]

    records = df_in.to_dict("records")

    config = RebalanceConfig(
        reaction_col=reaction_col,
        id_col=id_col,
        n_jobs=n_jobs,
        batch_size=batch_size,
        enable_logging=enable_logging,
        use_default_reduction=use_default_reduction,
    )

    user_logger = None
    if configure_warnings_and_logs is not None:
        configure_warnings_and_logs(True, True)

    rebalancer = ReactionRebalancer(config=config, user_logger=user_logger)
    result_records = rebalancer.rebalance(records, keep_extra=True)

    if isinstance(result_records, pd.DataFrame):
        return result_records.to_dict("records")
    return list(result_records)


def _fallback_save_df_gz(df, out_path: str | Path) -> None:
    pd = _ensure_pandas()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")


def build_subsets_from_rbl_records(
    rbl_records: List[Dict[str, Any]],
    name: str,
) -> Tuple[Any, Any, Any]:
    """
    Build MOS / MBS / MNC splits from SynRBL records.

    - carbon: success & solved_by == 'mcs-based' & confidence > 0.9
    - mnc:    success & solved_by == 'rule-based'
    - mos:    subset of carbon with missing on one side
    - mbs:    remaining carbon (both / balanced)
    """
    pd = _ensure_pandas()

    carbon: List[Dict[str, Any]] = []
    mnc: List[Dict[str, Any]] = []
    mos: List[Dict[str, Any]] = []
    mbs: List[Dict[str, Any]] = []

    for value in rbl_records:
        if not value.get("success"):
            continue
        solved_by = value.get("solved_by")
        if solved_by == "mcs-based":
            conf = value.get("confidence", 0.0)
            try:
                conf_val = float(conf)
            except Exception:
                conf_val = 0.0
            if conf_val > 0.9:
                carbon.append(value)
        elif solved_by == "rule-based":
            mnc.append(value)

    for value in carbon:
        rxn = value.get("input_reaction")
        if not isinstance(rxn, str) or not rxn.strip():
            continue
        side = reaction_missing_side(rxn)
        if side == "one-side":
            mos.append(value)
        else:
            mbs.append(value)

    mos_df = curate_records(mos)
    mbs_df = curate_records(mbs)
    mnc_df = curate_records(mnc)

    # Add r_id with dataset-specific prefix
    mos_df = _add_r_id_column(mos_df, prefix="mos")
    mbs_df = _add_r_id_column(mbs_df, prefix="mbs")
    mnc_df = _add_r_id_column(mnc_df, prefix="mnc")

    logger.info(
        "Subsets: MOS=%d, MBS=%d, MNC=%d",
        len(mos_df),
        len(mbs_df),
        len(mnc_df),
    )
    return mos_df, mbs_df, mnc_df


def process_single_entry(
    name: str,
    src: str,
    out: str,
    reaction_col: Optional[str],
    dataset_include: Optional[Sequence[str]] = None,
    *,
    make_subsets: bool = False,
    dry_run: bool = False,
    retries: int = 1,
) -> List[Dict[str, Any]]:
    """
    Run pipeline (load -> SynRBL -> clean -> save) for one config entry.

    Returns a list of result rows:
      - main dataset (name)
      - optionally: name_mos, name_mbs, name_mnc
    """
    saver = save_df_gz or _fallback_save_df_gz
    pd = _ensure_pandas()

    if reaction_col is None:
        raise ValueError(
            f"Config entry '{name}' does not define 'reaction_col'. "
            "Please specify it explicitly."
        )

    start = time.time()
    input_size: Optional[int] = None
    processed_count: Optional[int] = None
    status = "failed"
    message = ""
    saved_main = False

    mos_df = mbs_df = mnc_df = None
    mos_path = mbs_path = mnc_path = None

    for attempt in range(1, retries + 1):
        try:
            logger.info(
                "[%s] Loading raw CSV %s (attempt %d/%d)...",
                name,
                src,
                attempt,
                retries,
            )
            df_raw = _load_raw_csv(src)
            input_size = int(len(df_raw))

            logger.info("[%s] Running SynRBL rebalancing", name)
            rbl_records = _run_synrbl_on_df(df_raw, reaction_col=reaction_col)

            logger.info("[%s] Cleaning SynRBL records", name)
            df_clean = clean_synrbl(
                rbl_records,
                dataset_include=dataset_include,
            )
            processed_count = len(df_clean)

            # add r_id to main dataset
            df_clean = _add_r_id_column(df_clean, prefix=name)

            if dry_run:
                logger.info("[%s] dry-run: skipping save to %s", name, out)
            else:
                # saver(df_clean, out)
                saved_main = True
                logger.info("[%s] Saved cleaned dataset to %s", name, out)

                if make_subsets:
                    subset_dir = Path("Data/rbl")
                    subset_dir.mkdir(parents=True, exist_ok=True)

                    mos_df, mbs_df, mnc_df = build_subsets_from_rbl_records(
                        rbl_records,
                        name=name,
                    )

                    mos_path = subset_dir / "mos.csv.gz"
                    mbs_path = subset_dir / "mbs.csv.gz"
                    mnc_path = subset_dir / "mnc.csv.gz"

                    # MOS / MBS / MNC are saved even if they have 0 rows;
                    # we already ensured they always have the correct columns.
                    saver(mos_df, mos_path)
                    saver(mbs_df, mbs_path)
                    saver(mnc_df, mnc_path)

                    logger.info(
                        "[%s] Saved MOS/MBS/MNC subsets to %s",
                        name,
                        subset_dir,
                    )

            status = "success"
            message = "Processed OK"
            break

        except Exception as exc:
            message = str(exc)
            logger.exception("[%s] attempt %d failed: %s", name, attempt, exc)

    end = time.time()
    time_s = round(end - start, 3)
    if len(message) > 400:
        message = message[:400] + "...(truncated)"

    rows: List[Dict[str, Any]] = []

    # Main dataset row
    rows.append(
        {
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
            "saved": bool(saved_main),
            "time_s": time_s,
        }
    )

    # Subset rows
    if make_subsets and not dry_run and mos_df is not None and mos_path is not None:
        rows.append(
            {
                "name": f"{name}_mos",
                "src": src,
                "out": str(mos_path),
                "status": "success",
                "message": "",
                "input_size": None,
                "processed_items": int(len(mos_df)),
                "saved": True,
                "time_s": time_s,
            }
        )

    if make_subsets and not dry_run and mbs_df is not None and mbs_path is not None:
        rows.append(
            {
                "name": f"{name}_mbs",
                "src": src,
                "out": str(mbs_path),
                "status": "success",
                "message": "",
                "input_size": None,
                "processed_items": int(len(mbs_df)),
                "saved": True,
                "time_s": time_s,
            }
        )

    if make_subsets and not dry_run and mnc_df is not None and mnc_path is not None:
        rows.append(
            {
                "name": f"{name}_mnc",
                "src": src,
                "out": str(mnc_path),
                "status": "success",
                "message": "",
                "input_size": None,
                "processed_items": int(len(mnc_df)),
                "saved": True,
                "time_s": time_s,
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Complex dataset from validation set
# ---------------------------------------------------------------------------


def _extract_first_str(value: Any) -> Optional[str]:
    """
    Given a possibly list-like or string representation of a reaction, return the first
    non-empty string element, or None.
    """
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
        # attempt to parse stringified list (e.g. "['a','b']") and pick first element
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("(") and s.endswith(")")
        ):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)) and parsed:
                    return str(parsed[0]).strip() or None
            except Exception:
                # not a literal list -> treat as raw string
                pass
        return s or None
    # fallback: coerce to str
    s = str(value).strip()
    return s if s else None


def clean_synrbl_complex(
    data: pd.DataFrame,
    std: Optional[Any] = None,
    drop_cols: Optional[List[str]] = None,
    dataset_include: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Clean and curate a SynRBL dataframe for the complex dataset.
    """
    if std is None:
        # lazy import so function can be imported without synkit installed
        from synkit.Chem.Reaction.standardize import Standardize

        std = Standardize()

    if drop_cols is None:
        drop_cols = ["R-ids", "wrong_reactions"]

    if dataset_include is None:
        dataset_include = ["golden_dataset", "Jaworski"]

    # normalize dataset_include to lower-case strings for substring matching
    dataset_include_lc = [str(x).lower() for x in dataset_include]

    df = data.copy()

    # drop configured columns if present
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # require expected_reaction present
    if "expected_reaction" not in df.columns:
        raise ValueError("Input dataframe must contain an 'expected_reaction' column.")
    # drop rows with missing expected_reaction
    df = df.dropna(subset=["expected_reaction"])

    # if datasets column exists, filter rows to keep only matching datasets
    if "datasets" in df.columns:
        keep_mask = []
        for idx, row in df.iterrows():
            ds_raw = row["datasets"]
            ds_list = _normalize_to_list(ds_raw)
            # check any dataset element contains any include substring (case-insensitive)
            matched = False
            for ds in ds_list:
                ds_lc = ds.lower()
                if any(key in ds_lc for key in dataset_include_lc):
                    matched = True
                    break
            keep_mask.append(matched)
        # apply mask and log counts
        keep_series = pd.Series(keep_mask, index=df.index)
        kept = keep_series.sum()
        logger.info(
            "Dataset filter: keeping %d / %d rows matching %s",
            kept,
            len(df),
            list(dataset_include),
        )
        df = df[keep_series]
    else:
        # if no datasets column, log and proceed without filtering
        logger.info("No 'datasets' column found — skipping dataset filtering.")

    records: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        rec: Dict[str, Any] = {}
        # preferred identifier: 'id' else index
        rec["R-id"] = row.get("id", idx)

        # ground truth extraction & standardization
        gt_raw = _extract_first_str(row.get("expected_reaction"))
        gt_err: Optional[str] = None
        try:
            rec["ground_truth"] = std.fit(gt_raw) if gt_raw is not None else None
        except Exception as exc:
            logger.exception(
                "Standardization failed for ground_truth at R-id=%s", rec["R-id"]
            )
            rec["ground_truth"] = None
            gt_err = str(exc)

        # reaction extraction & standardization (may be missing)
        rxn_raw = (
            _extract_first_str(row.get("reaction"))
            if "reaction" in df.columns
            else None
        )
        rxn_err: Optional[str] = None
        if rxn_raw is None:
            rec["rxn"] = None
        else:
            try:
                rec["rxn"] = std.fit(rxn_raw)
            except Exception as exc:
                logger.exception(
                    "Standardization failed for reaction at R-id=%s", rec["R-id"]
                )
                rec["rxn"] = None
                rxn_err = str(exc)

        # attach error dict only when there were errors
        errors: Dict[str, str] = {}
        if gt_err:
            errors["ground_truth"] = gt_err
        if rxn_err:
            errors["reaction"] = rxn_err
        rec["error"] = errors if errors else None

        records.append(rec)

    result_df = pd.DataFrame.from_records(records)

    # ordered columns
    cols_order = ["R-id", "rxn", "ground_truth", "error"]
    for c in cols_order:
        if c not in result_df.columns:
            result_df[c] = None
    result_df = result_df[cols_order]

    # reset index for cleanliness
    result_df = result_df.reset_index(drop=True)
    return result_df


def build_complex_dataset(*, dry_run: bool = False) -> Dict[str, Any]:
    """
    Build the "complex" dataset from the SynRBL validation set.

      - load VALIDATION_SET_URL
      - clean with dataset_include=["golden_dataset","Jaworski"]
      - add r_id = complex_{i+1}
      - drop 'error'
      - save to Data/rbl/complex.csv.gz (unless dry_run)
    """
    pd = _ensure_pandas()
    saver = save_df_gz or _fallback_save_df_gz

    name = "complex"
    out_path = Path("Data/rbl/complex.csv.gz")
    status = "failed"
    message = ""
    processed_count: Optional[int] = None

    start = time.time()
    try:
        logger.info("Loading validation set from %s", VALIDATION_SET_URL)
        df = pd.read_csv(VALIDATION_SET_URL)
        print(df)

        complex_df = clean_synrbl_complex(
            df,
            dataset_include=["golden_dataset", "Jaworski"],
        )
        print(complex_df)

        # add r_id = complex_{i+1}
        complex_df = _add_r_id_column(complex_df, prefix=name)

        complex_df.drop(columns={"error"}, inplace=True, errors="ignore")
        processed_count = len(complex_df)

        if dry_run:
            logger.info(
                "dry-run: skipping save of complex dataset (n=%d)", processed_count
            )
        else:
            saver(complex_df, out_path)
            logger.info("Saved complex dataset to %s (n=%d)", out_path, processed_count)

        status = "success"
        message = "Processed OK"
    except Exception as exc:
        message = str(exc)
        logger.exception("Failed to build complex dataset: %s", exc)

    end = time.time()
    time_s = round(end - start, 3)
    if len(message) > 400:
        message = message[:400] + "...(truncated)"

    return {
        "name": name,
        "src": VALIDATION_SET_URL,
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
# CLI + summary
# ---------------------------------------------------------------------------


def _print_summary_table(results: List[Dict[str, Any]]) -> None:
    cols = [
        "name",
        "status",
        "input_size",
        "processed_items",
        "saved",
        "out",
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
        for c in cols:
            if c not in df.columns:
                df[c] = None
        print(df[cols].to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(
        description="Build reaction-balancing (RBL) datasets using SynRBL."
    )
    p.add_argument(
        "--config",
        help="JSON/YAML config overriding DEFAULT_CONFIG (same structure).",
    )
    p.add_argument(
        "--entries",
        help="Comma-separated subset of config keys to process (default: all).",
    )
    p.add_argument(
        "--src",
        help=(
            "Single-source raw CSV path/URL (overrides config); "
            "must be used with --out and --reaction-col."
        ),
    )
    p.add_argument(
        "--out",
        help="Output path for --src (required when --src is used).",
    )
    p.add_argument(
        "--reaction-col",
        help="Column name containing reaction SMILES for --src.",
    )
    p.add_argument(
        "--dataset-include",
        help="Comma-separated dataset labels to include (e.g. 'golden_dataset,Jaworski').",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do everything except saving files.",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retries for load/curate (default: 1).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG/INFO/WARNING/ERROR).",
    )
    p.add_argument(
        "--summary-out",
        default="reports/rbl_summary.csv.gz",
        help="Path to write summary CSV (gzipped).",
    )
    p.add_argument(
        "--no-complex",
        action="store_true",
        help="Skip building the complex dataset from the validation set.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s - %(message)s",
    )
    logger.info("build_rbl_dataset starting")

    pd = _ensure_pandas()
    results: List[Dict[str, Any]] = []

    # Single-source override
    if args.src:
        if not args.out:
            raise SystemExit("When --src is provided, you must also provide --out.")
        if not args.reaction_col:
            raise SystemExit(
                "When --src is provided, you must also specify --reaction-col."
            )

        dataset_include = (
            _normalize_to_list(args.dataset_include) if args.dataset_include else None
        )

        rows = process_single_entry(
            "single",
            args.src,
            args.out,
            reaction_col=args.reaction_col,
            dataset_include=dataset_include,
            make_subsets=False,
            dry_run=args.dry_run,
            retries=args.retries,
        )
        results.extend(rows)

    else:
        # Config-driven path
        if args.config:
            cfg_path = Path(args.config)
            if not cfg_path.exists():
                raise SystemExit(f"Config file not found: {cfg_path}")
            import json as _json

            text = cfg_path.read_text(encoding="utf-8")
            try:
                cfg = _json.loads(text)
            except Exception:
                try:
                    import yaml  # type: ignore

                    cfg = yaml.safe_load(text)
                except Exception as e:
                    raise SystemExit(f"Failed to parse config as JSON or YAML: {e}")
            if not isinstance(cfg, dict):
                raise SystemExit("Top-level config must be a dict.")
        else:
            cfg = DEFAULT_CONFIG.copy()
            logger.info(
                "No config provided: using DEFAULT_CONFIG with %d entries", len(cfg)
            )

        entries: Optional[List[str]] = None
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
            out = info.get("out") or f"Data/rbl/{name}.csv.gz"
            reaction_col = info.get("reaction_col")
            dataset_include = info.get("dataset_include")
            make_subsets = bool(info.get("make_subsets", False))

            logger.info(
                "Entry '%s': %s -> %s (reaction_col=%r, make_subsets=%s)",
                name,
                src,
                out,
                reaction_col,
                make_subsets,
            )

            try:
                rows = process_single_entry(
                    name,
                    src,
                    out,
                    reaction_col=reaction_col,
                    dataset_include=dataset_include,
                    make_subsets=make_subsets,
                    dry_run=args.dry_run,
                    retries=args.retries,
                )
                results.extend(rows)
            except Exception as exc:
                msg = str(exc)
                if len(msg) > 400:
                    msg = msg[:400] + "...(truncated)"
                logger.exception("Failed to process entry %s: %s", name, exc)
                results.append(
                    {
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
                )

    # Complex dataset row
    if not args.no_complex:
        logger.info("Building complex dataset from validation set...")
        complex_row = build_complex_dataset(dry_run=args.dry_run)
        results.append(complex_row)

    # Print summary
    _print_summary_table(results)

    # Save summary CSV
    try:
        summary_df = pd.DataFrame(results)
        summary_out = Path(args.summary_out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_out, index=False, compression="gzip")
        logger.info("Wrote summary to %s", summary_out)
    except Exception:
        logger.exception("Failed to write summary CSV; printing only.")

    logger.info("build_rbl_dataset finished")


if __name__ == "__main__":
    main()
