from collections import defaultdict
from rdkit import Chem
from typing import Optional, List, Dict, Any, Iterable, Tuple, Union
import pandas as pd
import logging
import ast

logger = logging.getLogger(__name__)


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
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles!r}")
    mol_h = Chem.AddHs(mol)
    counts = defaultdict(int)
    for a in mol_h.GetAtoms():
        counts[a.GetSymbol()] += 1
    return dict(counts)


def _sum_counts(smiles_list: List[str]) -> Dict[str, int]:
    total = defaultdict(int)
    for s in smiles_list:
        c = _mol_element_counts(s)
        for el, n in c.items():
            total[el] += n
    return dict(total)


def reaction_missing_side(
    reaction: str, *, return_details: bool = False
) -> Union[str, Dict]:
    """
    Analyze reaction SMILES and return one of:
      - "one-side"  : all non-zero element differences have the same sign
      - "both"      : some elements are more on left and some more on right
      - "balanced"  : no element differences

    If return_details=True, returns dict:
      {"status": <label>, "diff": {element: left-right, ...}}

    Example:
      reaction_missing_side("A>>B") -> "one-side" / "both" / "balanced"
    """
    left_mols, right_mols = _split_reaction(reaction)
    left_counts = _sum_counts(left_mols)
    right_counts = _sum_counts(right_mols)

    elems = set(left_counts) | set(right_counts)
    diff = {el: left_counts.get(el, 0) - right_counts.get(el, 0) for el in elems}

    pos = any(v > 0 for v in diff.values())
    neg = any(v < 0 for v in diff.values())

    if not pos and not neg:
        status = "balanced"
    elif pos and neg:
        status = "both"
    else:
        status = "one-side"

    if return_details:
        return {"status": status, "diff": diff}
    return status


from typing import List, Dict, Any, Optional


def _get_first_present(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in ("", None):
            v = d[k]
            # if it's a string, strip whitespace
            return v.strip() if isinstance(v, str) else v
    return None


def curate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Curate a list of reaction-record dicts.

    For each record produce a new dict with exactly:
      - "R-id"         : from 'R-id' (fallbacks: 'R_id', 'rid', 'id')
      - "rxn"          : from 'input_reaction' (fallbacks: 'reactions', 'rxn')
      - "ground_truth" : from 'standardized_reactions' (fallbacks: 'standardized_reaction',
                         'new_products', 'ground_truth')

    Records missing any of these will still be included but the missing values will be None.
    Records missing any reasonable identifier will be skipped.
    """
    curated: List[Dict[str, Any]] = []
    for rec in records:
        # find R-id (several common alternatives)
        rid = _get_first_present(rec, ["R-id", "R_id", "rid", "id"])
        if rid is None:
            # skip records with no identifier
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
    return pd.DataFrame(curated)


def _normalize_to_list(value: Any) -> List[str]:
    """
    Normalize `datasets`-like or list-like values into a Python list of strings.
    Accepts:
      - actual list/tuple of str -> returns list
      - string that is a Python literal list ("['a','b']") -> ast.literal_eval -> list
      - simple string -> [string]
    Any non-string objects inside lists/tuples are converted to str.
    Empty or None -> []
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if x is not None and str(x).strip() != ""]
    if isinstance(value, str):
        s = value.strip()
        # try to parse python literal list/tuple e.g. "['a','b']"
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith("(") and s.endswith(")")
        ):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [
                        str(x) for x in parsed if x is not None and str(x).strip() != ""
                    ]
            except Exception:
                # fall through to treat as plain string
                pass
        if s == "":
            return []
        return [s]
    # fallback: convert to single string
    return [str(value)]


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


def clean_synrbl(
    data: pd.DataFrame,
    std: Optional[Any] = None,
    drop_cols: Optional[List[str]] = None,
    dataset_include: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Clean and curate a SynRBL dataframe.

    Enhancements over the original:
      - Filters rows by presence of any substring in `dataset_include` inside the `datasets` column.
        Default dataset_include = ['golden_dataset', 'Jaworski'].
      - Robust parsing for `datasets` that may be lists, tuples, or string representations.
      - Safely extracts the first element from list-like `expected_reaction` / `reaction` fields.
      - Adds logging and better error capture.

    Parameters
    ----------
    data
        Input DataFrame. Must contain `expected_reaction` column and ideally a `datasets` column.
    std
        Standardizer instance exposing `.fit(x)` used to normalize reactions.
        If None, will instantiate `synkit.Chem.Reaction.standardize.Standardize()`.
    drop_cols
        Columns to drop before processing. Defaults to `['R-ids', 'wrong_reactions']`.
    dataset_include
        Iterable of substrings to match against entries in `datasets` column (case-insensitive).
        Rows are kept if any element of `datasets` contains any of these substrings.
        Default: ['golden_dataset', 'Jaworski'].

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['R-id','rxn','ground_truth','error'], filtered and standardized.
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
