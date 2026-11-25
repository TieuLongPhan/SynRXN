# Direct implementation; historical source identity is explicit below.
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('script/process/rbl.py')
from collections import defaultdict
from rdkit import Chem
from typing import Optional, List, Dict, Any, Iterable, Tuple, Union
import pandas as pd
import logging
import ast
logger = logging.getLogger(__name__)

def _split_reaction(reaction: str) -> Tuple[List[str], List[str]]:
    if '>>' in reaction:
        left, right = reaction.split('>>', 1)
    elif '>' in reaction:
        left, right = reaction.split('>', 1)
    else:
        raise ValueError("Reaction string must contain '>' or '>>' as separator.")
    left_mols = [s.strip() for s in left.split('.') if s.strip()]
    right_mols = [s.strip() for s in right.split('.') if s.strip()]
    return (left_mols, right_mols)

def _mol_element_counts(smiles: str) -> Dict[str, int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Cannot parse SMILES: {smiles!r}')
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

def reaction_missing_side(reaction: str, *, return_details: bool=False) -> Union[str, Dict]:
    """Classify the direction of elemental imbalance in reaction SMILES.

    :param reaction: Reaction SMILES, including both sides of the transformation.
    :type reaction: str
    :param return_details: Include the left-minus-right elemental inventory.
    :type return_details: bool
    :return: ``"balanced"`` when all element differences are zero,
        ``"one-side"`` when nonzero differences share one sign, or ``"both"``
        when differences have both signs. With ``return_details=True``, return
        a dictionary containing ``status`` and ``diff`` instead.
    :rtype: str or dict

    .. code-block:: python

        reaction_missing_side("C>>C")  # "balanced"
        reaction_missing_side("C>>[CH3]")  # "one-side"
    """
    left_mols, right_mols = _split_reaction(reaction)
    left_counts = _sum_counts(left_mols)
    right_counts = _sum_counts(right_mols)
    elems = set(left_counts) | set(right_counts)
    diff = {el: left_counts.get(el, 0) - right_counts.get(el, 0) for el in elems}
    pos = any((v > 0 for v in diff.values()))
    neg = any((v < 0 for v in diff.values()))
    if not pos and (not neg):
        status = 'balanced'
    elif pos and neg:
        status = 'both'
    else:
        status = 'one-side'
    if return_details:
        return {'status': status, 'diff': diff}
    return status
from typing import List, Dict, Any, Optional

def _get_first_present(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in ('', None):
            v = d[k]
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
        rid = _get_first_present(rec, ['R-id', 'R_id', 'rid', 'id'])
        if rid is None:
            continue
        rxn = _get_first_present(rec, ['input_reaction', 'reactions', 'rxn'])
        ground_truth = _get_first_present(rec, ['standardized_reactions', 'standardized_reaction', 'new_products', 'ground_truth'])
        curated.append({'R-id': rid, 'rxn': rxn, 'ground_truth': ground_truth})
    return pd.DataFrame(curated)

def _normalize_to_list(value: Any) -> List[str]:
    """Normalize a datasets cell to a list of nonempty strings.

    Lists and tuples are converted element by element. Strings containing a
    literal list or tuple are parsed with ``ast.literal_eval``; other nonempty
    strings become single-element lists. Empty or missing cells become empty
    lists. Non-string elements are converted to strings.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if x is not None and str(x).strip() != '']
    if isinstance(value, str):
        s = value.strip()
        if s.startswith('[') and s.endswith(']') or (s.startswith('(') and s.endswith(')')):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed if x is not None and str(x).strip() != '']
            except Exception:
                pass
        if s == '':
            return []
        return [s]
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
        if s.startswith('[') and s.endswith(']') or (s.startswith('(') and s.endswith(')')):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)) and parsed:
                    return str(parsed[0]).strip() or None
            except Exception:
                pass
        return s or None
    s = str(value).strip()
    return s if s else None

def clean_synrbl(data: pd.DataFrame, std: Optional[Any]=None, drop_cols: Optional[List[str]]=None, dataset_include: Optional[Iterable[str]]=None) -> pd.DataFrame:
    """Clean and curate a SynRBL dataframe.

    Enhancements over the original:
      - Filters rows by presence of any substring in `dataset_include` inside the `datasets` column.
        Default dataset_include = ['golden_dataset', 'Jaworski'].
      - Robust parsing for `datasets` that may be lists, tuples, or string representations.
      - Safely extracts the first element from list-like `expected_reaction` / `reaction` fields.
      - Adds logging and better error capture.

    :param data: Input DataFrame. Must contain `expected_reaction` column and ideally a `datasets` column.
    :param std: Standardizer instance exposing `.fit(x)` used to normalize reactions.
                If None, will instantiate `synkit.Chem.Reaction.standardize.Standardize()`.
    :param drop_cols: Columns to drop before processing. Defaults to `['R-ids', 'wrong_reactions']`.
    :param dataset_include: Iterable of substrings to match against entries in `datasets` column (case-insensitive).
                            Rows are kept if any element of `datasets` contains any of these substrings.
                            Default: ['golden_dataset', 'Jaworski'].

    :return: DataFrame with columns ['R-id','rxn','ground_truth','error'], filtered and standardized.
    :rtype: pd.DataFrame
    """
    if std is None:
        from synkit.Chem.Reaction.standardize import Standardize
        std = Standardize()
    if drop_cols is None:
        drop_cols = ['R-ids', 'wrong_reactions']
    if dataset_include is None:
        dataset_include = ['golden_dataset', 'Jaworski']
    dataset_include_lc = [str(x).lower() for x in dataset_include]
    df = data.copy()
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    if 'expected_reaction' not in df.columns:
        raise ValueError("Input dataframe must contain an 'expected_reaction' column.")
    df = df.dropna(subset=['expected_reaction'])
    if 'datasets' in df.columns:
        keep_mask = []
        for idx, row in df.iterrows():
            ds_raw = row['datasets']
            ds_list = _normalize_to_list(ds_raw)
            matched = False
            for ds in ds_list:
                ds_lc = ds.lower()
                if any((key in ds_lc for key in dataset_include_lc)):
                    matched = True
                    break
            keep_mask.append(matched)
        keep_series = pd.Series(keep_mask, index=df.index)
        kept = keep_series.sum()
        logger.info('Dataset filter: keeping %d / %d rows matching %s', kept, len(df), list(dataset_include))
        df = df[keep_series]
    else:
        logger.info("No 'datasets' column found — skipping dataset filtering.")
    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        rec: Dict[str, Any] = {}
        rec['R-id'] = row.get('id', idx)
        gt_raw = _extract_first_str(row.get('expected_reaction'))
        gt_err: Optional[str] = None
        try:
            rec['ground_truth'] = std.fit(gt_raw) if gt_raw is not None else None
        except Exception as exc:
            logger.exception('Standardization failed for ground_truth at R-id=%s', rec['R-id'])
            rec['ground_truth'] = None
            gt_err = str(exc)
        rxn_raw = _extract_first_str(row.get('reaction')) if 'reaction' in df.columns else None
        rxn_err: Optional[str] = None
        if rxn_raw is None:
            rec['rxn'] = None
        else:
            try:
                rec['rxn'] = std.fit(rxn_raw)
            except Exception as exc:
                logger.exception('Standardization failed for reaction at R-id=%s', rec['R-id'])
                rec['rxn'] = None
                rxn_err = str(exc)
        errors: Dict[str, str] = {}
        if gt_err:
            errors['ground_truth'] = gt_err
        if rxn_err:
            errors['reaction'] = rxn_err
        rec['error'] = errors if errors else None
        records.append(rec)
    result_df = pd.DataFrame.from_records(records)
    cols_order = ['R-id', 'rxn', 'ground_truth', 'error']
    for c in cols_order:
        if c not in result_df.columns:
            result_df[c] = None
    result_df = result_df[cols_order]
    result_df = result_df.reset_index(drop=True)
    return result_df
__configure_runtime(globals(), 'script/process/rbl.py')
