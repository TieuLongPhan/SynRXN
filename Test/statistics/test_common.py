# synrxn/statistics/common.py
# -*- coding: utf-8 -*-
"""
Common utilities for synrxn.statistics package.

Provides:
- ensure_dir / safe_name helpers
- extract_scoring_dfs: normalize wide/long scoring tables while preserving
  original method column names (case preserved).
"""

from typing import Dict, Iterable, List, Optional, Tuple, Union
import os
import re
import pandas as pd


def ensure_dir(path: str) -> str:
    """
    Ensure a directory exists.

    :param path: Directory path to create.
    :type path: str
    :returns: The input path (created if necessary).
    :rtype: str
    """
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(s: Union[str, int, float]) -> str:
    """
    Convert a string/number to a filesystem-safe, lowercase name.

    :param s: Input value to sanitize.
    :type s: Union[str,int,float]
    :returns: Sanitized lowercase string safe for filenames.
    :rtype: str
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s).lower())


def detect_long_format_columns(columns: Iterable[str]) -> bool:
    """
    Heuristic to decide if the table is in long format based on available columns.

    :param columns: Iterable of column names.
    :type columns: Iterable[str]
    :returns: True if 'method' and 'value' columns present (case-insensitive).
    :rtype: bool
    """
    lower_cols = {str(c).lower() for c in columns}
    return {"method", "value"}.issubset(lower_cols)


def _col_lookup_map(columns: Iterable[str]) -> Dict[str, str]:
    """
    Build a dict mapping lower-case column name -> original column name.

    :param columns: Iterable of column names.
    :type columns: Iterable[str]
    :returns: dict(lower_name -> original_name)
    :rtype: Dict[str,str]
    """
    return {str(c).lower(): c for c in columns}


def extract_scoring_dfs(
    report_df: pd.DataFrame,
    scoring_list: Optional[Union[List[str], str]] = None,
    method_list: Optional[Union[List[str], str]] = None,
    melt: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Normalize an input scoring DataFrame (wide or long) to predictable structure.

    Accepts:
     - wide format: columns ['scoring','cv_cycle', <method1>, <method2>, ...]
     - long format: columns ['scoring','cv_cycle','method','value']

    Returns a tuple (df, metrics, methods) where:
     - df is long-format (if melt=True) with columns ['scoring','cv_cycle','method','value']
       and 'scoring' values are lowercased,
     - metrics is a list of metric names (lowercased),
     - methods is a list of method names preserving original column casing.

    :param report_df: Input scoring table (wide or long).
    :type report_df: pandas.DataFrame
    :param scoring_list: If provided, subset to these metrics. Accepts str or list.
    :type scoring_list: Optional[Union[list,str]]
    :param method_list: If provided, subset to these methods/columns. Accepts str or list.
    :type method_list: Optional[Union[list,str]]
    :param melt: If True, return long-format DataFrame (with columns 'scoring','cv_cycle','method','value').
    :type melt: bool
    :returns: (normalized_df, metrics, methods)
    :rtype: Tuple[pandas.DataFrame, List[str], List[str]]
    """
    if not isinstance(report_df, pd.DataFrame):
        raise ValueError("report_df must be a pandas DataFrame")

    df = report_df.copy()
    col_map = _col_lookup_map(df.columns)

    # required meta columns (case-insensitive)
    if "scoring" not in col_map or "cv_cycle" not in col_map:
        raise ValueError(
            "Input must include 'scoring' and 'cv_cycle' columns (case-insensitive)."
        )

    scoring_col = col_map["scoring"]
    cv_col = col_map["cv_cycle"]

    # scoring_list normalization
    if isinstance(scoring_list, str):
        scoring_list = [scoring_list]
    if scoring_list is None:
        # preserve metric values as in data, but return lowercased list
        scoring_list = list(pd.unique(df[scoring_col]))
    scoring_list = [str(s).lower() for s in scoring_list]

    # detect long vs wide using original columns
    is_long = detect_long_format_columns(df.columns)

    if is_long:
        method_col = col_map["method"]
        value_col = col_map["value"]
        if method_list is None:
            method_list = list(pd.unique(df[method_col]))
        elif isinstance(method_list, str):
            method_list = [method_list]
        # select rows: match scoring (case-insensitive) and methods (exact match on original)
        mask = df[scoring_col].astype(str).str.lower().isin(scoring_list) & df[
            method_col
        ].isin(method_list)
        out = df.loc[mask, [scoring_col, cv_col, method_col, value_col]].copy()
        # rename to standard names and lowercase scoring
        out = out.rename(
            columns={
                scoring_col: "scoring",
                cv_col: "cv_cycle",
                method_col: "method",
                value_col: "value",
            }
        )
        out["scoring"] = out["scoring"].astype(str).str.lower()
        methods_out = list(pd.unique(out["method"]))
        return out, scoring_list, methods_out

    # wide format
    # methods are all columns except the scoring & cv columns (preserve original casing)
    candidate_methods = [c for c in df.columns if c not in {scoring_col, cv_col}]
    if len(candidate_methods) == 0:
        raise ValueError("No method columns detected in wide-format table.")
    if method_list is None:
        method_list = candidate_methods
    elif isinstance(method_list, str):
        method_list = [method_list]

    # select relevant rows and columns
    mask = df[scoring_col].astype(str).str.lower().isin(scoring_list)
    wide = df.loc[mask, [scoring_col, cv_col] + list(method_list)].copy()
    # rename scoring to lower-case canonical column for long-format melting
    wide = wide.rename(columns={scoring_col: "scoring", cv_col: "cv_cycle"})
    wide["scoring"] = wide["scoring"].astype(str).str.lower()

    if melt:
        long = wide.melt(
            id_vars=["scoring", "cv_cycle"], var_name="method", value_name="value"
        )
        methods_out = list(pd.unique(long["method"]))
        return long, scoring_list, methods_out

    # return wide (unchanged method column names) if requested not to melt
    return wide, scoring_list, list(method_list)
