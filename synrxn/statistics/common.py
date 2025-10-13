"""
Common utilities used by the split statistical pipelines.

Provides:
- file/directory helpers
- input normalization (wide ↔ long)
"""

from typing import List, Optional, Tuple, Union

import os
import re

import pandas as pd


def ensure_dir(path: str) -> str:
    """
    Ensure a directory exists.

    :param path: Directory path to create.
    :type path: str
    :returns: The input path.
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


def detect_long_format(df: pd.DataFrame) -> bool:
    """
    Heuristic for detecting long-format scoring tables.

    Long format is assumed when columns 'method' and 'value' exist.

    :param df: DataFrame to inspect.
    :type df: pandas.DataFrame
    :returns: True if table appears long-format.
    :rtype: bool
    """
    cols = set(c.lower() for c in df.columns.astype(str))
    return {"method", "value"}.issubset(cols)


def extract_scoring_dfs(
    report_df: pd.DataFrame,
    scoring_list: Optional[Union[List[str], str]] = None,
    method_list: Optional[Union[List[str], str]] = None,
    melt: bool = True,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Normalize an input scoring DataFrame (wide or long) to a predictable structure.

    Accepts:
    - wide format: columns ['scoring','cv_cycle', <method1>, <method2>, ...]
    - long format: columns ['scoring','cv_cycle','method','value']

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

    .. code-block:: python

        >>> import pandas as pd
        >>> from statistics_pipeline.common import extract_scoring_dfs
        >>> df = pd.DataFrame({
        ...     "scoring": ["acc","acc","mcc","mcc"],
        ...     "cv_cycle": [1,2,1,2],
        ...     "A":[0.9,0.91,0.85,0.84],
        ...     "B":[0.88,0.87,0.82,0.83],
        ... })
        >>> long_df, metrics, methods = extract_scoring_dfs(df, melt=True)
        >>> sorted(metrics)
        ['acc', 'mcc']
    """
    df = report_df.copy()
    # Normalize column names to strings and lower-case mapping map
    df.columns = [c if isinstance(c, str) else str(c) for c in df.columns]
    lower_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower_map)

    if "scoring" not in df or "cv_cycle" not in df:
        raise ValueError("Input must include 'scoring' and 'cv_cycle' columns.")

    # scoring list
    if isinstance(scoring_list, str):
        scoring_list = [scoring_list]
    if scoring_list is None:
        scoring_list = list(pd.unique(df["scoring"]))
    scoring_list = [str(s).lower() for s in scoring_list]

    # detect long vs wide
    if detect_long_format(df):
        if "method" not in df or "value" not in df:
            raise ValueError("Long format requires 'method' and 'value' columns.")
        if method_list is None:
            method_list = list(pd.unique(df["method"]))
        elif isinstance(method_list, str):
            method_list = [method_list]
        keep_mask = df["scoring"].str.lower().isin(scoring_list) & df["method"].isin(
            method_list
        )
        out = df.loc[keep_mask, ["scoring", "cv_cycle", "method", "value"]].copy()
        out["scoring"] = out["scoring"].str.lower()
        return out, scoring_list, list(pd.unique(out["method"]))
    else:
        # wide format
        non_method_cols = {"scoring", "cv_cycle"}
        candidate_methods = [c for c in df.columns if c not in non_method_cols]
        if len(candidate_methods) == 0:
            raise ValueError("No method columns detected in wide-format table.")
        if method_list is None:
            method_list = candidate_methods
        elif isinstance(method_list, str):
            method_list = [method_list]
        keep_mask = df["scoring"].str.lower().isin(scoring_list)
        wide = df.loc[keep_mask, ["scoring", "cv_cycle"] + list(method_list)].copy()
        wide["scoring"] = wide["scoring"].str.lower()
        if melt:
            long = wide.melt(
                id_vars=["scoring", "cv_cycle"], var_name="method", value_name="value"
            )
            return long, scoring_list, list(pd.unique(long["method"]))
        return wide, scoring_list, method_list
