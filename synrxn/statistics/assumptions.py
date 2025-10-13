"""
Assumption checks: homogeneity of variance (Levene) and normality diagnostics.

Produces:
- statistics/assumptions/<metric>/variance_homogeneity.csv
- statistics/assumptions/<metric>/normality.pdf
"""

from typing import Dict, List, Optional, Union

import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

from .common import ensure_dir, extract_scoring_dfs, safe_name


def _assump_metric_dir(save_root: str, metric: str) -> str:
    return ensure_dir(os.path.join(save_root, "assumptions", safe_name(metric)))


def run_assumptions(
    report_df: pd.DataFrame,
    scoring_list: Optional[Union[List[str], str]] = None,
    method_list: Optional[Union[List[str], str]] = None,
    save_root: str = "statistics",
) -> Dict[str, str]:
    """
    Run assumption checks: Levene's test for variance homogeneity and visual normality.

    :param report_df: Wide or long scoring DataFrame. Must include 'scoring' and 'cv_cycle'.
    :type report_df: pandas.DataFrame
    :param scoring_list: Metrics to analyze (None -> infer all).
    :type scoring_list: Optional[Union[list,str]]
    :param method_list: Methods to include (None -> infer all).
    :type method_list: Optional[Union[list,str]]
    :param save_root: Root directory to save outputs (default: "statistics").
    :type save_root: str
    :returns: Mapping metric -> output directory path.
    :rtype: Dict[str,str]

    .. code-block:: python

        >>> import pandas as pd
        >>> from statistics_pipeline.assumptions import run_assumptions
        >>> df = pd.DataFrame({
        ...     "scoring":["acc","acc","acc","acc"],
        ...     "cv_cycle":[1,1,2,2],
        ...     "A":[0.9,0.91,0.88,0.87],
        ...     "B":[0.85,0.86,0.84,0.83],
        ... })
        >>> out = run_assumptions(df, save_root="statistics_demo")
        >>> "acc" in out
        True
    """
    df_long, metrics, methods = extract_scoring_dfs(
        report_df=report_df,
        scoring_list=scoring_list,
        method_list=method_list,
        melt=True,
    )

    out_dirs: Dict[str, str] = {}
    sns.set_context("notebook")
    sns.set_style("whitegrid")

    for metric in metrics:
        mdir = _assump_metric_dir(save_root, metric)
        sub = df_long[df_long["scoring"] == metric].copy()

        # Levene's test
        groups = [g["value"].dropna().values for _, g in sub.groupby("method")]
        if len(groups) >= 2:
            stat, p = stats.levene(*groups, center="median")
            var_by_method = sub.groupby("method")["value"].var()
            fold = (
                float(var_by_method.max() / var_by_method.min())
                if (var_by_method.min() > 0)
                else float("inf")
            )
            pd.DataFrame(
                {
                    "variance_fold_difference": [fold],
                    "levene_stat": [stat],
                    "p_value": [p],
                }
            ).to_csv(os.path.join(mdir, "variance_homogeneity.csv"), index=False)

        # Normality visuals: center by method then histogram + Q-Q
        sub["value_centered"] = sub.groupby("method")["value"].transform(
            lambda x: x - x.mean()
        )
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        sns.histplot(sub["value_centered"].dropna(), kde=True, bins=20, ax=axes[0])
        axes[0].set_title(f"{metric.upper()} — centered histogram")
        stats.probplot = stats.probplot  # keep linter happy
        import scipy.stats as _sci_stats

        _sci_stats.probplot(sub["value_centered"].dropna(), dist="norm", plot=axes[1])
        axes[1].set_title("Q–Q plot")
        fig.tight_layout()
        fig.savefig(os.path.join(mdir, "normality.pdf"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        out_dirs[metric] = mdir
    return out_dirs
