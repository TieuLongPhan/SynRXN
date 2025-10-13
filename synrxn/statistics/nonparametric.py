"""
Non-parametric tests: Friedman + Conover-Friedman posthoc.

Produces:
- statistics/nonparametric/friedman/<metric>/friedman_test.pdf
- statistics/nonparametric/conover/<metric>/cofried_pc_<metric>.csv
- statistics/nonparametric/conover/<metric>/cofried_sign_plot.pdf
- statistics/nonparametric/conover/<metric>/cofried_ccd.pdf
"""

from typing import Dict, List, Optional, Union

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
from scipy import stats

from .common import ensure_dir, extract_scoring_dfs, safe_name


def _friedman_dir(root: str, metric: str) -> str:
    return ensure_dir(
        os.path.join(root, "nonparametric", "friedman", safe_name(metric))
    )


def _conover_dir(root: str, metric: str) -> str:
    return ensure_dir(os.path.join(root, "nonparametric", "conover", safe_name(metric)))


def run_nonparametric(
    report_df: pd.DataFrame,
    scoring_list: Optional[Union[List[str], str]] = None,
    method_list: Optional[Union[List[str], str]] = None,
    save_root: str = "statistics",
) -> Dict[str, Dict[str, str]]:
    """
    Run Friedman test and Conover-Friedman posthoc for each metric.

    :param report_df: Wide or long scoring DataFrame.
    :type report_df: pandas.DataFrame
    :param scoring_list: Metrics to analyze (None -> infer all).
    :type scoring_list: Optional[Union[list,str]]
    :param method_list: Methods to include (None -> infer all).
    :type method_list: Optional[Union[list,str]]
    :param save_root: Root directory for outputs (default 'statistics').
    :type save_root: str
    :returns: Mapping metric -> dict with 'friedman_dir' and 'conover_dir'.
    :rtype: Dict[str, Dict[str,str]]

    .. code-block:: python

        >>> import pandas as pd
        >>> from statistics_pipeline.nonparametric import run_nonparametric
        >>> df = pd.DataFrame({
        ...     "scoring":["acc","acc","acc","acc"],
        ...     "cv_cycle":[1,1,2,2],
        ...     "A":[0.9,0.91,0.88,0.87],
        ...     "B":[0.85,0.86,0.84,0.83],
        ... })
        >>> out = run_nonparametric(df, save_root="statistics_demo")
        >>> "acc" in out
        True
    """
    df_wide, metrics, methods = extract_scoring_dfs(
        report_df=report_df,
        scoring_list=scoring_list,
        method_list=method_list,
        melt=False,
    )

    out: Dict[str, Dict[str, str]] = {}
    sns.set_context("notebook")
    sns.set_style("whitegrid")

    for metric in metrics:
        sub = (
            df_wide[df_wide["scoring"] == metric]
            .drop(columns=["scoring"])
            .set_index("cv_cycle")
        )
        friedman_dir = _friedman_dir(save_root, metric)

        # Friedman test
        matrix = [sub[m].dropna().values for m in methods]
        if len(matrix) < 2:
            continue
        stat, p = stats.friedmanchisquare(*matrix)
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(methods)), 6))
        # boxplot of raw scores (wide)
        sub[methods].boxplot(ax=ax)
        ax.set_title(f"Friedman χ²={stat:.2f}, p={p:.2e}")
        ax.set_ylabel(metric.upper())
        fig.tight_layout()
        fig.savefig(
            os.path.join(friedman_dir, "friedman_test.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Conover-Friedman posthoc
        conover_dir = _conover_dir(save_root, metric)
        pc = sp.posthoc_conover_friedman(sub[methods], p_adjust="holm")
        pc.to_csv(os.path.join(conover_dir, f"cofried_pc_{safe_name(metric)}.csv"))

        # Sign plot
        fig, ax = plt.subplots(figsize=(1.6 * len(methods), 1.2 * len(methods)))
        sp.sign_plot(
            pc, ax=ax, linewidths=0.25, linecolor="black", square=True, clip_on=True
        )
        ax.set_title(metric.upper())
        fig.tight_layout()
        fig.savefig(
            os.path.join(conover_dir, "cofried_sign_plot.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Critical difference diagram (average ranks)
        ranks = sub[methods].rank(axis=1, method="average", pct=True).mean(axis=0)
        fig, ax = plt.subplots(figsize=(12, 2.8))
        sp.critical_difference_diagram(ranks, pc, ax=ax, label_props={"fontsize": 10})
        ax.set_title(metric.upper())
        fig.tight_layout()
        fig.savefig(
            os.path.join(conover_dir, "cofried_ccd.pdf"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

        out[metric] = {"friedman_dir": friedman_dir, "conover_dir": conover_dir}

    return out
