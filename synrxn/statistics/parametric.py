"""
Parametric tests: repeated-measures ANOVA (AnovaRM) and Tukey HSD (RM-style).

Produces:
- statistics/parametric/anova/<metric>/anova_test.pdf
- statistics/parametric/tukey/<metric>/tukey_result_tab_<metric>.csv
- statistics/parametric/tukey/<metric>/tukey_df_means_<metric>.csv
- statistics/parametric/tukey/<metric>/tukey_df_means_diff_<metric>.csv
- statistics/parametric/tukey/<metric>/tukey_pc_<metric>.csv
- statistics/parametric/tukey/<metric>/tukey_mcs.pdf
- statistics/parametric/tukey/<metric>/tukey_ci.pdf
"""

from typing import Dict, List, Optional, Union

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.libqsturng import psturng, qsturng

from .common import ensure_dir, extract_scoring_dfs, safe_name


def _anova_dir(root: str, metric: str) -> str:
    return ensure_dir(os.path.join(root, "parametric", "anova", safe_name(metric)))


def _tukey_dir(root: str, metric: str) -> str:
    return ensure_dir(os.path.join(root, "parametric", "tukey", safe_name(metric)))


def _tukey_rm_tables(
    long_df: pd.DataFrame, alpha: float = 0.05
) -> Dict[str, pd.DataFrame]:
    """
    Compute Tukey HSD style tables for repeated-measures data.

    :param long_df: Long-format DataFrame with 'method' and 'value' columns.
    :type long_df: pandas.DataFrame
    :param alpha: Significance level for confidence intervals.
    :type alpha: float
    :returns: Dict with keys: result_tab, df_means, df_means_diff, pc
    :rtype: Dict[str,pandas.DataFrame]
    """
    # Fit AnovaRM to get error MS and df
    aov = AnovaRM(
        data=long_df, depvar="value", subject="cv_cycle", within=["method"]
    ).fit()
    tbl = aov.anova_table.reset_index()
    # get the residual/error row (not 'method')
    error_row = tbl.loc[tbl["index"] != "method"].iloc[0]
    # statsmodels naming can differ; attempt both
    mse = error_row.get("Mean Sq", error_row.get("MS"))
    df_resid = int(error_row.get("Num DF", error_row.get("DF")))

    df_means = (
        long_df.groupby("method", as_index=True)["value"].mean().to_frame("value")
    )
    methods = df_means.index.tolist()
    n_groups = len(methods)
    n_per_group = long_df["method"].value_counts().mean()

    se = np.sqrt(2 * mse / n_per_group)
    qcrit = qsturng(1 - alpha, n_groups, df_resid)

    num = n_groups * (n_groups - 1) // 2
    result_tab = pd.DataFrame(
        index=range(num),
        columns=["group1", "group2", "meandiff", "lower", "upper", "p-adj"],
    )
    df_means_diff = pd.DataFrame(index=methods, columns=methods, data=0.0)
    pc = pd.DataFrame(index=methods, columns=methods, data=1.0)

    row = 0
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j:
                g1 = long_df.loc[long_df["method"] == m1, "value"].dropna().values
                g2 = long_df.loc[long_df["method"] == m2, "value"].dropna().values
                mdiff = float(g1.mean() - g2.mean())
                sr = abs(mdiff) / se
                padj = psturng(sr * np.sqrt(2), n_groups, df_resid)
                padj = float(padj if np.isscalar(padj) else padj[0])
                half = (qcrit / np.sqrt(2)) * se
                lower, upper = mdiff - half, mdiff + half
                result_tab.loc[row] = [m1, m2, mdiff, lower, upper, padj]
                pc.loc[m1, m2] = padj
                pc.loc[m2, m1] = padj
                df_means_diff.loc[m1, m2] = mdiff
                df_means_diff.loc[m2, m1] = -mdiff
                row += 1

    result_tab["group1_mean"] = result_tab["group1"].map(df_means["value"])
    result_tab["group2_mean"] = result_tab["group2"].map(df_means["value"])
    result_tab.index = result_tab["group1"] + " - " + result_tab["group2"]

    return {
        "result_tab": result_tab,
        "df_means": df_means,
        "df_means_diff": df_means_diff.astype(float),
        "pc": pc,
    }


def _mcs_heatmap(
    pc: pd.DataFrame,
    effect_size: pd.DataFrame,
    means: pd.DataFrame,
    save_path: str,
    effect_clip: float = 0.1,
    maximize: bool = True,
    cell_text_size: int = 11,
    axis_text_size: int = 10,
    title: str = None,
) -> None:
    """
    Draw matrix-of-comparisons (MCS) heatmap with significance stars and mean annotations.

    :param pc: Matrix of pairwise adjusted p-values (DataFrame).
    :type pc: pandas.DataFrame
    :param effect_size: Matrix of pairwise mean differences.
    :type effect_size: pandas.DataFrame
    :param means: Per-method means (DataFrame with index == methods).
    :type means: pandas.DataFrame
    :param save_path: Path to write PDF.
    :type save_path: str
    :param effect_clip: Clip value for color-scaling.
    :type effect_clip: float
    :param maximize: If False, reverse the colormap.
    :type maximize: bool
    """
    sns.set_context("notebook")
    sns.set_style("whitegrid")

    sig = pc.copy().astype(object)
    sig[(pc < 0.001) & (pc >= 0)] = "***"
    sig[(pc < 0.01) & (pc >= 0.001)] = "**"
    sig[(pc < 0.05) & (pc >= 0.01)] = "*"
    sig[(pc >= 0.05)] = ""
    import numpy as _np

    _np.fill_diagonal(sig.values, "")

    annot = effect_size.round(3).astype(str) + sig

    cmap = "coolwarm" if maximize else "coolwarm_r"
    v = float(effect_clip)
    fig, ax = plt.subplots(figsize=(1.6 * len(effect_size), 1.2 * len(effect_size)))
    sns.heatmap(
        effect_size,
        cmap=cmap,
        annot=annot,
        fmt="",
        cbar=True,
        vmin=-2 * v,
        vmax=2 * v,
        annot_kws={"size": cell_text_size},
        ax=ax,
    )

    xlabels = [f"{m}\n{means.loc[m].values[0]:.3f}" for m in means.index]
    ax.set_xticklabels(
        xlabels, rotation=0, ha="center", va="top", fontsize=axis_text_size
    )
    ax.set_yticklabels(xlabels, rotation=90, va="center", fontsize=axis_text_size)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _tukey_ci_plot(
    result_tab: pd.DataFrame,
    save_path: str,
    left_xlim: float = -0.5,
    right_xlim: float = 0.5,
    title: str = None,
) -> None:
    """
    Plot mean differences with confidence intervals.

    :param result_tab: DataFrame produced by _tukey_rm_tables()['result_tab'].
    :type result_tab: pandas.DataFrame
    :param save_path: Output PDF path.
    :type save_path: str
    :param left_xlim: Left x-limit.
    :type left_xlim: float
    :param right_xlim: Right x-limit.
    :type right_xlim: float
    """
    sns.set_context("notebook")
    sns.set_style("whitegrid")
    err = (
        result_tab["meandiff"] - result_tab["lower"],
        result_tab["upper"] - result_tab["meandiff"],
    )
    x = result_tab["meandiff"].values
    y = result_tab.index.values
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(result_tab) + 2))
    ax.errorbar(x=x, y=y, xerr=err, fmt="o", capsize=3)
    ax.axvline(0, ls="--")
    ax.set_xlim(left_xlim, right_xlim)
    ax.set_xlabel("Mean difference")
    ax.set_ylabel("")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_parametric(
    report_df: pd.DataFrame,
    scoring_list: Optional[Union[List[str], str]] = None,
    method_list: Optional[Union[List[str], str]] = None,
    save_root: str = "statistics",
    tukey_alpha: float = 0.05,
    direction_dict: Optional[Dict[str, str]] = None,
    effect_dict: Optional[Dict[str, float]] = None,
    left_xlim: float = -0.5,
    right_xlim: float = 0.5,
) -> Dict[str, Dict[str, str]]:
    """
    Run repeated-measures ANOVA and Tukey-style posthoc for each metric.

    :param report_df: Wide or long scoring DataFrame.
    :type report_df: pandas.DataFrame
    :param scoring_list: Metrics to analyze (None -> infer all).
    :type scoring_list: Optional[Union[list,str]]
    :param method_list: Methods to include (None -> infer all).
    :type method_list: Optional[Union[list,str]]
    :param save_root: Root folder for outputs.
    :type save_root: str
    :param tukey_alpha: Alpha for Tukey CIs.
    :type tukey_alpha: float
    :param direction_dict: Map metric -> 'maximize'|'minimize' for heatmap polarity.
    :type direction_dict: Optional[Dict[str,str]]
    :param effect_dict: Map metric -> clip value for colormap scale.
    :type effect_dict: Optional[Dict[str,float]]
    :param left_xlim: Left x-limit for Tukey CI plots.
    :type left_xlim: float
    :param right_xlim: Right x-limit for Tukey CI plots.
    :type right_xlim: float
    :returns: Mapping metric -> dict with 'anova_dir' and 'tukey_dir'.
    :rtype: Dict[str, Dict[str,str]]

    .. code-block:: python

        >>> import pandas as pd
        >>> from statistics_pipeline.parametric import run_parametric
        >>> df = pd.DataFrame({
        ...    "scoring":["acc","acc","acc","acc"],
        ...    "cv_cycle":[1,1,2,2],
        ...    "A":[0.9,0.91,0.88,0.87],
        ...    "B":[0.85,0.86,0.84,0.83],
        ... })
        >>> out = run_parametric(df, save_root="statistics_demo")
        >>> "acc" in out
        True
    """
    df_long, metrics, methods = extract_scoring_dfs(
        report_df=report_df,
        scoring_list=scoring_list,
        method_list=method_list,
        melt=True,
    )

    direction_dict = {str(k).lower(): v for k, v in (direction_dict or {}).items()}
    effect_dict = {str(k).lower(): v for k, v in (effect_dict or {}).items()}

    out: Dict[str, Dict[str, str]] = {}
    sns.set_context("notebook")
    sns.set_style("whitegrid")

    for metric in metrics:
        sub = df_long[df_long["scoring"] == metric].copy()

        # ANOVA plot
        anova_dir = _anova_dir(save_root, metric)
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(methods)), 6))
        sns.boxplot(data=sub, x="method", y="value", ax=ax, showmeans=True)
        model = AnovaRM(
            data=sub, depvar="value", subject="cv_cycle", within=["method"]
        ).fit()
        pval = model.anova_table.loc["method", "Pr > F"]
        ax.set_title(f"RM-ANOVA p={pval:.2e}")
        ax.set_xlabel("")
        ax.set_ylabel(metric.upper())
        fig.tight_layout()
        fig.savefig(
            os.path.join(anova_dir, "anova_test.pdf"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

        # Tukey RM-style
        tukey_dir = _tukey_dir(save_root, metric)
        tables = _tukey_rm_tables(sub, alpha=tukey_alpha)
        tables["result_tab"].to_csv(
            os.path.join(tukey_dir, f"tukey_result_tab_{safe_name(metric)}.csv")
        )
        tables["df_means"].to_csv(
            os.path.join(tukey_dir, f"tukey_df_means_{safe_name(metric)}.csv")
        )
        tables["df_means_diff"].to_csv(
            os.path.join(tukey_dir, f"tukey_df_means_diff_{safe_name(metric)}.csv")
        )
        tables["pc"].to_csv(
            os.path.join(tukey_dir, f"tukey_pc_{safe_name(metric)}.csv")
        )

        maximize = (
            True if direction_dict.get(metric, "maximize") == "maximize" else False
        )
        clip = float(effect_dict.get(metric, 0.1))
        _mcs_heatmap(
            pc=tables["pc"],
            effect_size=tables["df_means_diff"],
            means=tables["df_means"],
            save_path=os.path.join(tukey_dir, "tukey_mcs.pdf"),
            effect_clip=clip,
            maximize=maximize,
            title=metric.upper(),
        )

        _tukey_ci_plot(
            result_tab=tables["result_tab"],
            save_path=os.path.join(tukey_dir, "tukey_ci.pdf"),
            left_xlim=left_xlim,
            right_xlim=right_xlim,
            title=metric.upper(),
        )

        out[metric] = {"anova_dir": anova_dir, "tukey_dir": tukey_dir}

    return out
