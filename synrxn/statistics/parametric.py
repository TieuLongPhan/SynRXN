# synrxn/statistics/parametric.py
# -*- coding: utf-8 -*-
"""
Parametric tests: repeated-measures ANOVA (AnovaRM) and Tukey HSD (RM-style).

Robust: falls back to OLS residual-based MSE when AnovaRM table lacks a residual row.

Produces:
- statistics/parametric/anova/<metric>/anova_test.pdf
- statistics/parametric/tukey/<metric>/* (csv + pdf)
"""

from typing import Dict, List, Optional, Union

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.libqsturng import psturng, qsturng

from .common import ensure_dir, extract_scoring_dfs, safe_name


def _anova_dir(root: str, metric: str) -> str:
    return ensure_dir(os.path.join(root, "parametric", "anova", safe_name(metric)))


def _tukey_dir(root: str, metric: str) -> str:
    return ensure_dir(os.path.join(root, "parametric", "tukey", safe_name(metric)))


def _estimate_mse_and_df_from_anovarm(aov_result) -> tuple:
    """
    Try to extract MSE and df_resid from a statsmodels AnovaRM result (anova_table).
    Return (mse, df_resid). Raises ValueError if not available.
    """
    tbl = aov_result.anova_table
    idx = [str(i).lower() for i in tbl.index]
    resid_candidates = [
        i for i, name in enumerate(idx) if "resid" in name or "residual" in name
    ]
    if resid_candidates:
        row = tbl.iloc[resid_candidates[0]]
    else:
        non_method_indices = [i for i, name in enumerate(idx) if "method" not in name]
        if non_method_indices:
            row = tbl.iloc[non_method_indices[0]]
        else:
            raise ValueError("No residual/error row found in AnovaRM.anova_table")
    # Mean square
    if "Mean Sq" in row.index:
        mse = float(row["Mean Sq"])
    elif "MS" in row.index:
        mse = float(row["MS"])
    elif "mean_sq" in row.index:
        mse = float(row["mean_sq"])
    else:
        possible = [
            col
            for col in tbl.columns
            if "mean" in str(col).lower() or "ms" in str(col).lower()
        ]
        if possible:
            mse = float(row[possible[0]])
        else:
            raise ValueError("Cannot determine Mean Square from AnovaRM table columns.")
    # df
    if "Num DF" in row.index:
        df_resid = int(row["Num DF"])
    elif "DF" in row.index:
        df_resid = int(row["DF"])
    elif "df" in row.index:
        df_resid = int(row["df"])
    else:
        possible_df = [col for col in tbl.columns if "df" in str(col).lower()]
        if possible_df:
            df_resid = int(row[possible_df[0]])
        else:
            raise ValueError("Cannot determine DF from AnovaRM table.")
    return mse, df_resid


def _estimate_mse_and_df_via_ols(long_df: pd.DataFrame) -> tuple:
    """
    Robust fallback: fit OLS with subject and method dummies (both as fixed effects),
    then take residual mean square and df_resid from the OLS fit.

    Ensures the design matrix (exog) and response (endog) are numeric (float).
    This avoids 'Pandas data cast to numpy dtype of object' errors.
    """
    subj_dummies = pd.get_dummies(
        long_df["cv_cycle"].astype(str), prefix="subj", drop_first=True
    )
    method_dummies = pd.get_dummies(
        long_df["method"].astype(str), prefix="method", drop_first=True
    )

    X = pd.concat([subj_dummies, method_dummies], axis=1)
    X = sm.add_constant(X, has_constant="add")

    # coerce to numeric, fill NaN with 0.0, cast float
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

    y = pd.to_numeric(long_df["value"], errors="coerce")
    mask_valid = ~y.isna()
    if mask_valid.sum() == 0:
        raise RuntimeError(
            "No valid numeric observations found in 'value' column for OLS fallback."
        )

    X = X.loc[mask_valid].astype(float)
    y = y.loc[mask_valid].astype(float)

    if X.shape[1] == 0:
        raise RuntimeError(
            "Design matrix for OLS has zero columns after dummy encoding."
        )

    try:
        ols = sm.OLS(y, X).fit()
    except Exception as exc:
        raise RuntimeError(f"OLS fallback failed: {exc}") from exc

    mse = float(ols.mse_resid)
    df_resid = int(ols.df_resid)
    return mse, df_resid


def _tukey_rm_tables(
    long_df: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """
    Tukey HSD for repeated measures:
    - Get MSE and df_res from a one-way RM-ANOVA (with robust fallback to OLS)
    - Use studentized range to build CIs and adjusted p-values

    Returns dict with:
      result_tab, df_means, df_means_diff, pc
    """
    try:
        aov = AnovaRM(
            data=long_df, depvar="value", subject="cv_cycle", within=["method"]
        ).fit()
    except Exception:
        aov = None

    mse = None
    df_resid = None
    if aov is not None:
        try:
            mse, df_resid = _estimate_mse_and_df_from_anovarm(aov)
        except Exception:
            mse = None
            df_resid = None

    if mse is None or df_resid is None:
        mse, df_resid = _estimate_mse_and_df_via_ols(long_df)

    df_means = (
        long_df.groupby("method", as_index=True)["value"].mean().to_frame("value")
    )
    methods = df_means.index.tolist()
    n_groups = len(methods)
    n_per_group = long_df["method"].value_counts().mean()

    tukey_se = np.sqrt(2 * mse / n_per_group)
    qcrit = qsturng(1 - alpha, n_groups, df_resid)

    num_comparisons = n_groups * (n_groups - 1) // 2
    result_tab = pd.DataFrame(
        index=range(num_comparisons),
        columns=["group1", "group2", "meandiff", "lower", "upper", "p-adj"],
    )

    df_means_diff = pd.DataFrame(index=methods, columns=methods, data=0.0)
    pc = pd.DataFrame(index=methods, columns=methods, data=1.0)

    row_idx = 0
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                group1 = (
                    long_df.loc[long_df["method"] == method1, "value"].dropna().values
                )
                group2 = (
                    long_df.loc[long_df["method"] == method2, "value"].dropna().values
                )
                mean_diff = float(group1.mean() - group2.mean())
                studentized_range = abs(mean_diff) / tukey_se
                adjusted_p = psturng(studentized_range * np.sqrt(2), n_groups, df_resid)
                if isinstance(adjusted_p, (np.ndarray, list)):
                    adjusted_p = float(adjusted_p[0])
                lower = mean_diff - (qcrit / np.sqrt(2) * tukey_se)
                upper = mean_diff + (qcrit / np.sqrt(2) * tukey_se)
                result_tab.loc[row_idx] = [
                    method1,
                    method2,
                    mean_diff,
                    lower,
                    upper,
                    adjusted_p,
                ]
                pc.loc[method1, method2] = adjusted_p
                pc.loc[method2, method1] = adjusted_p
                df_means_diff.loc[method1, method2] = mean_diff
                df_means_diff.loc[method2, method1] = -mean_diff
                row_idx += 1

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
    title: Optional[str] = None,
):
    """Matrix of comparisons with significance stars."""
    sns.set_context("notebook")
    sns.set_style("whitegrid")

    sig = pc.copy().astype(object)
    sig[(pc < 0.001) & (pc >= 0)] = "***"
    sig[(pc < 0.01) & (pc >= 0.001)] = "**"
    sig[(pc < 0.05) & (pc >= 0.01)] = "*"
    sig[(pc >= 0.05)] = ""
    np.fill_diagonal(sig.values, "")

    annot = effect_size.round(3).astype(str) + sig

    v = effect_clip
    cmap = "coolwarm_r" if maximize is False else "coolwarm"

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
    if title:
        ax.set_title(title, fontsize=12)
    xlabels = [f"{m}\n{means.loc[m].values[0]:.3f}" for m in means.index]
    ax.set_xticklabels(
        xlabels, rotation=0, ha="center", va="top", fontsize=axis_text_size
    )
    ax.set_yticklabels(xlabels, rotation=90, va="center", fontsize=axis_text_size)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _tukey_ci_plot(
    tab: pd.DataFrame,
    save_path: str,
    left_xlim: float,
    right_xlim: float,
    title: str,
):
    """Mean differences with Tukey CIs."""
    sns.set_context("notebook")
    sns.set_style("whitegrid")
    result_err = np.array(
        [tab["meandiff"] - tab["lower"], tab["upper"] - tab["meandiff"]]
    )
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(tab) + 2))
    ax.errorbar(
        x=tab["meandiff"],
        y=tab.index,
        xerr=result_err,
        fmt="o",
        capsize=4,
        color="red",
        markersize=5,
    )
    ax.axvline(0, ls="--", lw=1)
    ax.set_xlabel("Mean Difference")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.set_xlim(left_xlim, right_xlim)
    ax.grid(True, axis="x")
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
    Parametric family:
      - RM-ANOVA per metric
      - Tukey HSD (RM) + MCS heatmap + CI plot

    :param report_df: Wide or long format DataFrame with 'scoring' and 'cv_cycle'.
    :type report_df: pandas.DataFrame
    :param scoring_list: Metrics to analyze (None => all).
    :type scoring_list: Optional[Union[list,str]]
    :param method_list: Methods to include (None => all).
    :type method_list: Optional[Union[list,str]]
    :param save_root: Root folder for outputs (default 'statistics').
    :type save_root: str
    :param tukey_alpha: Alpha for Tukey CIs (default 0.05).
    :type tukey_alpha: float
    :param direction_dict: Map metric -> 'maximize'|'minimize' for MCS polarity.
    :type direction_dict: Optional[Dict[str,str]]
    :param effect_dict: Map metric -> effect clipping limit for MCS color-scaling.
    :type effect_dict: Optional[Dict[str,float]]
    :param left_xlim: Left x-limit for CI plots.
    :type left_xlim: float
    :param right_xlim: Right x-limit for CI plots.
    :type right_xlim: float
    :returns: Mapping metric -> {"anova_dir":..., "tukey_dir":...}
    :rtype: Dict[str, Dict[str,str]]
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
        if sub.empty:
            continue

        anova_dir = _anova_dir(save_root, metric)
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(methods)), 6))
        sns.boxplot(data=sub, x="method", y="value", ax=ax, showmeans=True)

        try:
            anova_model = AnovaRM(
                data=sub, depvar="value", subject="cv_cycle", within=["method"]
            ).fit()
            method_row_candidates = [
                i for i in anova_model.anova_table.index if "method" in str(i).lower()
            ]
            if method_row_candidates:
                method_row = method_row_candidates[0]
            else:
                method_row = anova_model.anova_table.index[0]
            pval = float(anova_model.anova_table.loc[method_row, "Pr > F"])
        except Exception:
            pval = float("nan")

        ax.set_title(f"RM-ANOVA p={pval:.2e}")
        ax.set_xlabel("")
        ax.set_ylabel(metric.upper())
        fig.tight_layout()
        fig.savefig(
            os.path.join(anova_dir, "anova_test.pdf"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

        tukey_dir = _tukey_dir(save_root, metric)
        tables = _tukey_rm_tables(sub, alpha=tukey_alpha)

        tables["result_tab"].to_csv(
            os.path.join(tukey_dir, f"tukey_result_tab_{safe_name(metric)}.csv"),
            index=True,
        )
        tables["df_means"].to_csv(
            os.path.join(tukey_dir, f"tukey_df_means_{safe_name(metric)}.csv"),
            index=True,
        )
        tables["df_means_diff"].to_csv(
            os.path.join(tukey_dir, f"tukey_df_means_diff_{safe_name(metric)}.csv"),
            index=True,
        )
        tables["pc"].to_csv(
            os.path.join(tukey_dir, f"tukey_pc_{safe_name(metric)}.csv"), index=True
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
            tab=tables["result_tab"],
            save_path=os.path.join(tukey_dir, "tukey_ci.pdf"),
            left_xlim=left_xlim,
            right_xlim=right_xlim,
            title=metric.upper(),
        )

        out[metric] = {"anova_dir": anova_dir, "tukey_dir": tukey_dir}

    return out
