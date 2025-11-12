from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats import multitest

try:
    import scikit_posthocs as sp  # optional; used for Conover posthoc

    _HAS_SP = True
except Exception:
    _HAS_SP = False


def p_to_stars(p: float) -> str:
    """Map p-value to NS/*/**/***/**** and treat NaN as 'NS'."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NS"
    if p >= 0.05:
        return "NS"
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


def _rank_biserial(diff: np.ndarray) -> float:
    """Compute rank-biserial effect size for paired differences."""
    diff = np.asarray(diff)
    mask = ~np.isclose(diff, 0)
    if mask.sum() == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff[mask]))
    sum_pos = ranks[diff[mask] > 0].sum()
    sum_neg = ranks[diff[mask] < 0].sum()
    return float((sum_pos - sum_neg) / ranks.sum())


def _cohen_d_paired(diff: np.ndarray) -> Optional[float]:
    """Cohen's d for paired samples: mean(diff)/sd(diff). Returns None if sd == 0."""
    diff = np.asarray(diff)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return None
    return float(np.mean(diff) / sd)


def _validate_and_prepare(
    groups: Sequence[Sequence[float]], subjects: Optional[Sequence[Any]]
) -> Tuple[list, int, np.ndarray]:
    """Validate shapes and convert groups to float numpy arrays; return groups, k, subjects_arr."""
    groups_arr = [np.asarray(g, dtype=float) for g in groups]
    k = len(groups_arr)
    if k < 2:
        raise ValueError("Need at least two groups.")
    n = len(groups_arr[0])
    if not all(len(g) == n for g in groups_arr):
        raise ValueError("All groups must have the same length (repeated measures).")
    if subjects is None:
        subjects_arr = np.arange(n)
    else:
        subjects_arr = np.asarray(subjects)
        if len(subjects_arr) != n:
            raise ValueError("subjects must have the same length as groups.")
    return groups_arr, k, subjects_arr


def _build_long_df(
    groups: Sequence[np.ndarray], subjects: Iterable[Any]
) -> pd.DataFrame:
    """Construct the long-format DataFrame for statsmodels/posthoc functions."""
    k = len(groups)
    n = len(groups[0])
    cond_labels = [f"cond_{i}" for i in range(k)]
    long = pd.DataFrame(
        {
            "subject": np.repeat(subjects, k),
            "condition": np.tile(cond_labels, n),
            "score": np.asarray(groups).T.ravel(),
        }
    )
    return long


def _build_pairwise_df(
    pvals: Sequence[float], pairs: Sequence[Tuple[str, str]]
) -> pd.DataFrame:
    """Return a standardized DataFrame with p-value, stars and conclusion columns."""
    df = pd.DataFrame(
        {
            "group1": [a for a, _ in pairs],
            "group2": [b for _, b in pairs],
            "p-value": np.asarray(pvals, dtype=float),
        }
    )
    df["stars"] = df["p-value"].apply(
        lambda x: p_to_stars(float(x) if not (x is None or np.isnan(x)) else np.nan)
    )
    df["conclusion"] = df["stars"]
    return df


def _parametric_posthoc_tukey(long: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Run Tukey HSD; raise if it fails so caller can fallback."""
    tuk = pairwise_tukeyhsd(endog=long["score"], groups=long["condition"], alpha=alpha)
    summary = pd.DataFrame(
        data=tuk._results_table.data[1:], columns=tuk._results_table.data[0]
    )
    pairs = list(zip(summary["group1"], summary["group2"]))
    pvals = summary["p-adj"].astype(float).values
    return _build_pairwise_df(pvals, pairs)


def _pairwise_paired_ttests(
    groups: Sequence[np.ndarray], cond_labels: Sequence[str], alpha: float
) -> pd.DataFrame:
    """Paired t-tests for all pairs with Holm correction."""
    pairs = []
    pvals = []
    k = len(groups)
    for i in range(k):
        for j in range(i + 1, k):
            t, p = stats.ttest_rel(groups[i], groups[j])
            pairs.append((cond_labels[i], cond_labels[j]))
            pvals.append(float(p))
    adj = multitest.multipletests(pvals, alpha=alpha, method="holm")[1]
    return _build_pairwise_df(adj, pairs)


def _nonparametric_pairwise_wilcoxon_with_holm(
    groups: Sequence[np.ndarray], cond_labels: Sequence[str], alpha: float
) -> pd.DataFrame:
    """Pairwise Wilcoxon tests with Holm correction and robust fallbacks."""
    pairs = []
    pvals = []
    k = len(groups)
    for i in range(k):
        for j in range(i + 1, k):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    stat, p = stats.wilcoxon(groups[i], groups[j])
                except Exception:
                    p = np.nan
            if np.isnan(p):
                if np.allclose(groups[i], groups[j]):
                    p = 1.0
                else:
                    try:
                        _, p = stats.ttest_rel(groups[i], groups[j])
                    except Exception:
                        p = 1.0
            pairs.append((cond_labels[i], cond_labels[j]))
            pvals.append(float(p))
    adj_p = multitest.multipletests(pvals, alpha=alpha, method="holm")[1]
    return _build_pairwise_df(adj_p, pairs)


def _conover_posthoc_if_available(long: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Try scikit-posthocs Conover; return DataFrame or None on failure/unavailable."""
    if not _HAS_SP:
        return None
    try:
        ph = sp.posthoc_conover(
            long,
            val_col="score",
            group_col="condition",
            block_col="subject",
            p_adjust="holm",
        )
        pairs = []
        pvals = []
        for i in range(ph.shape[0]):
            for j in range(i + 1, ph.shape[1]):
                pairs.append((ph.index[i], ph.columns[j]))
                pvals.append(float(ph.iloc[i, j]))
        return _build_pairwise_df(pvals, pairs)
    except Exception:
        return None


def auto_test_repeated(
    *groups: Sequence[float],
    subjects: Optional[Sequence[Any]] = None,
    alpha: float = 0.05,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Automatic statistical testing for related (repeated measures) groups.

    Decision flow:
      1. Levene test for homogeneity of variances across groups.
         - If Levene p > alpha -> nonparametric branch (user rule).
         - If Levene p <= alpha -> parametric branch.
      2. Parametric branch:
         - k == 2: paired t-test fallback when AnovaRM unavailable.
         - k >= 3: AnovaRM; if significant -> Tukey HSD (or Holm-corrected t-tests).
      3. Nonparametric branch:
         - k == 2: Wilcoxon signed-rank (robust handling for ties/degenerate).
         - k >= 3: Friedman; if significant -> Conover (scikit-posthocs) or Holm-corrected pairwise Wilcoxon.

    :param groups: Two or more sequences of repeated-measure observations. Each sequence should
                   have length n (number of subjects), and all sequences must have the same length.
    :type groups: Sequence[Sequence[float]]
    :param subjects: Optional sequence of subject identifiers of length n. If None, subjects are
                     set to range(n).
    :type subjects: Optional[Sequence[Any]]
    :param alpha: Significance threshold (default 0.05).
    :type alpha: float
    :param verbose: If True, print intermediate test results.
    :type verbose: bool

    :return: Dictionary containing keys:
             - 'levene_p' (float): p-value from Levene test.
             - 'branch' (str): 'parametric' or 'nonparametric'.
             - 'diagnostics' (dict): basic diagnostics (n, k, diffs summary for k==2).
             - 'global_test' (dict): global test result and effect sizes where applicable.
             - 'posthoc' (pd.DataFrame or dict): pairwise posthoc results or a note if not applicable.
             - 'conclusion' (str): top-level conclusion as stars string.
             - 'notes' (list): fallback/diagnostic notes.
    :rtype: Dict[str, Any]

    :raises ValueError: If fewer than two groups are provided or group lengths mismatch.
    :raises RuntimeError: If a required global test (e.g., Friedman) fails internally.

    Example (Sphinx-style)
    ----------------------
    .. code-block:: python

       # two related groups (n subjects)
       r2_drfp = [0.30, 0.31, 0.29, 0.33, 0.28]
       r2_rxnfp = [0.28, 0.32, 0.27, 0.31, 0.29]
       res = auto_test_repeated(r2_drfp, r2_rxnfp, verbose=True)
       print("Levene p:", res['levene_p'])
       print("Branch:", res['branch'])
       print("Global test:", res['global_test'])
       print("Posthoc:", res['posthoc'])
       print("Conclusion:", res['conclusion'])

    """
    # Validate & prepare
    groups_arr, k, subjects_arr = _validate_and_prepare(groups, subjects)
    n = len(groups_arr[0])

    # two-group diagnostics
    diffs = None
    fraction_zeros = None
    mean_diff = median_diff = var_diff = None
    if k == 2:
        diffs = groups_arr[0] - groups_arr[1]
        fraction_zeros = float(np.isclose(diffs, 0).sum()) / float(n)
        mean_diff = float(np.mean(diffs))
        median_diff = float(np.median(diffs))
        var_diff = float(np.var(diffs, ddof=1))

    diagnostics = {
        "n": int(n),
        "k": int(k),
        "fraction_zeros_if_two_groups": fraction_zeros,
        "mean_diff_if_two_groups": mean_diff,
        "median_diff_if_two_groups": median_diff,
        "var_diff_if_two_groups": var_diff,
    }

    # Levene test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        levene_stat, levene_p = stats.levene(*groups_arr)

    if verbose:
        print(f"Levene: stat={levene_stat:.6f}, p={levene_p:.6g}")

    branch = "nonparametric" if (levene_p > alpha) else "parametric"

    result: Dict[str, Any] = {
        "levene_p": float(levene_p),
        "branch": branch,
        "diagnostics": diagnostics,
        "notes": [],
    }

    # long dataframe and labels
    cond_labels = [f"cond_{i}" for i in range(k)]
    long = _build_long_df(groups_arr, subjects_arr)

    # === Parametric branch ===
    if branch == "parametric":
        # Global test: prefer AnovaRM
        try:
            aov = AnovaRM(
                long, depvar="score", subject="subject", within=["condition"]
            ).fit()
            p_anova = float(aov.anova_table["Pr > F"].iloc[0])
            global_test = {
                "test": "AnovaRM",
                "p": p_anova,
                "stars": p_to_stars(p_anova),
                "effect_size": None,
            }
            if verbose:
                print("AnovaRM p:", p_anova)
        except Exception:
            # fallback: paired t-test (k==2) or OLS+ANOVA (k>2)
            if k == 2:
                tstat, p_t = stats.ttest_rel(groups_arr[0], groups_arr[1])
                d = _cohen_d_paired(groups_arr[0] - groups_arr[1])
                global_test = {
                    "test": "paired_ttest (fallback)",
                    "p": float(p_t),
                    "stars": p_to_stars(p_t),
                    "effect_size_cohen_d": d,
                }
                result["notes"].append("AnovaRM failed; used paired t-test fallback.")
                if verbose:
                    print("paired t-test p:", p_t)
            else:
                import statsmodels.formula.api as smf
                import statsmodels.api as sm

                model = smf.ols("score ~ C(condition) + C(subject)", data=long).fit()
                aov_table = sm.stats.anova_lm(model, typ=2)
                try:
                    p_cond = float(aov_table.loc["C(condition)", "PR(>F)"])
                except Exception:
                    p_cond = float(aov_table["PR(>F)"].iloc[0])
                global_test = {
                    "test": "OLS+ANOVA (fallback)",
                    "p": p_cond,
                    "stars": p_to_stars(p_cond),
                }
                result["notes"].append("AnovaRM failed; used OLS + ANOVA fallback.")
                if verbose:
                    print("OLS ANOVA p:", p_cond)

        result["global_test"] = global_test
        result["conclusion"] = global_test.get(
            "stars", p_to_stars(global_test.get("p", np.nan))
        )

        # If not significant, return early
        if global_test["p"] >= alpha:
            result["posthoc"] = {
                "note": "Global parametric test not significant",
                "stars": global_test["stars"],
                "conclusion": result["conclusion"],
            }
            return result

        # Try Tukey HSD, else Holm-corrected paired t-tests
        try:
            post = _parametric_posthoc_tukey(long, alpha)
            result["posthoc"] = post
        except Exception:
            result["notes"].append(
                "Tukey HSD failed; used paired t-tests with Holm correction."
            )
            result["posthoc"] = _pairwise_paired_ttests(groups_arr, cond_labels, alpha)
        return result

    # === Nonparametric branch ===
    # k == 2: Wilcoxon signed-rank (robust fallbacks)
    if k == 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stat, p = stats.wilcoxon(groups_arr[0], groups_arr[1])
            except Exception:
                stat, p = np.nan, np.nan

        if np.isnan(p):
            if np.allclose(groups_arr[0], groups_arr[1]):
                p = 1.0
                result["notes"].append(
                    "Wilcoxon degenerate: all paired differences are (nearly) zero."
                )
            else:
                try:
                    _, p_t = stats.ttest_rel(groups_arr[0], groups_arr[1])
                    p = float(p_t)
                    result["notes"].append(
                        "Wilcoxon returned NaN; used paired t-test fallback."
                    )
                except Exception:
                    p = 1.0
                    result["notes"].append(
                        "Wilcoxon returned NaN; paired t-test fallback failed; set p=1.0."
                    )

        rb = _rank_biserial(groups_arr[0] - groups_arr[1])
        d = _cohen_d_paired(groups_arr[0] - groups_arr[1])

        global_test = {
            "test": "Wilcoxon",
            "p": float(p),
            "stars": p_to_stars(p),
            "rank_biserial": rb,
            "cohen_d_if_applicable": d,
        }
        result["global_test"] = global_test
        result["conclusion"] = global_test["stars"]
        result["posthoc"] = {
            "pair": (cond_labels[0], cond_labels[1]),
            "p-value": float(p),
            "stars": global_test["stars"],
            "conclusion": global_test["stars"],
        }
        return result

    # k >= 3: Friedman
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fr_stat, fr_p = stats.friedmanchisquare(*groups_arr)
        except Exception as exc:
            raise RuntimeError("Friedman test failed: " + str(exc))

    kendalls_w = float(fr_stat / (n * (k - 1))) if (n * (k - 1)) != 0 else None
    global_test = {
        "test": "Friedman",
        "statistic": float(fr_stat),
        "p": float(fr_p),
        "stars": p_to_stars(fr_p),
        "kendalls_w": kendalls_w,
    }
    result["global_test"] = global_test
    result["conclusion"] = global_test["stars"]
    if verbose:
        print("Friedman p:", fr_p)

    if fr_p >= alpha:
        result["posthoc"] = {
            "note": "Global Friedman not significant",
            "stars": global_test["stars"],
            "conclusion": result["conclusion"],
        }
        return result

    # Try Conover (scikit-posthocs), else pairwise Wilcoxon with Holm
    conover_df = _conover_posthoc_if_available(long)
    if conover_df is not None:
        result["posthoc"] = conover_df
        return result

    result["notes"].append(
        "scikit-posthocs Conover unavailable or failed; falling back to Wilcoxon pairwise with Holm correction."
    )
    result["posthoc"] = _nonparametric_pairwise_wilcoxon_with_holm(
        groups_arr, cond_labels, alpha
    )
    return result
