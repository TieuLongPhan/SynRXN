from typing import Any, Dict, Optional, Sequence, Tuple

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
         - If Levene p > alpha -> nonparametric branch (this function's rule).
         - If Levene p <= alpha -> parametric branch.
      2. Parametric branch:
         - k == 2: paired t-test (AnovaRM fallback if used).
         - k >= 3: AnovaRM; if significant -> Tukey HSD (or Holm-corrected t-tests).
      3. Nonparametric branch:
         - k == 2: Wilcoxon signed-rank (robust handling for ties/degenerate).
         - k >= 3: Friedman; if significant -> Conover (scikit-posthocs) or Holm-corrected pairwise Wilcoxon.

    Return structure:
      - 'levene_p' : float
      - 'branch'   : 'parametric'|'nonparametric'
      - 'diagnostics': dict with n, k, and two-group diagnostics if applicable
      - 'global_test': dict (test name, p, stars, effect sizes where applicable)
      - 'posthoc': DataFrame or dict for pairwise results (includes 'p-value', 'stars', 'conclusion')
      - 'conclusion': top-level conclusion string (same as global_test['stars'])
      - 'notes': list of strings with fallback/diagnostic notes

    :param groups: Two or more sequences of repeated-measure observations (length n each).
    :param subjects: Optional sequence of subject identifiers (length n). If None, range(n) used.
    :param alpha: Significance threshold (default 0.05).
    :param verbose: If True, print intermediate test results.

    Example (Sphinx-style)
    ----------------------
    .. code-block:: python

       # two related groups (n subjects)
       res = auto_test_repeated(r2_drfp, r2_rxnfp, verbose=True)
       print(res['levene_p'], res['branch'])
       print(res['global_test'])
       print(res['posthoc'])
       print("Conclusion:", res['conclusion'])
    """
    # Convert groups to numpy arrays
    groups = [np.asarray(g, dtype=float) for g in groups]
    k = len(groups)
    if k < 2:
        raise ValueError("Need at least two groups.")
    n = len(groups[0])
    if not all(len(g) == n for g in groups):
        raise ValueError("All groups must have the same length (repeated measures).")
    if subjects is None:
        subjects = np.arange(n)
    else:
        subjects = np.asarray(subjects)
        if len(subjects) != n:
            raise ValueError("subjects must have the same length as groups.")

    # Basic diagnostics for two-group case
    diffs = None
    if k == 2:
        diffs = groups[0] - groups[1]
        fraction_zeros = float(np.isclose(diffs, 0).sum()) / float(n)
        mean_diff = float(np.mean(diffs))
        median_diff = float(np.median(diffs))
        var_diff = float(np.var(diffs, ddof=1))
    else:
        fraction_zeros = None
        mean_diff = None
        median_diff = None
        var_diff = None

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
        levene_stat, levene_p = stats.levene(*groups)

    if verbose:
        print(f"Levene: stat={levene_stat:.6f}, p={levene_p:.6g}")

    # Decision rule: (user requested) Levene p > alpha => nonparametric
    branch = "nonparametric" if (levene_p > alpha) else "parametric"

    result: Dict[str, Any] = {
        "levene_p": float(levene_p),
        "branch": branch,
        "diagnostics": diagnostics,
        "notes": [],
    }

    cond_labels = [f"cond_{i}" for i in range(k)]
    long = pd.DataFrame(
        {
            "subject": np.repeat(subjects, k),
            "condition": np.tile(cond_labels, n),
            "score": np.asarray(groups).T.ravel(),
        }
    )

    def build_pairwise_df(
        pvals: Sequence[float], pairs: Sequence[Tuple[str, str]]
    ) -> pd.DataFrame:
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
        df["conclusion"] = df["stars"]  # duplicate for clarity
        return df

    # ==== PARAMETRIC BRANCH ====
    if branch == "parametric":
        # Try AnovaRM
        try:
            aov = AnovaRM(
                long, depvar="score", subject="subject", within=["condition"]
            ).fit()
            # avoid deprecated positional indexing: use iloc[0]
            p_anova = float(aov.anova_table["Pr > F"].iloc[0])
            global_test = {
                "test": "AnovaRM",
                "p": p_anova,
                "stars": p_to_stars(p_anova),
            }
            global_test["effect_size"] = None
            if verbose:
                print("AnovaRM p:", p_anova)
        except Exception:
            # Fallbacks
            if k == 2:
                tstat, p_t = stats.ttest_rel(groups[0], groups[1])
                d = _cohen_d_paired(groups[0] - groups[1])
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
        # top-level conclusion
        result["conclusion"] = global_test.get(
            "stars", p_to_stars(global_test.get("p", np.nan))
        )

        if global_test["p"] >= alpha:
            result["posthoc"] = {
                "note": "Global parametric test not significant",
                "stars": global_test["stars"],
                "conclusion": result["conclusion"],
            }
            return result

        # Parametric post-hoc: Tukey HSD
        try:
            tuk = pairwise_tukeyhsd(
                endog=long["score"], groups=long["condition"], alpha=alpha
            )
            summary = pd.DataFrame(
                data=tuk._results_table.data[1:], columns=tuk._results_table.data[0]
            )
            pairs = list(zip(summary["group1"], summary["group2"]))
            pvals = summary["p-adj"].astype(float).values
            post = build_pairwise_df(pvals, pairs)
            result["posthoc"] = post
            return result
        except Exception:
            # fallback: pairwise paired t-tests with Holm correction
            pairs = []
            pvals = []
            for i in range(k):
                for j in range(i + 1, k):
                    t, p = stats.ttest_rel(groups[i], groups[j])
                    pairs.append((cond_labels[i], cond_labels[j]))
                    pvals.append(float(p))
            adj = multitest.multipletests(pvals, alpha=alpha, method="holm")[1]
            result["notes"].append(
                "Tukey HSD failed; used paired t-tests with Holm correction."
            )
            post = build_pairwise_df(adj, pairs)
            result["posthoc"] = post
            return result

    # ==== NONPARAMETRIC BRANCH ====
    else:
        # k == 2 -> Wilcoxon signed-rank test
        if k == 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    stat, p = stats.wilcoxon(groups[0], groups[1])
                except Exception:
                    stat, p = np.nan, np.nan

            if np.isnan(p):
                if np.allclose(groups[0], groups[1]):
                    p = 1.0
                    result["notes"].append(
                        "Wilcoxon degenerate: all paired differences are (nearly) zero."
                    )
                else:
                    try:
                        _, p_t = stats.ttest_rel(groups[0], groups[1])
                        p = float(p_t)
                        result["notes"].append(
                            "Wilcoxon returned NaN; used paired t-test fallback."
                        )
                    except Exception:
                        p = 1.0
                        result["notes"].append(
                            "Wilcoxon returned NaN; paired t-test fallback failed; set p=1.0."
                        )

            rb = _rank_biserial(groups[0] - groups[1])
            d = _cohen_d_paired(groups[0] - groups[1])

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
                fr_stat, fr_p = stats.friedmanchisquare(*groups)
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

        # Conover posthoc or fallback
        if _HAS_SP:
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
                post = build_pairwise_df(pvals, pairs)
                result["posthoc"] = post
                return result
            except Exception:
                result["notes"].append(
                    "scikit-posthocs Conover failed; falling back to Wilcoxon pairwise with Holm correction."
                )

        pairs = []
        pvals = []
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
        post = build_pairwise_df(adj_p, pairs)
        result["posthoc"] = post
        return result
