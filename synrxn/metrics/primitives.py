"""Numerical scorers with fixed evaluation populations and explicit missingness.

These functions score already qualified, source-specific targets. They do not
construct chemical references or establish experimental correctness.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Mapping


def _paired(reference, prediction):
    reference, prediction = list(reference), list(prediction)
    if len(reference) != len(prediction):
        raise ValueError("reference and prediction lengths must agree; use None for missing outputs")
    if not reference:
        raise ValueError("the eligible reference set must be nonempty")
    return reference, prediction


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _valid_label(value):
    if value is None or isinstance(value, Real) and not math.isfinite(value):
        return False
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _coverage(n, valid):
    return {"eligible_count": n, "valid_count": valid, "invalid_or_missing_count": n - valid,
            "coverage": valid / n, "error_denominator": valid,
            "denominator_policy": "The eligible population is fixed. Numerical errors use valid outputs; report coverage alongside errors."}


def _units(unit):
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("declare the native target unit, including any logarithm or normalization")
    return unit


def _ranks(values):
    """Average ranks, including ties; ranks start at zero."""
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    first = 0
    while first < len(ordered):
        last = first + 1
        while last < len(ordered) and values[ordered[last]] == values[ordered[first]]:
            last += 1
        for index in ordered[first:last]:
            result[index] = (first + last - 1) / 2
        first = last
    return result


def _correlation(left, right):
    if len(left) < 2:
        return None
    a, b = sum(left) / len(left), sum(right) / len(right)
    covariance = sum((x - a) * (y - b) for x, y in zip(left, right))
    scale = math.sqrt(sum((x - a) ** 2 for x in left) * sum((y - b) ** 2 for y in right))
    return covariance / scale if scale else None


def regression(reference, prediction, *, unit, signed=False, **_):
    """Scalar native-unit errors; no implicit logarithm, pooling or imputation."""
    reference, prediction = _paired(reference, prediction)
    if type(signed) is not bool:
        raise ValueError("signed must explicitly be True or False")
    truth = [_finite(value) for value in reference]
    if any(value is None for value in truth):
        raise ValueError("regression references must all be finite")
    pairs = [(t, p) for t, value in zip(truth, prediction) if (p := _finite(value)) is not None]
    result = {**_coverage(len(truth), len(pairs)), "unit": _units(unit),
              "mae": None, "rmse": None, "r2": None, "spearman": None,
              "direction_accuracy": None, "direction_denominator": 0}
    if pairs:
        errors = [p - t for t, p in pairs]
        result.update(mae=sum(abs(e) for e in errors) / len(errors),
                      rmse=math.sqrt(sum(e * e for e in errors) / len(errors)))
        observed = [t for t, _ in pairs]
        mean = sum(observed) / len(observed)
        variance = sum((t - mean) ** 2 for t in observed)
        if len(pairs) >= 2 and variance:
            result["r2"] = 1 - sum(e * e for e in errors) / variance
        result["spearman"] = _correlation(_ranks(observed), _ranks([p for _, p in pairs]))
        nonzero = [(t, p) for t, p in pairs if signed and t != 0]
        result["direction_denominator"] = len(nonzero)
        if nonzero:
            result["direction_accuracy"] = sum((t > 0) == (p > 0) and p != 0 for t, p in nonzero) / len(nonzero)
    result["undefined_policy"] = "R2 requires nonconstant references and at least two valid outputs; Spearman requires nonconstant ranks. Undefined metrics are None."
    return result


def classification(reference, prediction, *, labels=None, **_):
    """Label accuracy and F1; invalid outputs remain false negatives."""
    from sklearn.metrics import f1_score, matthews_corrcoef, recall_score

    reference, prediction = _paired(reference, prediction)
    if any(not _valid_label(value) for value in reference):
        raise ValueError("classification reference labels must be finite, nonmissing and hashable")
    classes = list(dict.fromkeys(reference)) if labels is None else list(labels)
    if not classes or any(not _valid_label(value) for value in classes) or len(set(classes)) != len(classes):
        raise ValueError("labels must be unique, nonempty and cannot include None")
    indices = {label: index for index, label in enumerate(classes)}
    if any(value not in indices for value in reference):
        raise ValueError("reference label absent from declared label universe")
    truth = [indices[value] for value in reference]
    guess = []
    for value in prediction:
        try:
            guess.append(indices.get(value, -1))
        except TypeError:
            guess.append(-1)
    valid = sum(value != -1 for value in guess)
    result = {**_coverage(len(truth), valid), "accuracy": sum(t == p for t, p in zip(truth, guess)) / len(truth),
              "macro_f1": float(f1_score(truth, guess, labels=list(range(len(classes))), average="macro", zero_division=0)),
              "balanced_accuracy": float(recall_score(truth, guess, labels=sorted(set(truth)), average="macro", zero_division=0)),
              "mcc": float(matthews_corrcoef(truth, guess)) if valid == len(truth) and len(set(truth)) > 1 and len(set(guess)) > 1 else None,
              "labels": classes, "score_denominator": len(truth)}
    result["denominator_policy"] = "Accuracy, macro F1 and balanced accuracy include every eligible row; invalid predictions are wrong. MCC is undefined when outputs are missing/invalid or either label vector is constant."
    return result


def binary(reference, prediction, *, threshold=0.5, bins=10, **_):
    """Positive-class probabilities, discrimination, calibration and abstention.

    AUROC/AP and probability losses use valid probability outputs only. Their
    denominator is reported and no full-population performance is inferred.
    Equal confidence ties are selected together in risk/coverage output.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    reference, prediction = _paired(reference, prediction)
    if any(value not in (0, 1) for value in reference):
        raise ValueError("binary references must be 0 or 1; missing labels are not negatives")
    if not 0 <= threshold <= 1 or type(bins) is not int or bins < 1:
        raise ValueError("threshold must be in [0,1] and bins must be a positive integer")
    valid = [(int(t), p) for t, value in zip(reference, prediction)
             if (p := _finite(value)) is not None and 0 <= p <= 1]
    labels = [None if (p := _finite(value)) is None or not 0 <= p <= 1 else int(p >= threshold)
              for value in prediction]
    result = classification(reference, labels, labels=[0, 1])
    result.update(probability_denominator=len(valid), positive_count=sum(reference), positive_label=1, threshold=threshold,
                  auroc=None, average_precision=None, brier_score=None, log_loss=None,
                  expected_calibration_error=None, reliability_diagram=[], selective_risk_coverage=[],
                  probability_denominator_policy="Probability losses/discrimination/calibration use valid probabilities only; coverage and all-eligible classification scores remain separate.",
                  calibration_bins=bins,
                  undefined_policy="AUROC needs both classes. This reporting policy also suppresses average precision for single-class populations, where it is uninformative; AP is not thereby mathematically undefined. Confidently wrong endpoint probabilities have infinite log loss.")
    if not valid:
        return result
    truth, probabilities = zip(*valid)
    if len(set(truth)) == 2:
        result["auroc"] = float(roc_auc_score(truth, probabilities))
        result["average_precision"] = float(average_precision_score(truth, probabilities))
    result["brier_score"] = sum((p - t) ** 2 for t, p in valid) / len(valid)
    losses = [-(math.log(p) if t else math.log1p(-p)) if 0 < p < 1 else (0.0 if p == t else math.inf) for t, p in valid]
    # Infinite loss for a confidently wrong prediction is retained, not clipped.
    result["log_loss"] = sum(losses) / len(losses)
    ece = 0.0
    for index in range(bins):
        members = [(t, p) for t, p in valid if min(int(p * bins), bins - 1) == index]
        if members:
            observed, mean = sum(t for t, _ in members) / len(members), sum(p for _, p in members) / len(members)
            ece += len(members) * abs(observed - mean) / len(valid)
            result["reliability_diagram"].append({"lower": index / bins, "upper": (index + 1) / bins,
                "count": len(members), "mean_probability": mean, "positive_fraction": observed})
    result["expected_calibration_error"] = ece
    # Sort once: a threshold sweep must not scan an entire large feasibility
    # corpus again for every distinct probability. Equal confidence ties enter
    # together. Confidence follows the declared threshold's selected label.
    ordered = sorted(((p if p >= threshold else 1 - p, int(p >= threshold) != t)
                      for t, p in valid), reverse=True)
    retained, wrong = 0, 0
    while retained < len(ordered):
        confidence = ordered[retained][0]
        stop = retained
        while stop < len(ordered) and ordered[stop][0] == confidence:
            wrong += ordered[stop][1]
            stop += 1
        retained = stop
        result["selective_risk_coverage"].append({"confidence_threshold": confidence,
            "retained_count": retained, "coverage": retained / len(reference),
            "risk": wrong / retained})
    return result


def set_recovery(reference, prediction, *, observed_only=False, **_):
    """Sets of already normalized labels/edits; no chemical normalization."""
    reference, prediction = _paired(reference, prediction)
    tp = fp = fn = valid = matches = 0
    macro_f1, recall = [], []
    no_edit, valid_no_edit, false_edits = 0, 0, 0
    for truth, value in zip(reference, prediction):
        if truth is None or isinstance(truth, (str, bytes)):
            raise ValueError("set references must be collections, including explicitly empty sets")
        truth = set(truth)
        ok = value is not None and not isinstance(value, (str, bytes))
        try:
            guess = set(value) if ok else set()
        except TypeError:
            guess, ok = set(), False
        valid += ok
        matches += bool(ok and truth == guess)
        hit, extra, missed = len(truth & guess), len(guess - truth), len(truth - guess)
        tp, fp, fn = tp + hit, fp + extra, fn + missed
        macro_f1.append(2 * hit / (2 * hit + extra + missed) if truth or guess else float(ok))
        if truth:
            recall.append(hit / len(truth))
        if not truth:
            no_edit += 1
            valid_no_edit += ok
            false_edits += bool(ok and guess)
    result = {**_coverage(len(reference), valid), "observed_recall": tp / (tp + fn) if tp + fn else None,
              "query_macro_recall": sum(recall) / len(recall) if recall else None,
              "true_positives": tp, "false_positives": None if observed_only else fp,
              "false_negatives": fn, "micro_f1": None, "macro_f1": None, "exact_match": None,
              "false_edit_rate": None, "unchanged_reference_count": no_edit, "false_edit_denominator": valid_no_edit, "observed_only": observed_only}
    if not observed_only:
        result.update(micro_f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else valid / len(reference),
                      macro_f1=sum(macro_f1) / len(reference), exact_match=matches / len(reference),
                      false_edit_rate=false_edits / valid_no_edit if valid_no_edit else None)
    result["denominator_policy"] = "Missing outputs contribute no recovered labels and remain in eligible counts. Precision/F1/exact match are not scored when unreported labels are unknown."
    return result


def ranked_retrieval(reference, prediction, *, top_k=(1, 3, 5), observed_only=False, **_):
    """Macro retrieval scores against complete, nonnegative graded judgments.

    Each reference is a candidate-ID -> relevance mapping. Unknown candidates
    and duplicates consume their submitted rank and receive zero gain. They
    are counted explicitly. A query without positive judgments has undefined
    recall, reciprocal rank and nDCG, never an invented perfect score.

    In observed-only mode, zero and absent judgments are unjudged rather than
    negative. Only supplied-positive recall, hit rate and reciprocal rank are
    scored; nDCG is suppressed. Without a complete gallery, coverage measures
    syntactically supplied candidate IDs, not gallery or chemical validity.
    """
    reference, prediction = _paired(reference, prediction)
    if not top_k or any(type(k) is not int or k < 1 for k in top_k):
        raise ValueError("top_k must contain positive integers")
    if type(observed_only) is not bool:
        raise ValueError("observed_only must explicitly be True or False")
    cutoffs = sorted(set(top_k))
    records, valid, invalid_candidates, supplied, unjudged_candidates = [], 0, 0, 0, 0
    for judgments, ranked in zip(reference, prediction):
        if not isinstance(judgments, Mapping) or not judgments:
            raise ValueError("each reference query requires a nonempty judged candidate universe")
        if any(_finite(v) is None or _finite(v) < 0 for v in judgments.values()):
            raise ValueError("retrieval relevance must be finite and nonnegative")
        judgments = {key: float(value) for key, value in judgments.items()}
        if isinstance(ranked, (str, bytes)):
            raise ValueError("ranked predictions require candidate-ID lists")
        ranked = [] if ranked is None else list(ranked)
        supplied += bool(ranked)
        positives = sum(v > 0 for v in judgments.values())
        seen, gains, hits = set(), [], []
        for candidate in ranked[:max(cutoffs)]:
            try:
                duplicate = candidate in seen
                in_reference = candidate in judgments
                seen.add(candidate)
                known = in_reference and not duplicate
                invalid = duplicate or (not in_reference and not observed_only)
                unjudged = observed_only and not duplicate and (not in_reference or judgments[candidate] == 0)
            except TypeError:
                known, invalid, unjudged = False, True, False
            invalid_candidates += invalid
            unjudged_candidates += unjudged
            relevance = judgments[candidate] if known else 0
            gains.append(relevance)
            hits.append(relevance > 0)
        valid += bool(seen) if observed_only else bool(seen & set(judgments))
        row = {}
        scale = max(judgments.values())
        def gain(value):
            return 2 ** (value - scale) - 2 ** (-scale) if scale else 0.0
        for k in cutoffs:
            dcg = sum(gain(value) / math.log2(rank + 2) for rank, value in enumerate(gains[:k]))
            ideal = sum(gain(value) / math.log2(rank + 2) for rank, value in enumerate(sorted(judgments.values(), reverse=True)[:k]))
            first = next((index + 1 for index, hit in enumerate(hits[:k]) if hit), None)
            row[k] = {"recall": sum(hits[:k]) / positives if positives else None,
                      "ndcg": dcg / ideal if ideal and not observed_only else None,
                      "reciprocal_rank": 1 / first if first else (0.0 if positives else None),
                      "hit_rate": float(any(hits[:k])) if positives else None}
        records.append(row)
    scores = {}
    for k in cutoffs:
        scores[k] = {}
        for metric in ("recall", "ndcg", "reciprocal_rank", "hit_rate"):
            values = [row[k][metric] for row in records if row[k][metric] is not None]
            scores[k][metric] = sum(values) / len(values) if values else None
            scores[k][metric + "_denominator"] = len(values)
    return {**_coverage(len(reference), valid), "top_k": scores,
            "invalid_or_duplicate_candidates": invalid_candidates,
            "unjudged_candidates": unjudged_candidates, "queries_with_submissions": supplied,
            "observed_only": observed_only,
            "coverage_policy": "Syntactically supplied hashable IDs; gallery validity is unverified." if observed_only else "At least one candidate belongs to the declared judged universe.",
            "judgment_policy": "Recovery of supplied positive associations only. Zero/absent judgments are unknown relevance; nDCG is suppressed. Unjudged IDs and duplicates consume rank." if observed_only else "Explicit judged candidate universe; unknown/duplicate candidates get zero gain and consume rank. No-positive-reference queries are reported as undefined."}


def observed_optimization(reference, prediction, *, budgets=(1, 3, 5), maximize=True, unit, hit_threshold=None, candidate_groups=None, **_):
    """Measured finite-oracle best-response/regret curves, without extrapolation.

    :param candidate_groups: Optional complete candidate-ID to audited substrate-
        parent ID mapping for each panel. Parent IDs must be nonempty strings.
        Group recovery uses the supplied hit threshold and measured pool only.
    """
    reference, prediction = _paired(reference, prediction)
    if not budgets or any(type(k) is not int or k < 1 for k in budgets):
        raise ValueError("budgets must contain positive integers")
    if type(maximize) is not bool:
        raise ValueError("maximize must explicitly be True or False")
    if hit_threshold is not None:
        if _finite(hit_threshold) is None:
            raise ValueError("hit_threshold must be finite")
        hit_threshold = float(hit_threshold)
    if candidate_groups is None:
        group_maps = [None] * len(reference)
    else:
        if isinstance(candidate_groups, (str, bytes, Mapping)):
            raise ValueError("candidate_groups requires one mapping per measured panel")
        try:
            group_maps = list(candidate_groups)
        except TypeError as exc:
            raise ValueError("candidate_groups requires one mapping per measured panel") from exc
        if len(group_maps) != len(reference):
            raise ValueError("candidate_groups requires one mapping per measured panel")
        if hit_threshold is None:
            raise ValueError("candidate_groups requires an explicit source hit_threshold")
    curves, valid, invalid, repeated = [], 0, 0, 0
    for oracle, proposed, groups in zip(reference, prediction, group_maps):
        if not isinstance(oracle, Mapping) or not oracle or any(_finite(v) is None for v in oracle.values()):
            raise ValueError("each oracle must contain finite observed candidate responses")
        oracle = {key: float(value) for key, value in oracle.items()}
        if candidate_groups is not None and (not isinstance(groups, Mapping) or set(groups) != set(oracle)
                or any(not isinstance(parent, str) or not parent.strip() for parent in groups.values())):
            raise ValueError("each candidate_groups mapping must exactly cover its measured panel with nonempty parent IDs")
        if isinstance(proposed, (str, bytes)):
            raise ValueError("optimization proposals require candidate-ID lists")
        proposed = [] if proposed is None else list(proposed)
        optimum = (max if maximize else min)(oracle.values())
        available, row, seen = [], {}, set()
        hit_total = sum(value >= hit_threshold if maximize else value <= hit_threshold for value in oracle.values()) if hit_threshold is not None else None
        hit_candidates = {candidate for candidate, value in oracle.items()
                          if (value >= hit_threshold if maximize else value <= hit_threshold)} if groups is not None else None
        hit_groups = {groups[candidate] for candidate in hit_candidates} if groups is not None else None
        for step in range(1, max(budgets) + 1):
            if step <= len(proposed):
                candidate = proposed[step - 1]
                try:
                    known = candidate in oracle
                except TypeError:
                    known = False
                if known:
                    available.append(oracle[candidate])
                    repeated += candidate in seen
                    seen.add(candidate)
                else:
                    invalid += 1
            if step in budgets:
                best = (max if maximize else min)(available) if available else None
                recovered_groups = {groups[candidate] for candidate in seen & hit_candidates} if groups is not None else None
                row[step] = {"best_response": best,
                             "observed_regret": (optimum - best if maximize else best - optimum) if best is not None else None,
                             "hit": (best >= hit_threshold if maximize else best <= hit_threshold) if best is not None and hit_threshold is not None else None,
                             "observed_candidate_coverage": len(seen) / len(oracle),
                             "source_hit_recovery": sum(oracle[c] >= hit_threshold if maximize else oracle[c] <= hit_threshold for c in seen) / hit_total if hit_total else None,
                             "scope_hit_groups_recovered": len(recovered_groups) if recovered_groups is not None else None,
                             "scope_hit_group_denominator": len(hit_groups) if hit_groups is not None else None,
                             "scope_hit_group_recovery": len(recovered_groups) / len(hit_groups) if hit_groups else None,
                             "observed_failures": sum(oracle[c] < hit_threshold if maximize else oracle[c] > hit_threshold for c in seen) if hit_threshold is not None else None}
        valid += bool(available)
        curves.append(row)
    scores = {}
    for budget in sorted(set(budgets)):
        records = [row[budget] for row in curves]
        available = [row for row in records if row["best_response"] is not None]
        grouped = [row for row in records if row["scope_hit_group_denominator"] is not None]
        scope_fractions = [row["scope_hit_group_recovery"] for row in grouped if row["scope_hit_group_recovery"] is not None]
        scores[budget] = {"valid_oracle_count": len(available), "coverage": len(available) / len(reference),
            "mean_best_response": sum(row["best_response"] for row in available) / len(available) if available else None,
            "mean_observed_regret": sum(row["observed_regret"] for row in available) / len(available) if available else None,
            "hit_rate": sum(bool(row["hit"]) for row in records) / len(reference) if hit_threshold is not None else None,
            "observed_candidate_coverage": sum(row["observed_candidate_coverage"] for row in records) / len(reference),
            "source_hit_recovery": sum(row["source_hit_recovery"] for row in records if row["source_hit_recovery"] is not None) / sum(row["source_hit_recovery"] is not None for row in records) if any(row["source_hit_recovery"] is not None for row in records) else None,
            "source_hit_recovery_denominator": sum(row["source_hit_recovery"] is not None for row in records),
            "mean_scope_hit_groups_recovered": sum(row["scope_hit_groups_recovered"] for row in grouped) / len(grouped) if grouped else None,
            "scope_hit_group_recovery": sum(scope_fractions) / len(scope_fractions) if scope_fractions else None,
            "scope_hit_group_recovery_panel_denominator": len(scope_fractions),
            "scope_hit_group_panel_denominator": len(grouped),
            "scope_hit_groups_recovered_total": sum(row["scope_hit_groups_recovered"] for row in grouped) if grouped else None,
            "scope_hit_group_denominator_total": sum(row["scope_hit_group_denominator"] for row in grouped) if grouped else None,
            "mean_observed_failures": sum(row["observed_failures"] for row in records) / len(reference) if hit_threshold is not None else None}
    return {**_coverage(len(reference), valid), "unit": _units(unit), "maximize": bool(maximize), "budgets": scores,
            "per_panel": [{"panel_index": index, "budgets": row} for index, row in enumerate(curves)],
            "scope_recovery_policy": "Distinct audited substrate-parent IDs with an observed hit in the fixed measured pool. Multiple hit attempts for one parent count once. Fractions are panel-macro over pools with at least one hit-bearing parent; count totals sum panel-specific groups, not a global union. Missing group metadata and no-positive group fractions are undefined.",
            "invalid_candidates": invalid, "repeated_candidates": repeated, "oracle_policy": "Only supplied measured candidates have responses. Invalid/repeated proposals consume budget; repeats do not discover new points. No unmeasured outcomes are imputed."}


def sequence_recovery(reference, prediction, **_):
    """Action/token sequences: Levenshtein edits and complete recovery."""
    reference, prediction = _paired(reference, prediction)
    distances, normalized, exact, valid = [], [], 0, 0
    for truth, guess in zip(reference, prediction):
        if truth is None or isinstance(truth, (str, bytes)):
            raise ValueError("sequence references require explicit action/token lists")
        truth = list(truth)
        ok = guess is not None and not isinstance(guess, (str, bytes))
        guess = list(guess) if ok else []
        valid += ok
        previous = list(range(len(guess) + 1))
        for i, item in enumerate(truth, start=1):
            current = [i]
            for j, other in enumerate(guess, start=1):
                current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (item != other)))
            previous = current
        distance = previous[-1]
        distances.append(distance if ok else max(1, len(truth)))
        normalized.append(distance / max(1, len(truth), len(guess)) if ok else 1.0)
        exact += ok and truth == guess
    return {**_coverage(len(reference), valid), "exact_match": exact / len(reference),
            "mean_edit_distance": sum(distances) / len(reference),
            "mean_normalized_edit_distance": sum(normalized) / len(reference),
            "denominator_policy": "Every eligible sequence is scored. Missing output has normalized error one, including an empty reference. Tokens are not chemically interpreted."}


def composition(reference, prediction, *, unit, component_ids=None, **_):
    """Aligned native component errors without normalization or clipping.

    References must be nonnegative. Finite negative model outputs remain in
    numerical errors and are also counted as physical-bound violations.
    Component identity/order is declared once for the full population.
    """
    import numpy as np

    reference, prediction = _paired(reference, prediction)
    errors, total_errors, negative_components, negative_rows = [], [], 0, 0
    width = None
    for truth, guess in zip(reference, prediction):
        truth = np.asarray(truth, dtype=float)
        if truth.ndim != 1 or not len(truth) or not np.isfinite(truth).all() or (truth < 0).any():
            raise ValueError("composition references must be finite nonnegative component vectors")
        if width is None:
            width = len(truth)
        elif len(truth) != width:
            raise ValueError("composition references require one shared component order and width")
        try:
            guess = np.asarray(guess, dtype=float)
            ok = guess.shape == truth.shape and np.isfinite(guess).all()
        except (TypeError, ValueError):
            ok = False
        if ok:
            errors.append(np.abs(truth - guess))
            total_errors.append(abs(float(truth.sum() - guess.sum())))
            negative_components += int((guess < 0).sum())
            negative_rows += bool((guess < 0).any())
    component_ids = ["component_" + str(i) for i in range(width)] if component_ids is None else list(component_ids)
    if (len(component_ids) != width or any(not isinstance(value, str) or not value.strip() for value in component_ids)
            or len(set(component_ids)) != width):
        raise ValueError("component_ids must be unique nonempty strings matching the shared component width")
    by_component = {name: float(sum(error[index] for error in errors) / len(errors)) if errors else None
                    for index, name in enumerate(component_ids)}
    return {**_coverage(len(reference), len(errors)), "unit": _units(unit),
            "component_mae": float(sum(float(e.sum()) for e in errors) / sum(len(e) for e in errors)) if errors else None,
            "component_ids": component_ids, "component_mae_by_id": by_component,
            "component_error_denominator": len(errors),
            "mean_l1_error": float(sum(float(e.sum()) for e in errors) / len(errors)) if errors else None,
            "mean_total_mass_error": sum(total_errors) / len(total_errors) if total_errors else None,
            "negative_prediction_component_count": negative_components,
            "negative_prediction_row_count": negative_rows,
            "physical_bound_policy": "Finite negative predictions contribute their raw numeric errors and are counted separately; outputs are not clipped.",
            "total_error_policy": "Absolute disagreement between predicted and recorded component sums; not physical mass conservation or a measurement of unreported source mass.",
            "normalization_policy": "One shared declared component order; no renormalization or missing-mass imputation. L1 is macro over valid observations, MAE is micro over valid components."}


def clustering(reference, prediction, **_):
    """External label agreement; label permutation does not alter scores."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    reference, prediction = _paired(reference, prediction)
    if any(not _valid_label(value) for value in reference):
        raise ValueError("clustering references require finite nonmissing hashable labels")
    pairs = [(t, p) for t, p in zip(reference, prediction) if _valid_label(p)]
    result = {**_coverage(len(reference), len(pairs)), "adjusted_rand_index": None, "normalized_mutual_information": None}
    if len(pairs) >= 2:
        truth, guess = zip(*pairs)
        truth_ids = {v: i for i, v in enumerate(dict.fromkeys(truth))}
        guess_ids = {v: i for i, v in enumerate(dict.fromkeys(guess))}
        truth, guess = [truth_ids[v] for v in truth], [guess_ids[v] for v in guess]
        result.update(adjusted_rand_index=float(adjusted_rand_score(truth, guess)),
                      normalized_mutual_information=float(normalized_mutual_info_score(truth, guess)))
    result["qualification"] = "External label agreement on valid outputs; not clustering stability. Withhold labels during fitting and model selection. Noise labels must be declared by the caller."
    return result


def geometry(reference, prediction, *, unit, permutations=None, **_):
    """Proper-rotation Kabsch alignment over declared atom correspondences.

    Optional per-case permutations enumerate accepted atom equivalences. No
    element identities, reaction saddle validity or symmetry are inferred.
    """
    import numpy as np

    reference, prediction = _paired(reference, prediction)
    if permutations is not None and len(permutations) != len(reference):
        raise ValueError("one permutation list is required per case")
    errors = []
    for index, (truth, guess) in enumerate(zip(reference, prediction)):
        truth = np.asarray(truth, dtype=float)
        if truth.ndim != 2 or truth.shape[1] != 3 or not len(truth) or not np.isfinite(truth).all():
            raise ValueError("geometry reference must be a finite nonempty N by 3 array")
        accepted = permutations[index] if permutations is not None else [list(range(len(truth))) ]
        if not accepted or any(any(not isinstance(i, Integral) or isinstance(i, bool) for i in p) or sorted(p) != list(range(len(truth))) for p in accepted):
            raise ValueError("accepted atom permutations must be nonempty complete bijections")
        try:
            guess = np.asarray(guess, dtype=float)
            ok = guess.shape == truth.shape and np.isfinite(guess).all()
        except (ValueError, TypeError):
            ok = False
        if not ok:
            continue
        target = truth - truth.mean(axis=0)
        distances = []
        for permutation in accepted:
            mobile = guess[permutation] - guess.mean(axis=0)
            u, _, vt = np.linalg.svd(mobile.T @ target)
            correction = np.diag([1., 1., 1. if np.linalg.det(u @ vt) >= 0 else -1.])
            rotation = u @ correction @ vt
            distances.append(float(np.sqrt(np.mean(np.sum((mobile @ rotation - target) ** 2, axis=1)))))
        errors.append(min(distances))
    return {**_coverage(len(reference), len(errors)), "unit": _units(unit),
            "mean_aligned_rmsd": sum(errors) / len(errors) if errors else None,
            "per_case_rmsd": errors, "correspondence_policy": "Identity atom order unless explicit accepted permutations are supplied; proper rotations only, no reflection. Not independent saddle-point verification."}


def trajectory(reference, prediction, *, times, unit, time_unit, **_):
    """Scalar target trajectories with per-series errors and time integrals."""
    import numpy as np

    reference, prediction = _paired(reference, prediction)
    if len(times) != len(reference):
        raise ValueError("supply one time grid per trajectory")
    records = []
    for truth, guess, time in zip(reference, prediction, times):
        truth, time = np.asarray(truth, dtype=float), np.asarray(time, dtype=float)
        if truth.ndim != 1 or len(truth) < 2 or time.shape != truth.shape or not np.isfinite(time).all() or not np.isfinite(truth).all() or not (np.diff(time) > 0).all():
            raise ValueError("trajectory references need finite targets and strictly increasing time grids of length at least two")
        result = regression(truth, [None] * len(truth) if guess is None else guess, unit=unit)
        integral = None
        if result["coverage"] == 1:
            integral = float(np.trapezoid((np.asarray(guess, dtype=float) - truth) ** 2, time))
        records.append({**result, "integrated_squared_error": integral})
    valid = [r for r in records if r["coverage"] == 1]
    return {**_coverage(len(reference), len(valid)), "unit": _units(unit),
            "macro_mae": sum(r["mae"] for r in valid) / len(valid) if valid else None,
            "macro_rmse": sum(r["rmse"] for r in valid) / len(valid) if valid else None,
            "mean_integrated_squared_error": sum(r["integrated_squared_error"] for r in valid) / len(valid) if valid else None,
            "per_trajectory": records, "time_unit": _units(time_unit), "integral_unit": "(" + unit + ")^2 * " + time_unit,
            "denominator_policy": "Macro summary uses complete valid trajectories; partial point errors and point coverage remain visible per trajectory. No interpolation or temporal resampling."}


def control(reference, prediction, *, times, unit, time_unit, actions, action_unit, action_bounds, state_bounds=None, **_):
    """Score supplied closed-loop state trajectories and controller actions.

    This is a scorer, not a plant simulator. Reference is the desired scalar
    state trajectory; prediction is the observed/simulated controlled state.
    """
    import numpy as np

    reference, prediction = _paired(reference, prediction)
    result = trajectory(reference, prediction, times=times, unit=unit, time_unit=time_unit)
    if len(actions) != len(reference):
        raise ValueError("supply one action trajectory per controlled state trajectory")
    lower, upper = action_bounds
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("action_bounds must be finite increasing limits")
    if state_bounds is not None and (len(state_bounds) != 2 or not all(math.isfinite(v) for v in state_bounds) or state_bounds[0] > state_bounds[1]):
        raise ValueError("state_bounds must be finite increasing limits")
    records = []
    for time, action, state in zip(times, actions, prediction):
        time = np.asarray(time, dtype=float)
        try:
            action = np.asarray(action, dtype=float)
            ok = action.shape == time.shape and np.isfinite(action).all()
        except (TypeError, ValueError):
            ok = False
        item = {"action_valid": bool(ok), "integrated_squared_action": None, "action_bound_violations": None, "state_bound_violations": None}
        if ok:
            item.update(integrated_squared_action=float(np.trapezoid(action ** 2, time)),
                        action_bound_violations=int(((action < lower) | (action > upper)).sum()))
        if state_bounds is not None and state is not None:
            state = np.asarray(state, dtype=float)
            if state.shape == time.shape and np.isfinite(state).all():
                item["state_bound_violations"] = int(((state < state_bounds[0]) | (state > state_bounds[1])).sum())
        records.append(item)
    result.update(control_records=records, action_bounds=list(action_bounds),
                  state_bounds=list(state_bounds) if state_bounds is not None else None,
                  action_valid_count=sum(item["action_valid"] for item in records), action_unit=_units(action_unit),
                  control_effort_unit="(" + action_unit + ")^2 * " + time_unit,
                  control_effort_policy="Integral of squared supplied action, in supplied action unit squared times native time; bound violations count sampled points, not continuous-time safety.")
    return result


def residue_probabilities(reference, prediction, *, threshold=0.5, bins=10, **_):
    """Protein-macro binary metrics; references are complete residue label lists."""
    reference, prediction = _paired(reference, prediction)
    rows = [binary(t, [None] * len(t) if p is None else p, threshold=threshold, bins=bins) for t, p in zip(reference, prediction)]
    ap = [r["average_precision"] for r in rows if r["coverage"] == 1 and r["average_precision"] is not None]
    return {**_coverage(len(reference), sum(r["coverage"] == 1 for r in rows)), "protein_macro_average_precision": sum(ap) / len(ap) if ap else None,
            "average_precision_protein_denominator": len(ap), "per_protein": rows,
            "eligible_residue_count": sum(r["eligible_count"] for r in rows),
            "valid_residue_count": sum(r["valid_count"] for r in rows),
            "average_precision_denominator_policy": "Macro AP includes only completely predicted proteins with both reference classes; partial per-protein scores remain diagnostic.",
            "qualification": "Each supplied protein is one macro unit. Unknown/unannotated residues must be excluded by a declared eligibility mask before scoring; they are not negative labels."}


def energy_forces(reference, prediction, *, energy_unit, force_unit, **_):
    """Separate native energy and flattened Cartesian force errors per trajectory."""
    import numpy as np

    if not isinstance(reference, Mapping) or not isinstance(prediction, Mapping):
        raise ValueError("supply mappings with energy and forces arrays")
    energy_reference = list(reference["energy"])
    energy_guess = list(prediction.get("energy", [None] * len(energy_reference)))
    energy = regression(energy_reference, energy_guess, unit=energy_unit)
    truth = np.asarray(reference["forces"], dtype=float)
    if truth.ndim < 2 or truth.shape[-1] != 3 or not truth.size or not np.isfinite(truth).all():
        raise ValueError("reference forces require finite Cartesian vectors")
    if truth.ndim == 2:
        truth = truth[None, ...]
    if truth.ndim != 3 or truth.shape[0] != energy["eligible_count"]:
        raise ValueError("reference energies and forces require matching configurations and fixed atom order")
    guess = prediction.get("forces")
    try:
        guess = np.asarray(guess, dtype=float)
        if guess.ndim == 2 and len(truth) == 1:
            guess = guess[None, ...]
        ok = guess.shape == truth.shape
    except (ValueError, TypeError):
        ok = False
    force = regression(truth.reshape(-1), guess.reshape(-1) if ok else [None] * truth.size, unit=force_unit)
    force_frames = np.isfinite(guess).all(axis=(1, 2)) if ok else np.zeros(len(truth), dtype=bool)
    joint_frames = sum(bool(force_ok) and _finite(value) is not None for force_ok, value in zip(force_frames, energy_guess))
    return {**_coverage(len(truth), joint_frames), "energy": energy, "force_components": force,
            "force_configuration_coverage": _coverage(len(truth), int(force_frames.sum())),
            "configuration_policy": "Top-level coverage requires energy and all Cartesian force components for the same configuration; partial component errors are diagnostic.",
            "qualification": "Force errors average Cartesian components. Call once per declared trajectory or campaign to preserve its scientific weighting. No energy/force unit conversion or gradient consistency check."}
