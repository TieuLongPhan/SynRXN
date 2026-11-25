"""Failure-inclusive mechanism recovery using SynKit's supplied-mechanism API.

Event and trajectory recovery require strict replay validity. Net equivalence
compares endpoints and is only a diagnostic when the product is supplied in
the input. Replay proves consistency under the declared representation, not
experimental confirmation of a physical mechanism.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import lru_cache
from typing import Any

METRIC_CONTRACT = "synrxn.mechanism-metrics.v1"
POLICY_FIELDS = ("endpoint_resource_policy", "closed_shell_atom_maps")


@lru_cache(maxsize=1)
def _backend():
    try:
        from synkit.Mechanism import (
            MechanismRecord,
            MechanismReplayer,
            mechanism_equivalent,
            mechanism_from_legacy_epd,
        )
    except ImportError as exc:
        raise RuntimeError("Mechanism metrics require SynKit's MechanismRecord/replay/equivalence API.") from exc
    # Capability check for a known representation failure in older backends:
    # participating explicit hydrogens must survive event/trajectory matching.
    left = mechanism_from_legacy_epd(
        "[NH3+:2][H:4].[O-:1][CH3:3]>>[NH3:2].[O:1]([CH3:3])[H:4]",
        [["LP-/Sigma+", [1], [1, 4]], ["Sigma-/LP+", [2, 4], [2]]],
    )
    right = mechanism_from_legacy_epd(
        "[NH3+:12][H:14].[O-:11][CH3:13]>>[NH3:12].[O:11]([CH3:13])[H:14]",
        [["LP-/Sigma+", [11], [11, 14]], ["Sigma-/LP+", [12, 14], [12]]],
    )
    if not all(mechanism_equivalent(left, right, level=level) for level in ("events", "trajectory")):
        raise RuntimeError("SynKit comparator drops mapped hydrogens; use the fixed, pinned mechanism backend.")
    return MechanismRecord, MechanismReplayer, mechanism_equivalent


def _record(value: Any, record_type: type):
    return value if isinstance(value, record_type) else record_type.from_dict(value)


def _policy(record) -> dict:
    return {key: record.metadata[key] for key in POLICY_FIELDS if key in record.metadata}


def _apply_policy(record, policy: dict):
    metadata = {key: value for key, value in record.metadata.items() if key not in POLICY_FIELDS}
    return replace(record, metadata={**metadata, **policy})


def _validated_references(alternatives, record_type, replayer, levels, case_id):
    if not isinstance(alternatives, Sequence) or isinstance(alternatives, (str, bytes)) or not alternatives:
        raise ValueError("no reference annotations for " + case_id)
    accepted = [_record(value, record_type) for value in alternatives]
    policy = _policy(accepted[0])
    if any(_policy(record) != policy for record in accepted):
        raise ValueError("reference endpoint policies differ within " + case_id)
    for record in accepted:
        certificate = replayer.replay(record).certificate
        if certificate.status != "VALID":
            raise ValueError("reference fails strict replay: " + case_id)
        if certificate.verification_level not in levels:
            raise ValueError("reference verification level is outside the declared evaluation population: " + case_id)
    return accepted, policy


def evaluate_mechanism_predictions(
    references: Mapping[str, Sequence[Any]],
    predictions: Mapping[str, Sequence[Any]],
    *,
    top_k: Sequence[int] = (1, 3, 5),
    allowed_verification_levels: Sequence[str] = ("EXACT", "NORMALIZED"),
) -> dict:
    """Score ranked predictions against one or more accepted annotations per case.

    Values are sequences of typed MechanismRecord instances or their serialized
    dictionaries. Missing, malformed and replay-invalid predictions remain in
    the denominator. Reference failures raise rather than silently changing the
    eligible population. A prediction cannot relax the reference's endpoint policy.
    """
    if not top_k or any(type(rank) is not int or rank < 1 for rank in top_k):
        raise ValueError("top_k must contain positive integers")
    ranks = tuple(sorted(set(top_k)))
    levels = tuple(sorted(set(allowed_verification_levels)))
    if not levels or set(levels) - {"EXACT", "NORMALIZED", "DELTA_CONSISTENT"}:
        raise ValueError("select explicit supported verification levels")
    unknown = set(predictions) - set(references)
    if unknown:
        raise ValueError("prediction IDs absent from references: " + ", ".join(sorted(unknown)))
    record_type, replayer_type, equivalent = _backend()
    replayer = replayer_type(validation="strict", repair=False, verify_stereo="stepwise")
    cases = []
    totals = {rank: {"net": 0, "events": 0, "trajectory": 0, "valid": 0} for rank in ranks}
    maximum = max(ranks)
    covered = 0
    for case_id, alternatives in references.items():
        accepted, policy = _validated_references(alternatives, record_type, replayer, levels, case_id)
        ranked = predictions.get(case_id, ())
        if not isinstance(ranked, Sequence) or isinstance(ranked, (str, bytes)):
            raise ValueError("predictions must be a ranked sequence for " + case_id)
        covered += bool(ranked)
        results = []
        for rank, value in enumerate(ranked[:maximum], start=1):
            result = {"rank": rank, "valid": False, "net": False, "events": False, "trajectory": False}
            try:
                record = _apply_policy(_record(value, record_type), policy)
                certificate = replayer.replay(record).certificate
                result.update(
                    valid=certificate.status == "VALID" and certificate.verification_level in levels,
                    replay_status=certificate.status,
                    verification_level=certificate.verification_level,
                    issue_codes=[issue.code for issue in certificate.issues],
                )
                for level in ("net", "events", "trajectory"):
                    result[level] = any(equivalent(reference, record, level=level) for reference in accepted)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
        hits = {}
        for rank in ranks:
            prefix = results[:rank]
            hit = {
                "net": any(item["net"] for item in prefix),
                "events": any(item["valid"] and item["events"] for item in prefix),
                "trajectory": any(item["valid"] and item["trajectory"] for item in prefix),
                "valid": any(item["valid"] for item in prefix),
            }
            for name, matched in hit.items():
                totals[rank][name] += matched
            hits[str(rank)] = hit
        cases.append(
            {
                "case_id": case_id,
                "reference_annotations": len(accepted),
                "predictions_supplied": len(ranked),
                "reference_endpoint_policy": policy,
                "predictions": results,
                "top_k_hits": hits,
            }
        )
    denominator = len(references)
    metrics = {}
    for rank, counts in totals.items():
        metrics[str(rank)] = {
            "denominator": denominator,
            "counts": counts,
            "net_equivalence": counts["net"] / denominator if denominator else None,
            "event_recovery": counts["events"] / denominator if denominator else None,
            "trajectory_recovery": counts["trajectory"] / denominator if denominator else None,
            "strict_validity": counts["valid"] / denominator if denominator else None,
        }
    return {
        "contract": METRIC_CONTRACT,
        "allowed_verification_levels": list(levels),
        "cases": denominator,
        "cases_with_predictions": covered,
        "prediction_coverage": covered / denominator if denominator else None,
        "top_k": metrics,
        "case_results": cases,
        "meaning": {
            "events": "Equivalent supplied electron events, allowing independent event swaps.",
            "trajectory": "Equivalent supplied ordered steps and groups; not observed physical timing.",
            "validity": "Strict representation/replay consistency; not experimental mechanism truth.",
            "net": "Endpoint equivalence; uninformative for mechanism recovery if endpoints are given.",
        },
    }
