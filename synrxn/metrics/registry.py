"""Discover task metric profiles and dispatch their shared scorers."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from importlib import import_module
from importlib.resources import files
from inspect import Parameter, signature

from . import primitives
from ._profiles import MetricVariant, TaskMetricProfile

__all__ = [
    "MetricVariant", "TaskMetricProfile", "evaluate_task", "get_task_metrics",
    "list_task_metrics", "metric_protocol_coverage",
]


@lru_cache(maxsize=1)
def _task_profiles():
    """Load the family/task modules shipped with SynRXN."""
    profiles = {}
    for family in sorted(files(__package__).iterdir(), key=lambda entry: entry.name):
        if not family.is_dir() or family.name.startswith("_"):
            continue
        for module in sorted(family.iterdir(), key=lambda entry: entry.name):
            task = module.name.removesuffix(".py")
            if not module.is_file() or not module.name.endswith(".py"):
                continue
            if not task.isidentifier() or task.startswith("_"):
                continue
            profile = import_module(f".{family.name}.{task}", __package__).PROFILE
            if profile.task_id != task or profile.family_id != family.name or task in profiles:
                raise ValueError(f"Invalid metric profile identity: {family.name}/{task}")
            profiles[task] = profile
    return profiles


def _score_value(score, name):
    """Read a scalar or labeled score, including an explicit rank or budget."""
    for key in name.split("."):
        if key.isdecimal() and key not in score:
            key = int(key)
        if key not in score:
            return None
        score = score[key]
    return score


def list_task_metrics(family_id: str | None = None) -> dict[str, TaskMetricProfile]:
    """Return independent profile snapshots indexed by scientific task ID."""
    profiles = _task_profiles()
    if family_id is not None and family_id not in {p.family_id for p in profiles.values()}:
        raise KeyError("unknown scientific family: " + family_id)
    return deepcopy({
        key: value for key, value in profiles.items() if family_id is None or value.family_id == family_id
    })


def get_task_metrics(task_id: str) -> TaskMetricProfile:
    """Return the metric variants and reporting choices for a task."""
    try:
        return deepcopy(_task_profiles()[task_id])
    except KeyError:
        raise KeyError("unknown scientific task: " + task_id) from None


def metric_protocol_coverage(registry=None):
    """Audit task/family coverage, optionally against a loaded registry dict."""
    profiles = list_task_metrics()
    result = {
        "implemented_task_count": len(profiles),
        "family_count": len({p.family_id for p in profiles.values()}),
        "variant_count": sum(len(p.variants) for p in profiles.values()),
        "scorers": sorted({v.scorer for p in profiles.values() for v in p.variants.values()}),
        "qualification": (
            "Executable metric contracts; not a claim of completed source-specific benchmark qualification "
            "or model results."
        ),
    }
    if registry is not None:
        expected = {t["id"]: t["family_id"] for t in registry["tasks"]}
        result.update(
            missing_task_ids=sorted(set(expected) - set(profiles)),
            extra_task_ids=sorted(set(profiles) - set(expected)),
            family_mismatches=sorted(
                task for task in profiles.keys() & expected.keys() if profiles[task].family_id != expected[task]
            ),
        )
        result["complete"] = not any(
            result[name] for name in ("missing_task_ids", "extra_task_ids", "family_mismatches")
        )
    return result


def _reference_contract(reference, prediction, *, instances, contract):
    from .reference import evaluate_predictions
    return evaluate_predictions(instances, reference, prediction, contract)


def _mechanism(reference, prediction, *, top_k=(1, 3, 5), allowed_verification_levels=("EXACT", "NORMALIZED")):
    from .mechanism import evaluate_mechanism_predictions
    return evaluate_mechanism_predictions(
        reference, prediction, top_k=top_k, allowed_verification_levels=allowed_verification_levels,
    )


def evaluate_task(task_id, reference, prediction, *, variant=None, source_id=None, cohort_id=None, **context):
    """Score one source-qualified population, with explicit target variant/context.

    Arrays must have one entry per eligible reference; use None for missing
    outputs. The mapping/forward/mechanism variants use their existing typed
    contract schemas instead. Unit-bearing variants require declared units.
    The result includes the designated primary and optional secondary scores
    in `reported_metrics`, alongside full scorer output. A missing cutoff
    or undefined value is reported as None.
    """
    profile = get_task_metrics(task_id)
    variant_id = profile.default_variant if variant is None else variant
    if variant_id not in profile.variants:
        raise KeyError(f"unknown metric variant {task_id}/{variant_id}")
    protocol = profile.variants[variant_id]
    scorer = {"reference_contract": _reference_contract, "mechanism": _mechanism}.get(protocol.scorer)
    if scorer is None:
        scorer = getattr(primitives, protocol.scorer)
    options = {**protocol.defaults, **context}
    accepted = {name for name, p in signature(scorer).parameters.items()
                if name not in {"reference", "prediction"} and p.kind != Parameter.VAR_KEYWORD}
    unknown = set(options) - accepted
    if unknown:
        raise TypeError("unknown metric context: " + ", ".join(sorted(unknown)))
    if protocol.scorer == "reference_contract":
        expected = "mapping" if task_id == "atom_mapping" else "forward"
        if getattr(options.get("contract"), "task", None) != expected:
            raise ValueError("reference contract task does not match scientific metric task")
    score = scorer(reference, prediction, **options)
    return {"metric_contract": "synrxn.task-metrics.v1", "task_id": task_id, "family_id": profile.family_id,
            "variant": variant_id, "source_id": source_id, "cohort_id": cohort_id,
            "metrics": score, "directions": dict(protocol.metrics), "qualification": profile.qualification,
            "primary_metric": protocol.primary_metric, "secondary_metric": protocol.secondary_metric,
            "reported_metrics": {name: _score_value(score, name)
                                 for name in (protocol.primary_metric, protocol.secondary_metric) if name},
            "interpretation": protocol.interpretation}
