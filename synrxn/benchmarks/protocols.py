"""Audit scientific task coverage by metric profiles and split requirements.

This catalogue joins configured software interfaces to scientific definitions.
It does not certify table eligibility, available grouping metadata or completed
predictive experiments. Source-specific task qualifications remain attached.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


def _split_metadata_audit(task):
    """Separate declared assignment columns from potential dependence fields."""
    from synrxn.split.strategies import _is_membership_field

    rows = []
    for binding in task.get("bindings", ()):
        fields = binding.get("task_fields", {}).get(task["id"], {}).get("group_fields", ())
        memberships = [field for field in fields if _is_membership_field(field)]
        context = [field for field in fields if field in {"scale", "parent_dataset"}]
        dependencies = [field for field in fields if field not in memberships and field not in context]
        rows.append({
            "binding_id": "/".join(filter(None, (binding.get("namespace"), binding.get("key")))),
            "declared_fields": list(fields), "membership_fields": memberships,
            "context_or_transfer_fields": context, "candidate_dependency_fields": dependencies,
            "metadata_status": "not_source_qualified_by_this_audit",
            "qualification": (
                "Assignments are not dependency identifiers. Candidate fields require source qualification, "
                "namespacing and lineage closure before splitting. Native membership provenance is not "
                "established by a field named native_split. This audit reads no observation metadata."
            ),
        })
    return rows


def task_protocol_catalog(registry=None):
    """Return a JSON-compatible audit of all registered scientific tasks.

    ``registry`` can be a scientific registry mapping or a path. With no
    argument, use the local checkout's provenance registry. Installed users can
    supply that registry explicitly; metric and split profiles themselves are
    packaged and can be inspected independently of a checkout.
    """
    from synrxn.metrics import list_task_metrics
    from synrxn.baseline.dummy import list_dummy_protocols
    from synrxn.split import get_task_split_protocols
    from synrxn.reconstruction.paths import physical_path

    registry_path = None
    if registry is None:
        registry = physical_path("kdd/task_registry.json")
    if isinstance(registry, (str, Path)):
        supplied = Path(registry)
        # Explicit files follow the caller's working directory. Historical
        # repository identities still resolve through the storage aliases.
        registry_path = supplied.resolve() if supplied.exists() else physical_path(supplied)
        payload = registry_path.read_bytes()
        registry = json.loads(payload)
    elif isinstance(registry, Mapping):
        payload = json.dumps(registry, sort_keys=True, ensure_ascii=False).encode()
    else:
        raise TypeError("Provide a scientific task registry mapping or path")

    definitions = registry["tasks"]
    tasks = {task["id"]: task for task in definitions}
    if len(tasks) != len(definitions):
        raise ValueError("Duplicate scientific task IDs in registry")
    families = {family["id"]: family["title"] for family in registry["families"]}
    if any(task["family_id"] not in families for task in definitions):
        raise ValueError("Scientific task refers to an unknown family")
    metrics = list_task_metrics()
    splits = get_task_split_protocols()
    dummies = list_dummy_protocols()
    issues = {
        "missing_metric_profiles": sorted(tasks.keys() - metrics.keys()),
        "extra_metric_profiles": sorted(metrics.keys() - tasks.keys()),
        "missing_split_protocols": sorted(tasks.keys() - splits.keys()),
        "extra_split_protocols": sorted(splits.keys() - tasks.keys()),
        "missing_dummy_protocols": sorted(tasks.keys() - dummies.keys()),
        "extra_dummy_protocols": sorted(dummies.keys() - tasks.keys()),
        "metric_family_mismatches": sorted(
            task for task in tasks.keys() & metrics.keys()
            if tasks[task]["family_id"] != metrics[task].family_id),
        "split_family_mismatches": sorted(
            task for task in tasks.keys() & splits.keys()
            if tasks[task]["family_id"] != splits[task].family_id),
        "invalid_metric_defaults": sorted(
            task for task, profile in metrics.items()
            if not profile.variants or profile.default_variant not in profile.variants),
    }
    rows = []
    for task in definitions:
        task_id = task["id"]
        rows.append({
            "task_id": task_id,
            "task_title": task["title"],
            "family_id": task["family_id"],
            "family_title": families[task["family_id"]],
            "registry_metrics": task.get("metrics", task.get("evaluation")),
            "registry_qualification": task.get("qualification", task.get("required_data")),
            "metric_profile": metrics[task_id].as_dict() if task_id in metrics else None,
            "split_protocol": splits[task_id].as_dict() if task_id in splits else None,
            "dummy_protocols": {variant: profile.as_dict()
                                for variant, profile in dummies.get(task_id, {}).items()},
            "binding_split_metadata": _split_metadata_audit(task),
        })
    default_dummies = [dummies[task][metrics[task].default_variant]
                       for task in tasks.keys() & metrics.keys() & dummies.keys()]
    return {
        "schema": "synrxn.task-evaluation-catalog.v1",
        "status": "passed" if not any(issues.values()) else "failed",
        "summary": {
            "scientific_families": len(families),
            "scientific_tasks": len(tasks),
            "metric_profiles": len(metrics),
            "split_protocols": len(splits),
            "dummy_protocols": len(dummies),
            "generic_dummy_defaults": sum(p.status == "generic_control" for p in default_dummies),
            "specialized_dummy_defaults": sum(p.status == "specialized_adapter_required" for p in default_dummies),
        },
        "registry": {
            "path": str(registry_path) if registry_path is not None else None,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "as_of": registry.get("as_of"),
        },
        "issues": issues,
        "qualification": (
            "Configured evaluators, dummy controls and split requirements are software coverage. "
            "Execute the explicitly selected source-qualified variant, supply its "
            "required metadata and units, and retain denominators and failures. "
            "Binding field roles are inspected without certifying their metadata or native split provenance. "
            "This audit reads no observations and runs no predictive experiment."
        ),
        "tasks": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = task_protocol_catalog(args.registry)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status": report["status"], **report["summary"],
                          "output": str(args.output)}, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
