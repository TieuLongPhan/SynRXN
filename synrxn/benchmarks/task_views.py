"""Versioned, source-linked input/target views for the eight database tasks."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, contextmanager
import csv
import gzip
import hashlib
from importlib.resources import files
import io
import json
from pathlib import Path

from synrxn.data.catalog import DatasetCatalog
from synrxn.data.release import artifact_path, selected_datasets
from synrxn.benchmarks.readiness import NUMERIC_TARGETS, matches_selector, present, target_state


def definitions(path: Path | None = None) -> dict:
    raw = path.read_text() if path else files("synrxn.data").joinpath("task_protocols.json").read_text()
    result = json.loads(raw)
    if result["schema"] != "synrxn.task-views.v1":
        raise ValueError("unsupported task-view definition schema")
    return result


def task_definition(task_id: str, path: Path | None = None) -> dict:
    for task in definitions(path)["tasks"]:
        if task["id"] == task_id:
            return task
    raise KeyError("unknown task view: " + task_id)


def _endpoints(row: dict) -> tuple[str, str]:
    reaction = next(
        (row[name] for name in ["reaction_smiles", "rxn", "aam", "rsmi", "reaction"] if present(row.get(name))), ""
    )
    parts = reaction.split(">")
    left, right = (parts[0], parts[2]) if len(parts) == 3 else ("", "")
    return row["reactants"] if present(row.get("reactants")) else left, (
        row["products"] if present(row.get("products")) else right
    )


def make_instance(
    row: dict, task: dict, dataset, source_sha256: str, index: int, seed: int = 42,
    *, propose_split: bool = True,
) -> dict | None:
    """Apply explicit field eligibility; this does not claim model featurizability."""
    left, right = _endpoints(row)
    if not left.strip() or (task["require_product"] and not right.strip()):
        return None
    if task["id"] == "mechanistic_prediction" and (
        row.get("reference_replay_status") != "VALID"
        or row.get("verification_level") not in task["allowed_verification_levels"]
    ):
        return None
    targets, mask = {}, {}
    fields = [target for target in dataset.targets if target in task["targets"]]
    for target in fields:
        value = row.get(target)
        valid = target_state(value, target) == "present"
        if target == "feasible":
            valid = value in {"0", "1"}
        if target == "temperature_c" and row.get("temperature_plausible") == "false":
            valid = False
        mask[target] = valid
        if valid:
            targets[target] = (
                int(value) if target == "feasible" else float(value) if target in NUMERIC_TARGETS else value
            )
    if not targets:
        return None
    if task["id"] == "role_aware_condition" and not set(task["primary_role_targets"]).intersection(targets):
        return None
    if task["id"] == "mechanistic_prediction":
        for target in targets:
            targets[target] = json.loads(targets[target])
        inputs = {"mapped_overall_reaction": left + ">>" + right}
    else:
        inputs = {"reactants": left, "products": right or None}
        if task["id"] == "yield":
            inputs["context"] = {name: row[name] for name in task["context_fields"] if present(row.get(name))}
    group = next((row[name] for name in task["group_priority"] if present(row.get(name))), None)
    if group is None and task.get("grouping_rule") == "directionless_exact_endpoints":
        group = (
            "raw-undirected-"
            + hashlib.sha256(json.dumps(sorted([left, right]), separators=(",", ":")).encode()).hexdigest()
        )
    group = group or "raw-" + hashlib.sha256((left + ">>" + right).encode()).hexdigest()
    proposed = None
    if propose_split:
        number = int(hashlib.sha256((str(seed) + ":" + group).encode()).hexdigest()[:16], 16) % 10000
        proposed = "train" if number < 8000 else "validation" if number < 9000 else "test"
        # The expert set remains held out rather than becoming training data.
        if dataset.dataset_id.key == "feasibility/frea_expert.csv.gz":
            proposed = "test"
    native_split = row.get(dataset.split_column) or row.get("upstream_split") or None
    locator = dataset.dataset_id.key + ":" + source_sha256 + ":" + str(index)
    observation_id = "row-" + hashlib.sha256(locator.encode()).hexdigest()
    return {
        "instance_id": task["id"] + ":" + observation_id,
        "observation_id": observation_id,
        "inputs": inputs,
        "targets": targets,
        "target_mask": mask,
        "group": group,
        "source_split": native_split,
        "source_split_status": dataset.split_status,
        "proposed_split": proposed,
        "source_record_id": row.get(dataset.row_identifier),
        "source_row": index,
        "evidence_tier": row.get("evidence_tier") or row.get("verification_level"),
        "source_dataset": dataset.dataset_id.key,
    }


@contextmanager
def _jsonl(path: Path):
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=1) as gz,
        io.TextIOWrapper(gz, encoding="utf8") as handle,
    ):
        yield handle


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def export_task_view(
    task_id: str,
    dataset_key: str,
    data_dir: Path,
    output: Path,
    *,
    protocol_path: Path | None = None,
    seed: int = 42,
    limit: int | None = None,
    propose_split: bool = True,
) -> dict:
    """Write paired input and target files with optional proposed assignments.

    :param propose_split: Generate the standard diagnostic assignment when true.
        Recorded source assignments are retained in either mode.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    task = task_definition(task_id, protocol_path)
    catalog = DatasetCatalog(data_dir / "metadata.yaml" if (data_dir / "metadata.yaml").exists() else None)
    candidates = [item for partition in ("main", "sharealike") for item in selected_datasets(catalog, partition)]
    dataset = next((item for item in candidates if item.dataset_id.key == dataset_key), None)
    if dataset is None or not any(
        matches_selector({"key": dataset_key, "targets": list(dataset.targets)}, selector)
        for selector in task["selectors"]
    ):
        raise ValueError("dataset is not a canonical member of this task view")
    source = artifact_path(data_dir, dataset_key)
    before = source.stat()
    source_sha = _digest(source)
    expert_groups = set()
    if dataset_key == "feasibility/frea_corpus.csv.gz":
        expert_path = artifact_path(data_dir, "feasibility/frea_expert.csv.gz")
        if expert_path.exists():
            with gzip.open(expert_path, "rt", encoding="utf8", newline="") as handle:
                expert_groups = {row["exact_reaction_group"] for row in csv.DictReader(handle)}
    if output.exists():
        raise ValueError("task-view output must be a fresh directory")
    output.mkdir(parents=True)
    paths = [
        output / name for name in ("inputs.jsonl.gz", "targets.jsonl.gz", "membership.jsonl.gz", "exclusions.jsonl.gz")
    ]
    counts, tiers, fields, excluded = Counter(), Counter(), Counter(), Counter()
    expert_overlap = 0
    accepted = scanned = 0
    csv.field_size_limit(16 * 1024 * 1024)
    with ExitStack() as stack:
        handles = [stack.enter_context(_jsonl(path)) for path in paths]
        handle = stack.enter_context(gzip.open(source, "rt", newline="", encoding="utf8"))
        for index, row in enumerate(csv.DictReader(handle)):
            if limit is not None and accepted >= limit:
                break
            scanned += 1
            instance = make_instance(row, task, dataset, source_sha, index, seed, propose_split=propose_split)
            if instance is None:
                left, right = _endpoints(row)
                reason = (
                    "missing_required_endpoints"
                    if not left.strip() or (task["require_product"] and not right.strip())
                    else (
                        "outside_primary_mechanism_tier"
                        if task_id == "mechanistic_prediction"
                        else "no_observed_valid_task_target"
                    )
                )
                excluded[reason] += 1
                handles[3].write(json.dumps({"source_row": index, "reason": reason}, sort_keys=True) + "\n")
                continue
            if instance["group"] in expert_groups:
                if propose_split:
                    instance["proposed_split"] = "test"
                expert_overlap += 1
            input_record = {"instance_id": instance["instance_id"], "inputs": instance["inputs"]}
            target_record = {key: value for key, value in instance.items() if key not in {"inputs", "proposed_split"}}
            membership = {
                "instance_id": instance["instance_id"],
                "group": instance["group"],
                "proposed_split": instance["proposed_split"],
                "source_split": instance["source_split"],
            }
            for stream, payload in zip(handles[:3], (input_record, target_record, membership)):
                stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            accepted += 1
            if propose_split:
                counts[instance["proposed_split"]] += 1
            tiers[instance["evidence_tier"] or "source_annotation"] += 1
            fields.update(instance["targets"].keys())
            if accepted % 500000 == 0:
                print(f"{task_id}/{dataset.name}: exported {accepted:,} field-eligible records", flush=True)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or _digest(source) != source_sha:
        raise ValueError("source changed during task-view export")
    report = {
        "schema": "synrxn.task-view-export.v1",
        "task": task_id,
        "dataset": dataset_key,
        "source_sha256": source_sha,
        "protocol": task,
        "source_rows_scanned": scanned,
        "field_eligible_rows": accepted,
        "field_excluded_rows": scanned - accepted,
        "preview_limit": limit,
        "seed": seed if propose_split else None,
        "proposed_split_counts": dict(counts),
        "evidence_tiers": dict(tiers),
        "exclusion_reasons": dict(excluded),
        "corpus_rows_with_exact_expert_group": expert_overlap,
        "observed_target_counts": dict(fields),
        "files_sha256": {p.name: _digest(p) for p in paths},
        "target_units": dataset.target_units,
        "target_kinds": dataset.target_kinds,
        "limits": [
            "Field eligibility does not certify model featurization or scientific independence.",
            "Proposed group allocation uses recorded or exact-text keys; source splits remain separate."
            if propose_split else "No evaluation split was assigned. Source splits are recorded metadata.",
            "This is data preparation, without predictor fitting, inference or benchmark scores.",
        ],
    }
    (output / "view-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
