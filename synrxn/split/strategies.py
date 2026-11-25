"""Deterministic task splits with explicit lineage and leakage checks.

All indices are zero-based input positions. No records are silently excluded.
Known dependence components take precedence over row balance and stratification.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
import hashlib
import json
import math
import random
from typing import Any

from .protocols import get_task_split_protocol


PARTITIONS = ("train", "validation", "test")
_INTRINSIC_FIELDS = (
    "observation_group", "parent_observation_id", "parent_record_id",
    "parent_reaction_id", "parent_mechanism_id", "trajectory_id", "episode_id",
    "split_component",
)
_MEMBERSHIP_FIELDS = {
    "split", "native_split", "derived_split", "proposed_split", "source_split",
    "upstream_split", "outer_fold", "inner_fold", "fold", "fold_id",
    "fold_index", "partition", "partition_id", "membership", "split_label",
    "fold_label", "partition_label",
}
_TRANSFER_FIELDS = {
    "source_disjoint": ("source_group", "source_document_id", "source_publication_doi", "reference_doi"),
    "family_disjoint": ("family_group", "reaction_family", "mechanism_family"),
    "scaffold_disjoint": ("scaffold_group", "scaffold_id"),
    "protein_disjoint": ("protein_cluster", "homology_group", "native_cluster_40", "native_cluster_80"),
    "campaign_disjoint": ("campaign_id", "campaign_assay_group", "campaign_group"),
    "entity_disjoint": ("entity_group", "reactor_geometry_id", "feedstock_family", "system_id"),
}
_HARD_RELATIONSHIPS = {
    "same_observation", "reprinted_observation", "shared_parent", "same_trajectory",
    "same_campaign", "same_assay",
}
STRATEGIES = (
    "group", "stratified_group", "random_diagnostic", "native", "chronological",
    "group_chronological", *_TRANSFER_FIELDS,
)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, float) and not math.isfinite(value))


def _key(value: Any) -> str:
    if _missing(value):
        raise ValueError("Missing/non-finite grouping metadata")
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Grouping metadata must be JSON-serializable") from exc


class _Union:
    def __init__(self, n):
        self.parents = list(range(n))

    def find(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def join(self, a, b):
        left, right = self.find(a), self.find(b)
        self.parents[max(left, right)] = min(left, right)

    def components(self):
        groups = defaultdict(list)
        for position in range(len(self.parents)):
            groups[self.find(position)].append(position)
        return list(groups.values())


def _require_field(records, field):
    missing = [position for position, row in enumerate(records) if _missing(row.get(field))]
    if missing:
        raise ValueError(f"Metadata field {field!r} is required for every row; missing at positions {missing[:8]}")


def _select_field(records, candidates, description):
    for field in candidates:
        if all(not _missing(row.get(field)) for row in records):
            return field
    raise ValueError(f"{description} requires complete, audited metadata in one of {tuple(candidates)!r}; none is available")


def _is_membership_field(field):
    normalized = field.lower()
    return normalized in _MEMBERSHIP_FIELDS or normalized.endswith(
        ("_split", "_fold", "_partition", "_membership"))


def _join_field(union, records, field, *, require_complete):
    if require_complete:
        _require_field(records, field)
    anchors = {}
    for position, row in enumerate(records):
        if _missing(row.get(field)):
            continue
        value = _key(row[field])
        if value in anchors:
            union.join(anchors[value], position)
        else:
            anchors[value] = position


def _time(value):
    if isinstance(value, bool):
        raise ValueError("Boolean values are not timestamps")
    if isinstance(value, (float, int)):
        if not math.isfinite(value):
            raise ValueError("Timestamps must be finite")
        return "numeric", float(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Timestamp is not ISO-8601: {value!r}") from exc
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    else:
        raise ValueError("Use finite numerical timestamps or ISO-8601 dates/times")
    # Naive dates/times have an explicit, deterministic UTC interpretation.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return "datetime", parsed.timestamp()


def _chronological_components(union, times):
    """Join interleaved intervals so every split boundary is strictly causal."""
    intervals = sorted((min(times[i] for i in group), max(times[i] for i in group), group)
                       for group in union.components())
    active = None
    end = -math.inf
    for start, stop, group in intervals:
        if active is not None and start <= end:
            union.join(active, group[0])
            end = max(end, stop)
        else:
            active, end = group[0], stop
    return sorted(union.components(), key=lambda group: (min(times[i] for i in group), group[0]))


def _ratios(ratios):
    if len(ratios) != 3 or any(not math.isfinite(x) or x < 0 for x in ratios):
        raise ValueError("ratios must be three finite nonnegative values")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("ratios must sum to one")
    if sum(x > 0 for x in ratios) < 2:
        raise ValueError("At least two partitions must have positive ratios")
    return tuple(float(x) for x in ratios)


def _balanced_assignment(groups, labels, ratios, seed):
    rng = random.Random(seed)
    groups = list(groups)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    targets = [sum(map(len, groups)) * ratio for ratio in ratios]
    enabled = [index for index, ratio in enumerate(ratios) if ratio]
    if len(groups) < len(enabled):
        raise ValueError(f"Only {len(groups)} independent components for {len(enabled)} requested nonempty partitions")
    counts = [0, 0, 0]
    assigned = [[] for _ in PARTITIONS]
    total_labels = Counter(labels) if labels is not None else Counter()
    assigned_labels = [Counter() for _ in PARTITIONS]
    for ordinal, group in enumerate(groups):
        empty = [index for index in enabled if not assigned[index]]
        choices = empty if len(groups) - ordinal == len(empty) else enabled
        local_labels = Counter(labels[i] for i in group) if labels is not None else Counter()

        def cost(index):
            size_cost = sum(((counts[j] + (len(group) if j == index else 0) - targets[j]) ** 2)
                            / max(targets[j], 1) for j in enabled)
            label_cost = 0.0
            for label, total in total_labels.items():
                for j in enabled:
                    target = total * ratios[j]
                    actual = assigned_labels[j][label] + (local_labels[label] if j == index else 0)
                    label_cost += (actual - target) ** 2 / max(target, 1)
            return size_cost + label_cost

        chosen = min(choices, key=lambda index: (cost(index), index))
        assigned[chosen].extend(group)
        counts[chosen] += len(group)
        assigned_labels[chosen].update(local_labels)
    return {position: PARTITIONS[index] for index, group in enumerate(assigned) for position in group}


def _ordered_assignment(groups, ratios):
    enabled = [index for index, ratio in enumerate(ratios) if ratio]
    if len(groups) < len(enabled):
        raise ValueError(f"Only {len(groups)} temporally separable components for {len(enabled)} requested nonempty partitions")
    # Select contiguous boundaries nearest the desired cumulative row counts,
    # reserving at least one whole independent component per remaining split.
    total = sum(map(len, groups))
    cumulative = [0]
    for group in groups:
        cumulative.append(cumulative[-1] + len(group))
    cursor = 0
    membership = {}
    target = 0.0
    for offset, partition in enumerate(enabled):
        target += ratios[partition] * total
        remaining = len(enabled) - offset - 1
        if remaining:
            stop = min(range(cursor + 1, len(groups) - remaining + 1),
                       key=lambda boundary: (abs(cumulative[boundary] - target), boundary))
        else:
            stop = len(groups)
        for group in groups[cursor:stop]:
            for position in group:
                membership[position] = PARTITIONS[partition]
        cursor = stop
    return membership


def make_task_split(
    task_id: str, records: Iterable[Mapping[str, Any]], *, strategy: str | None = None,
    group_fields: Iterable[str] | None = None, seed: int = 0,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    id_field: str = "instance_id", native_field: str = "native_split",
    timestamp_field: str | None = None, label_field: str | None = None,
    relationships: Iterable[Mapping[str, Any]] = (),
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Split input positions without fabricating missing lineage.

    ``group_fields`` declares fields whose equal values must each remain whole:
    this is a transitive union, not grouping by the tuple of their values. The
    default uses the task's first complete recommended field. Assignment columns
    such as ``native_split`` and ``outer_fold`` are not dependency identifiers.
    Physical parent-observation and trajectory fields are enforced whenever
    present. Document, panel, mechanism and trajectory requirements declared by
    the task also apply to native, transfer and random diagnostic strategies.
    Campaign and broad parent groups are caller-selected transfer axes, rather
    than unconditional physical-duplicate constraints. Accepted relationship edges
    use row identifiers from ``id_field``. Native split aliases ``val`` and
    ``valid`` normalize to ``validation``; other unknown labels are rejected.

    Chronological strategies merge overlapping dependence-group time intervals;
    if this prevents the requested number of partitions, they fail explicitly.
    Label distributions/unseen labels are always reported when supplied; group
    integrity takes priority over an approximate stratification ratio.
    """
    protocol = get_task_split_protocol(task_id)
    strategy = strategy or protocol.recommended_strategy
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown split strategy {strategy!r}; choose from {STRATEGIES!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    ratios = _ratios(ratios)
    records = list(records)
    if not records or any(not isinstance(row, Mapping) for row in records):
        raise ValueError("records must be a nonempty iterable of mapping records")
    _require_field(records, id_field)
    identifiers = [_key(row[id_field]) for row in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate {id_field!r} values are not allowed")
    by_id = {identifier: index for index, identifier in enumerate(identifiers)}
    relationships = [dict(edge) for edge in relationships]
    accepted_relationships = []
    relationship_positions = set()
    for edge in relationships:
        left, right = _key(edge.get("left")), _key(edge.get("right"))
        if left not in by_id or right not in by_id:
            raise ValueError("Relationship endpoint lies outside the selected records")
        if edge.get("status") == "accepted" and edge.get("kind") in _HARD_RELATIONSHIPS:
            positions = (by_id[left], by_id[right])
            accepted_relationships.append(positions)
            if positions[0] != positions[1]:
                relationship_positions.update(positions)
    relationship_coverage_complete = len(relationship_positions) == len(records)
    if isinstance(group_fields, str):
        raise ValueError("group_fields must be a sequence of field names, not a string")
    declared = tuple(dict.fromkeys(group_fields or ()))
    if any(not isinstance(field, str) or not field for field in declared):
        raise ValueError("group_fields entries must be nonempty field names")
    if any(_is_membership_field(field) for field in declared):
        raise ValueError("Split/fold membership labels are assignments, not dependency group identifiers")
    explicit_group_fields = declared
    required_candidates = protocol.required_group_field_candidates
    required_field = next((field for field in required_candidates
                           if all(not _missing(row.get(field)) for row in records)), None)
    required_resolution = None
    if required_candidates:
        if required_field:
            declared = tuple(dict.fromkeys((*declared, required_field)))
            required_resolution = "protocol_field"
        elif explicit_group_fields:
            required_resolution = "explicit_group_fields"
        elif relationship_coverage_complete:
            required_resolution = "accepted_relationships"
        else:
            raise ValueError(
                f"{protocol.independence_unit} requires complete, audited metadata in "
                f"one of {required_candidates!r}, explicit whole-unit-equivalent "
                "group_fields, or accepted hard relationships covering every row")
    if not declared and strategy in {"group", "stratified_group", "group_chronological"}:
        if required_resolution != "accepted_relationships":
            declared = (_select_field(records, protocol.group_field_candidates, protocol.independence_unit),)
    if not declared and strategy == "native":
        # Published labels do not override known dependence. If the primary
        # grouping metadata exists, validate it instead of merely echoing labels.
        declared = tuple(field for field in protocol.group_field_candidates
                         if all(not _missing(row.get(field)) for row in records))[:1]
    transfer = None
    if strategy in _TRANSFER_FIELDS:
        candidates = _TRANSFER_FIELDS[strategy]
        transfer = _select_field(records, candidates, strategy)
        declared = tuple(dict.fromkeys((*declared, transfer)))
    union = _Union(len(records))
    intrinsic = [field for field in _INTRINSIC_FIELDS if any(not _missing(row.get(field)) for row in records)]
    for field in declared:
        _join_field(union, records, field, require_complete=True)
    for field in intrinsic:
        _join_field(union, records, field, require_complete=False)
    for left, right in accepted_relationships:
        union.join(left, right)
    applied_relationships = len(accepted_relationships)
    timestamp_values = None
    if strategy in {"chronological", "group_chronological"}:
        if not timestamp_field:
            raise ValueError("A timestamp_field is required for chronological splitting")
        _require_field(records, timestamp_field)
        parsed = [_time(row[timestamp_field]) for row in records]
        if len({kind for kind, value in parsed}) != 1:
            raise ValueError("Do not mix numerical and ISO-8601 timestamps")
        timestamp_values = [value for kind, value in parsed]
        groups = _chronological_components(union, timestamp_values)
        membership = _ordered_assignment(groups, ratios)
    else:
        groups = union.components()
        if strategy == "native":
            _require_field(records, native_field)
            aliases = {"train": "train", "validation": "validation", "val": "validation", "valid": "validation", "test": "test"}
            membership = {}
            for index, row in enumerate(records):
                if not isinstance(row[native_field], str) or row[native_field] not in aliases:
                    raise ValueError(f"Unknown native partition {row[native_field]!r} at position {index}")
                membership[index] = aliases[row[native_field]]
            if not all(any(partition == name for partition in membership.values()) for name in ("train", "test")):
                raise ValueError("Native partitions must contain training and test records")
            for group in groups:
                if len({membership[index] for index in group}) != 1:
                    raise ValueError("Native partitions split a declared dependence group; source labels cannot be silently repaired")
        else:
            if strategy == "stratified_group" and not label_field:
                raise ValueError("stratified_group requires a label_field")
            labels = None
            if strategy == "stratified_group":
                _require_field(records, label_field)
                labels = [_key(row[label_field]) for row in records]
            membership = _balanced_assignment(groups, labels, ratios, seed)
    if set(membership) != set(range(len(records))):
        raise RuntimeError("Split failed to cover every input row exactly once")
    for group in groups:
        if len({membership[position] for position in group}) != 1:
            raise RuntimeError("Split violated a declared dependence component")
    indices = {partition: sorted(index for index, value in membership.items() if value == partition)
               for partition in PARTITIONS}
    group_ids = {}
    for group in groups:
        group_id = _digest(sorted(identifiers[index] for index in group))
        for index in group:
            group_ids[str(index)] = group_id
    label_summary = None
    if label_field:
        _require_field(records, label_field)
        labels = [_key(row[label_field]) for row in records]
        distributions = {partition: dict(sorted(Counter(labels[index] for index in positions).items()))
                         for partition, positions in indices.items()}
        train_labels = set(distributions["train"])
        all_labels = set(labels)
        label_summary = {
            "field": label_field, "counts": distributions,
            "label_keys": "JSON-encoded original values; numeric and string classes stay distinct",
            "unseen_in_training": {partition: sorted(set(distributions[partition]) - train_labels)
                                   for partition in ("validation", "test")},
            "missing_from_partition": {partition: sorted(all_labels - set(distributions[partition]))
                                       for partition in PARTITIONS},
            "policy": "No labels or rows dropped; rare labels may prevent class-balanced group splits.",
        }
    metadata_fields = tuple(dict.fromkeys((id_field, *declared, *intrinsic,
                                          *([native_field] if strategy == "native" else []),
                                          *([timestamp_field] if timestamp_values is not None else []),
                                          *([label_field] if label_field else []))))
    # Hash normalized metadata only; do not pretend this is a source payload hash.
    metadata = [{field: (None if _missing(row.get(field)) else
                        str(row[field]) if isinstance(row.get(field), (date, datetime)) else row.get(field))
                 for field in metadata_fields} for row in records]
    provenance = dict(provenance or {})
    available = sorted({field for candidates in _TRANSFER_FIELDS.values() for field in candidates
                        if all(not _missing(row.get(field)) for row in records)})
    partial_intrinsic = {field: sum(_missing(row.get(field)) for row in records)
                         for field in intrinsic if any(_missing(row.get(field)) for row in records)}
    result = {
        "schema": "synrxn.task-split.v1", "task_id": task_id,
        "family_id": protocol.family_id, "strategy": strategy, "seed": seed,
        "requested_ratios": dict(zip(PARTITIONS, ratios)) if strategy != "native" else None,
        "indices": indices, "membership": {str(index): membership[index] for index in sorted(membership)},
        "row_ids": [row[id_field] for row in records], "id_field": id_field,
        "counts": {partition: len(positions) for partition, positions in indices.items()},
        "observed_ratios": {partition: len(positions) / len(records) for partition, positions in indices.items()},
        "group_ids": group_ids, "component_count": len(groups),
        "largest_component": max(map(len, groups)),
        "declared_group_fields": list(declared), "enforced_intrinsic_fields": intrinsic,
        "explicit_group_fields": list(explicit_group_fields),
        "selected_transfer_field": transfer,
        "required_independence_unit": {
            "unit": protocol.independence_unit if required_candidates else None,
            "resolution": required_resolution,
            "selected_field": required_field,
            "accepted_relationship_rows": len(relationship_positions),
        },
        "incomplete_intrinsic_metadata": partial_intrinsic,
        "available_transfer_metadata": available,
        "applied_hard_relationships": applied_relationships,
        "unapplied_relationships": len(relationships) - applied_relationships,
        "empty_partitions": [partition for partition in PARTITIONS if not indices[partition]],
        "excluded_rows": [], "label_summary": label_summary,
        "timestamp_field": timestamp_field if timestamp_values is not None else None,
        "temporal_ranges": ({partition: [min(timestamp_values[index] for index in positions),
                                         max(timestamp_values[index] for index in positions)] if positions else None
                             for partition, positions in indices.items()} if timestamp_values is not None else None),
        "protocol": protocol.as_dict(), "protocol_sha256": _digest(protocol.as_dict()),
        "input_metadata_sha256": _digest(metadata), "relationships_sha256": _digest(relationships),
        "source_provenance": provenance,
        "qualification": [
            protocol.qualification,
            "Only supplied group identifiers and accepted hard relationships are enforced; absent lineage is not inferred.",
            "A split does not enforce causal feature construction, train-only preprocessing, candidate-library policy or target-specific eligibility.",
            "Reuse one manifest across task projections; independently splitting different record sets does not ensure cross-task isolation.",
            "Random diagnostic assignment preserves enforced hard units; it is not an unrestricted row shuffle or evidence of source/chemistry transfer." if strategy == "random_diagnostic" else
            "Transfer claims are limited to the explicitly enforced group fields; other available fields do not imply disjointness.",
        ],
    }
    if partial_intrinsic:
        result["qualification"].append("Some intrinsic lineage values are missing; known equal values are grouped but independence for unknown rows is unqualified.")
    if required_resolution in {"explicit_group_fields", "accepted_relationships"}:
        result["qualification"].append("The caller declares the supplied groups/relationships to cover the required whole unit; semantic completeness requires a source audit.")
    # A reproducible report must itself be serializable (including caller provenance).
    _digest(result)
    return result
