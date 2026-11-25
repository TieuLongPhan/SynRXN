"""Auditable source/family partitions with typed hard-dependence constraints."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from typing import Any

HARD_RELATIONSHIPS = {"same_observation", "reprinted_observation", "shared_parent"}


class _EvidenceGroups:
    def __init__(self, identifiers):
        self.parent = {key: key for key in identifiers}

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def join(self, left, right):
        a, b = self.find(left), self.find(right)
        self.parent[max(a, b)] = min(a, b)


def _group_records(records, relationships, fields):
    by_id = {r["instance_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate task instances")
    union = _EvidenceGroups(by_id)
    for relationship in relationships:
        if not {relationship["left"], relationship["right"]} <= set(by_id):
            raise ValueError("relationship points outside the selected records")
        if relationship.get("status") == "accepted" and relationship["kind"] in HARD_RELATIONSHIPS:
            union.join(relationship["left"], relationship["right"])
    excluded = {}
    for record in records:
        for name in fields:
            if not record.get(name):
                excluded[record["instance_id"]] = f"unknown_{name}"
    excluded_roots = {union.find(key) for key in excluded}
    for record in records:
        if union.find(record["instance_id"]) in excluded_roots:
            excluded.setdefault(record["instance_id"], "hard_group_has_unknown_lineage")
    for name in fields:
        anchors = {}
        for record in records:
            key = record["instance_id"]
            if key in excluded:
                continue
            group = record[name]
            if group in anchors:
                union.join(anchors[group], key)
            else:
                anchors[group] = key
    groups = defaultdict(list)
    for key in by_id:
        if key not in excluded:
            groups[union.find(key)].append(key)
    return by_id, groups, excluded


def _assign_groups(by_id, groups, regime, seed, ratios, preserve_upstream_test):
    membership, group_ids = {}, {}
    for keys in groups.values():
        group_id = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
        value = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16], 16) / 2**64
        partition = "train" if value < ratios[0] else "validation" if value < sum(ratios[:2]) else "test"
        if preserve_upstream_test and any(by_id[k].get("upstream_split") == "test" for k in keys):
            partition = "test"
        for key in keys:
            group_ids[key] = group_id
            if regime == "continuity":
                original = by_id[key].get("upstream_split")
                if original not in {"train", "val", "valid", "validation", "test"}:
                    raise ValueError("continuity requires known upstream memberships")
                membership[key] = "validation" if original in {"val", "valid"} else original
            else:
                membership[key] = partition
    return membership, group_ids


def _group_violations(by_id, groups, membership):
    violations = []
    for name in ["transformation_group", "source_group", "family_group"]:
        positions = defaultdict(set)
        for key, partition in membership.items():
            value = by_id[key].get(name)
            if value:
                positions[value].add(partition)
        violations.extend({"kind": name, "group": group, "partitions": sorted(parts)}
                          for group, parts in positions.items() if len(parts) > 1)
    for group, keys in groups.items():
        parts = {membership[k] for k in keys}
        if len(parts) > 1:
            violations.append({"kind": "hard_dependence", "group": group,
                               "partitions": sorted(parts)})
    return sorted(violations, key=lambda v: (v["kind"], v["group"]))


def build_partitions(
    records: list[dict[str, Any]], relationships: list[dict[str, str]],
    regime: str, seed: int = 1,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    preserve_upstream_test: bool = True,
) -> dict[str, Any]:
    """Keep evidence groups whole; chemical families stay a separate relation.

    Source/family views conservatively group identical transformations too.
    Unknown lineage is excluded from source-independence claims. Continuity
    preserves upstream labels and reports violations rather than changing them.
    """
    if regime not in {"continuity", "source_disjoint", "family_disjoint", "intersection"}:
        raise ValueError("unknown partition regime")
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("split ratios must be nonnegative and sum to one")
    fields = []
    if regime != "continuity":
        fields.append("transformation_group")
    if regime in {"source_disjoint", "intersection"}:
        fields.append("source_group")
    if regime in {"family_disjoint", "intersection"}:
        fields.append("family_group")
    by_id, groups, excluded = _group_records(records, relationships, fields)
    membership, group_ids = _assign_groups(by_id, groups, regime, seed, ratios, preserve_upstream_test)
    violations = _group_violations(by_id, groups, membership)
    if regime != "continuity" and any(v["kind"] in fields + ["hard_dependence"] for v in violations):
        raise RuntimeError("partition generation broke a declared independence constraint")
    counts = Counter(membership.values())
    return {
        "protocol_version": "pilot-groups-v1", "regime": regime, "seed": seed,
        "source_unit": "reported_document_identifier", "family_definition": "mapped-changed-bond-element-signature-v1",
        "membership": membership, "group_ids": group_ids, "excluded": excluded,
        "counts": dict(counts), "group_count": len(groups),
        "largest_group": max((len(keys) for keys in groups.values()), default=0),
        "unknown_relationships": sum(r.get("status") != "accepted" for r in relationships),
        "cross_partition_relationships": violations,
        "empty_partitions": [p for p in ("train", "validation", "test") if not counts[p]],
        "preserve_upstream_test": preserve_upstream_test,
        "limitations": ["Document identifiers are not audited patent-family lineage.",
                        "Family signatures depend on supplied mapping annotations and do not define mechanisms.",
                        "The limited pilot population cannot establish population-wide generalization."],
    }
