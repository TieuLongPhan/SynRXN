import pytest

from synrxn.split.dependence import build_partitions


def rows():
    return [dict(instance_id=str(i), source_group=f"doc:{i // 2}",
                 transformation_group=f"chem:{i}", family_group=f"family:{i // 3}",
                 upstream_split="test" if i == 11 else "train") for i in range(12)]


@pytest.mark.parametrize("regime", ["source_disjoint", "family_disjoint", "intersection"])
def test_claimed_groups_never_cross_partitions_and_upstream_test_is_not_training(regime):
    records = rows()
    relationships = [dict(left="0", right="7", kind="shared_parent", status="accepted")]
    manifest = build_partitions(records, relationships, regime)
    assert manifest["membership"]["0"] == manifest["membership"]["7"]
    assert manifest["membership"]["11"] == "test"
    fields = ["source_group"] if regime == "source_disjoint" else ["family_group"]
    if regime == "intersection":
        fields.append("source_group")
    for name in fields:
        for a in records:
            for b in records:
                if a[name] == b[name]:
                    assert manifest["membership"][a["instance_id"]] == manifest["membership"][b["instance_id"]]
    assert manifest == build_partitions(list(reversed(records)), relationships, regime)


def test_unknown_primary_source_is_excluded_with_its_hard_group():
    records = rows()
    records[0]["source_group"] = None
    relations = [dict(left="0", right="3", kind="same_observation", status="accepted")]
    manifest = build_partitions(records, relations, "source_disjoint")
    assert "0" in manifest["excluded"] and "3" in manifest["excluded"]
    assert "0" not in manifest["membership"]


def test_uncertain_relationship_is_reported_and_does_not_assert_shared_evidence():
    relation = dict(left="0", right="4", kind="same_observation", status="uncertain")
    manifest = build_partitions(rows(), [relation], "source_disjoint")
    assert manifest["unknown_relationships"] == 1


def test_continuity_reports_existing_overlap_without_changing_upstream_membership():
    records = rows()
    records[10]["upstream_split"] = "train"
    records[11]["source_group"] = records[10]["source_group"]
    manifest = build_partitions(records, [], "continuity")
    assert manifest["membership"]["10"] == "train"
    assert manifest["membership"]["11"] == "test"
    assert any(v["kind"] == "source_group" for v in manifest["cross_partition_relationships"])
