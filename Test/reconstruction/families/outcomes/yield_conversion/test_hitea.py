"""Check relation closure and absence of labels in preparation decisions."""

from copy import deepcopy

import pytest

from synrxn.reconstruction.families.outcomes.yield_conversion import _hitea as prepare


def test_transitive_relations_stay_together():
    rows = [
        {"sample_id": "a", "screen_id": "s1", "chemical_split_group": "r1"},
        {"sample_id": "b", "screen_id": "s2", "chemical_split_group": "r1"},
        {"sample_id": "c", "screen_id": "s2", "chemical_split_group": "r2"},
        {"sample_id": "d", "screen_id": "s3", "chemical_split_group": "r3"},
    ]
    groups = prepare.component_ids(rows)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]
    assert prepare.component_ids(list(reversed(rows))) == list(reversed(groups))


def test_missing_relations_do_not_merge_observations():
    rows = [{"sample_id": "a", "screen_id": "s1", "notebook_id": ""},
            {"sample_id": "b", "screen_id": "s2", "notebook_id": ""}]
    assert len(set(prepare.component_ids(rows))) == 2
    with pytest.raises(ValueError):
        prepare.component_ids([{"sample_id": "unknown"}])


def test_folds_are_invariant_to_targets_and_input_order():
    rows = [{"sample_id": str(i), "screen_id": str(i), "yield_pct": str(i)} for i in range(15)]
    groups = prepare.component_ids(rows)
    assignment = prepare.assign_folds(groups)
    changed = deepcopy(rows)
    for row in changed:
        row["yield_pct"] = "100"
    assert prepare.assign_folds(prepare.component_ids(changed)) == assignment
    assert prepare.assign_folds(list(reversed(groups))) == assignment
    assert set(assignment.values()) == set(range(5))


def test_context_keeps_zero_and_withholds_outcomes_and_identifiers():
    row = {"Reaction_T": "0.0", "Catalyst_2_Short_Hand": "ligand A", "Reagent_1_eq": "NaN",
           "Product_Selectivity": "0.8", "REACTION_ID": "experiment1", "yie_react": "87"}
    ctx = prepare.context(row, prepare.RAW_CONTEXT)
    assert ctx == {"Reaction_T": "0", "Catalyst_2_Short_Hand": "ligand A"}
    assert prepare.clean("25.00", True) == prepare.clean("25", True)
    assert prepare.clean("", True) != prepare.clean("0", True)


def test_each_declared_relation_is_disjoint_in_grouped_folds():
    rows = [{"sample_id": str(i), "screen_id": str(i // 2), "reactants": str(i // 4)} for i in range(40)]
    groups = prepare.component_ids(rows)
    assignment = prepare.assign_folds(groups)
    folds = [assignment[group] for group in groups]
    assert prepare.relation_overlap(rows, folds, "screen_id") == 0
    assert prepare.relation_overlap(rows, folds, "reactants") == 0
    assert prepare.relation_overlap(rows, list(range(40)), "screen_id") > 0


def test_preparation_does_not_overwrite_edited_review_cards(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    cards = output / "panel_cards.csv"
    cards.write_text("human annotation must be preserved\n")
    (output / "audit.json").write_text('{"initial_panel_cards_sha256": "prior-template-hash"}')
    with pytest.raises(ValueError, match="Panel cards have edits"):
        prepare.prepare(tmp_path / "data", tmp_path / "raw.csv", output)
    assert cards.read_text() == "human annotation must be preserved\n"
