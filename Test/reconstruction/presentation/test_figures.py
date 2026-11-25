"""Keep the publication hierarchy complete and its source counts unambiguous."""
import copy
import json

import pytest

from synrxn.reconstruction.presentation.figures import build_inventory, generate, render_tikz

from synrxn.reconstruction.paths import REPO, physical_path


def inputs():
    return [json.loads(physical_path(path, repo=REPO).read_text()) for path in (
        "kdd/task_registry.json", "kdd/hierarchy/hierarchy.json")]


def test_figure_covers_tasks_and_deduplicates_reused_sources():
    registry, hierarchy = inputs()
    inventory = build_inventory(registry, hierarchy)
    counts = inventory["summary"]
    assert (counts["families"], counts["tasks"], counts["catalogue_tables"]) == (12, 41, 105)
    assert counts["unique_primary_tables"] == 98
    assert counts["overlapping_views"] == 7
    assert counts["all_task_table_bindings"] == 146
    example = inventory["example"]
    assert example["ncs"]["rows"] == 27438
    assert (len(example["minot_tables"]), example["minot_rows"]) == (4, 6865)
    source = render_tikz(inventory)
    assert {line[8:] for line in source.splitlines() if line.startswith("% task: ")} == {
        task["id"] for task in registry["tasks"]}


def test_unlabelled_tasks_and_changed_bindings_are_rejected():
    registry, hierarchy = inputs()
    changed = copy.deepcopy(registry)
    changed["tasks"][0]["id"] = "new_unlabelled_task"
    with pytest.raises(ValueError, match="labels"):
        build_inventory(changed, hierarchy)
    changed = copy.deepcopy(hierarchy)
    changed["families"][0]["tasks"][0]["tables"][0]["rows"] += 1
    with pytest.raises(ValueError, match="bindings"):
        build_inventory(registry, changed)


def test_portable_sources_and_provenance_are_deterministic(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = generate(REPO, first, build=False, preview=tmp_path / "preview1")
    generate(REPO, second, build=False, preview=tmp_path / "preview2")
    assert [path.name for path in first.iterdir()] == ["hierarchy.tex"]
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    assert len(manifest["input_sha256"]) == 2
    assert set(manifest["style"]["origins"]) == {
        "PhD_thesis/chapters/shared/style/tokens.json",
        "PhD_thesis/chapters/shared/style/figure_style.tex",
        "PhD_thesis/chapters/11/fig/synrxn_layout.tex",
        "PhD_thesis/chapters/11/fig/synrxn_concept.tex",
    }
    assert "PhD_thesis" not in (first / "hierarchy.tex").read_text()
