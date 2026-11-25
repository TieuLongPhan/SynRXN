"""Task coverage, metadata navigation and explicit output generation."""

import copy
import json
import os
import subprocess
import sys

import pytest

from synrxn.reconstruction.catalogue import index as module
from synrxn.reconstruction.families import suite
from synrxn.reconstruction.coverage import report, sync_plan
from synrxn.reconstruction.paths import REPO, RETIRED_METADATA, physical_path, source_path

ROOT = physical_path("kdd")


def registry():
    return json.loads((ROOT / "task_registry.json").read_text())


def test_all_original_interfaces_are_preserved_with_distinct_condition_variants():
    data = registry()
    original = json.loads((source_path(REPO / "synrxn/data/task_protocols.json")).read_text())["tasks"]
    assert set(data["legacy_view_map"]) == {task["id"] for task in original}
    by_id = {task["id"]: task for task in data["tasks"]}
    condition = by_id[data["legacy_view_map"]["reagent_set"]]
    assert condition["id"] == data["legacy_view_map"]["role_aware_condition"]
    assert set(condition["legacy_view_ids"]) == {"reagent_set", "role_aware_condition"}
    assert {binding["key"] for binding in condition["bindings"]} >= {
        "synthesis/uspto_500", "condition_sharealike/ord_hte_condition"}


def test_every_local_source_table_is_assigned_without_counting_overlap_as_primary():
    data = registry()
    expected = {item["id"] for path in ["doc/_static/catalog-data.json", "doc/_static/catalog-sharealike-data.json"]
                for item in json.loads((source_path(REPO / path)).read_text())["datasets"]}
    actual = {item["key"] for task in data["tasks"] for item in task["bindings"] if item["namespace"] == "canonical"}
    assert expected == actual
    expansion = json.loads((source_path(REPO / "Data/expansion/catalog.json")).read_text())["tables"]
    assigned = {item["key"] for task in data["tasks"] for item in task["bindings"] if item["namespace"] == "expansion"}
    assert assigned == {item["key"] for item in expansion}
    for task in data["tasks"]:
        primary = [item for item in task["bindings"] if item["role"] != "overlapping_view"]
        assert task["primary_table_count"] == len(primary)
        assert task["primary_source_rows"] == sum(item["rows"] for item in primary)


def test_barriers_and_reaction_energies_remain_separate_despite_shared_rows():
    tasks = {task["id"]: task for task in registry()["tasks"]}
    barrier = {item["key"] for item in tasks["activation_barrier"]["bindings"]}
    energy = {item["key"] for item in tasks["reaction_energetics"]["bindings"]}
    assert {"property/b97xd3", "property/cycloadd"} <= barrier & energy
    assert "property/rad6re" in energy - barrier
    assert "property/lograte" in {item["key"] for item in tasks["reaction_rates"]["bindings"]}


def test_audited_overlaps_count_the_primary_observations_once():
    tasks = {task["id"]: task for task in registry()["tasks"]}
    pairs = [
        ("atom_mapping", "aam/natcomm", "canonical:aam/golden", 491, 1758),
        ("activation_barrier", "property/snar", "coverage:solution_context/snar_solution_context", 503, 503),
        ("activation_barrier", "property/cycloadd", "coverage:solution_context/cycloadd_solution_context", 5269, 5269),
        ("reaction_energetics", "property/cycloadd", "coverage:solution_context/cycloadd_solution_context", 5269, 5269),
        ("product_distribution", "C06/single_solvent", "expansion:C06/catechol", 656, 1227),
        ("sequential_optimization", "C06/single_solvent", "expansion:C06/catechol", 656, 1227),
    ]
    for task, overlap, primary_id, subset_rows, primary_rows in pairs:
        bindings = {b["key"]: b for b in tasks[task]["bindings"]}
        assert bindings[overlap]["counted_under"] == primary_id
        assert bindings[overlap]["role"] == "overlapping_view"
        assert bindings[overlap]["rows"] == subset_rows
        namespace, key = primary_id.split(":", 1)
        assert (bindings[key]["role"], bindings[key]["namespace"], bindings[key]["rows"]) == (
            "primary", namespace, primary_rows
        )


def test_feasibility_success_and_assay_activity_do_not_share_a_label_contract():
    tasks = {task["id"]: task for task in registry()["tasks"]}
    feasibility = {item["key"] for item in tasks["reaction_feasibility"]["bindings"]}
    success = {item["key"] for item in tasks["reaction_success"]["bindings"]}
    activity = {item["key"] for item in tasks["enzyme_activity"]["bindings"]}
    assert feasibility.isdisjoint(success)
    assert activity == {"classification/ehreact_activity"}
    assert feasibility.isdisjoint(activity)


def test_task_definitions_are_not_reported_as_completed_benchmarks():
    data = registry()
    assert all(task["benchmark_result_status"] == "not completed by this planning work" for task in data["tasks"])
    active = {task["id"] for task in data["tasks"]}
    assert active.isdisjoint(task["id"] for task in data["future_tasks"])
    assert all(task["bindings"] for task in data["tasks"])
    assert {key for family in data["families"] for key in family["task_ids"]} == active


def test_roadmap_families_have_explicit_tasks_without_invented_dataset_coverage():
    data = registry()
    policy = json.loads((ROOT / "scope_policy.json").read_text())
    assert len(data["families"]) == len(policy["families"]) == 12
    local_families = {family["id"] for family in data["families"] if family["task_ids"]}
    empty_families = {family["id"] for family in data["families"] if not family["task_ids"]}
    assert len(local_families) == data["summary"]["families_with_local_task_data"]
    assert len(local_families) + len(empty_families) == 12
    future = {task["id"]: task for task in data["future_tasks"]}
    assert len(future) == data["summary"]["future_task_definitions"]
    initial_ids = {task["id"] for task in policy["future_tasks"]}
    promoted = set(data["promoted_roadmap_task_ids"])
    assert len(initial_ids) == data["summary"]["initial_roadmap_task_definitions"] == 19
    assert initial_ids == promoted | set(future)
    assert promoted.isdisjoint(future)
    assert len(promoted) == data["summary"]["promoted_roadmap_task_definitions"]
    for family in data["families"]:
        assert family["task_ids"] or family["future_task_ids"]
        assert all(future[key]["family_id"] == family["id"] for key in family["future_task_ids"])
    assert {key for family in data["families"] for key in family["future_task_ids"]} == set(future)
    for task in future.values():
        assert task["bindings"] == []
        assert task["required_data"] and task["evaluation"]
        assert task["benchmark_result_status"] == "not completed by this planning work"


def test_route_planning_is_excluded_but_single_step_and_physical_networks_remain():
    data = registry()
    active = {task["id"] for task in data["tasks"]}
    future = {task["id"] for task in data["future_tasks"]}
    excluded = {task["id"] for task in data["excluded_tasks"]}
    assert "multistep_planning" in excluded
    assert excluded.isdisjoint(active | future)
    assert "retrosynthesis" in active
    assert "sequential_optimization" in active
    assert "reaction_network_modeling" in active | future
    assert data["extension_scope_reconciliation"]["multistep_planning"].startswith("excluded")
    # The scope overlay must not rewrite the historical acquisition inventory.
    source = json.loads((source_path(REPO / "synrxn/data/expansion_tasks.json")).read_text())
    assert "multistep_planning" in {task["id"] for task in source["tasks"]}


def test_primary_family_assignments_are_unique_and_consistent_with_scope_policy():
    data = registry()
    policy = json.loads((ROOT / "scope_policy.json").read_text())
    declared = [(key, family["id"]) for family in policy["families"] for key in family["current_task_ids"]]
    assert len(declared) == len(dict(declared)) == 22
    declared += [(task["id"], task["family_id"]) for task in policy["future_tasks"]]
    assert len(declared) == len(dict(declared)) == 41
    assert dict(declared) == {task["id"]: task["family_id"] for task in data["tasks"] + data["future_tasks"]}
    listed = [key for family in data["families"] for key in family["task_ids"]]
    assert len(listed) == len(set(listed)) == len(data["tasks"])


def test_source_expansion_preserves_the_initial_matrix_without_claiming_old_budget_covers_new_tasks():
    data = registry()
    suite = json.loads((ROOT / "study_protocol.json").read_text())
    original_ids = {task["id"] for task in data["tasks"] if task["origin"] == "initial_22_source_backed_definition"}
    assert len(original_ids) == data["summary"]["initial_local_data_backed_task_definitions"] == 22
    assert original_ids <= set(suite["task_ids"])
    assert data["summary"]["original_primary_task_dataset_bindings_preserved"] == 96
    assert data["summary"]["primary_task_dataset_bindings"] == sum(task["primary_table_count"] for task in data["tasks"])
    assert "historical" in data["delivery_policy"]


def test_family_source_counts_deduplicate_reused_task_references():
    data = registry()
    families = {family["id"]: family for family in data["families"]}
    # Classification/clustering share eight tables; mapping adds ten and balancing four.
    assert families["representation"]["primary_table_count"] >= 22
    # Eight barrier tables share B97XD3/Cycloadd with thermochemistry; RAD6 and rate add two.
    assert families["physics"]["primary_table_count"] >= 10
    # Both directions and centers reuse the same three synthesis tables.
    assert families["transformations"]["primary_table_count"] == 3
    assert families["transformations"]["primary_source_rows"] == 540062
    for family in families.values():
        bindings = {(binding["namespace"], binding["key"]): binding for task in data["tasks"]
                    if task["family_id"] == family["id"] for binding in task["bindings"]}
        primary = [binding for binding in bindings.values() if binding["role"] != "overlapping_view"]
        assert family["primary_table_count"] == len(primary)
        assert family["counting_units"] == sorted({binding["counting_unit"] for binding in primary})


def test_assay_binary_success_has_its_own_contract_without_recounting_the_source():
    data = registry()
    tasks = {task["id"]: task for task in data["tasks"]}
    success = tasks["reaction_success"]
    binding = next(item for item in success["bindings"] if item["key"] == "yield/buchwald_hartwig_ood")
    contract = binding["task_fields"]["reaction_success"]
    assert binding["binding_origin"] == "qualified_after_initial_matrix"
    assert binding["rows"] == 27438
    assert binding["release_eligible"] is True
    assert contract["native_label_selector"] == "source_metadata_json.upstream_labels"
    assert set(contract["inputs"]).isdisjoint(contract["targets"] + contract["excluded_inputs"])
    assert contract["targets"] == ["success"]
    assert success["primary_table_count"] == 5
    assert success["primary_source_rows"] == 34303
    assert data["summary"]["original_primary_task_dataset_bindings_preserved"] == 96
    assert data["summary"]["followup_canonical_task_dataset_bindings"] == 1
    outcomes = next(family for family in data["families"] if family["id"] == "outcomes")
    grouped = [task for group in outcomes["task_groups"] for task in group["task_ids"]]
    assert set(grouped) == set(outcomes["task_ids"])
    assert len(grouped) == len(set(grouped))


def test_promotions_have_explicit_nonleaking_contracts_and_source_provenance():
    data = registry()
    promoted = set(data["promoted_roadmap_task_ids"])
    for task in data["tasks"]:
        if task["id"] in promoted:
            assert task["origin"] == "promoted_from_initial_19_roadmap_definitions"
            assert task["required_data"] and task["evaluation"]
            assert task["release_eligible_table_count"] > 0
        for binding in task["bindings"]:
            if binding["namespace"] != "coverage":
                continue
            assert binding["release_eligible"] is True
            assert binding["key"].startswith(binding["coverage_branch"] + "/")
            assert (source_path(REPO / binding["path"])).is_file()
            assert binding["source_url"].startswith("https://")
            assert len(binding["source_sha256"]) == len(binding["sha256"]) == 64
            assert binding["reference_evidence"] and binding["qualification"] and binding["counting_unit"]
            contract = binding["task_fields"][task["id"]]
            assert contract["inputs"] and contract["targets"]
            assert set(contract["inputs"]).isdisjoint(contract["targets"])


def test_legacy_rights_failures_are_visible_and_do_not_remove_historical_bindings():
    data = registry()
    rights = json.loads((ROOT / "coverage/legacy_rights.json").read_text())
    by_key = {(row["namespace"], row["key"]): row for row in rights["tables"]}
    seen = set()
    for task in data["tasks"]:
        for binding in task["bindings"]:
            if binding["namespace"] not in {"canonical", "expansion"}:
                continue
            key = (binding["namespace"], binding["key"])
            assert binding["release_eligible"] == by_key[key]["release_eligible"]
            assert binding["rights_status"] == by_key[key]["status"]
            seen.add(key)
    assert seen == set(by_key)
    assert by_key[("canonical", "property/lograte")]["release_eligible"] is True
    assert by_key[("canonical", "property/lograte")]["commercial_use_allowed"] is False
    assert by_key[("canonical", "property/lograte")]["partition"] == 'noncommercial'
    task = next(task for task in data["tasks"] if task["id"] == "reaction_rates")
    assert any(binding["namespace"] == "coverage" and binding["release_eligible"] for binding in task["bindings"])


def test_registry_serialization_is_identical_across_python_hash_seeds(tmp_path):
    outputs = []
    for seed in ("17", "2027"):
        output = tmp_path / seed
        subprocess.run([sys.executable, "-m", "synrxn.reconstruction", "registry", "--output", str(output)],
                       cwd=REPO, env={**os.environ, "PYTHONHASHSEED": seed},
                       check=True, capture_output=True, text=True)
        outputs.append(output)
    for output in outputs:
        assert {path.name for path in output.iterdir()} == {"task_registry.json"}
    assert (outputs[0] / "task_registry.json").read_bytes() == (outputs[1] / "task_registry.json").read_bytes()
    assert (outputs[0] / "task_registry.json").read_bytes() == (ROOT / "task_registry.json").read_bytes()


def test_optional_registry_reports_require_an_explicit_request(tmp_path):
    subprocess.run([sys.executable, "-m", "synrxn.reconstruction", "registry", "--reports", "--output", str(tmp_path)],
                   cwd=REPO, check=True, capture_output=True, text=True)
    assert {path.name for path in tmp_path.iterdir()} == {"task_registry.json", "family_coverage.csv", "task_structure.md"}
    assert json.loads((tmp_path / "task_registry.json").read_text())["summary"]["local_data_backed_task_definitions"] == 41


def test_real_inventory_preserves_counts_shared_sources_and_pending_boundary():
    data = registry()
    index = module.build_index(data, module.dataset_inventory(REPO))
    assert (index["summary"]["families"], index["summary"]["tasks"], index["summary"]["catalogue_table_views"]) == (12, 41, 105)
    assert index["summary"]["tables_by_namespace"] == {"canonical": 49, "expansion": 16, "coverage": 40}
    assay = [t for t in index["tables"] if t["key"] == "yield/buchwald_hartwig_ood"]
    assert len(assay) == 1
    assert {"yield_conversion", "reaction_success", "condition_ranking"} <= set(assay[0]["task_ids"])
    assert assay[0]["source_boundary"] == "frozen_verified_only"
    assert assay[0]["verification_status"].startswith("pending")
    assert all("pending" in table["regeneration_status"] for table in index["tables"] if table["namespace"] != "canonical")
    assert index["summary"]["completed_predictive_runs_by_index"] == 0


def test_family_deduplication_and_rejection_of_uncatalogued_task_source():
    data = registry()
    inventory = module.dataset_inventory(REPO)
    index = module.build_index(data, inventory)
    by_id = {t["id"]: t for t in data["tasks"]}
    for family in index["families"]:
        expected = {module.identity(binding) for task_id in family["task_ids"] for binding in by_id[task_id]["bindings"]}
        assert family["table_count"] == len(expected)
    damaged = copy.deepcopy(data)
    damaged["tasks"][0]["bindings"][0]["key"] = "invented/missing-source"
    with pytest.raises(ValueError, match="Uncatalogued binding"):
        module.build_index(damaged, inventory)


def test_task_inventory_reports_source_rows_and_overlaps_without_export(monkeypatch, capsys):
    def unexpected_export(*args, **kwargs):
        pytest.fail("Single-task inspection must only read catalogue metadata")

    monkeypatch.setattr(module, "generate", unexpected_export)
    assert module.main(["--task", "product_distribution"]) == 0
    output = capsys.readouterr().out
    assert "| `expansion:C06/catechol` | 1,227 |" in output
    assert "| `expansion:C06/claisen` | 283 |" in output
    assert "| `expansion:C06/single_solvent` | 656 | flow-condition observation | overlapping view | Yes |" in output
    assert "Data/families/selectivity/product_distribution/expansion/C06/claisen/data.parquet" in output
    assert module.main(["--task", "forward_prediction"]) == 0
    assert "| Held |" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        module.main(["--task", "unknown_task"])
    assert error.value.code == 2


def test_every_command_targets_exact_selector_without_inventing_tasks():
    data = registry()
    index = module.build_index(data, module.dataset_inventory(REPO))
    assert {task["id"] for task in index["tasks"]} == {task["id"] for task in data["tasks"]}
    for kind in ("family", "task"):
        entries = index["families"] if kind == "family" else index["tasks"]
        for entry in entries:
            for mode, cmd in entry["commands"].items():
                args = module.shlex.split(cmd)
                assert args[args.index("--" + kind) + 1] == entry["id"]
                assert args[3:5] == ["families", "build" if mode == "rebuild" else "verify"]
                assert "--check-rows" in args
                assert args[args.index("--output") + 1] == f".cache/synrxn/runs/reproduction/{kind}/{entry['id']}/{mode}"


@pytest.mark.parametrize('reports', [False, True])
def test_command_index_exports_only_requested_formats(tmp_path, reports):
    result = module.generate(REPO, tmp_path, reports=reports)
    expected = {'index.json', 'index.md'} if reports else {'index.json'}
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert result['summary']['tasks'] == 41
    assert result['summary']['completed_predictive_runs_by_index'] == 0


def test_protocol_sync_changes_counts_without_modifying_policies_or_input():
    registry = json.loads(physical_path('kdd/task_registry.json').read_text())
    protocol = json.loads(physical_path('kdd/study_protocol.json').read_text())
    original = copy.deepcopy(protocol)
    # Stale derived counts must be repaired from the registry, without another plan.
    protocol['compute_allocation']['suite_execution_ceiling'] = -1
    updated = sync_plan.synchronize(registry, protocol)
    assert protocol['compute_allocation']['suite_execution_ceiling'] == -1
    assert updated['compute_allocation']['suite_execution_ceiling'] == 1620
    assert updated['task_ids'] == [task['id'] for task in registry['tasks']]
    assert updated['family_baselines'] == original['family_baselines']
    assert updated['training_policy'] == original['training_policy']
    assert updated['reproducibility']['entrypoint'] == 'python -m synrxn.reconstruction full'


def test_coverage_export_is_explicit_and_cannot_write_into_source_metadata(tmp_path):
    result = report.build()
    assert result['summary'] == {'families': 12, 'local_definitions': 41,
                                 'release_eligible_definitions': 41, 'binding_rows': 146}
    report.export(result, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {'coverage.json'}
    with pytest.raises(ValueError, match='outside source'):
        report.export(result, physical_path('kdd/coverage'))
    with pytest.raises(ValueError, match='new or empty'):
        report.export(result, tmp_path, reports=True)


def test_suite_snapshots_current_inputs_without_requiring_retired_plans(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, 'snapshot_physical_files', lambda run, names, repo: set(names))
    names = suite.snapshot(tmp_path, [], REPO)
    assert {'kdd/task_registry.json', 'kdd/study_protocol.json', 'kdd/scope_policy.json'} <= names
    assert not {'kdd/' + name for name in RETIRED_METADATA} & names
    assert suite.REGISTRY_OUTPUTS == ['task_registry.json']

