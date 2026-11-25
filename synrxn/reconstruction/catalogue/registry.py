#!/usr/bin/env python3
"""Build the proposed scientific task taxonomy from local source inventories.

Definitions describe research decisions, not completed model benchmarks. The
existing eight export IDs remain intact and map to the broader task structure.
"""

from collections import defaultdict
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from synrxn.reconstruction.paths import REPO, source_path, physical_path, logical_relative

ROOT = physical_path("kdd")
# These roles describe how current task counts use the source inventories.
# Keep the original bindings for reproduction and the historical matrix.
PRIMARY_OBSERVATION_TABLES = {
    ("canonical", "aam/natcomm"): "canonical:aam/golden",
    ("canonical", "property/snar"): "coverage:solution_context/snar_solution_context",
    ("canonical", "property/cycloadd"): "coverage:solution_context/cycloadd_solution_context",
    ("expansion", "C06/single_solvent"): "expansion:C06/catechol",
}
PREFERRED_CONTEXT_TABLES = {
    "solution_context/snar_solution_context",
    "solution_context/cycloadd_solution_context",
}
# Additional uses of frozen source tables are explicit, so the original matrix
# remains independently countable when a new endpoint is qualified later.
FOLLOWUP_CANONICAL_CONTRACTS = {
    ("reaction_success", "yield/buchwald_hartwig_ood"): {
        "inputs": ["reactants", "reagents"],
        "targets": ["success"],
        "target_path": "kdd/case_studies/ncs_yield/success_targets.csv",
        "target_join_key": "sample_id",
        "native_label_selector": "source_metadata_json.upstream_labels",
        "split_metadata": "kdd/case_studies/ncs_yield/native_splits.csv",
        "split_protocol": "Ten native cluster initializations, four held-out cycles each; calibration and model selection use outer training only. Cluster OOD does not establish unseen-substrate evaluation.",
        "excluded_inputs": ["yield_pct", "source_metadata_json", "success", "sample_id", "source_record_id", "source_index", "upstream_split", "split_metadata"],
        "qualification": "Observed success under source-specific recorded conditions. Preserve native labels, including threshold exceptions; a negative does not mean chemical impossibility. Reactant/reagent-only input contract; full-reaction author reproduction is a separately declared input variant.",
        "reference_evidence": "https://doi.org/10.1038/s43588-026-01017-6",
    },
}
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from synrxn.reconstruction.coverage.rights import table_has_evidence
from synrxn.reconstruction.coverage.license_policy import ALLOWED_DATA_LICENSES, permitted, partition_for, commercial_allowed


def legacy_release_decision(record):
    expression = record.get("release_license_expression", record.get("verified_upstream_license_spdx", record.get("declared_license_spdx", "")))
    return bool(record.get("release_eligible") and permitted(expression))


def coverage_tables():
    """Normalize admitted coverage catalogs without rewriting source inventories.

    Explicit table-level decisions take precedence over raw-archive eligibility:
    several biological archives contain separately cleared and quarantined
    subsets. The canonical49/expansion16 remain visible under a separate rights
    gate, even when a historical declaration is insufficient for redistribution.
    """
    normalized, catalog_paths, admitted_paths = [], [], {}
    for path in sorted(physical_path("Data/coverage_expansion").glob("*/catalog.json")):
        branch = path.parent.name
        catalog_paths.append(logical_relative(path))
        data = json.loads(path.read_text())
        sources = {source["id"]: source for source in data.get("sources", [])}
        for table in data.get("tables", []):
            source = sources.get(table.get("source_id"), {})
            allowed = table.get("release_eligible", source.get("redistribution_allowed", True))
            expression = table.get("license_expression", table.get("license", source.get("license", "")))
            if not allowed or not permitted(expression) or not table_has_evidence(table, source, data, REPO) or table.get("quarantined") or table.get("redistribution_status", "").lower() in {'quarantined_not_for_redistribution', 'internal_use_only', 'not_for_redistribution'}:
                continue
            task_fields = table.get("task_fields", {})
            ids = table.get("task_ids", list(task_fields))
            if not set(ids) <= set(task_fields):
                raise ValueError("Coverage table lacks task contracts: " + table["id"])
            path_value = table["path"]
            if path_value in admitted_paths:
                raise ValueError("Same physical coverage table appears in multiple catalogs: " + path_value + " in " + admitted_paths[path_value] + " and " + branch)
            admitted_paths[path_value] = branch
            if not physical_path(path_value).is_file():
                raise ValueError("Missing admitted coverage data: " + path_value)
            for task_id in ids:
                contract = task_fields[task_id]
                if not contract.get("inputs") or not contract.get("targets"):
                    raise ValueError("Empty coverage input/target contract: " + table["id"])
                if set(contract["inputs"]) & set(contract["targets"]):
                    raise ValueError("Coverage input/target leakage: " + table["id"])
            normalized.append({"namespace": "coverage", "key": branch + "/" + table["id"],
                "coverage_branch": branch, "catalog_path": logical_relative(path),
                "path": path_value, "sha256": table["sha256"], "rows": int(table["rows"]),
                "role": table.get("role", "primary"), "counting_unit": table.get("counting_unit", "source-qualified task row"),
                "task_ids": list(ids), "task_fields": task_fields,
                "source_id": table.get("source_id"), "source_url": table.get("source_url", source.get("url")),
                "source_sha256": table.get("source_sha256"), "license": table.get("license", source.get("license")),
                "qualification": table.get("qualification"), "reference_evidence": table.get("reference_evidence"),
                "release_eligible": True, "rights_status": "admitted_source_or_table_specific_grant",
                "release_license_expression": expression, 'partition': partition_for(expression),
                'commercial_use_allowed': commercial_allowed(expression),
                **{key: table[key] for key in ["atomic_configurations", "time_points", "distinct_mechanism_families", "target_units", "native_split_counts", "native_split_trajectories", "parent_groups", "derived_view_of", "primary_eligible_rows", "primary_replay_eligible_rows", "strict_replay_eligible_rows", "preferred_primary_rows", "reference_tier_counts", "native_context_rows"] if key in table}})
    keys = [row["key"] for row in normalized]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate normalized coverage keys")
    return normalized, catalog_paths


def update_task_counts(task):
    primary = [binding for binding in task["bindings"] if binding["role"] != "overlapping_view"]
    task["primary_table_count"] = len(primary)
    task["primary_source_rows"] = sum(binding["rows"] for binding in primary)
    task["release_eligible_table_count"] = sum(binding.get("release_eligible", False) for binding in primary)
    task["release_blocked_table_count"] = len(primary) - task["release_eligible_table_count"]
    task["counting_units"] = sorted({binding["counting_unit"] for binding in primary})
    task["readiness"] = ("source_qualified_coverage_contract_available; model_evaluation_pending"
                          if any(binding["namespace"] == "coverage" for binding in task["bindings"]) else task["readiness"])
    task["release_readiness"] = ("at_least_one_source_qualified_redistributable_reference"
                                 if task["release_eligible_table_count"] else "local_data_present; source_rights_gate_unresolved_or_blocked")


def apply_observation_roles(bindings):
    by_id = {f"{b['namespace']}:{b['key']}": b for b in bindings}
    for binding in bindings:
        identity = binding["namespace"], binding["key"]
        if identity[0] == "coverage" and identity[1] in PREFERRED_CONTEXT_TABLES:
            binding["role"] = "primary"
    for binding in bindings:
        primary_id = PRIMARY_OBSERVATION_TABLES.get((binding["namespace"], binding["key"]))
        if primary_id is None:
            continue
        primary = by_id.get(primary_id)
        if (primary is None or primary["role"] != "primary" or primary["rows"] < binding["rows"]
                or (binding.get("release_eligible") and not primary.get("release_eligible"))):
            raise ValueError(f"Invalid primary observation table for {binding['key']}: {primary_id}")
        binding.update(role="overlapping_view", counted_under=primary_id)

def definition(key, title, canonical, extension, legacy, question, metrics, qualification):
    return {"id": key, "title": title, "canonical_keys": canonical,
            "extension_adapter_ids": extension, "legacy_view_ids": legacy,
            "question": question, "metrics": metrics, "qualification": qualification}


def specifications(catalog):
    keys = lambda prefix: sorted(key for key in catalog if key.startswith(prefix + "/"))
    classes = [key for key in keys("classification") if not key.endswith("ehreact_activity")]
    synthesis = ["synthesis/da", "synthesis/uspto_50k", "synthesis/uspto_mit"]
    barriers = [key for key in keys("property") if set(catalog[key]["targets"]) & {"ea", "G_act"}]
    return [
        definition("atom_mapping", "Atom mapping", keys("aam"), [], [],
            "Unmapped reaction -> correspondence between reactant and product atoms",
            ["reference_mapping_agreement", "valid_mapping_coverage"],
            "Compare by reference tier and symmetry; computational mapping QC is not independent chemical validation."),
        definition("reaction_balancing", "Reaction completion and balancing", keys("rbl"), [], [],
            "Incomplete recorded reaction -> supplied complete balanced reaction",
            ["reference_completion_match", "atom_and_charge_accounting", "coverage"],
            "Missing-species variants reuse observations; a recovered source reference is not proof of a unique physical reaction."),
        definition("forward_prediction", "Forward product prediction", synthesis, [], [],
            "Recorded reactants and permitted context -> recorded product structures",
            ["product_recovery_top_k", "valid_output_fraction", "coverage"],
            "Derive a target-isolated view from endpoints; remove product-derived mappings, reaction centers and identifiers from inputs."),
        definition("retrosynthesis", "Single-step retrosynthesis", synthesis, [], [],
            "Recorded product -> precursor sets",
            ["recorded_precursor_recovery_top_k", "valid_output_fraction", "coverage"],
            "A single recorded precursor set is not exhaustive acceptable chemistry; preserve source groups and alternatives. No multi-step route search is included."),
        definition("reaction_center", "Reaction-center prediction", ["synthesis/uspto_mit"], [], [],
            "Reactant structures -> source-annotated reactive atom/bond centers",
            ["center_precision", "center_recall", "center_F1", "coverage"],
            "The rc target exists, but its source index basis and label derivation must be qualified before a new benchmark view is advertised."),
        definition("reaction_classification", "Reaction classification", classes, [], ["classification"],
            "Recorded reaction -> source class, template family or enzyme class",
            ["macro_F1", "accuracy", "hierarchical_scores"],
            "Different taxonomies remain source-specific; balanced/unbalanced variants are related data."),
        definition("reaction_clustering", "Reaction clustering", classes, [], ["clustering"],
            "Recorded reactions -> unsupervised groups",
            ["ARI", "NMI", "coverage", "stability"],
            "Withhold reference labels from fitting and selection; any label-informed k policy is a separately declared diagnostic."),
        definition("yield_conversion", "Yield, assay response and conversion", keys("yield") + ["property/phosphatase"],
            ["yield_conversion"], ["yield"],
            "Reaction and recorded context -> source-specific measured response",
            ["MAE", "RMSE", "coverage"],
            "Yield, UV response and conversion are distinct endpoints within this task family. Keep units, measurement methods and source-specific scores separate."),
        definition("reaction_success", "Experimental reaction success", ["yield/buchwald_hartwig_ood"], ["reaction_success"], [],
            "Recorded attempted experiment -> source-defined success label",
            ["AUROC", "AUPRC", "balanced_accuracy", "MCC", "Brier_score", "log_loss", "reliability_diagram", "selective_risk_vs_coverage"],
            "Preserve the native label and establish its threshold definition; missing observations and generated distractors are not failed experiments."),
        definition("reaction_feasibility", "Proposal feasibility", keys("feasibility"), [], ["feasibility"],
            "Reactants and proposed products -> source feasibility judgment",
            ["AUPRC", "AUROC", "balanced_accuracy", "MCC", "calibration"],
            "Expert judgments and generated supervision are separate. FREA already supplies these labels and source-dependent evaluations; do not claim new labels."),
        definition("enzyme_activity", "Enzyme or assay activity", ["classification/ehreact_activity"], [], [],
            "Reaction and recorded assay identity -> binary activity",
            ["macro_F1", "AUPRC", "coverage"],
            "This tests the supplied assay setting, not generic feasibility or transfer to unseen enzyme sequences; sequences are not provided by this table."),
        definition("condition_recovery", "Recorded reagent and condition recovery",
            ["synthesis/uspto_500", "condition/orderly_condition", "condition_sharealike/ord_hte_condition"], [],
            ["reagent_set", "role_aware_condition"],
            "Recorded endpoints -> reported generic agents, explicit role sets or operating conditions",
            ["set_micro_F1", "set_macro_F1", "per_role_F1", "numeric_target_MAE", "coverage"],
            "Keep generic-agent and role-aware outputs as named task variants; missing roles are unknown and a recorded recipe is not uniquely correct."),
        definition("condition_ranking", "Ranking measured conditions",
            ["yield/hitea", "yield/buchwald_hartwig_ood"], ["condition_ranking"], [],
            "Reaction and observed candidate settings -> ordering by recorded response",
            ["observed_regret_at_1", "observed_regret_at_3", "panel_coverage"],
            "Requires comparable observed panels; the initial HiTEA design has 429. Other source rows are candidates, not already qualified panels."),
        definition("sequential_optimization", "Sequential experiment selection", [], ["sequential_optimization"], [],
            "Observed candidate pool and prior observations -> next experiment choice",
            ["best_response_vs_budget", "observed_regret", "hit_rate"],
            "Offline replay uses measured candidate points. It does not establish prospective laboratory performance or infer unmeasured outcomes."),
        definition("procedure_generation", "Experimental procedure generation", ["condition_sharealike/openexp"], [], [],
            "Recorded reaction and allowed recipe fields -> supplied experimental action sequence",
            ["action_sequence_recovery", "sequence_edit_distance", "coverage"],
            "Withhold procedure text and actions from inputs; action/reference agreement is not validation of laboratory execution."),
        definition("activation_barrier", "Activation-barrier prediction", barriers, [], ["energy_barrier"],
            "Recorded reaction and source context -> activation quantity",
            ["source_specific_MAE", "source_specific_RMSE", "coverage"],
            "Electronic barriers, Arrhenius activation energies and activation Gibbs free energies retain their units, theory, phase, solvent, temperature and standard state. Experimental and computed references are scored separately; do not pool numerical values. Solvent-aware views of existing rows are overlapping context variants."),
        definition("reaction_energetics", "Reaction energy and thermochemistry",
            ["property/b97xd3", "property/cycloadd", "property/rad6re"], [], [],
            "Recorded reaction -> source reaction enthalpy or free energy",
            ["source_specific_MAE", "source_specific_RMSE", "coverage"],
            "Reaction energy differs from an activation barrier. Preserve direction, quantity, units and theory; shared source rows do not become new reactions."),
        definition("reaction_rates", "Reaction-rate prediction", ["property/lograte"], [], [],
            "Recorded reaction and available source context -> source-defined rate coefficient or its documented transform",
            ["native_target_MAE", "native_target_RMSE", "coverage"],
            "Preserve log base, rate units, temperature and computed-versus-measured evidence. Arrhenius parameters are reference labels, not hidden inputs. Enzyme kcat needs enzyme context; Km and Ki are distinct parameters, not rate constants."),
        definition("regioselectivity", "Regioselectivity", ["selectivity/regiosqm_eas"], ["regioselectivity"], [],
            "Substrate and permitted context -> observed reactive sites or regioselective product annotation",
            ["site_set_recovery", "symmetry_aware_site_accuracy", "product_annotation_recovery", "coverage"],
            "Native atom sites and product-only annotations are separate variants. Minot has 1030 rows and 924 nonempty product annotations, not 1030 validated site labels."),
        definition("enantioselectivity", "Enantioselectivity", ["selectivity/hitea_chiral_resolution"], ["enantioselectivity"], [],
            "Substrates, catalyst and recorded conditions -> source selectivity endpoint",
            ["native_unit_MAE", "orientation_accuracy_when_defined", "coverage"],
            "Signed ee, absolute ee and derived energy differences are distinct; do not infer a favored stereoisomer when its convention is unresolved."),
        definition("product_distribution", "Condition-dependent product distributions", [], ["product_distribution"], [],
            "Reaction system and measured conditions -> per-component response vector",
            ["per_product_MAE", "composition_error", "coverage"],
            "Residence time is an input, not a physical time trajectory. Missing mass is not silently renormalized."),
        definition("mechanistic_prediction", "Electron-pushing mechanism prediction", ["mechanism/synepd_polar"], [], ["mechanistic_prediction"],
            "Supplied mapped reactants and products -> electron-event and ordered-mechanism annotations",
            ["event_recovery_top_k", "trajectory_recovery_top_k", "strict_replay_validity", "coverage"],
            "Endpoints are supplied; endpoint agreement is not event recovery. SynEPD has 1924 primary-tier records. M-CSA variants retain observed/reconstructed input flags, replay/reference tiers, native entry/proposal groups and development exposure.")
    ]


def build(output_dir=None, *, reports=False):
    output = Path(output_dir) if output_dir is not None else ROOT
    output.mkdir(parents=True, exist_ok=True)
    paths = ["doc/_static/catalog-data.json", "doc/_static/catalog-sharealike-data.json",
             "Data/expansion/catalog.json", "synrxn/data/task_protocols.json",
             "synrxn/data/expansion_tasks.json", "ncs/artifacts/task-views-002/summary.json",
             "kdd/scope_policy.json"]
    coverage, coverage_paths = coverage_tables()
    paths.extend(coverage_paths)
    rights_path = "kdd/coverage/legacy_rights.json"
    if physical_path(rights_path).exists():
        paths.append(rights_path)
    sources = {name: json.loads(physical_path(name).read_text()) for name in paths}
    legacy_rights = {(row["namespace"], row["key"]): row for row in sources.get(rights_path, {}).get("tables", [])}
    catalog = {item["id"]: item for path in paths[:2] for item in sources[path]["datasets"]}
    expansion = sources["Data/expansion/catalog.json"]
    policy = sources["kdd/scope_policy.json"]
    task_family = {}
    for family in policy["families"]:
        for task_id in family["current_task_ids"]:
            if task_id in task_family:
                raise ValueError("Task assigned to multiple primary families: " + task_id)
            task_family[task_id] = family["id"]
    specs = specifications(catalog)
    if set(task_family) != {spec["id"] for spec in specs}:
        raise ValueError("Scope policy and source-backed task definitions differ")
    old_exports = defaultdict(dict)
    for item in sources["ncs/artifacts/task-views-002/summary.json"]["exports"]:
        report = item["report"]
        old_exports[report["task"]][report["dataset"].removesuffix(".csv.gz")] = report["field_eligible_rows"]
    tasks, old_map = [], {}
    legacy_primary_bindings = 0
    for spec in specs:
        task = dict(spec)
        task["family_id"] = task_family[spec["id"]]
        task["bindings"] = []
        for key in spec["canonical_keys"]:
            item = catalog[key]
            task["bindings"].append({"namespace": "canonical", "key": key, "rows": item["rows"],
                "role": "primary", "counting_unit": "source-table row",
                "release_eligible": legacy_release_decision(legacy_rights.get(("canonical", key), {})),
                "release_license_expression": legacy_rights.get(("canonical", key), {}).get("release_license_expression", legacy_rights.get(("canonical", key), {}).get("verified_upstream_license_spdx", legacy_rights.get(("canonical", key), {}).get("declared_license_spdx"))),
                'partition': legacy_rights.get(('canonical', key), {}).get('partition'),
                'commercial_use_allowed': legacy_rights.get(('canonical', key), {}).get('commercial_use_allowed', False),
                "rights_status": legacy_rights.get(("canonical", key), {}).get("status", "not_audited"),
                "legacy_field_eligible_rows": {view: old_exports[view][key] for view in spec["legacy_view_ids"] if key in old_exports[view]}})
            contract = FOLLOWUP_CANONICAL_CONTRACTS.get((spec["id"], key))
            if contract:
                task["bindings"][-1].update({
                    "binding_origin": "qualified_after_initial_matrix",
                    "task_fields": {spec["id"]: contract},
                    "qualification": contract["qualification"],
                    "reference_evidence": contract["reference_evidence"],
                })
        for item in expansion["tables"]:
            implemented = set(spec["extension_adapter_ids"]) & set(item.get("task_fields", {}))
            if implemented:
                task["bindings"].append({"namespace": "expansion", "key": item["key"],
                    "rows": item["rows"], "role": item["role"], "counting_unit": item["counting_unit"],
                    "release_eligible": legacy_release_decision(legacy_rights.get(("expansion", item["key"]), {})),
                    "release_license_expression": legacy_rights.get(("expansion", item["key"]), {}).get("release_license_expression", legacy_rights.get(("expansion", item["key"]), {}).get("verified_upstream_license_spdx", legacy_rights.get(("expansion", item["key"]), {}).get("declared_license_spdx"))),
                    'partition': legacy_rights.get(('expansion', item['key']), {}).get('partition'),
                    'commercial_use_allowed': legacy_rights.get(('expansion', item['key']), {}).get('commercial_use_allowed', False),
                    "rights_status": legacy_rights.get(("expansion", item["key"]), {}).get("status", "not_audited"),
                    "adapter_ids": sorted(implemented), "qualification": item.get("qualification")})
        for item in coverage:
            if spec["id"] in item["task_ids"]:
                task["bindings"].append({**item, "task_fields": {spec["id"]: item["task_fields"][spec["id"]]}})
        legacy_primary_bindings += sum(
            item["namespace"] in {"canonical", "expansion"} and item["role"] != "overlapping_view"
            and item.get("binding_origin") != "qualified_after_initial_matrix" for item in task["bindings"]
        )
        apply_observation_roles(task["bindings"])
        task["readiness"] = ("legacy_export_variants_available" if spec["legacy_view_ids"] else
            "extension_adapter_available; standardized_contract_and_qualification_pending" if any(item["namespace"] == "expansion" for item in task["bindings"]) else
            "local_source_data; standardized_task_contract_needed")
        task["benchmark_result_status"] = "not completed by this planning work"
        task["origin"] = "initial_22_source_backed_definition"
        update_task_counts(task)
        for view in spec["legacy_view_ids"]:
            if view in old_map:
                raise ValueError("Legacy view mapped twice: " + view)
            old_map[view] = spec["id"]
        tasks.append(task)
    expected_old = {task["id"] for task in sources["synrxn/data/task_protocols.json"]["tasks"]}
    if set(old_map) != expected_old:
        raise ValueError("The original eight views are not completely preserved")
    covered = {item["key"] for task in tasks for item in task["bindings"] if item["namespace"] == "canonical"}
    if covered != set(catalog):
        raise ValueError("Unassigned canonical tables: " + repr(set(catalog) - covered))
    extension_specs = {task["id"]: task for task in sources["synrxn/data/expansion_tasks.json"]["tasks"]}
    future = []
    promoted = []
    for spec in policy["future_tasks"]:
        task = dict(spec)
        task["candidate_ids"] = extension_specs[spec["extension_id"]]["candidate_ids"] if spec.get("extension_id") else []
        task["candidate_status"] = ("Inherited acquisition candidates; integration and suitability are unverified"
                                    if task["candidate_ids"] else "Source acquisition and qualification required")
        task["bindings"] = [{**item, "task_fields": {spec["id"]: item["task_fields"][spec["id"]]}}
                            for item in coverage if spec["id"] in item["task_ids"]]
        task["readiness"] = "roadmap; no integrated benchmark table for this task in the captured snapshot"
        task["benchmark_result_status"] = "not completed by this planning work"
        task["legacy_view_ids"] = []
        task["canonical_keys"] = []
        task["extension_adapter_ids"] = [spec["extension_id"]] if spec.get("extension_id") else []
        task["metrics"] = sorted({metric for item in task["bindings"] for metric in item["task_fields"][spec["id"]].get("metrics", [])})
        task["qualification"] = "Source-qualified variants are defined per binding; required-data scope remains recorded. Acquisition and reference checks do not establish completed model performance."
        if task["bindings"]:
            task["origin"] = "promoted_from_initial_19_roadmap_definitions"
            update_task_counts(task)
            tasks.append(task)
            promoted.append(task["id"])
        else:
            task["origin"] = "initial_19_roadmap_definition_still_unbound"
            task["primary_table_count"] = 0
            task["primary_source_rows"] = 0
            task["release_eligible_table_count"] = 0
            future.append(task)
    universe = {task["id"] for task in tasks + future}
    if len(universe) != 41 or universe != {spec["id"] for spec in specs} | {spec["id"] for spec in policy["future_tasks"]}:
        raise ValueError("Coverage promotion changed the fixed41-definition task universe")
    unknown_coverage_ids = {task_id for binding in coverage for task_id in binding["task_ids"]} - universe
    if unknown_coverage_ids:
        raise ValueError("Coverage catalogs introduced undeclared tasks: " + repr(unknown_coverage_ids))
    future_ids = {task["id"] for task in future}
    excluded_ids = {task["id"] for task in policy["excluded_tasks"]}
    active_ids = {task["id"] for task in tasks}
    if len(future_ids) != len(future) or active_ids & future_ids or (active_ids | future_ids) & excluded_ids:
        raise ValueError("Duplicate or conflicting active, roadmap and excluded task IDs")
    families = []
    for spec in policy["families"]:
        family = {key: value for key, value in spec.items() if key != "current_task_ids"}
        family["task_ids"] = [task["id"] for task in tasks if task["family_id"] == family["id"]]
        family["future_task_ids"] = [task["id"] for task in future if task["family_id"] == family["id"]]
        family["data_status"] = "local task references present; qualification varies" if family["task_ids"] else "roadmap only; no integrated task references"
        bindings = {(item["namespace"], item["key"]): item
                    for task in tasks if task["family_id"] == family["id"] for item in task["bindings"]}
        primary = [item for item in bindings.values() if item["role"] != "overlapping_view"]
        family["primary_table_count"] = len(primary)
        family["primary_source_rows"] = sum(item["rows"] for item in primary)
        family["overlapping_view_count"] = len(bindings) - len(primary)
        family["counting_units"] = sorted({item["counting_unit"] for item in primary})
        family["release_eligible_table_count"] = sum(item.get("release_eligible", False) for item in primary)
        family["data_counting_policy"] = "Deduplicated by namespace/source-table key within this family only; row units can differ, source views can overlap and rows are not unique reactions. Inspect each binding's counting_unit and physical configuration counts."
        if not family["task_ids"] and not family["future_task_ids"]:
            raise ValueError("Family has neither current nor roadmap tasks: " + family["id"])
        families.append(family)
    if {task["family_id"] for task in future} - {family["id"] for family in families}:
        raise ValueError("Roadmap task assigned to an unknown family")
    result = {"schema": "synrxn.proposed-scientific-task-registry.v1", "version": "0.5.0", "as_of": "2026-10-21",
        "status": "proposed taxonomy with source-backed bindings; not a declaration of completed benchmarks",
        "counting_policy": "Count scientific input/output decisions. Role heads, context views, sources, seeds, folds and metric granularities are not automatically independent tasks. Generic and role-aware condition recovery retain separate legacy variants under one parent task.",
        "source_sha256": {name: hashlib.sha256(source_path(name, repo=REPO).read_bytes()).hexdigest() for name in paths + ['kdd/coverage/license_policy.py', 'kdd/coverage/rights.py']},
        "summary": {"scientific_families": len(families),
                    "families_with_local_task_data": sum(bool(family["task_ids"]) for family in families),
                    "roadmap_only_families": sum(not family["task_ids"] for family in families),
                    "local_data_backed_task_definitions": len(tasks),
                    "initial_local_data_backed_task_definitions": len(specs),
                    "initial_roadmap_task_definitions": len(policy["future_tasks"]),
                    "promoted_roadmap_task_definitions": len(promoted),
                    "release_eligible_task_definitions": sum(task["release_eligible_table_count"] > 0 for task in tasks),
                    "legacy_export_variants_preserved": len(old_map), "legacy_parent_tasks": len(set(old_map.values())),
                    "future_task_definitions": len(future), "canonical_tables_covered": len(covered),
                    "total_task_definitions": len(tasks) + len(future),
                    "expansion_tables_covered": len({item["key"] for task in tasks for item in task["bindings"] if item["namespace"] == "expansion"}),
                    "coverage_tables_covered": len({item["key"] for task in tasks for item in task["bindings"] if item["namespace"] == "coverage"}),
                    "original_primary_task_dataset_bindings_preserved": legacy_primary_bindings,
                    "followup_canonical_task_dataset_bindings": sum(binding.get("binding_origin") == "qualified_after_initial_matrix" for task in tasks for binding in task["bindings"]),
                    "release_eligible_primary_task_dataset_bindings": sum(task["release_eligible_table_count"] for task in tasks),
                    "primary_task_dataset_bindings": sum(task["primary_table_count"] for task in tasks)},
        "families": families, "tasks": tasks, "legacy_view_map": old_map,
        "future_tasks": future, "excluded_tasks": policy["excluded_tasks"],
        "initial_roadmap_task_ids": [task["id"] for task in policy["future_tasks"]],
        "promoted_roadmap_task_ids": promoted,
        "coverage_catalogs": coverage_paths,
        "rights_audit": "coverage/legacy_rights.json" if rights_path in sources else None,
        "rights_policy": "Local source availability and release eligibility are separate. Verified CC BY-NC data are accepted in separate noncommercial partitions; copyleft and database obligations remain explicit. Historical bindings with unresolved original permissions remain visible without being selected for publication. New promoted bindings require admitted table/source grants; quarantined views are excluded.",
        "scope_policy": "scope_policy.json", "coverage_audit": "../../../doc/data_concept.rst",
        "cross_cutting_axes": policy["cross_cutting_axes"], "initial_delivery_policy": policy["delivery_policy"],
        "delivery_policy": "The source-qualified coverage matrix extends the initial22-definition/96-binding preparation scope. Its promoted contracts require task-specific baselines and evaluation before benchmark completion is claimed. Initial budget estimates are historical and do not certify execution of the expanded matrix.",
        "extension_scope_reconciliation": {
            "yield_conversion": "yield_conversion", "reaction_success": "reaction_success", "recorded_condition_recovery": "condition_recovery",
            "condition_ranking": "condition_ranking", "regioselectivity": "regioselectivity", "enantioselectivity": "enantioselectivity",
            "sequential_optimization": "sequential_optimization", "product_distribution": "product_distribution",
            "solvent_effects": "context variant within activation_barrier/reaction_energetics, not a new endpoint by itself",
            "rate_constants": "reaction_rates; enzyme-specific parameters require separately qualified units and inputs",
            "reaction_completion": "reaction_balancing", **{key: "future task" for key in sorted(future_ids) if key in extension_specs},
            **{key: "excluded from KDD scope by user decision; retained only in the historical source inventory" for key in sorted(excluded_ids)}},
        "case_studies": [{"task_ids": ["yield_conversion", "condition_ranking"], "source": "HiTEA",
                          "protocol": "case_studies/hitea_protocol.json", "role": "One detailed example within the full suite"}],
        "companion": {"source": "M-CSA", "task": "mechanistic_prediction",
                      "role": "Observed-input mechanism references plus overlapping full, chain and step views with reference/exposure qualifications; not four extra tasks"}}
    (output / "task_registry.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if reports:
        fields = ["family_id", "family", "local_task_definitions", "roadmap_task_definitions",
                  "primary_source_tables", "source_table_rows", "overlapping_views", "data_status"]
        with (output / "family_coverage.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for family in families:
                writer.writerow({"family_id": family["id"], "family": family["title"],
                                 "local_task_definitions": len(family["task_ids"]),
                                 "roadmap_task_definitions": len(family["future_task_ids"]),
                                 "primary_source_tables": family["primary_table_count"],
                                 "source_table_rows": family["primary_source_rows"],
                                 "overlapping_views": family["overlapping_view_count"],
                                 "data_status": family["data_status"]})
        summary = result["summary"]
        lines = ["# Proposed SynRXN task structure", "",
                 f"The working taxonomy covers **{len(families)} higher-level families and {len(tasks) + len(future)} proposed task definitions**: **{len(tasks)} backed by local source data** and **{len(future)} roadmap definitions**. {summary['families_with_local_task_data']} families have some local task data; {summary['roadmap_only_families']} currently have roadmap definitions only. These counts describe scope and planned interfaces, not completed model benchmarks.", "",
                 "All eight existing export IDs remain preserved. Reagent-set and role-aware condition prediction are two variants of condition recovery under this taxonomy. Source quantities and reference qualifications remain separate within each task.", "",
                 "**Multi-step synthesis route planning is excluded.** Single-step precursor prediction, per-reaction experimental decisions and elementary mechanistic networks remain included. The scope policy records excluded tasks and source-specific qualifications.", "",
                 "## Families and tasks", "", "| Family | Current source-backed tasks | Current count | Roadmap count |", "| --- | --- | ---: | ---: |"]
        for family in families:
            names = [task["title"] for task in tasks if task["family_id"] == family["id"]]
            lines.append(f"| {family['title']} | {'; '.join(names) or 'No integrated task references yet'} | {len(names)} | {len(family['future_task_ids'])} |")
        lines += ["", "## Data by family", "",
                  "A source table is counted once within each family, even when reused by several tasks. Tables can still be shared across families. Row units include documents, physical trajectories, computational examples and measurements. Primary counts exclude explicitly overlapping views. A zero indicates no integrated task reference in this snapshot.", "",
                  "| Family | Primary source tables | Source-table rows |", "| --- | ---: | ---: |"]
        for family in families:
            lines.append(f"| {family['title']} | {family['primary_table_count']} | {family['primary_source_rows']:,} |")
        lines += ["", "[Family counts as CSV](family_coverage.csv); the registry retains all task–table bindings."]
        lines += ["", "## Data and readiness", "", f"Counts below are source-table rows, with units retained per binding. They are not uniformly eligible rows, unique reactions, tested condition panels or independent observations. Task totals overlap. All 49 frozen tables and 16 original expansion tables remain assigned, with {summary['coverage_tables_covered']} additional source-qualified coverage tables. Transition1x trajectory counts have a separate atomic-configuration total; its final TS view reuses the same physical source.", "",
                  "| Task | Primary tables | Source-table rows | Existing interface evidence |", "| --- | ---: | ---: | --- |"]
        for task in tasks:
            evidence = ", ".join(task["legacy_view_ids"]) or ("Coverage contract; model evaluation pending" if any(item["namespace"] == "coverage" for item in task["bindings"]) else "Expansion adapter; qualification pending" if any(item["namespace"] == "expansion" for item in task["bindings"]) else "Source data; standardized contract needed")
            lines.append(f"| {task['title']} | {task['primary_table_count']} | {task['primary_source_rows']:,} | {evidence} |")
        lines += ["", "## Original eight export views", "", "| Existing view | Scientific task |", "| --- | --- |"]
        for view, task_id in old_map.items():
            lines.append(f"| `{view}` | {next(task['title'] for task in tasks if task['id'] == task_id)} |")
        lines += ["", "## Roadmap definitions", "", (f"{len(promoted)} of the initial 19 roadmap definitions now have admitted source-qualified references. The {len(future)} still-unbound definitions remain below. Each retained task specifies its required data and evaluation boundary. New references still require task-specific baseline and evaluation work." if future else "All initial 19 roadmap definitions now have admitted source-qualified reference variants. No unbound definitions remain. This is source and interface coverage; model benchmark execution and broader scientific qualification remain pending."), "",
                  "| Family | Roadmap tasks |", "| --- | --- |"]
        for family in families:
            names = [task["title"] for task in future if task["family_id"] == family["id"]]
            if names:
                lines.append(f"| {family['title']} | {'; '.join(names)} |")
        lines += ["", "## Redistribution readiness", "",
                  f"{summary['release_eligible_task_definitions']} local task definitions have at least one reference admitted for redistribution. This is separate from the {len(tasks)} definitions with local source data. The [legacy rights audit](coverage/legacy_rights.json) retains source-specific grants, including verified noncommercial terms, GNU licenses and database obligations, and identifies unresolved original-source permissions. Original historical bindings remain visible without bypassing the release gate. New quarantined tables are excluded from promotions."]
        lines += ["", "## Scope and exclusions", "",
                  "The [scope policy](scope_policy.json) assigns every current and roadmap task one primary family and records the explicit route-planning exclusion. Chemical domains, input modalities, evidence tiers, robustness and uncertainty are cross-cutting dimensions. Their names do not establish dataset coverage. This is a working taxonomy of major areas, not proof that every problem in reaction informatics has an implemented benchmark.", "",
                  "## Counting rules", "", "Balanced/unbalanced tables, source populations and the same records reused for classification, clustering or inverse synthesis are not independent datasets. A new temperature or solvent split is a generalization condition, not automatically a new task. Event and trajectory scoring are granularities of mechanism prediction. Role-specific heads remain variants of condition recovery. Yield, conversion, energy and rate quantities retain source-specific definitions; they are not numerically pooled.", "",
                  "The eight exported views are the minimum backward-compatible evaluation coverage. The new task definitions need their own source-qualified contracts and baseline evidence before the paper calls them completed benchmarks. A blocked task remains visible in the readiness matrix rather than disappearing from the suite.", "",
                  "[Machine-readable registry](task_registry.json) includes every task–dataset binding, target qualification and source hash. [Detailed HiTEA case](case_studies/hitea_protocol.json) remains available within the full scope.", ""]
        (output / "task_structure.md").write_text("\n".join(lines))
    print(json.dumps(result["summary"]))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT,
                        help="Write generated artifacts here; source inventories still come from this checkout")
    parser.add_argument("--reports", action="store_true",
                        help="Also export optional family CSV and task Markdown summaries")
    args = parser.parse_args()
    build(args.output, reports=args.reports)
