#!/usr/bin/env python3
"""Render scientific task definitions, source-table counts and release decisions.

The inventory and its source identities are separate from the presentation.
HTML supplements the pinned hierarchy with task contracts and licensing from
the registry and release manifest; held records have no payload link.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import json
import os
from pathlib import Path

from synrxn.reconstruction.paths import REPO, physical_path
LABELS = {
    "aam/natcomm": "NatComm subset of Golden",
    "property/phosphatase": "Phosphatase assay responses",
    "yield/hitea": "HiTEA assay responses",
    "yield/buchwald_hartwig_ood": "NCS 2026 / BH-HTE-OOD yield collection",
    "property/cycloadd": "Cycloaddition original source format",
    "property/snar": "SNAr original source format",
    "C02/minot_al1_data": "Minot borylation AL1",
    "C02/minot_al2_data": "Minot borylation AL2",
    "C02/minot_al3_data": "Minot borylation AL3",
    "C02/minot_preal_data": "Minot borylation pre-AL",
    "C02/regio": "Minot borylation regioselectivity",
    "C03/cn_screen": "Cernak C–N coupling screen",
    "C06/catechol": "Catechol flow conditions",
    "C06/claisen": "Claisen flow conditions",
    "C06/single_solvent": "Single-solvent subset (overlapping view)",
    "C16/aldol_reaction_dataset": "OrgAIcat aldol reactions",
    "C16/michael_addition_dataset": "OrgAIcat Michael additions",
    "C16/validation_in_house_acetone": "OrgAIcat in-house acetone validation",
    "C16/validation_in_house_hydroxyacetone": "OrgAIcat in-house hydroxyacetone validation",
    "C16/validation_new_literature_aldol": "OrgAIcat literature aldol validation",
    "C16/validation_new_literature_michael": "OrgAIcat literature Michael validation",
    "C16/validation_virtual_chemist": "OrgAIcat Virtual Chemist validation",
    "knowledge/EnzChemRED2024/documents": "EnzChemRED source documents",
    "knowledge/EnzChemRED2024/curation": "EnzChemRED controlled curation cases",
    "knowledge/EnzChemRED2024/retrieval": "EnzChemRED precedent queries",
    "knowledge/CatDRX_ORD2025/inverse_components": "CatDRX / ORD component design",
    "knowledge/MinotScope2026/scope_campaigns": "Minot substrate-scope campaigns",
    "impurity/pba_benzaldehyde_side_product": "PBA benzaldehyde side products",
    "impurity/solid_state_reported_impurity_phases": "Solid-state reported impurity phases",
    "physics_rates/quantumpioneer_hAT_kinetics": "QuantumPioneer HAT kinetics",
    "physics/transition1x_reactive_trajectories": "Transition1x reactive trajectories",
    "physics/transition1x_final_ts_geometry": "Transition1x endpoint / TS triplets",
    "physics/orgreact_ts_geometry": "OrgReact endpoint / TS triplets",
    "physics/chemsindy_reference_trajectories": "ChemSINDy reference trajectories",
    "process/nmr_saccharose_hydrolysis_snapshots": "Sucrose hydrolysis NMR spectra",
    "process/nmr_pmg_284_snapshots": "PMG-284 NMR spectra",
    "process/nmr_pmg_287_snapshots": "PMG-287 NMR spectra",
    "process/nmr_saccharose_hydrolysis_progress": "NMR saccharose hydrolysis progress",
    "process/nmr_pmg_284_progress": "NMR PMG-284 progress",
    "process/nmr_pmg_287_progress": "NMR PMG-287 progress",
    "process/lowfield_nmr_yield_progress": "Low-field NMR yield progress",
    "process/reac_discovery_cs1": "Reac-Discovery case study 1",
    "process/reac_discovery_cs2": "Reac-Discovery case study 2",
    "process/fcc_lab_source_full": "FCC laboratory source data",
    "process/fcc_pilot_train": "FCC pilot training data",
    "process/fcc_pilot_validation": "FCC pilot validation data",
    "process/fcc_pilot_test": "FCC pilot test data",
    "process/cstr_control_transitions": "CSTR control transitions",
    "bio/enzeng_inverse_biocatalyst_design": "EnzEng inverse biocatalyst design",
    "bio/reactzyme_retrieval": "ReactZyme enzyme–reaction retrieval",
    "bio/enzeng_mutation_effects": "EnzEng mutation effects",
    "bio/mcsa_reference_active_sites": "M-CSA reference active sites",
    "bio/brenda_record_kcat": "BRENDA kcat records",
    "bio/brenda_record_km": "BRENDA Km records",
    "bio/brenda_record_ki": "BRENDA Ki records",
    "mechanisms/mcsa_observed_mechanisms": "M-CSA observed-start complete mechanisms",
    "mechanisms/mcsa_full_mechanisms": "M-CSA full mechanisms (overlapping)",
    "mechanisms/mcsa_continuous_chains": "M-CSA continuous chains (overlapping)",
    "mechanisms/mcsa_elementary_steps": "M-CSA elementary steps (overlapping)",
    "solution_context/snar_solution_context": "SNAr activation free energies with solvent and temperature",
    "solution_context/cycloadd_solution_context": "Cycloaddition activation and reaction free energies in water",
    "solution_barriers/lipred_solution_activation_gibbs": "LiPRED solution activation Gibbs free energies",
}

# Browsing names describe the observations; catalogue keys retain source identity.
NAVIGATION_KEYS = {
    ("coverage", "process/nmr_saccharose_hydrolysis_snapshots"):
        "process/nmr_saccharose_hydrolysis_spectra",
    ("coverage", "process/nmr_pmg_284_snapshots"):
        "process/nmr_pmg_284_spectra",
    ("coverage", "process/nmr_pmg_287_snapshots"):
        "process/nmr_pmg_287_spectra",
}


def load(path):
    return json.loads(physical_path(path).read_text())


def sha256(path):
    return hashlib.sha256(physical_path(path).read_bytes()).hexdigest()


def build_inventory(repo):
    registry_path = repo / "kdd/task_registry.json"
    registry = load(registry_path)
    audit_path = repo / "kdd/coverage/release_manifest.json"
    audit = load(audit_path)
    catalogue_keys = {(t["namespace"], t["key"]) for t in
                      audit["release_tables"] + audit["held_tables"]
                      if t["namespace"] != "coverage_quarantine"}
    # Held quarantine records can retain the coverage namespace. The registry
    # defines the admitted inventory; the release manifest supplies file pins.
    manifest = {(t["namespace"], t["key"]): t for t in
                audit["release_tables"] + audit["held_tables"]}
    inputs = [registry_path, audit_path, Path(__file__).resolve()]
    metadata = {}
    for rel in ("doc/_static/catalog-data.json", "doc/_static/catalog-sharealike-data.json"):
        path = repo / rel
        inputs.append(path)
        for table in load(path)["datasets"]:
            key = table["id"]
            metadata[("canonical", key)] = {
                "title": LABELS.get(key, table["title"]),
                "catalogue_rows": table["rows"], "source_catalogue": rel,
                "citations": table.get("citations", []),
            }
    path = repo / "Data/expansion/catalog.json"
    inputs.append(path)
    catalog = load(path)
    sources = {s["id"]: s for s in catalog["sources"]}
    for table in catalog["tables"]:
        key = table["key"]
        source = sources[table["source_id"]]
        metadata[("expansion", key)] = {
            "title": LABELS.get(key, key), "catalogue_rows": table["rows"],
            "source_catalogue": path.relative_to(repo).as_posix(),
            "source_id": table["source_id"],
            "source_urls": source.get("data_urls", []) + source.get("paper_urls", []),
        }
    for rel in registry["coverage_catalogs"]:
        path = repo / rel
        inputs.append(path)
        branch = path.parent.name
        for table in load(path)["tables"]:
            key = branch + "/" + table["id"]
            metadata[("coverage", key)] = {
                "title": LABELS.get(key, table.get("title", key)), "catalogue_rows": table["rows"],
                "source_catalogue": rel, "source_id": table.get("source_id"),
                "source_urls": [table["source_url"]] if table.get("source_url") else [],
                "qualification": table.get("qualification"),
                **{field: table[field] for field in ("native_context_rows", "strict_replay_eligible_rows", "preferred_primary_rows") if field in table},
            }
    families, unique, flat = [], {}, []
    for family in registry["families"]:
        node = {"id": family["id"], "title": family["title"], "tasks": []}
        family_tables = {}
        for task in registry["tasks"]:
            if task["family_id"] != family["id"]:
                continue
            item = {"id": task["id"], "title": task["title"], "tables": [],
                    "primary_table_count": task["primary_table_count"],
                    "primary_source_rows": task["primary_source_rows"],
                    "qualification": task.get("qualification", "")}
            for binding in task["bindings"]:
                identity = (binding["namespace"], binding["key"])
                meta = metadata[identity]
                pin = manifest[identity]
                if binding["rows"] != meta["catalogue_rows"] or binding["rows"] != pin["expected_rows"]:
                    raise ValueError(f"Row-count disagreement: {identity}")
                if not physical_path(pin["path"], repo).is_file():
                    raise ValueError(f"Missing catalogue data: {pin['path']}")
                table = {"namespace": identity[0], "key": identity[1],
                         "title": meta["title"], "rows": binding["rows"],
                         "role": binding["role"], "counting_unit": binding["counting_unit"],
                         "path": pin["path"], "sha256": pin["expected_sha256"],
                         **{k: v for k, v in meta.items() if k not in {"title", "catalogue_rows"}},
                         **{k: binding[k] for k in ("atomic_configurations", "time_points",
                            "distinct_mechanism_families", "derived_view_of", "counted_under",
                            "legacy_field_eligible_rows") if k in binding}}
                if identity in unique and unique[identity] != table:
                    # Legacy field eligibility is task/view-specific metadata.
                    common = set(table) - {"legacy_field_eligible_rows"}
                    if any(unique[identity].get(k) != table[k] for k in common):
                        raise ValueError(f"Inconsistent reused table: {identity}")
                unique[identity] = table
                family_tables[identity] = table
                item["tables"].append(table)
                flat.append({"family_id": node["id"], "family": node["title"],
                             "task_id": item["id"], "task": item["title"],
                             **{k: table[k] for k in ("namespace", "key", "title", "rows",
                                "counting_unit", "role", "path", "sha256", "source_catalogue")},
                             "source_id": table.get("source_id", ""),
                             "source_urls": " | ".join(table.get("source_urls", [])),
                             "citations": " | ".join(table.get("citations", []))})
            primary = [t for t in item["tables"] if t["role"] != "overlapping_view"]
            if (len(primary), sum(t["rows"] for t in primary)) != (item["primary_table_count"], item["primary_source_rows"]):
                raise ValueError(f"Task summary disagreement: {task['id']}")
            node["tasks"].append(item)
        primary = [t for t in family_tables.values() if t["role"] != "overlapping_view"]
        node.update(primary_table_count=len(primary), primary_source_rows=sum(t["rows"] for t in primary),
                    overlapping_view_count=len(family_tables) - len(primary))
        if (node["primary_table_count"], node["primary_source_rows"]) != (family["primary_table_count"], family["primary_source_rows"]):
            raise ValueError(f"Family summary disagreement: {family['id']}")
        families.append(node)
    # Every canonical/expansion table and every qualified coverage table occurs.
    expected = {k for k in metadata if k[0] != "coverage" or k in catalogue_keys}
    if set(unique) != expected:
        raise ValueError(f"Unassigned or unexpected tables: {set(unique) ^ expected}")
    primary = [t for t in unique.values() if t["role"] != "overlapping_view"]
    prep_path = repo / "kdd/preparation/audit.json"
    inputs.append(prep_path)
    prep = load(prep_path)
    if prep["source"]["canonical_rows"] != unique[("canonical", "yield/hitea")]["rows"]:
        raise ValueError("HiTEA preparation population differs from the catalogue")
    # Scientific coverage includes held entries as metadata; redistribution
    # decisions remain in the source-specific release manifest.
    report = {"schema": "synrxn.catalogue-hierarchy.v1", "registry_as_of": registry["as_of"],
              "scope": "Reaction informatics excluding multistep synthesis planning. All catalogue views are represented; held entries are metadata only and source-specific redistribution decisions remain in the release manifest.",
              "counting_policy": "Row counts are stored table records in their stated units, not necessarily task-eligible rows or unique reactions. Primary family/task totals exclude explicitly overlapping views. Tables are deduplicated within a family, but reused across families and tasks.",
              "summary": {"families": len(families), "tasks": len(registry["tasks"]),
                  "catalogue_tables": len(unique), "catalogue_rows": sum(t["rows"] for t in unique.values()),
                  "unique_primary_tables": len(primary), "unique_primary_rows": sum(t["rows"] for t in primary),
                  "overlapping_views": len(unique) - len(primary),
                  "primary_task_table_bindings": sum(t["primary_table_count"] for f in families for t in f["tasks"]),
                  "all_task_table_bindings": len(flat)},
              "case_study_populations": {
                  "hitea_source_rows": prep["source"]["canonical_rows"],
                  "hitea_response_eligible_rows": prep["source"]["eligible_rows"],
                  "hitea_ranking_panels": prep["ranking_panels_at_least_5_candidates"],
                  "hitea_ranking_rows": prep["ranking_panel_rows"],
                  "bh_source_rows": prep["bh_replication"]["rows"],
                  "bh_rows_after_overlap_exclusion": prep["bh_replication"]["rows_after_union_of_exclusions"],
              },
              "input_sha256": {p.relative_to(repo).as_posix(): sha256(p) for p in inputs},
              "families": families}
    return report, flat


def _presentation_metadata(repo):
    """Read display fields without changing the checksum-pinned inventory."""
    registry = json.loads(physical_path("kdd/task_registry.json", repo).read_text())
    release = json.loads(physical_path("kdd/coverage/release_manifest.json", repo).read_text())
    families = {family["id"]: family for family in registry["families"]}
    tasks = {task["id"]: task for task in registry["tasks"]}
    admitted = {(table["namespace"], table["key"]) for table in release["release_tables"]}
    rights = {(table["namespace"], table["key"]): table for table in
              release["release_tables"] + release["held_tables"]}
    return families, tasks, admitted, rights


def _field_list(values):
    if isinstance(values, str):
        return html.escape(values)
    return ", ".join(f"<code>{html.escape(str(value))}</code>" for value in values)


def _source_references(table, repo):
    """Supplement canonical catalogue references from their existing locks.

    A source lock can retain an earlier license assessment. Only bibliographic
    fields are read here; the release manifest supplies the current decision.
    """
    urls = list(table.get("source_urls", []))
    if table["namespace"] != "canonical":
        return urls
    lock = physical_path(f'Data/sources/{table["key"]}.json', repo)
    if not lock.is_file():
        return urls
    source = json.loads(lock.read_text())
    if source.get("source_url"):
        urls.append(source["source_url"])
    for citation in source.get("citations", []):
        if citation.get("doi"):
            urls.append("https://doi.org/" + citation["doi"])
        elif citation.get("url"):
            urls.append(citation["url"])
    for attribution in source.get("required_attribution", []):
        if isinstance(attribution, str) and attribution.startswith("10.") and " " not in attribution:
            urls.append("https://doi.org/" + attribution)
    return list(dict.fromkeys(urls))


def public_inventory(report, *, repo=REPO):
    """Enrich an export with current task definitions and admission decisions.

    The historical hierarchy remains byte-for-byte unchanged. Export consumers
    can distinguish metadata-only records without interpreting a local path.
    """
    repo = Path(repo).resolve()
    families, tasks, admitted, rights = _presentation_metadata(repo)
    result = copy.deepcopy(report)
    result["schema"] = "synrxn.public-catalogue-hierarchy.v1"
    result["presentation_input_sha256"] = {
        relative: sha256(repo / relative) for relative in (
            "Data/metadata/reconstruction/hierarchy/hierarchy.json",
            "Data/metadata/reconstruction/task_registry.json",
            "Data/metadata/reconstruction/coverage/release_manifest.json",
            "synrxn/reconstruction/catalogue/hierarchy.py",
        )
    }
    result["scope"] = (
        "Reaction informatics excluding multistep synthesis planning. "
        "The catalogue includes admitted data views and metadata-only entries; "
        "release_eligible records the source-specific redistribution decision."
    )
    identities = set()
    for family in result["families"]:
        family["scope"] = families[family["id"]].get("scope", "")
        for task in family["tasks"]:
            definition = tasks[task["id"]]
            task["question"] = definition.get("question", "")
            task["metrics"] = definition.get("metrics", [])
            bindings = {(b["namespace"], b["key"]): b for b in definition["bindings"]}
            for table in task["tables"]:
                identity = table["namespace"], table["key"]
                identities.add(identity)
                pin = rights[identity]
                table["release_eligible"] = identity in admitted
                table["admission_state"] = "admitted" if identity in admitted else "metadata_only"
                table["release_license_expression"] = pin.get("license_expression")
                table["rights_status"] = pin.get("rights_status")
                table["license_obligations"] = pin.get("license_obligations", [])
                table["task_fields"] = bindings[identity].get("task_fields", {}).get(task["id"], {})
                table["source_urls"] = _source_references(table, repo)
                # Historical paths remain source identities, never promises of
                # a hosted bulk download in a documentation-only publication.
                table["data_path"] = (physical_path(table["path"], repo).relative_to(repo).as_posix()
                                      if identity in admitted else None)
    result["summary"]["admitted_table_views"] = len(identities & admitted)
    result["summary"]["metadata_only_table_views"] = len(identities - admitted)
    return result


def _definition(label, content):
    return f"<dt>{html.escape(label)}</dt><dd>{content}</dd>" if content else ""


def _catalogue_link(path, label, *, repo, output):
    if output is None:
        raise ValueError("A local catalogue link requires an output directory")
    href = Path(os.path.relpath(physical_path(path, repo), output)).as_posix()
    return f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>'


def _source_links(urls):
    links = []
    for url in dict.fromkeys(urls):
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            continue
        label = url.removeprefix("https://").removeprefix("http://").rstrip("/")
        links.append(f'<a href="{html.escape(url, quote=True)}">{html.escape(label)}</a>')
    return "<br>".join(links)


def _table_method_fields(table, binding, task_id, pin, *, repo, output, data_links, is_admitted):
    """Describe a source view using its task contract and current release terms."""
    e = html.escape
    source_urls = _source_references(table, repo)
    if binding.get("source_url"):
        source_urls.append(binding["source_url"])
    reference = binding.get("reference_evidence")
    if isinstance(reference, str) and reference.startswith(("https://", "http://")):
        source_urls.append(reference)
    contract = binding.get("task_fields", {}).get(task_id, {})
    methods = ""
    for field, label in (("inputs", "Input fields"), ("targets", "Target fields"),
                         ("excluded_inputs", "Excluded input fields"),
                         ("group_fields", "Grouping fields")):
        methods += _definition(label, _field_list(contract.get(field, [])))
    target_units = binding.get("target_units")
    if target_units:
        unit_text = (json.dumps(target_units, ensure_ascii=False)
                     if isinstance(target_units, dict) else target_units)
        methods += _definition("Target units", e(str(unit_text)))
    for field, label in (("split_protocol", "Evaluation split"),
                         ("temporal_policy", "Temporal scope"),
                         ("split_status", "Split status")):
        methods += _definition(label, e(str(contract.get(field, ""))))
    qualification = (contract.get("qualification") or binding.get("qualification")
                     or table.get("qualification"))
    methods += _definition("Reference scope", e(qualification or ""))
    if counted_under := binding.get("counted_under"):
        methods += _definition("Counted under", f"<code>{e(counted_under)}</code>")
    if reference and not source_urls:
        methods += _definition("Reference evidence", e(str(reference)))
    methods += _definition("Source", _source_links(source_urls))
    methods += _definition("Source identity", e(table.get("source_id", "")))
    methods += _definition("Citation keys", _field_list(table.get("citations", [])))
    if data_links:
        catalogue = _catalogue_link(table["source_catalogue"], "Source metadata", repo=repo, output=output)
        methods += _definition("Catalogue", catalogue)
    methods += _definition("SHA-256", f'<code class="checksum">{e(table["sha256"])}</code>')
    obligations = pin.get("license_obligations", [])
    methods += _definition("License conditions", " ".join(e(str(item)) for item in obligations))
    if not is_admitted:
        decision = pin.get("qualification") or pin.get("reason", "")
        methods += _definition("Redistribution decision", e(decision))
    return methods


def _table_row(table, binding, task_id, pin, *, repo, output, data_links, is_admitted):
    e = html.escape
    license_expression = pin.get("license_expression")
    license_display = e(license_expression) if license_expression else "Unresolved dataset grant"
    admission = "Admitted" if is_admitted else "Metadata only"
    title = e(table["title"])
    if data_links and is_admitted:
        title = _catalogue_link(table["path"], table["title"], repo=repo, output=output)
    role = "Overlapping view" if table["role"] == "overlapping_view" else "Primary view"
    facts = []
    for field, label in (("atomic_configurations", "atomic configurations"),
                         ("time_points", "time points"),
                         ("distinct_mechanism_families", "mechanism families"),
                         ("native_context_rows", "rows with native context"),
                         ("strict_replay_eligible_rows", "strict replay references"),
                         ("preferred_primary_rows", "preferred primary references")):
        value = binding.get(field, table.get(field))
        if value is not None:
            facts.append(f"{value:,} {label}")
    methods = _table_method_fields(table, binding, task_id, pin, repo=repo, output=output,
                                   data_links=data_links, is_admitted=is_admitted)
    special = ""
    if table["key"] == "feasibility/frea_corpus":
        special = '<span class="reference-kind">Corpus with generated negative examples</span>'
    elif table["key"] == "feasibility/frea_expert":
        special = '<span class="reference-kind">Expert feasibility judgments</span>'
    details = ('<details class="dataset-methods"><summary>Methods and provenance</summary>'
               f'<dl class="method-fields">{methods}</dl></details>')
    facts_html = f'<span class="counting-facts">{e("; ".join(facts))}</span>' if facts else ""
    state = "admitted" if is_admitted else "held"
    return (
        f'<tr class="dataset-row" data-admission="{state}" data-role="{e(table["role"], quote=True)}">'
        f'<th scope="row"><span class="dataset-title">{title}</span>'
        f'<code class="dataset-key">{e(table["namespace"])}:{e(table["key"])}</code>'
        f'<span class="dataset-role">{role}</span>{special}{details}</th>'
        f'<td class="record-count"><span>{table["rows"]:,}</span>'
        f'<span class="counting-unit">{e(table["counting_unit"])}</span>{facts_html}</td>'
        f'<td class="license-cell"><span class="license-expression">{license_display}</span>'
        f'<span class="admission-state {state}">{admission}</span></td></tr>')


def _task_panel(task, definition, *, rights, admitted, repo, output, data_links):
    e = html.escape
    question = definition.get("question", "")
    if " -> " in question:
        inputs, targets = question.split(" -> ", 1)
        task_definition = _definition("Input", e(inputs)) + _definition("Target", e(targets))
    else:
        task_definition = _definition("Question", e(question))
    task_definition += _definition("Evaluation", _field_list(definition.get("metrics", [])))
    bindings = {(binding["namespace"], binding["key"]): binding for binding in definition["bindings"]}
    rows = []
    for table in task["tables"]:
        identity = table["namespace"], table["key"]
        rows.append(_table_row(table, bindings[identity], task["id"], rights[identity],
                               repo=repo, output=output, data_links=data_links,
                               is_admitted=identity in admitted))
    qualification = definition.get("qualification", "")
    if qualification.startswith("Source-qualified variants are defined per binding;"):
        qualification = ""
    scope = f'<p class="task-scope">{e(qualification)}</p>' if qualification else ""
    return (
        f'<details class="catalogue-task" id="task-{e(task["id"], quote=True)}">'
        f'<summary><span class="task-title">{e(task["title"])}</span>'
        f'<span class="task-size">{len(task["tables"])} table '
        f'{"view" if len(task["tables"]) == 1 else "views"}</span></summary>'
        f'<div class="task-content"><dl class="task-definition">{task_definition}</dl>{scope}'
        '<div class="catalogue-table-wrap"><table class="family-datasets">'
        f'<caption>Source tables for {e(task["title"].lower())}</caption>'
        '<thead><tr><th scope="col">Dataset and provenance</th>'
        '<th scope="col">Stored records and counting unit</th>'
        '<th scope="col">License and availability</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></div></details>')


def _family_section(number, family, definition, tasks):
    e = html.escape
    family_id = family["id"]
    return (
        f'<section class="catalogue-family" id="family-{e(family_id, quote=True)}" '
        f'aria-labelledby="heading-{e(family_id, quote=True)}">'
        f'<h2 id="heading-{e(family_id, quote=True)}"><span class="family-number">{number:02d}</span> '
        f'{e(family["title"])}</h2><p class="family-scope">{e(definition.get("scope", ""))}</p>'
        f'{"".join(tasks)}</section>')


def _family_navigation(number, family):
    e = html.escape
    tables = {(table["namespace"], table["key"]) for task in family["tasks"] for table in task["tables"]}
    return (
        f'<li><a href="#family-{e(family["id"], quote=True)}">'
        f'<span class="family-number">{number:02d}</span> {e(family["title"])}'
        f'<span class="family-size">{len(family["tasks"])} tasks · '
        f'{len(tables)} views</span></a></li>')


def _catalogue_introduction(report, admitted_count, held_count, exports):
    e = html.escape
    summary = report["summary"]
    exports_html = ""
    if exports:
        exports_html = '<p class="catalogue-exports">' + " · ".join(
            f'<a href="{e(url, quote=True)}" download>{e(label)}</a>'
            for label, url in exports.items()) + "</p>"
    known = {table["key"]: table for family in report["families"] for task in family["tasks"]
             for table in task["tables"]}
    frea = known.get("feasibility/frea_corpus")
    expert = known.get("feasibility/frea_expert")
    frea_note = (f' The {frea["rows"]:,}-record FREA corpus includes generated negative examples; '
                 f'{expert["rows"]:,} expert judgments are kept separately.' if frea and expert else "")
    return (
        '<div class="catalogue-introduction">'
        f'<p class="catalogue-summary"><strong>{summary["families"]} families</strong> · '
        f'<strong>{summary["tasks"]} scientific tasks</strong> · '
        f'<strong>{admitted_count} data tables</strong></p>'
        '<p>The catalogue relates reaction-informatics questions to source datasets, '
        'measured or computed targets, and evaluation criteria. '
        'Multistep synthesis planning is outside its scope.</p>'
        f'<p>{held_count} {"source record is" if held_count == 1 else "source records are"} '
        'metadata only. '
        f'The inventory contains {summary["unique_primary_tables"]} primary views and '
        f'{summary["overlapping_views"]} explicitly overlapping views. A table can support several tasks.</p>'
        '<p class="counting-policy">Rows retain their source-specific units; they are not a count of '
        f'independent reactions or test examples.{frea_note}</p>{exports_html}</div>')


def render_fragment(report, *, output=None, repo=REPO, data_links=False, exports=None):
    """Return the browser body for the standalone page or a Sphinx raw directive.

    Published documentation uses the default ``data_links=False``: the bulk
    assets are not part of a documentation build. A local standalone page may
    link to admitted payloads. ``exports`` maps labels to URLs copied or hosted
    by the caller; no repository-relative download links are assumed.
    """
    repo = Path(repo).resolve()
    family_definitions, task_definitions, admitted, rights = _presentation_metadata(repo)
    identities = {(table["namespace"], table["key"])
                  for family in report["families"] for task in family["tasks"]
                  for table in task["tables"]}
    unknown = identities - rights.keys()
    if unknown:
        raise ValueError(f"Catalogue views lack a release decision: {sorted(unknown)}")
    sections, navigation = [], []
    for number, family in enumerate(report["families"], 1):
        tasks = [_task_panel(task, task_definitions[task["id"]], rights=rights, admitted=admitted,
                             repo=repo, output=output, data_links=data_links) for task in family["tasks"]]
        sections.append(_family_section(number, family, family_definitions[family["id"]], tasks))
        navigation.append(_family_navigation(number, family))
    introduction = _catalogue_introduction(report, len(identities & admitted), len(identities - admitted), exports)
    return (
        '<div class="synrxn-family-browser" data-family-browser>'
        f'{introduction}'
        '<form class="catalogue-controls" role="search" onsubmit="return false" hidden>'
        '<div class="catalogue-search-field"><label for="family-search">Search the catalogue</label>'
        '<input type="search" id="family-search" name="q" placeholder="Dataset, task, target or license" '
        'autocomplete="off" aria-controls="family-sections"></div>'
        '<div class="catalogue-filter-field"><label for="family-availability">Availability</label>'
        '<select id="family-availability"><option value="all">All table views</option>'
        '<option value="admitted">Admitted data</option><option value="held">Metadata only</option></select></div>'
        '<button type="button" data-action="expand">Open tasks</button>'
        '<button type="button" data-action="collapse">Close tasks</button>'
        '<output class="catalogue-result-count" aria-live="polite"></output></form>'
        '<nav class="catalogue-family-index" aria-label="Scientific families"><ol>'
        f'{"".join(navigation)}</ol></nav>'
        '<p class="catalogue-no-results" hidden>No table views match these filters.</p>'
        f'<div class="catalogue-family-sections" id="family-sections">{"".join(sections)}</div>'
        '<p class="catalogue-source-note">Licenses and admission decisions are taken from the '
        'source-specific release manifest. Input and target fields are shown where the task contract '
        'specifies them. Reference availability does not imply that a model has been evaluated.</p></div>')


def render(report, output):
    """Render the local browser with the same body and assets as the docs site."""
    static = REPO / "doc/_static"
    css = (static / "hierarchy-browser.css").read_text()
    js = (static / "hierarchy-browser.js").read_text()
    body = render_fragment(report, output=output, data_links=True)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>SynRXN | Scientific task catalogue</title>'
            '<style>body{max-width:1180px;margin:0 auto;padding:36px 28px 64px;'
            'font:16px/1.6 system-ui,sans-serif;color:#202b33;background:#fff}'
            'body>header{border-bottom:2px solid #263e4a;margin-bottom:24px}'
            'body>header p{margin:0 0 4px;color:#53636c;font-size:.85rem}'
            'body>header h1{font:600 2rem/1.3 Georgia,serif;margin:0 0 20px}'
            'a{color:#195568}a:hover{text-decoration-thickness:2px}'
            '@media(max-width:640px){body{padding:24px 14px}}'
            f'{css}</style></head><body><header><p>SynRXN · Reaction informatics</p>'
            '<h1>Scientific task catalogue</h1></header>'
            f'{body}<script>{js}</script></body></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=physical_path("kdd/hierarchy"))
    args = parser.parse_args()
    report, flat = build_inventory(REPO)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "hierarchy.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (output / "hierarchy.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    (output / "index.html").write_text(render(report, output))
    print(json.dumps({"output_dir": str(output), **report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
