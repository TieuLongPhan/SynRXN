"""Generate manuscript inventory tables from the scientific metadata.

This does not verify native data or infer completed predictive evaluations.
Generated facts retain their input hashes and the inventory's counting units.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from synrxn.reconstruction.catalogue.release_scope import CATALOGUE, MAIN, SCOPES, select_scope
from synrxn.reconstruction.paths import REPO, physical_path


def latex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                    "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(value))


def build_facts(registry, hierarchy, release, ncs):
    tables = {}
    for family in hierarchy["families"]:
        for task in family["tasks"]:
            for table in task["tables"]:
                identity = (table["namespace"], table["key"])
                if identity in tables and any(tables[identity][field] != table[field]
                                              for field in ("path", "sha256", "rows", "role")):
                    raise ValueError("Conflicting physical table identity: " + repr(identity))
                tables[identity] = table
    tasks = registry["tasks"]
    bindings = {(t["id"], b["namespace"], b["key"]) for t in tasks for b in t["bindings"]}
    shown = {(t["id"], b["namespace"], b["key"]) for f in hierarchy["families"]
             for t in f["tasks"] for b in t["tables"]}
    if bindings != shown or len(tasks) != len({t["id"] for t in tasks}):
        raise ValueError("Manuscript inputs disagree about task bindings")
    admitted = {(t["namespace"], t["key"]) for t in release["release_tables"]}
    primary = {key: t for key, t in tables.items() if t["role"] != "overlapping_view"}
    corpus = ("canonical", "feasibility/frea_corpus")
    corpus_rows = tables[corpus]["rows"] if corpus in tables else 0
    counts = ncs["canonical"]["upstream_success_label_counts"]
    return {
        "schema": "synrxn.manuscript-facts.v1",
        "counting_policy": "Catalogue table views and source-specific stored-row units; overlapping sources are not globally deduplicated reactions",
        "families": len(registry["families"]), "tasks": len(tasks),
        "catalogue_table_views": len(tables),
        "stored_rows": sum(t["rows"] for t in tables.values()),
        "primary_table_views": len(primary),
        "primary_stored_rows": sum(t["rows"] for t in primary.values()),
        "overlapping_views": len(tables) - len(primary),
        "release_admitted_catalogue_views": len(set(tables) & admitted),
        "held_catalogue_views": len(set(tables) - admitted),
        "task_table_bindings": len(bindings),
        "primary_task_table_bindings": sum(b["role"] != "overlapping_view" for t in tasks for b in t["bindings"]),
        "whole_frea_corpus_rows": corpus_rows,
        "stored_rows_without_whole_frea_corpus": sum(t["rows"] for t in tables.values()) - corpus_rows,
        "primary_stored_rows_without_whole_frea_corpus": sum(t["rows"] for t in primary.values()) - corpus_rows,
        "ncs_canonical_rows": ncs["canonical"]["rows"],
        "ncs_positive": int(counts["1"]), "ncs_negative": int(counts["0"]),
        "predictive_results": "Pending; inventory and reconstruction checks do not establish predictive performance",
    }


def build_license_ledger(registry, release):
    """Join actual table rights; never substitute article or legacy code licenses."""
    rights = {}
    for category in ("release_tables", "held_tables"):
        for table in release[category]:
            identity = (table["namespace"], table["key"])
            if identity in rights:
                raise ValueError("Duplicate source-rights identity: " + repr(identity))
            admitted = category == "release_tables"
            if admitted and not table.get("license_expression"):
                raise ValueError("Admitted source has no license expression: " + repr(identity))
            rights[identity] = (table, admitted)
    sources = {}
    for task in registry["tasks"]:
        for binding in task["bindings"]:
            identity = (binding["namespace"], binding["key"])
            if identity not in rights:
                raise ValueError("Source missing from rights manifest: " + repr(identity))
            table, admitted = rights[identity]
            expression = table.get("license_expression")
            if (binding.get("release_license_expression") != expression
                    or binding["release_eligible"] != admitted):
                raise ValueError("Registry and source-rights manifest disagree: " + repr(identity))
            if identity not in sources:
                sources[identity] = {
                    "namespace": identity[0], "key": identity[1], "path": table["path"],
                    "sha256": table["expected_sha256"], "rows": table["expected_rows"],
                    "role": binding["role"], "release_admitted": admitted,
                    "license_expression": expression,
                    "rights_status": table.get("rights_status", binding["rights_status"]),
                    "partition": table["partition"],
                    "commercial_use_allowed": table["commercial_use_allowed"],
                    "evidence_ids": table.get("evidence_ids", []),
                    "source_id": table.get("source_id"),
                    "source_url": table.get("source_url"),
                    "reference_evidence": table.get("reference_evidence"),
                    "license_obligations": table["license_obligations"],
                    "qualification": table["qualification"],
                    "task_ids": [], "primary_task_ids": [],
                }
            source = sources[identity]
            if binding["rows"] != source["rows"] or binding["role"] != source["role"]:
                raise ValueError("Binding disagrees with licensed source identity: " + repr(identity))
            source["task_ids"].append(task["id"])
            if binding["role"] != "overlapping_view":
                source["primary_task_ids"].append(task["id"])
    expressions = sorted({source["license_expression"] for source in sources.values()
                          if source["license_expression"] is not None})
    codes = {expression: f"L{index}" for index, expression in enumerate(expressions, 1)}
    for source in sources.values():
        source["license_code"] = codes.get(source["license_expression"], "H")
        source["task_ids"].sort()
        source["primary_task_ids"].sort()
    return {
        "schema": "synrxn.manuscript-source-licenses.v1",
        "policy": "Each expression applies to its identified source table; codes listed together for a task are not a task-wide combined license. H identifies held sources, not permission to redistribute.",
        "license_key": [{"code": codes[expression], "license_expression": expression}
                        for expression in expressions],
        "tables": [sources[identity] for identity in sorted(sources)],
        "held_audit_views_outside_catalogue": [
            {"namespace": identity[0], "key": identity[1]}
            for identity, (_, admitted) in sorted(rights.items())
            if not admitted and identity not in sources
        ],
    }


def render_tasks(registry, ledger):
    sources = {(t["namespace"], t["key"]): t for t in ledger["tables"]}
    code_order = {item["code"]: index for index, item in enumerate(ledger["license_key"])}
    code_order["H"] = len(code_order)
    lines = ["% Generated by synrxn.reconstruction.presentation.manuscript; do not edit.",
             r"\begin{table*}[t]", r"\caption{Scientific task definitions and primary source-table references. Rows retain their source-specific units; task totals overlap. Source-license codes refer to separate source-specific expressions in Table~\ref{tab:source-license-key}, not a blanket task license. H denotes a held source, included in inventory counts but excluded from redistribution. These are reference quantities, not predictive test denominators.}",
             r"\label{tab:all-tasks}", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabularx}{\textwidth}{@{}Xlrr>{\raggedright\arraybackslash}p{1.12in}@{}}",
             r"\toprule Task & Family ID & Tables & Stored rows & Source licenses \\", r"\midrule"]
    for family in registry["families"]:
        for task in registry["tasks"]:
            if task["family_id"] == family["id"]:
                primary = [binding for binding in task["bindings"]
                           if binding["role"] != "overlapping_view"]
                codes = {sources[(b["namespace"], b["key"])]["license_code"] for b in primary}
                licenses = ", ".join(sorted(codes, key=code_order.__getitem__))
                lines.append(f"{latex(task['title'])} & {latex(family['id'])} & {task['primary_table_count']} & {task['primary_source_rows']:,} & {licenses}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table*}", ""])
    return "\n".join(lines)


def render_license_key(ledger):
    lines = ["% Generated by synrxn.reconstruction.presentation.manuscript; do not edit.",
             r"\begin{table*}[t]",
             r"\caption{Audited source-license expressions for Table~\ref{tab:all-tasks}. Expressions are retained verbatim from the release manifest: AND indicates joint obligations for that source; codes separated by commas in Table~\ref{tab:all-tasks} describe different source tables. Redistribution remains subject to attribution, notices and source-specific terms recorded in the source-license ledger.}",
             r"\label{tab:source-license-key}", r"\small",
             r"\begin{tabularx}{\textwidth}{@{}lX@{}}",
             r"\toprule Code & Effective source-license expression \\", r"\midrule"]
    for item in ledger["license_key"]:
        lines.append(latex(item["code"]) + " & " + r"\texttt{" +
                     latex(item["license_expression"]) + "}" + r" \\")
    held = [latex(t["namespace"] + "/" + t["key"]) for t in ledger["tables"]
            if not t["release_admitted"]]
    if held:
        lines.append("H & Held: no verified redistribution grant (" + ", ".join(held) +
                     r"); no license is inferred from an article or software notice. \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}",
        r"\par\smallskip\noindent The generated \texttt{source\_licenses.json} and \texttt{source\_licenses.csv} identify each of the " +
        str(len(ledger["tables"])) +
        " catalogue table views by source path and SHA-256, map it to these expressions and task bindings, and retain its admission status. " +
        str(len(ledger["held_audit_views_outside_catalogue"])) +
        r" additional quarantined views are outside this catalogue and these tables. The custom \texttt{LicenseRef-PNAS-NC-Educational} term retains its audited noncommercial educational-use qualification; it is not a standard Creative Commons grant.",
        r"\end{table*}", ""])
    return "\n".join(lines)


def generate(repo=REPO, output=None, *, scope=CATALOGUE):
    repo = Path(repo)
    output = Path(output) if output else repo / ".cache/synrxn/manuscript"
    names = ["kdd/task_registry.json", "kdd/hierarchy/hierarchy.json",
             "kdd/coverage/release_manifest.json", "kdd/case_studies/ncs_yield/report.json"]
    values = [json.loads(physical_path(name, repo).read_text()) for name in names]
    values[:2] = select_scope(*values[:2], scope)
    facts = build_facts(*values)
    facts["release_scope"] = scope
    ledger = build_license_ledger(values[0], values[2])
    facts["source_license_expression_count"] = len(ledger["license_key"])
    facts["input_sha256"] = {name: hashlib.sha256(physical_path(name, repo).read_bytes()).hexdigest() for name in names}
    facts["generator"] = {"path": "synrxn/reconstruction/presentation/manuscript.py",
                          "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    ledger["input_sha256"] = facts["input_sha256"]
    ledger["generator"] = facts["generator"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "facts.json").write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
    macros = {"Families": "families", "Tasks": "tasks", "TableViews": "catalogue_table_views",
              "StoredRows": "stored_rows", "PrimaryViews": "primary_table_views",
              "PrimaryRows": "primary_stored_rows", "AdmittedViews": "release_admitted_catalogue_views",
              "NCSRows": "ncs_canonical_rows", "NCSPositive": "ncs_positive", "NCSNegative": "ncs_negative"}
    lines = ["% Generated inventory facts; no model-performance claims."]
    for macro, key in macros.items():
        value = f"{facts[key]:,}".replace(",", r"\,")
        lines.append(r"\newcommand{\SynRXN" + macro + "}{" + value + "}")
    (output / "stats.tex").write_text("\n".join(lines) + "\n")
    (output / "tasks.tex").write_text(render_tasks(values[0], ledger))
    (output / "licenses.tex").write_text(render_license_key(ledger))
    (output / "source_licenses.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    fields = ["namespace", "key", "path", "sha256", "rows", "role", "release_admitted",
              "license_code", "license_expression", "rights_status", "partition",
              "commercial_use_allowed", "task_ids", "primary_task_ids", "evidence_ids",
              "source_id", "source_url", "reference_evidence", "license_obligations",
              "qualification"]
    with (output / "source_licenses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for source in ledger["tables"]:
            row = dict(source)
            for field in ("task_ids", "primary_task_ids", "evidence_ids", "license_obligations"):
                row[field] = json.dumps(row[field], separators=(",", ":"))
            writer.writerow(row)
    return facts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / ".cache/synrxn/manuscript")
    parser.add_argument("--scope", choices=SCOPES, default=MAIN)
    args = parser.parse_args(argv)
    facts = generate(output=args.output, scope=args.scope)
    print(json.dumps({key: facts[key] for key in ("families", "tasks", "catalogue_table_views", "stored_rows")}, sort_keys=True))


if __name__ == "__main__":
    main()
