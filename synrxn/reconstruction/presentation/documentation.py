"""Build website views from the catalogue and current redistribution decisions.

Documentation builds read metadata only. The recorded catalogue inputs remain
unchanged; public examples and license fields follow the current release review.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

from synrxn.reconstruction.catalogue import hierarchy
from synrxn.reconstruction.catalogue.release_scope import MAIN, select_scope

HIERARCHY = "Data/metadata/reconstruction/hierarchy/hierarchy.json"
REGISTRY = "Data/metadata/reconstruction/task_registry.json"
RELEASE = "Data/metadata/reconstruction/coverage/release_manifest.json"
CATALOGUES = ("catalog-data", "catalog-sharealike-data")


def catalog_citations(path):
    """Format the existing bibliography for the source-table browser.

    :param path: Documentation BibTeX file.
    :return: Citation text and publication links indexed by citation key.
    """
    from pybtex.database import parse_file
    from pybtex.plugin import find_plugin

    bibliography = parse_file(path)
    style = find_plugin("pybtex.style.formatting", "unsrt")()
    return {
        entry.key: {
            "text": entry.text.render_as("text"),
            "doi": bibliography.entries[entry.key].fields.get("doi", ""),
            "url": bibliography.entries[entry.key].fields.get("url", ""),
        }
        for entry in style.format_entries(bibliography.entries.values())
    }


def _add_citations(app):
    citations = catalog_citations(Path(app.srcdir) / "refs.bib")
    payload = json.dumps(citations, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e")
    app.add_js_file(None, body="window.SYNRXN_CITATIONS = " + payload + ";", priority=400)


def _read(repo, relative):
    return json.loads((repo / relative).read_text(encoding="utf-8"))


def release_report(repo):
    """Select the same task and table scope as the main DuckDB build."""
    registry = _read(repo, REGISTRY)
    report = _read(repo, HIERARCHY)
    return select_scope(registry, report, MAIN)[1]


def canonical_catalog(catalogue, release, *, identities=None):
    """Apply current table terms and omit examples from metadata-only entries."""
    result = copy.deepcopy(catalogue)
    admitted = {(item["namespace"], item["key"])
                for item in release["release_tables"]}
    decisions = {(item["namespace"], item["key"]): item for item in
                 release["release_tables"] + release["held_tables"]}
    if identities is not None:
        result["datasets"] = [item for item in result["datasets"]
                              if ("canonical", item["id"]) in identities]
    for item in result["datasets"]:
        identity = "canonical", item["id"]
        decision = decisions[identity]
        item["release_eligible"] = identity in admitted
        item["rights_status"] = decision["rights_status"]
        item["license"] = decision.get("license_expression") or "Unresolved"
        item["release_partition"] = decision["partition"]
        item["license_obligations"] = decision.get("license_obligations", [])
        if not item["release_eligible"]:
            for key, value in (("sample", []), ("depiction", None),
                               ("reaction_text", ""), ("target_summaries", {})):
                item[key] = value
    # Integrity checks establish artifact identity, not scientific validity.
    result["release"].pop("integrity", None)
    result["release"]["license_source"] = RELEASE
    result["release"]["dataset_count"] = len(result["datasets"])
    return result


def family_summary(report):
    """Return an RST table; source tables can be reused across families."""
    lines = [".. list-table:: Scientific families", "   :header-rows: 1",
             "   :widths: 58 12 15 15", "", "   * - Family", "     - Tasks",
             "     - Data tables", "     - Metadata only"]
    for family in report["families"]:
        tables = {(table["namespace"], table["key"]): table
                  for task in family["tasks"] for table in task["tables"]}
        admitted = sum(table["release_eligible"] for table in tables.values())
        lines.extend([
            f'   * - `{family["title"]} <families.html#family-{family["id"]}>`_',
            f'     - {len(family["tasks"])}',
            f'     - {admitted}',
            f'     - {len(tables) - admitted}',
        ])
    return "\n".join(lines)


def _source_read(app, docname, source):
    if docname in {"catalog", "catalog_sharealike"}:
        app.env.note_dependency(str(Path(app.srcdir) / "refs.bib"))
    if docname not in {"index", "families"}:
        return
    repo = Path(app.srcdir).parent
    report = release_report(repo)
    for relative in (HIERARCHY, REGISTRY, RELEASE):
        app.env.note_dependency(str(repo / relative))
    for module_path in (__file__, hierarchy.__file__, Path(hierarchy.__file__).with_name("release_scope.py")):
        app.env.note_dependency(module_path)
    if docname == "index":
        public = hierarchy.public_inventory(report, repo=repo)
        source[0] = source[0].replace(".. synrxn-family-summary", family_summary(public))
    else:
        fragment = hierarchy.render_fragment(report, repo=repo, exports={
            "Task–table inventory (CSV)": "_static/families.csv",
            "Definitions and source records (JSON)": "_static/families.json",
        })
        raw = ".. raw:: html\n\n" + "\n".join("   " + line for line in fragment.splitlines())
        source[0] = source[0].replace(".. synrxn-family-browser", raw)
        # Canonical source locks supply additional citation links.
        for path in (repo / "Data/metadata/sources").glob("*/*.json"):
            app.env.note_dependency(str(path))


def _write_exports(app, exception):
    if exception is not None or app.builder.format != "html":
        return
    repo = Path(app.srcdir).parent
    static = Path(app.outdir) / "_static"
    static.mkdir(parents=True, exist_ok=True)
    report = hierarchy.public_inventory(release_report(repo), repo=repo)
    report["release_scope"] = MAIN
    (static / "families.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ("family_id", "family", "task_id", "task", "namespace", "key", "title",
              "rows", "counting_unit", "role", "release_eligible",
              "release_license_expression", "rights_status", "sha256", "source_urls")
    with (static / "families.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for family in report["families"]:
            for task in family["tasks"]:
                for table in task["tables"]:
                    record = {key: table.get(key, "") for key in fields}
                    record.update(family_id=family["id"], family=family["title"],
                                  task_id=task["id"], task=task["title"],
                                  source_urls=" | ".join(table.get("source_urls", [])))
                    writer.writerow(record)

    identities = {(table["namespace"], table["key"])
                  for family in report["families"] for task in family["tasks"]
                  for table in task["tables"]}
    release = _read(repo, RELEASE)
    for name in CATALOGUES:
        original = _read(repo, f"doc/_static/{name}.json")
        catalogue = canonical_catalog(original, release, identities=identities)
        catalogue["release"]["license_source_sha256"] = hashlib.sha256(
            (repo / RELEASE).read_bytes()).hexdigest()
        payload = json.dumps(catalogue, ensure_ascii=False, indent=2) + "\n"
        (static / f"{name}.json").write_text(payload, encoding="utf-8")
        # The script form supports file:// previews without a fetch request.
        (static / f"{name}.js").write_text(
            "window.SYNRXN_CATALOG = " + payload.rstrip() + ";\n", encoding="utf-8")
        eligible = {item["id"] for item in catalogue["datasets"] if item["release_eligible"]}
        for old in original["datasets"]:
            if old["id"] not in eligible and old.get("depiction"):
                preview = (Path(app.outdir) / old["depiction"]).resolve()
                if preview.is_relative_to(static.resolve()):
                    preview.unlink(missing_ok=True)
                    preview.with_suffix(".cdk.svg").unlink(missing_ok=True)


def setup(app):
    app.add_css_file("hierarchy-browser.css")
    app.add_js_file("hierarchy-browser.js", defer="defer")
    app.connect("builder-inited", _add_citations)
    app.connect("source-read", _source_read)
    app.connect("build-finished", _write_exports)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
