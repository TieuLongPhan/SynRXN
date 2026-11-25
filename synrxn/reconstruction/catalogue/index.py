#!/usr/bin/env python3
"""Index family/task reproduction commands from metadata without scanning data.

This index declares available selectors and source boundaries. It is not a
verification certificate, a successful native rebuild or a model-run report.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex

from synrxn.reconstruction.families.suite import dataset_inventory
from synrxn.reconstruction.paths import REPO, physical_path, logical_relative


BOUNDARIES = {
    "canonical": "frozen_verified_only",
    "expansion": "declared_native_regeneration_supported",
    "coverage": "declared_native_regeneration_supported",
}


def identity(table: dict) -> tuple[str, str]:
    return table["namespace"], table["key"]


def command(kind: str, selector: str, mode: str) -> str:
    return shlex.join([
        "python", "-m", "synrxn.reconstruction", "families",
        "build" if mode == "rebuild" else "verify", "--" + kind, selector,
        "--check-rows", "--output",
        f".cache/synrxn/runs/reproduction/{kind}/{selector}/{mode}",
    ])


def scope_entry(kind: str, item: dict, tasks: list[dict]) -> dict:
    unique = {}
    for task in tasks:
        for binding in task["bindings"]:
            key = identity(binding)
            entry = {
                "namespace": key[0], "key": key[1], "rows": binding["rows"],
                "role": binding["role"], "counting_unit": binding["counting_unit"],
                "release_eligible": binding["release_eligible"],
                "source_boundary": BOUNDARIES[key[0]],
            }
            if key in unique and unique[key] != entry:
                raise ValueError("Conflicting shared-table metadata: " + repr(key))
            unique[key] = entry
    tables = [unique[key] for key in sorted(unique)]
    return {
        "id": item["id"], "title": item["title"],
        "task_ids": [task["id"] for task in tasks],
        "table_count": len(tables),
        "primary_table_count": sum(t["role"] != "overlapping_view" for t in tables),
        "stored_rows": sum(t["rows"] for t in tables),
        "primary_stored_rows": sum(t["rows"] for t in tables if t["role"] != "overlapping_view"),
        "release_eligible_table_count": sum(t["release_eligible"] for t in tables),
        "tables": tables,
        "commands": {mode: command(kind, item["id"], mode) for mode in ("verify", "rebuild")},
        "execution_status": "pending; consult a successful run manifest for certification",
    }


def build_index(registry: dict, inventory: list[dict]) -> dict:
    """Build selectors using catalogue declarations only, with no data-file IO."""
    sources = {identity(table): table for table in inventory}
    if len(sources) != len(inventory):
        raise ValueError("Duplicate catalogue source identity")
    tasks = registry["tasks"]
    by_id = {task["id"]: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Duplicate scientific task ID")
    assigned = [key for family in registry["families"] for key in family["task_ids"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(by_id):
        raise ValueError("Every scientific task must have exactly one family")
    referenced = set()
    source_bindings = {key: [] for key in sources}
    for task in tasks:
        for binding in task["bindings"]:
            key = identity(binding)
            if key not in sources:
                raise ValueError("Uncatalogued binding: " + repr(key))
            if binding["rows"] != sources[key]["expected_rows"]:
                raise ValueError("Binding/catalogue row mismatch: " + repr(key))
            referenced.add(key)
            source_bindings[key].append((task, binding))
    if referenced != set(sources):
        raise ValueError("Catalogue sources lack task assignments: " + repr(set(sources) - referenced))
    family_entries = []
    for family in registry["families"]:
        members = [by_id[key] for key in family["task_ids"]]
        if any(task["family_id"] != family["id"] for task in members):
            raise ValueError("Task family assignment disagrees with family membership")
        family_entries.append(scope_entry("family", family, members))
    task_entries = []
    for task in tasks:
        entry = scope_entry("task", task, [task])
        entry["family_id"] = task["family_id"]
        task_entries.append(entry)
    tables = []
    for key in sorted(sources):
        source = sources[key]
        bindings = source_bindings[key]
        eligible = {binding["release_eligible"] for _, binding in bindings}
        roles = {binding["role"] for _, binding in bindings}
        if len(eligible) != 1 or len(roles) != 1:
            raise ValueError("Conflicting table role or release eligibility: " + repr(key))
        tables.append({
            "namespace": key[0], "key": key[1], "path": source["path"],
            "format": source["format"], "rows": source["expected_rows"],
            "expected_sha256": source["expected_sha256"],
            "expected_bytes": source.get("expected_bytes"),
            "role": next(iter(roles)), "release_eligible": next(iter(eligible)),
            "counting_units": sorted({binding["counting_unit"] for _, binding in bindings}),
            "task_ids": sorted({task["id"] for task, _ in bindings}),
            "family_ids": sorted({task["family_id"] for task, _ in bindings}),
            "source_boundary": BOUNDARIES[key[0]],
            "verification_status": "pending; expected metadata only",
            "regeneration_status": ("not certified; frozen input is verified in place"
                                    if key[0] == "canonical" else
                                    "pending; native-source builder execution required"),
        })
    counts = Counter(table["namespace"] for table in tables)
    return {
        "schema": "synrxn.reproduction-index.v1",
        "as_of": registry.get("as_of"),
        "scope": "metadata index; no dataset scans, downloads, regeneration or predictive model runs",
        "summary": {
            "families": len(family_entries), "tasks": len(task_entries),
            "catalogue_table_views": len(tables), "tables_by_namespace": dict(sorted(counts.items())),
            "stored_rows": sum(table["rows"] for table in tables),
            "completed_predictive_runs_by_index": 0,
        },
        "counting_policy": (
            "Deduplicate source identities by (namespace, key) within each scope; "
            "rows have source-specific units and can overlap across views."
        ),
        "source_boundaries": BOUNDARIES,
        "output_policy": (
            "Commands require a new or empty output directory. "
            "Change the final directory component before repeating a run."
        ),
        "execution_status": "pending; index generation certifies no source verification or native regeneration",
        "families": family_entries, "tasks": task_entries, "tables": tables,
    }


def markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render(index: dict) -> str:
    summary = index["summary"]
    lines = [
        "# Family and task reproduction index", "",
        f"**{summary['families']} families · {summary['tasks']} tasks · "
        f"{summary['catalogue_table_views']} catalogue table views.**",
        "",
        "One shared runner selects each scientific family or task. "
        "This generated index reads metadata only. "
        "See doc/data_concept.rst for environment locks and the complete suite workflow.",
        "",
        "Commands require a new or empty output directory. Use a fresh directory for each run. "
        "The rebuild command verifies frozen inputs and runs the available native-source builders.",
        "",
        "| Source namespace | Table views | Reproduction boundary |",
        "| --- | ---: | --- |",
    ]
    for namespace, count in summary["tables_by_namespace"].items():
        lines.append(f"| {namespace} | {count} | `{BOUNDARIES[namespace]}` |")
    lines += [
        "", "Use the run manifest to confirm native regeneration.", "", "## Families", "",
        "| Family | Tasks | Unique table views | Stored rows |", "| --- | ---: | ---: | ---: |",
    ]
    for family in index["families"]:
        anchor = family["id"].replace("_", "-")
        lines.append(
            f"| [{markdown(family['title'])}](#{anchor}) | {len(family['task_ids'])} | "
            f"{family['table_count']} | {family['stored_rows']:,} |"
        )
    by_id = {task["id"]: task for task in index["tasks"]}
    for family in index["families"]:
        lines += [
            "", f"<a id=\"{family['id'].replace('_', '-')}\"></a>", "",
            f"### {family['title']} (`{family['id']}`)", "", "```bash",
            family["commands"]["verify"], family["commands"]["rebuild"], "```", "",
            "| Task | Table views | Stored rows | Verify command | Rebuild command |",
            "| --- | ---: | ---: | --- | --- |",
        ]
        for task_id in family["task_ids"]:
            task = by_id[task_id]
            lines.append(
                f"| {markdown(task['title'])} (`{task_id}`) | {task['table_count']} | {task['stored_rows']:,} | "
                f"`{task['commands']['verify']}` | `{task['commands']['rebuild']}` |"
            )
    lines += [
        "", "## Source table views", "",
        "Each table appears once below. A table can contribute to several family or task totals. "
        "Roles and release eligibility follow the registry.", "",
        "| Source identity | Rows | Counting unit(s) | Role | Release eligible |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for table in index["tables"]:
        source = markdown(f"{table['namespace']}:{table['key']}")
        units = markdown(", ".join(table["counting_units"]))
        eligible = "yes" if table["release_eligible"] else "no"
        lines.append(f"| `{source}` | {table['rows']:,} | {units} | {table['role']} | {eligible} |")
    lines += [
        "",
        "The [machine-readable index](index.json) records paths, hashes, selector membership and execution status.",
        "",
    ]
    return "\n".join(lines)


def load_index(repo: Path) -> dict:
    """Read the current task registry and catalogue metadata."""
    registry_path = physical_path("kdd/task_registry.json", repo)
    registry = json.loads(registry_path.read_text(encoding="utf8"))
    return build_index(registry, dataset_inventory(repo))


def render_task(index: dict, task_id: str, repo: Path) -> str:
    """Show the stored population and release status of each table for one task."""
    task = next((entry for entry in index["tasks"] if entry["id"] == task_id), None)
    if task is None:
        raise KeyError(task_id)
    sources = {identity(source): source for source in index["tables"]}
    lines = [
        f"## {task['title']}", "",
        f"{task['table_count']} table views with {task['stored_rows']:,} stored rows.",
        "Row units and overlaps follow the source metadata.", "",
        "| Table ID | Stored rows | Counting unit | Role | Release eligible |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for table in task["tables"]:
        namespace, key = identity(table)
        eligible = "Yes" if table["release_eligible"] else "Held"
        role = table["role"].replace("_", " ")
        unit = markdown(table["counting_unit"])
        lines.append(
            f"| `{namespace}:{markdown(key)}` | {table['rows']:,} | {unit} | {role} | {eligible} |"
        )
    lines += ["", "Table paths", ""]
    for table in task["tables"]:
        source = sources[identity(table)]
        path = physical_path(source["path"], repo).relative_to(repo.resolve())
        lines.append(f"- `{path.as_posix()}`")
    return "\n".join(lines)


def generate(repo: Path, output: Path, *, reports: bool = False) -> dict:
    index = load_index(repo)
    metadata_paths = [
        physical_path("kdd/task_registry.json", repo), repo / "doc/_static/catalog-data.json",
        repo / "doc/_static/catalog-sharealike-data.json", repo / "Data/expansion/catalog.json",
        repo / "synrxn/reconstruction/families/suite.py",
        repo / "synrxn/reconstruction/families/__init__.py",
        repo / "synrxn/reconstruction/families/__main__.py",
        repo / "synrxn/reconstruction/families/_selection.py",
        repo / "synrxn/reconstruction/families/_tables.py",
        repo / "synrxn/reconstruction/catalogue/index.py",
        repo / "synrxn/reconstruction/__main__.py", repo / "synrxn/reconstruction/catalogue/layout.py",
        *sorted(physical_path("Data/coverage_expansion", repo).glob("*/catalog.json")),
    ]
    index["source_sha256"] = {
        logical_relative(path, repo): hashlib.sha256(physical_path(path, repo).read_bytes()).hexdigest()
        for path in sorted(metadata_paths)
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    if reports:
        (output / "index.md").write_text(render(index), encoding="utf8")
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", help="Print one task's tables and row counts without writing an index")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reports", action="store_true",
                        help="Also export a Markdown command index")
    args = parser.parse_args(argv)
    if args.task is not None:
        if args.reports or args.output_dir is not None:
            parser.error("--task prints to standard output. Run without --task to export an index.")
        index = load_index(REPO)
        if not any(task["id"] == args.task for task in index["tasks"]):
            parser.error(f"unknown task ID {args.task!r}")
        print(render_task(index, args.task, REPO))
        return 0
    output = args.output_dir or REPO / ".cache/synrxn/reproduction-index"
    result = generate(REPO, output, reports=args.reports)
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
