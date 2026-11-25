"""Task membership for the default download and the full source catalogue."""

MAIN = "main"
CATALOGUE = "catalogue"
SCOPES = (MAIN, CATALOGUE)
OPTIONAL_TASKS = frozenset({"reaction_feasibility"})


def select_scope(registry, hierarchy, scope):
    """Select download tasks from the checked source definitions."""
    if scope not in SCOPES:
        raise ValueError(f"Unknown release scope: {scope}")
    if scope == CATALOGUE:
        return registry, hierarchy
    tasks = [task for task in registry["tasks"] if task["id"] not in OPTIONAL_TASKS]
    if len(tasks) == len(registry["tasks"]):
        return registry, hierarchy
    families = []
    for family in registry["families"]:
        members = [task for task in tasks if task["family_id"] == family["id"]]
        if not members:
            continue
        tables = {(binding["namespace"], binding["key"]): binding
                  for task in members for binding in task["bindings"]}
        primary = [binding for binding in tables.values() if binding["role"] != "overlapping_view"]
        families.append({
            **family,
            "task_ids": [task["id"] for task in members],
            "primary_table_count": len(primary),
            "primary_source_rows": sum(binding["rows"] for binding in primary),
            "overlapping_view_count": len(tables) - len(primary),
            "release_eligible_table_count": sum(binding.get("release_eligible", False) for binding in tables.values()),
        })
    by_family = {family["id"]: family for family in families}
    nodes = []
    for node in hierarchy["families"]:
        if node["id"] not in by_family:
            continue
        family = by_family[node["id"]]
        nodes.append({
            **node,
            "tasks": [task for task in node["tasks"] if task["id"] in family["task_ids"]],
            **{name: family[name] for name in (
                "primary_table_count", "primary_source_rows", "overlapping_view_count"
            )},
        })
    bindings = [b for task in tasks for b in task["bindings"]]
    tables = {(b["namespace"], b["key"]): b for b in bindings}
    overlaps = sum(b["role"] == "overlapping_view" for b in tables.values())
    summary = {
        "families": len(families), "tasks": len(tasks), "catalogue_tables": len(tables),
        "catalogue_rows": sum(table["rows"] for table in tables.values()),
        "unique_primary_tables": len(tables) - overlaps, "overlapping_views": overlaps,
        "unique_primary_rows": sum(table["rows"] for table in tables.values()
                                   if table["role"] != "overlapping_view"),
        "primary_task_table_bindings": sum(b["role"] != "overlapping_view" for b in bindings),
        "all_task_table_bindings": len(bindings),
    }
    return {**registry, "tasks": tasks, "families": families}, {**hierarchy, "families": nodes, "summary": summary}
