"""Select registered tasks and dispatch construction of their source tables."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ..paths import REPO, physical_path
from synrxn.reconstruction.families.suite import read_json
from ._tables import run_scope, table_plan


def read_registry(repo=REPO):
    """Read the scientific task definitions."""
    return read_json(physical_path('kdd/task_registry.json', repo))


def select_tasks(registry, families=(), tasks=(), select_all=False):
    """Resolve the union of explicit selectors; reject unknown/empty selections."""
    known_families = {item['id'] for item in registry['families']}
    known_tasks = {item['id'] for item in registry['tasks']}
    if set(families) - known_families or set(tasks) - known_tasks:
        raise ValueError('Unknown family/task selector: ' + ', '.join(sorted(
            (set(families) - known_families) | (set(tasks) - known_tasks))))
    if select_all and (families or tasks):
        raise ValueError('--all cannot be combined with --family/--task')
    selected = [item for item in registry['tasks'] if select_all or
                item['family_id'] in families or item['id'] in tasks]
    if not selected:
        raise ValueError('Choose --all, --family ID or --task ID; use --list for IDs')
    return selected


def scope_plan(repo=REPO, families=(), tasks=(), select_all=False):
    """Resolve the registered tasks and their source tables without reading rows."""
    repo = Path(repo).resolve()
    selected = select_tasks(read_registry(repo), families, tasks, select_all)
    return table_plan(repo, selected, families, tasks, select_all)


def plan(*, families=(), tasks=(), select_all=False, repo=REPO):
    """Inspect the source selection without reading data rows."""
    return scope_plan(repo, families, tasks, select_all)


def run(output, *, families=(), tasks=(), select_all=False, mode='verify',
        check_rows=False, repo=REPO):
    """Verify or rebuild the sources selected by task or family."""
    selected = plan(families=families, tasks=tasks, select_all=select_all, repo=repo)
    return run_scope(selected, output, mode=mode, check_rows=check_rows, repo=repo)


def plan_table(family_id, task_id, table, repo=REPO):
    """Select one registered table binding and its native builder dependencies."""
    repo = Path(repo).resolve()
    task, = select_tasks(read_registry(repo), tasks=(task_id,))
    if task['family_id'] != family_id:
        raise ValueError(f'{task_id} belongs to family {task["family_id"]}, not {family_id}')
    namespace, key = table
    bindings = [binding for binding in task['bindings']
                if (binding['namespace'], binding['key']) == table]
    if len(bindings) != 1:
        raise ValueError(f'Expected one {namespace}:{key} binding for {task_id}, found {len(bindings)}')
    selected = table_plan(repo, [{**task, 'bindings': bindings}], (), (task_id,), False)
    selected['selection']['table'] = {'namespace': namespace, 'key': key}
    return selected


def table_main(family_id, task_id, table, argv=None, *, build_table=None, source_pins=None, acquire=None):
    """Run a table entry point via the shared reconstruction workflow."""
    namespace, key = table
    parser = argparse.ArgumentParser(description=f'{task_id} source table {namespace}:{key}')
    parser.add_argument('action', choices=('prepare', 'plan', 'verify', 'build'), nargs='?', default='prepare')
    parser.add_argument('--check-rows', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--families-dir', type=Path, default=Path('Data/families'),
                        help='Destination for prepare')
    parser.add_argument('--cache', type=Path, help='Alternate cache with raw/ or prepared families/ under it')
    parser.add_argument('--offline', action='store_true', help='Use only available local inputs')
    parser.add_argument('--from-cache', action='store_true', help='Place the checksum-pinned prepared table')
    args = parser.parse_args(argv)
    try:
        selected = plan_table(family_id, task_id, table)
        if build_table is not None:
            selected.update(
                table_recipe={'source_pins': source_pins or {}},
                coverage_adapters=[],
                coverage_output_adapter_closure=[],
                coverage_branches=[key.split('/', 1)[0]],
                coverage_closure_table_keys=[key],
            )
        if args.action == 'plan':
            print(json.dumps(selected, indent=2))
            return 0
        if args.action == 'prepare':
            from ._prepare import prepare

            if args.output is not None:
                parser.error('use --families-dir for prepare, or choose build for a run report')
            result = prepare(selected, REPO, args.families_dir, args.cache, args.offline,
                             args.from_cache, build_table, acquire)
            print(json.dumps(result))
            return 0
        if args.cache or args.offline or args.from_cache or args.families_dir != Path('Data/families'):
            parser.error('cache and family destination options apply to prepare')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        output = args.output or REPO / '.cache/synrxn/runs/tables' / task_id / stamp
        result = run_scope(selected, output, mode='rebuild' if args.action == 'build' else 'verify',
                           check_rows=args.check_rows, build_table=build_table)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({key: result[key] for key in ('status', 'output_dir', 'counts', 'errors')}))
    return int(result['status'] != 'passed')
