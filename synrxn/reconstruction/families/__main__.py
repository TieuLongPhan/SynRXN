"""List, plan, verify or build the source tables for a task or family."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from synrxn.reconstruction.paths import REPO
from synrxn.reconstruction.families._selection import read_registry, run, scope_plan
from synrxn.reconstruction.families._selection import select_tasks as select_tasks
from synrxn.reconstruction.families._tables import run_scope as run_scope


def _run_selection(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List all family/task identifiers')
    parser.add_argument('--family', action='append', default=[], metavar='ID')
    parser.add_argument('--task', action='append', default=[], metavar='ID')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--mode', choices=['verify', 'rebuild'], default='verify')
    parser.add_argument('--plan', action='store_true', help='Print selection/dependency closure without hashing data or writing outputs')
    parser.add_argument('--check-rows', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.list:
        registry = read_registry()
        print(json.dumps([{'id': f['id'], 'title': f['title'], 'tasks': [
            {'id': t['id'], 'title': t['title']} for t in registry['tasks'] if t['family_id'] == f['id']]}
            for f in registry['families']], indent=2))
        return 0
    try:
        if args.plan:
            selected = scope_plan(REPO, args.family, args.task, args.all)
            print(json.dumps(selected, indent=2))
            return 0
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        result = run(args.output or REPO / '.cache/synrxn/runs/scope' / stamp,
                     families=args.family, tasks=args.task, select_all=args.all,
                     mode=args.mode, check_rows=args.check_rows)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({'status': result['status'], 'output_dir': result['output_dir'], 'counts': result['counts'], 'errors': result['errors'],
                      'predictive_models_run': 0}))
    return result['status'] != 'passed'


def main(argv=None):
    """Dispatch task and family reconstruction commands."""
    args = list(sys.argv[1:] if argv is None else argv)
    commands = ('list', 'plan', 'verify', 'build', 'import-source', 'organize', 'sync-entries')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=commands)
    if not args or args[0] in {'-h', '--help'}:
        parser.print_help()
        return 0
    command, rest = args[0], args[1:]
    if command.startswith('-'):
        return _run_selection(args)
    if command not in commands:
        parser.error('unknown command: ' + command)
    if command == 'organize':
        from synrxn.reconstruction.catalogue.layout import main as layout
        return layout(['build', *rest])
    if command == 'sync-entries':
        from synrxn.reconstruction.families._entries import main as sync_entries
        return sync_entries(rest)
    if command == 'import-source':
        from synrxn.reconstruction.families._importers import main as import_source
        return import_source(rest)
    prefix = {
        'list': ['--list'], 'plan': ['--plan'],
        'verify': ['--mode', 'verify'], 'build': ['--mode', 'rebuild'],
    }[command]
    return _run_selection([*prefix, *rest])


if __name__ == '__main__':
    command = _run_selection if __spec__ and __spec__.name.endswith('.scope') else main
    raise SystemExit(command())
