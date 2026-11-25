#!/usr/bin/env python3
"""Rebuild acquired coverage in an isolated workspace and compare exact bytes.

Uses an offline native-asset cache hardlinked read-only by convention. Does not
fetch or modify the canonical data, source cache or existing derived tables.
The isolated code, new outputs, command logs and comparison manifest remain
in the requested new directory. Native inputs are checked by audit.py.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from synrxn.reconstruction.paths import REPO, family_storage, source_path, physical_path, logical_relative
ADAPTERS = ('knowledge', 'bio', 'physics', 'quantumpioneer', 'process',
            'solution_context', 'mcsa', 'solution_barriers')
# process.py also emits the separate impurity catalogue when its native cache
# exists. The direct impurity entry point permits a narrow task-only rebuild.
SELECTABLE_ADAPTERS = (*ADAPTERS, 'impurity')


def output_adapter_closure(adapters):
    owners = set(adapters)
    if 'process' in owners:
        owners.add('impurity')
    return owners


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def compare_workspace(workspace, repo=REPO, branches=None):
    """Compare native historical output paths against current physical storage."""
    workspace, repo = Path(workspace), Path(repo)
    expected_paths, comparisons = set(), []
    metadata = physical_path('Data/coverage_expansion', repo)
    quarantine = physical_path('Data/coverage_expansion/bio/quarantine', repo)
    logical_paths = {Path(logical_relative(path, repo)) for base in (metadata, quarantine)
                     for path in base.rglob('*') if path.is_file()}
    logical_paths.update(Path(key) for key in family_storage(repo)
                         if key.startswith('Data/coverage_expansion/'))
    for relative in sorted(logical_paths):
        if branches is not None and relative.parts[2] not in branches:
            continue
        path = physical_path(relative, repo)
        if not path.is_file():
            continue
        expected_paths.add(relative)
        rebuilt = workspace / relative
        observed = sha(rebuilt) if rebuilt.is_file() else None
        expected = sha(path)
        comparisons.append({'path': relative.as_posix(), 'expected_sha256': expected,
                            'observed_sha256': observed, 'status': 'verified' if observed == expected else 'failed'})
    if not expected_paths:
        raise ValueError('No frozen coverage outputs match the selected branches')
    for path in sorted((workspace / 'Data/coverage_expansion').rglob('*')):
        if path.is_file() and path.relative_to(workspace) not in expected_paths:
            comparisons.append({'path': path.relative_to(workspace).as_posix(), 'expected_sha256': None,
                                'observed_sha256': sha(path), 'status': 'failed', 'error': 'Unexpected generated file'})
    return comparisons


def rebuild(output, repo=REPO, adapters=None):
    selected = tuple(adapters) if adapters is not None else ADAPTERS
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(SELECTABLE_ADAPTERS):
        raise ValueError('Select distinct known coverage adapters')
    output_owners = output_adapter_closure(selected)
    output_branches = set()
    for path in physical_path('Data/coverage_expansion', repo).glob('*/catalog.json'):
        catalog = json.loads(path.read_text())
        adapter = catalog.get('adapter', catalog.get('builder', ''))
        if isinstance(adapter, dict):
            adapter = adapter.get('path', '')
        if Path(adapter).stem in output_owners:
            output_branches.add(path.parent.name)
    if adapters is not None and not output_branches:
        raise ValueError('No source catalogues belong to the selected adapters')
    output = output.resolve()
    protected = [repo / name for name in ['script', 'kdd', 'ncs', 'synrxn', 'Data', 'archive/research', 'paper/kdd/literature']]
    if output == repo or any(output.is_relative_to(path) for path in protected) or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError('Use a new or empty output directory outside source/data trees')
    workspace = output / 'workspace'
    workspace.mkdir(parents=True)
    temporary_inputs, steps, comparisons, errors = [], [], [], []
    input_stats = {}

    def link_input(source, destination):
        source = Path(source)
        before = source.stat()
        input_stats[source] = (before.st_size, before.st_mtime_ns, before.st_ino)
        try:
            return os.link(source, destination)
        except OSError as error:
            import errno
            if error.errno != errno.EXDEV:
                raise
            return shutil.copyfile(source, destination)

    try:
        for name in ['kdd/coverage', 'kdd/integration/sources']:
            shutil.copytree(physical_path(name, repo), workspace / name,
                            ignore=shutil.ignore_patterns('__pycache__', 'release_audit.json', 'release_manifest.json'))
        # Execute the preserved original sources in their historical relative
        # layout. Data contracts and embedded source hashes remain byte-exact.
        archived = repo / 'archive/source-code/kdd/coverage'
        if archived.exists():
            shutil.copytree(archived, workspace / 'kdd/coverage', dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(physical_path('kdd/scope_policy.json', repo), workspace / 'kdd/scope_policy.json')
        (workspace / 'script').mkdir()
        shutil.copyfile(source_path('script/fetch_expansion_data.py', repo),
                        workspace / 'script/fetch_expansion_data.py')
        (workspace / 'Data').mkdir()
        for name in ['coverage', 'synrxn2']:
            source = physical_path(f'Data/raw/{name}', repo)
            if source.exists():
                shutil.copytree(source, workspace / 'Data/raw' / name,
                                copy_function=link_input, ignore=shutil.ignore_patterns('.ranges', '*.parts', '*.partial'))
        expansion = workspace / 'Data/expansion'
        expansion.mkdir()
        frozen = physical_path('Data/expansion', repo)
        expansion_paths = {path.relative_to(frozen) for path in frozen.rglob('*') if path.is_file()}
        expansion_paths.update(Path(key).relative_to('Data/expansion') for key in family_storage(repo)
                               if key.startswith('Data/expansion/'))
        for name in sorted(expansion_paths):
            native = expansion / name
            native.parent.mkdir(parents=True, exist_ok=True)
            native.symlink_to(physical_path(Path('Data/expansion') / name, repo))
        # Direct M-CSA provenance is a pinned native archive under ncs/sources,
        # rather than Data/raw. Include every source-lock dependency explicitly.
        for lock_path in sorted(physical_path('kdd/coverage/adapters', repo).glob('*_sources.json')):
            if adapters is not None and lock_path.name.removesuffix('_sources.json') not in output_owners:
                continue
            for asset in json.loads(lock_path.read_text()).get('assets', []):
                relative = Path(asset['path'])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Unsafe native source-lock path')
                source, destination = source_path(relative, repo), workspace / relative
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                link_input(source, destination)
                temporary_inputs.append(destination)
        env = {**os.environ, 'PYTHONHASHSEED': '2027', 'OMP_NUM_THREADS': '1',
               'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
        for name in selected:
            command = [sys.executable, str(workspace / 'kdd/coverage/adapters' / (name + '.py'))]
            print('Rebuilding ' + name, flush=True)
            with (output / (name + '.log')).open('w') as log:
                completed = subprocess.run(command, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            steps.append({'adapter': name, 'command': command, 'returncode': completed.returncode})
            if completed.returncode:
                break
        comparisons = compare_workspace(workspace, repo, output_branches if adapters is not None else None)
    except (Exception, KeyboardInterrupt) as error:
        errors.append(f'{type(error).__name__}: {error}')
    finally:
        # Retain rebuilt outputs/code/logs, not native carriers. Always remove
        # temporary links, including on setup, launch or comparison failures.
        raw_cache = workspace / 'Data/raw'
        if raw_cache.exists():
            shutil.rmtree(raw_cache)
        expansion = workspace / 'Data/expansion'
        if expansion.is_dir():
            shutil.rmtree(expansion)
        for path in temporary_inputs:
            if path.exists():
                path.unlink()
    changed = []
    for source, before in input_stats.items():
        try:
            after = source.stat()
            if before != (after.st_size, after.st_mtime_ns, after.st_ino):
                changed.append(str(source.relative_to(repo)))
        except OSError:
            changed.append(str(source.relative_to(repo)))
    if changed:
        errors.append('Native inputs changed during isolated rebuilding')
    status = 'passed' if not errors and len(steps) == len(selected) and all(s['returncode'] == 0 for s in steps) and all(c['status'] == 'verified' for c in comparisons) else 'failed'
    manifest = {'schema': 'synrxn.coverage-rebuild.v1', 'status': status, 'network': 'offline; --fetch never used',
                'scope': 'all_coverage_adapters' if adapters is None else 'selected_adapters_only',
                'selected_adapters': list(selected),
                'output_adapter_closure': sorted(output_owners),
                'steps': steps, 'comparisons': comparisons, 'errors': errors, 'changed_native_inputs': changed, 'verified_files': sum(c['status'] == 'verified' for c in comparisons),
                'failed_files': [c['path'] for c in comparisons if c['status'] != 'verified']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': status, 'verified_files': manifest['verified_files'], 'failed_files': manifest['failed_files']}), flush=True)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--adapters', nargs='+', choices=SELECTABLE_ADAPTERS,
                   help='Rebuild only these adapters; manifest explicitly records the limited scope')
    args = p.parse_args(argv)
    return rebuild(args.output, adapters=args.adapters)['status'] != 'passed'


if __name__ == '__main__':
    raise SystemExit(main())
