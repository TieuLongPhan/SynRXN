"""Resolve selected tables and run their construction and verification."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from synrxn.reconstruction.families import suite as reproduce
from ..coverage import rebuild
from ..paths import REPO, logical_relative, physical_path


def adapter_name(catalog):
    adapter = catalog.get('adapter', catalog.get('builder', ''))
    return Path(adapter.get('path', '') if isinstance(adapter, dict) else adapter).stem


def table_plan(repo, selected, families, tasks, select_all):
    """Resolve table bindings and shared builder dependencies."""
    inventory = {(r['namespace'], r['key']): r for r in reproduce.dataset_inventory(repo)}
    bindings = [{'task_id': task['id'], 'family_id': task['family_id'], **binding}
                for task in selected for binding in task['bindings']]
    identities = {(b['namespace'], b['key']) for b in bindings}
    missing = identities - set(inventory)
    if missing:
        raise ValueError('Registry binding has no catalogue table: ' + repr(sorted(missing)))
    primary_ids = {(b['namespace'], b['key']) for b in bindings if b['role'] != 'overlapping_view'}
    tables = [{**inventory[key], 'role': 'primary' if key in primary_ids else 'overlapping_view'}
              for key in sorted(identities)]
    catalogs = {p.parent.name: (p, reproduce.read_json(p))
                for p in physical_path('Data/coverage_expansion', repo).glob('*/catalog.json')}
    coverage_owners = set()
    for namespace, key in identities:
        if namespace == 'coverage':
            coverage_owners.add(adapter_name(catalogs[key.split('/')[0]][1]))
    unsupported = coverage_owners - set(rebuild.SELECTABLE_ADAPTERS)
    if unsupported:
        raise ValueError('Coverage tables have no registered rebuilder: ' + repr(sorted(unsupported)))
    owners = rebuild.output_adapter_closure(coverage_owners)
    closure_branches = sorted(branch for branch, (_, cat) in catalogs.items()
                              if adapter_name(cat) in owners)
    expansion_catalog = reproduce.read_json(repo / 'Data/expansion/catalog.json')
    expansion_keys = {key for namespace, key in identities if namespace == 'expansion'}
    expansion_sources = {t['source_id'] for t in expansion_catalog['tables'] if t['key'] in expansion_keys}
    # Knowledge's measured-scope view consumes the original Minot expansion.
    if 'knowledge' in coverage_owners:
        expansion_sources.add('C02')
    closure_tables = [(branch + '/' + t['id']) for branch in closure_branches
                      for t in catalogs[branch][1]['tables']]
    expansion_closure = sorted(t['key'] for t in expansion_catalog['tables'] if t['source_id'] in expansion_sources)
    primary = [t for t in tables if t['role'] != 'overlapping_view']
    return {'schema': 'synrxn.scope-plan.v1',
            'selection': {'families': sorted(set(families)), 'tasks': sorted(set(tasks)), 'all': select_all},
            'task_ids': [t['id'] for t in selected],
            'family_ids': sorted({t['family_id'] for t in selected}),
            'tables': tables, 'bindings': bindings,
            'counts': {'families': len({t['family_id'] for t in selected}), 'tasks': len(selected),
                       'unique_tables': len(tables), 'primary_tables': len(primary),
                       'overlapping_views': len(tables) - len(primary),
                       'stored_source_rows': sum(t['expected_rows'] for t in tables),
                       'primary_source_rows': sum(t['expected_rows'] for t in primary),
                       'task_table_bindings': len(bindings),
                       'release_eligible_bindings': sum(bool(b.get('release_eligible', True)) for b in bindings)},
            'coverage_adapters': sorted(coverage_owners),
            'coverage_output_adapter_closure': sorted(owners),
            'coverage_branches': closure_branches, 'coverage_closure_table_keys': sorted(closure_tables),
            'expansion_sources': sorted(expansion_sources),
            'expansion_closure_table_keys': expansion_closure,
            'canonical_reproduction': 'frozen inputs verified; native importers are not run',
            'counting_policy': 'Source-table rows retain their native units; tables and task bindings can overlap. Not globally unique reactions.',
            'predictive_models_run': 0}


def dependency_records(plan, repo):
    """Select checksum-pinned metadata, code and known native dependencies."""
    pins = {}
    def add(path, digest=None, size=None, namespace='metadata'):
        path = Path(path)
        relative = path.as_posix()
        full = reproduce.local_path(repo, relative)
        if digest is None:
            digest = reproduce.sha(full)
        record = {'namespace': namespace, 'key': relative, 'path': relative, 'expected_sha256': digest}
        if size is not None:
            record['expected_bytes'] = size
        if relative in pins and pins[relative]['expected_sha256'] != digest:
            raise ValueError('Conflicting source pins: ' + relative)
        pins[relative] = record

    registry = reproduce.read_json(repo / 'kdd/task_registry.json')
    for path in sorted((Path(repo) / 'synrxn').rglob('*.py')):
        add(path.relative_to(repo), namespace='runtime_code')
    for relative in ['kdd/task_registry.json', 'synrxn/reconstruction/families/_tables.py',
                     'synrxn/reconstruction/families/suite.py', 'synrxn/reconstruction/paths.py',
                     'synrxn/reconstruction/provenance/compat.py', 'synrxn/reconstruction/__main__.py',
                     'synrxn/reconstruction/__init__.py', 'synrxn/reconstruction/coverage/rebuild.py']:
        add(relative)
    selected_catalogs = {'Data/expansion/catalog.json'}
    if any(t['namespace'] == 'canonical' for t in plan['tables']):
        selected_catalogs.update(['doc/_static/catalog-data.json', 'doc/_static/catalog-sharealike-data.json'])
    selected_catalogs.update(f'Data/coverage_expansion/{b}/catalog.json' for b in plan['coverage_branches'])
    for relative in sorted(selected_catalogs):
        add(relative, registry['source_sha256'].get(relative))
    recipe = plan.get('table_recipe')
    if recipe:
        for path, checksum in recipe['source_pins'].items():
            add(path, checksum, namespace='native_source')
    selected_ids = {t['key'] for t in plan['tables']}
    for owner in plan['coverage_output_adapter_closure']:
        add(f'kdd/coverage/adapters/{owner}.py')
        lock = physical_path(f'kdd/coverage/adapters/{owner}_sources.json', repo)
        if lock.exists():
            add(logical_relative(lock, repo))
            for asset in reproduce.read_json(lock).get('assets', []):
                add(asset['path'], asset['sha256'], asset.get('bytes'), 'native_source')
    for branch in plan['coverage_branches']:
        catalog = reproduce.read_json(repo / f'Data/coverage_expansion/{branch}/catalog.json')
        if recipe:
            for table in catalog['tables']:
                if f"{branch}/{table['id']}" in selected_ids:
                    for asset in table['source_assets']:
                        add(asset['path'], asset['sha256'], asset.get('bytes'), 'native_source')
            continue
        owner = adapter_name(catalog)
        declared = catalog.get('adapter_sha256', catalog.get('builder_sha256'))
        if isinstance(catalog.get('adapter'), dict):
            declared = catalog['adapter'].get('sha256', declared)
        if declared:
            add(f'kdd/coverage/adapters/{owner}.py', declared)
        additional_lock = physical_path(f'kdd/coverage/{owner}_sources.json', repo)
        if additional_lock.exists():
            add(logical_relative(additional_lock, repo))
        # Some adapters place assets on source records rather than every table.
        for item in catalog.get('sources', []) + catalog['tables']:
            for asset in item.get('source_assets', []) + item.get('files', []):
                if asset.get('path') and asset.get('sha256'):
                    add(asset['path'], asset['sha256'], asset.get('bytes'), 'native_source')
    if plan['expansion_sources']:
        for relative in ['kdd/integration/source_audit.json', 'kdd/integration/input_survey.md', 'manifest.json']:
            add(relative)
        for path in sorted(physical_path('kdd/integration/acquisition', repo).glob('*.json')):
            record = reproduce.read_json(path)
            if record['source_id'] in plan['expansion_sources'] and record.get('status') == 'verified':
                add(logical_relative(path, repo))
                add(record['path'], record['sha256'], record.get('bytes_observed'), 'native_source')
    if any(t['namespace'] == 'canonical' and t['key'] == 'yield/buchwald_hartwig_ood' for t in plan['tables']):
        relative = 'kdd/case_studies/ncs_yield/report.json'
        report = reproduce.read_json(repo / relative)
        add(relative)
        add('kdd/prepare_ncs_yield_splits.py')
        for field in ['output', 'success_targets']:
            add('kdd/case_studies/ncs_yield/' + report[field]['file'], report[field]['sha256'])
        from synrxn.reconstruction.families.outcomes.reaction_success._bh_hte_ood import SOURCE_NAME
        candidates = [physical_path(Path('Data/raw/yield') / SOURCE_NAME, repo),
                      repo / '.cache/ncs-primary-evidence/BH-HTE-OOD-official/data' / SOURCE_NAME]
        found = next((p for p in candidates if p.exists()), None)
        if found:
            add(found.relative_to(repo), report['source']['sha256'], namespace='native_source')
    return [pins[key] for key in sorted(pins)]


def _source_stamp(path):
    info = path.stat()
    return info.st_size, info.st_mtime_ns, info.st_ino


def rebuild_expansion(sources, output, repo=REPO):
    """Run the legacy expansion builder without its live-repository side effects."""
    output = Path(output).resolve()
    output.mkdir(parents=True)
    workspace = output / 'workspace'
    for name in ['synrxn', 'kdd/integration']:
        shutil.copytree(physical_path(name, repo), workspace / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # A native workspace deliberately retains its original output layout so
    # embedded provenance keys and byte comparisons stay identical.
    (workspace / 'Data/expansion').mkdir(parents=True)
    shutil.copyfile(repo / 'manifest.json', workspace / 'manifest.json')
    source_links = []
    source_stats = {}
    try:
        for path in sorted(physical_path('kdd/integration/acquisition', repo).glob('*.json')):
            item = reproduce.read_json(path)
            if item['source_id'] not in sources or item.get('status') != 'verified':
                continue
            source = reproduce.local_path(repo, item['path'])
            destination = workspace / item['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_stats[item['path']] = _source_stamp(source)
            # Copies, rather than writable hardlinks, protect native bytes even
            # if a historical source parser changes its input unexpectedly.
            shutil.copyfile(source, destination)
            source_links.append(destination)
        command = [sys.executable, '-m', 'synrxn.reconstruction', 'expansion',
                   '--root', 'Data/expansion', 'build', '--repo', str(workspace)]
        for source in sources:
            command.extend(['--source', source])
        env = {**os.environ, 'PYTHONPATH': str(workspace), 'PYTHONHASHSEED': '2027',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        with (output / 'build.log').open('w') as log:
            result = subprocess.run(command, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        frozen = reproduce.read_json(repo / 'Data/expansion/catalog.json')
        generated = reproduce.read_json(workspace / 'Data/expansion/catalog.json')
        expected = {t['key']: t for t in frozen['tables'] if t['source_id'] in sources}
        observed = {t['key']: t for t in generated['tables']}
        comparisons = []
        for key in sorted(set(expected) | set(observed)):
            a, b = expected.get(key), observed.get(key)
            actual = reproduce.sha(workspace / 'Data/expansion' / b['path']) if b else None
            comparisons.append({'key': key, 'status': 'verified' if a and b and a == b and actual == a['sha256'] else 'failed',
                                'expected_sha256': a['sha256'] if a else None, 'observed_sha256': actual,
                                'table_contract_identical': a == b})
        changed = [name for name, before in source_stats.items() if before !=
                   _source_stamp(reproduce.local_path(repo, name))]
        expected_issues = [issue for issue in frozen.get('issues', []) if issue['source_id'] in sources]
        issues = generated.get('issues', [])
        manifest = {'schema': 'synrxn.scope-expansion-rebuild.v1', 'sources': list(sources),
                    'command': command, 'returncode': result.returncode, 'comparisons': comparisons,
                    'changed_native_inputs': changed, 'issues': issues, 'expected_issues': expected_issues,
                    'catalog_comparison': 'Exact table contracts and output bytes; run timestamp/environment are recorded separately.',
                    'status': 'passed' if result.returncode == 0 and comparisons and not changed and
                    all(t['status'] == 'verified' for t in comparisons) and issues == expected_issues else 'failed'}
        reproduce.write_json(output / 'manifest.json', manifest)
        return manifest
    finally:
        for path in source_links:
            path.unlink(missing_ok=True)


def run_scope(plan, output, mode='verify', check_rows=False, repo=REPO, *, build_table=None):
    if mode not in {'verify', 'rebuild'}:
        raise ValueError('Unknown reproduction mode')
    repo, output = Path(repo).resolve(), Path(output).resolve()
    protected = [repo / p for p in ['script', 'kdd', 'ncs', 'synrxn', 'Data', 'archive/research', 'paper/kdd/literature'] ]
    cache_input = output.is_relative_to(repo / '.cache') and not output.is_relative_to(repo / '.cache/synrxn/runs')
    if output == repo or cache_input or any(output.is_relative_to(p) for p in protected) or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError('Use a new or empty report directory outside code/data/cache trees')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 'synrxn.scope-reproduction.v1', 'status': 'running', 'mode': mode,
                'started_at': datetime.now(timezone.utc).isoformat(), 'selection': plan['selection'],
                'counts': plan['counts'], 'predictive_models_run': 0, 'network_access': 'offline',
                'canonical_reproduction': plan['canonical_reproduction'], 'errors': [], 'steps': [],
                'regenerated_selected_tables': 0, 'regenerated_closure_tables': 0,
                'output_dir': str(output), 'row_recount_requested': check_rows,
                'selected_table_paths': [t['path'] for t in plan['tables']]}
    reproduce.write_json(output / 'plan.json', plan)
    try:
        records = dependency_records(plan, repo)
        checks = [reproduce.audit_file(r, repo) for r in records]
        input_checks = list(checks)
        reproduce.write_json(output / 'dependency_verification.json', checks)
        if any(r['status'] != 'verified' for r in checks):
            raise ValueError('Pinned dependencies failed; inspect dependency_verification.json')
        checks = [reproduce.audit_file(t, repo, check_rows) for t in plan['tables']]
        input_checks.extend(checks)
        reproduce.write_json(output / 'dataset_verification.json', checks)
        if any(r['status'] != 'verified' for r in checks):
            raise ValueError('Selected table verification failed; inspect dataset_verification.json')
        manifest['verified_selected_tables'] = len(checks)
        manifest['verified_only_canonical_tables'] = sum(t['namespace'] == 'canonical' for t in plan['tables'])
        # Preserve small code/catalog snapshots without copying native carriers.
        snapshots = reproduce.snapshot_physical_files(
            output, [record['path'] for record in records if record['namespace'] != 'native_source'], repo)
        reproduce.write_json(output / 'snapshot_manifest.json', snapshots)
        reproduce.write_json(output / 'environment.json', reproduce.environment(repo))
        if mode == 'rebuild' and build_table is not None:
            table, = plan['tables']
            filename = Path(reproduce.local_path(repo, table['path'])).name
            relative = Path('table') / filename
            (output / relative).parent.mkdir()
            build_table(repo, output / relative)
            comparison = reproduce.audit_file({**table, 'path': relative.as_posix()}, output, check_rows=True)
            reproduce.write_json(output / 'table_comparison.json', comparison)
            manifest['steps'].append({'id': 'table_recipe', 'status': comparison['status'], 'table': table['key']})
            if comparison['status'] != 'verified':
                raise ValueError('Table rebuild differs; inspect table_comparison.json')
        if mode == 'rebuild' and plan['expansion_sources']:
            result = rebuild_expansion(plan['expansion_sources'], output / 'expansion_rebuild', repo)
            manifest['steps'].append({'id': 'expansion_rebuild', 'status': result['status'], 'sources': plan['expansion_sources']})
            if result['status'] != 'passed':
                raise ValueError('Expansion rebuild differs; inspect expansion_rebuild/manifest.json')
        if mode == 'rebuild' and plan['coverage_adapters']:
            result = rebuild.rebuild(output / 'coverage_rebuild', repo, plan['coverage_adapters'])
            manifest['steps'].append({'id': 'coverage_rebuild', 'status': result['status'], 'adapters': plan['coverage_adapters']})
            if result['status'] != 'passed':
                raise ValueError('Coverage rebuild differs; inspect coverage_rebuild/manifest.json')
        identities = {(t['namespace'], t['key']) for t in plan['tables']}
        if mode == 'rebuild' and ('canonical', 'yield/buchwald_hartwig_ood') in identities:
            from synrxn.reconstruction.families.outcomes.reaction_success import _bh_hte_ood as ncs
            result = ncs.prepare(ncs.find_source(), output / 'ncs_yield')
            comparisons = reproduce.compare_artifacts(repo / 'kdd/case_studies/ncs_yield', output / 'ncs_yield',
                                                     ['native_splits.csv', 'success_targets.csv', 'report.json'])
            reproduce.write_json(output / 'ncs_yield/comparisons.json', comparisons)
            status = 'passed' if all(c['status'] == 'verified' for c in comparisons) else 'failed'
            manifest['steps'].append({'id': 'ncs_splits_and_scoring_targets', 'status': status,
                                      'canonical_table_regenerated': False})
            if status != 'passed':
                raise ValueError('NCS split/target companions differ')
        if mode == 'rebuild':
            regenerated = {('coverage', key) for key in plan['coverage_closure_table_keys']}
            regenerated.update(('expansion', key) for key in plan['expansion_closure_table_keys'])
            manifest['regenerated_selected_tables'] = len(identities & regenerated)
            manifest['regenerated_closure_tables'] = len(regenerated)
            manifest['extra_regenerated_table_keys'] = [namespace + ':' + key for namespace, key in sorted(regenerated - identities)]
        manifest['physical_post_run_changes'] = reproduce.physical_snapshot_changes(snapshots, repo)
        manifest['dependency_post_run_changes'] = list(manifest['physical_post_run_changes'])
        for record in records:
            if record['namespace'] != 'native_source' and reproduce.sha(reproduce.local_path(repo, record['path'])) != record['expected_sha256']:
                manifest['dependency_post_run_changes'].append(record['path'])
        for record in input_checks:
            stat = reproduce.local_path(repo, record['path']).stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (record['observed_bytes'], record['observed_mtime_ns'], record['observed_inode']):
                manifest['dependency_post_run_changes'].append(record['path'])
        manifest['dependency_post_run_changes'] = sorted(set(manifest['dependency_post_run_changes']))
        if manifest['dependency_post_run_changes']:
            raise ValueError('Pinned metadata/code changed during reproduction')
        manifest['status'] = 'passed'
    except (Exception, KeyboardInterrupt) as error:
        manifest['status'] = 'failed'
        manifest['errors'].append(type(error).__name__ + ': ' + str(error))
    manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
    manifest['reproduction_scope'] = ('Selected source artifacts verified; no observations regenerated.' if mode == 'verify'
        else 'Supported source adapters and task companions regenerated; canonical release tables remain verified frozen inputs.')
    reproduce.write_json(output / 'manifest.json', manifest)
    return manifest
