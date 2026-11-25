"""Generate or recover a table and place it in the family data tree."""
from __future__ import annotations

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from synrxn.reconstruction.catalogue.layout import place_table
from synrxn.reconstruction.families._tables import run_scope
from synrxn.reconstruction.families.suite import physical_sha, read_json
from synrxn.reconstruction.paths import physical_path


def _acquire_expansion(sources, repo, cache, offline):
    """Restore missing expansion inputs using the existing acquisition receipts."""
    fetched = False
    receipts = physical_path('kdd/integration/acquisition', repo)
    for path in sorted(receipts.glob('*.json')):
        receipt = read_json(path)
        if receipt['source_id'] not in sources or receipt.get('status') != 'verified':
            continue
        target = physical_path(receipt['path'], repo)
        cached = cache / Path(receipt['path']).relative_to('Data')
        if target.is_file():
            continue
        if not cached.is_file() and offline:
            raise ValueError(f'Missing offline input: {cached}')
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.synrxn-source-', dir=target.parent) as temporary:
            staged = Path(temporary) / target.name
            if cached.is_file():
                shutil.copyfile(cached, staged)
            else:
                request = Request(receipt['url'], headers={'User-Agent': 'SynRXN-data/2.0'})
                try:
                    with urlopen(request, timeout=90) as response, staged.open('wb') as output:
                        shutil.copyfileobj(response, output)
                except OSError as error:
                    raise ValueError(f"Download failed for {receipt['path']}. Supply --cache DIR: {error}") from error
                fetched = True
            if physical_sha(staged) != receipt['sha256']:
                raise ValueError(f"Recovered input differs from the source pin: {receipt['path']}")
            staged.replace(target)
    return 'upstream and local inputs' if fetched else 'cached native inputs'


def prepare(plan, repo, destination, cache, offline, from_cache, build_table, acquire):
    """Prepare one admitted table using its recipe or its pinned cached artifact."""
    table, = plan['tables']
    binding, = plan['bindings']
    if not binding.get('release_eligible', True):
        raise ValueError(f"Table is held by the release manifest: {table['key']}")
    repo = Path(repo).resolve()
    cache = Path(cache).resolve() if cache else repo / 'Data'
    prepared_table = physical_path(table['path'], repo)
    run_root = repo / '.cache/synrxn/runs'
    run_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='prepare-', dir=run_root) as temporary:
        work = Path(temporary)
        if from_cache or (table['namespace'] == 'canonical' and build_table is None):
            artifact = prepared_table
            if cache != repo / 'Data':
                logical = Path(table['path']).relative_to('Data')
                previous = logical if table['namespace'] == 'canonical' else Path(*logical.parts[1:])
                previous = Path('tables', table['namespace'], previous)
                candidates = [cache / prepared_table.relative_to(repo / 'Data'), cache / previous,
                              cache / str(previous).replace('_snapshots.jsonl.gz', '_spectra.jsonl.gz')]
                artifact = next((path for path in candidates if path.is_file()), candidates[0])
            if not artifact.is_file():
                raise ValueError(f'Missing cached release table: {artifact}')
            source = 'cached release table'
        elif build_table is not None:
            source = acquire(repo, cache, offline)
            artifact = work / Path(table['path']).name
            build_table(repo, artifact)
        else:
            if cache != repo / 'Data' and table['namespace'] == 'coverage':
                raise ValueError('This shared recipe uses the native archive. Use --from-cache with an alternate table cache.')
            source = _acquire_expansion(plan['expansion_sources'], repo, cache, offline)
            result = run_scope(plan, work / 'rebuild', mode='rebuild', repo=repo)
            if result['status'] != 'passed':
                details = '; '.join(result['errors'])
                raise ValueError(f'{details}. Use --from-cache when the pinned released table is available.')
            workspace = f"{table['namespace']}_rebuild"
            artifact = work / 'rebuild' / workspace / 'workspace' / table['path']
        path = place_table(artifact, binding['family_id'], binding['task_id'],
                           table['namespace'], table['key'], repo, destination)
    return {'status': 'passed', 'table': table['key'], 'path': str(path), 'source': source}
