# Direct implementation; historical source identity is explicit below.
"""Download checksum-pinned expansion source data without executing source code."""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('script/fetch_expansion_data.py')
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
import threading
from synrxn.reconstruction.paths import HistoricalPath as Path
ROOT = Path(__source_file__).resolve().parents[1]
RAW = ROOT / 'Data/raw/synrxn2'
REPORT = ROOT / 'kdd/integration/acquisition.json'
SELECTED = {'C02': None, 'C07': ['Datasets.tar.bz2', 'Splits.tar.bz2', 'Molecules.tar.bz2', 'README.md'], 'C08': ['Transition1x.h5'], 'C09': ['ref_routes_n1.json', 'ref_routes_n5.json', 'targets_n1.txt', 'targets_n5.txt', 'stock_n1.txt', 'stock_n5.txt', 'ref_routes_reaction_hashes.txt'], 'C10': ['RxnSolvKSE_dataset_v1.1.zip', 'README.md'], 'C11': ['quantumpioneer_kinetics_dataset.csv', 'quantumpioneer_thermo_dataset_transition_states.csv', 'SCHEMA.md'], 'C12': None, 'C13': ['Data.zip', 'bib.zip'], 'C14': ['cleaned_uniprot_rhea.tsv', 'rhea_molecules.tsv', 'enzyme_smi_split.zip', 'reaction_smi_split.zip', 'time_split.zip'], 'C15': ['catpred-db.tar.gz'], 'C17': ['choriso_public.tar.gz', 'splits_choriso.tar.gz'], 'C21': None, 'C24': ['EnzChemRED.tar.gz', 'README.txt']}
GIT_FILES = {'C06': ['data/full_data/catechol_full_data_yields.csv', 'data/single_solvent/catechol_single_solvent_yields.csv', 'data/full_data/claisen_data_clean.csv', 'data/featurization_look_ups/smiles_lookup.csv', 'kaggle_data/catechol-benchmark-metadata.json', 'LICENSE', 'README.md'], 'C16': ['data/aldol_reaction-dataset.xlsx', 'data/mannich_reaction_Denmark.xlsx', 'data/michael_addition-dataset.xlsx', 'data/validation_data/validation_in_house_acetone.xlsx', 'data/validation_data/validation_in_house_hydroxyacetone.xlsx', 'data/validation_data/validation_new_literature_aldol.xlsx', 'data/validation_data/validation_new_literature_michael.xlsx', 'data/validation_data/validation_virtual_chemist.xlsx', 'LICENSE', 'README.md', 'data/README.md']}

def ranged_download(asset, target, workers=6):
    """Resume independent HTTP ranges and verify their lengths before assembly.

    Some public file gateways close long streams early without a Content-Length.
    Short, explicit ranges avoid silently treating such streams as full files.
    Chunks are only reusable at the same URL, size and upstream checksum.
    """
    size = asset['bytes']
    chunk_size = 1024 * 1024
    identity = hashlib.sha256(json.dumps(asset, sort_keys=True).encode()).hexdigest()[:16]
    directory = target.parent / '.chunks' / identity
    directory.mkdir(parents=True, exist_ok=True)
    url = asset['url']
    match = re.match('https://zenodo.org/api/records/(\\d+)/files/(.+)/content$', url)
    if match:
        url = f'https://zenodo.org/records/{match[1]}/files/{match[2]}?download=1'

    def chunk(start):
        end = min(size, start + chunk_size) - 1
        path = directory / f'{start:012d}.bin'
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        last_error = None
        for attempt in range(3):
            try:
                request_url = url
                if 'figshare.com' in url:
                    request_url += ('&' if '?' in url else '?') + f'synrxn={start}-{time.time_ns()}'
                req = urllib.request.Request(request_url, headers={'User-Agent': 'SynRXN-data-integration/2.0', 'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(req, timeout=45) as response:
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                        raise ValueError('server did not honor the requested byte range')
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise ValueError(f'truncated range: {len(data)}/{end - start + 1}')
                path.write_bytes(data)
                return path
            except Exception as error:
                last_error = error
                time.sleep(min(attempt + 1, 3))
        raise RuntimeError(f'range {start}-{end}: {last_error}')
    starts = list(range(0, size, chunk_size))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        paths = list(pool.map(chunk, starts))
    temporary = target.with_name(target.name + '.part')
    with temporary.open('wb') as output:
        for path in paths:
            with path.open('rb') as source:
                while (block := source.read(chunk_size)):
                    output.write(block)
    temporary.replace(target)

def download(source_id, asset):
    target = RAW / source_id / asset['name']
    if target.is_absolute() and (not target.resolve().is_relative_to((RAW / source_id).resolve())):
        raise ValueError('unsafe source path')
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = asset.get('checksum')
    expected_size = asset.get('bytes')
    result = {'source_id': source_id, **asset, 'path': target.relative_to(ROOT).as_posix()}
    try:
        needs_download = not target.exists() or (expected_size is not None and target.stat().st_size != expected_size)
        if needs_download and expected_size and (expected_size > 2 * 1024 * 1024):
            print(f"RANGE {source_id} {asset['name']} expected={expected_size}", flush=True)
            ranged_download(asset, target)
        elif needs_download:
            temporary = target.with_name(target.name + '.part')
            url = asset['url']
            if 'figshare.com' in url:
                url += ('&' if '?' in url else '?') + 'synrxn=' + str(time.time_ns())
            headers = {'User-Agent': 'SynRXN-data-integration/2.0'}
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(url, headers=headers)
            print(f"START {source_id} {asset['name']} expected={expected_size}", flush=True)
            with urllib.request.urlopen(request, timeout=90) as response:
                resumed = offset and response.status == 206 and response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')
                output = temporary.open('ab' if resumed else 'wb')
                n = offset if resumed else 0
                last = 0
                with output:
                    while (chunk := response.read(1024 * 1024)):
                        output.write(chunk)
                        n += len(chunk)
                        if n - last >= 256 * 1024 * 1024:
                            print(f"PROGRESS {source_id} {asset['name']} {n}", flush=True)
                            last = n
            temporary.replace(target)
        sha, md5 = (hashlib.sha256(), hashlib.md5())
        with target.open('rb') as handle:
            while (chunk := handle.read(4 * 1024 * 1024)):
                sha.update(chunk)
                md5.update(chunk)
        result.update(bytes_observed=target.stat().st_size, sha256=sha.hexdigest(), md5=md5.hexdigest())
        if expected_size is not None and result['bytes_observed'] != expected_size:
            raise ValueError('source byte size differs from the pinned release')
        if expected:
            algorithm, value = expected.split(':', 1)
            actual = result.get(algorithm)
            if actual is None or actual.lower() != value.lower():
                raise ValueError('source checksum differs from the pinned release')
        with target.open('rb') as handle:
            is_pointer = handle.read(128).startswith(b'version https://git-lfs')
        if is_pointer:
            raise ValueError('download is an LFS pointer rather than data')
        result['status'] = 'verified'
        print(f"OK {source_id} {asset['name']} {result['bytes_observed']}", flush=True)
    except Exception as error:
        result.update(status='unavailable', error=str(error))
        print(f"ERROR {source_id} {asset['name']} {error}", flush=True)
    report_dir = ROOT / 'kdd/integration/acquisition'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_name = source_id + '-' + hashlib.sha256(asset['name'].encode()).hexdigest()[:12] + '.json'
    (report_dir / report_name).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result

def assets():
    sources = json.loads((ROOT / 'kdd/integration/source_audit.json').read_text())['sources']
    jobs = []
    for source in sources:
        key = source['id']
        if key in SELECTED:
            selected = SELECTED[key]
            for asset in source['files']:
                if selected is None or asset['name'] in selected:
                    jobs.append((key, asset))
        if key in GIT_FILES:
            git = source['github']
            tree = {item['path']: item for item in git['tree']}
            for path in GIT_FILES[key]:
                if path in tree:
                    jobs.append((key, {'name': path, 'bytes': tree[path].get('size'), 'url': f"https://raw.githubusercontent.com/{git['repository']}/{git['commit']}/{path}"}))
        if key == 'C03':
            git = source['github']
            path = 'data/80/ord_dataset-805ad863feef48579d95d86a728035f4.parquet'
            pointer = (RAW / 'notes/C03' / path).read_text()
            checksum = re.search('oid (sha256:\\w+)', pointer).group(1)
            size = int(re.search('size (\\d+)', pointer).group(1))
            jobs.append((key, {'name': 'cernak_cn.parquet', 'bytes': size, 'checksum': checksum, 'url': f"https://media.githubusercontent.com/media/{git['repository']}/{git['commit']}/{path}"}))
        if key == 'C20':
            jobs.append((key, {'name': 'oc20neb_dft_trajectories_04_23_24.tar.gz', 'bytes': 1464594652, 'url': 'https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20neb/oc20neb_dft_trajectories_04_23_24.tar.gz'}))
    return jobs


def main(argv=None):
    """Fetch the explicitly selected expansion assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', help='Restrict acquisition to a candidate ID')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-bytes', type=int, help='Skip individual optional assets larger than this many bytes')
    args = parser.parse_args(argv)
    jobs = [(source, asset) for source, asset in assets() if not args.source or source in args.source]
    if args.max_bytes:
        jobs = [(source, asset) for source, asset in jobs if not asset.get('bytes') or asset['bytes'] <= args.max_bytes]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda job: download(*job), jobs))
    previous = json.loads(REPORT.read_text()).get('assets', []) if REPORT.exists() else []
    merged = {(r['source_id'], r['name']): r for r in previous + results}
    REPORT.write_text(json.dumps({'schema': 'synrxn.acquisition.v1', 'assets': list(merged.values())}, indent=2, ensure_ascii=False) + '\n')
    print('COMPLETE', sum((r['status'] == 'verified' for r in results)), '/', len(results), flush=True)


__configure_runtime(globals(), 'script/fetch_expansion_data.py')

if __name__ == "__main__":
    main()
