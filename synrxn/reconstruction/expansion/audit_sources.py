# Direct implementation; historical source identity is explicit below.
"""Record primary release metadata for the 26 SynRXN expansion candidates."""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('script/audit_expansion_sources.py')
import concurrent.futures
import hashlib
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from synrxn.reconstruction.paths import HistoricalPath as Path
ROOT = Path(__source_file__).resolve().parents[1]
OUT = ROOT / 'kdd/integration'
CACHE = ROOT / 'Data/raw/synrxn2/metadata'
ZENODO = {'C01': 19636649, 'C02': 20773622, 'C07': 11493786, 'C09': 7341155, 'C10': 8423911, 'C11': 20060606, 'C13': 17310823, 'C15': 14775076, 'C21': 8396100, 'C24': 11067998, 'C14': 11494913}
FIGSHARE = {'C04': 23298467, 'C08': 19614657, 'C12': 31668682, 'C17': 22598230}
GITHUB = {'C03': 'open-reaction-database/ord-data', 'C06': 'jpfolch/catechol_solvent_selection', 'C14': 'WillHua127/ReactZyme', 'C16': 'DeepSynthesis/orgaicat', 'C19': 'wangxr0526/Parrot', 'C22': 'ReactionMechanismGenerator/RMG-database', 'C23': 'wangxr0526/EasIFA'}

def request(url):
    CACHE.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(url.encode()).hexdigest()
    path = CACHE / (name + '.json')
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get('status') == 200:
            return cached
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'SynRXN-source-audit/2.0'})
        with urllib.request.urlopen(req, timeout=35) as response:
            raw = response.read()
            body = json.loads(raw) if 'json' in response.headers.get('Content-Type', '') else raw.decode(errors='replace')
            result = {'url': url, 'status': response.status, 'body': body, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    except urllib.error.HTTPError as error:
        result = {'url': url, 'status': error.code, 'error': error.read().decode(errors='replace')[:1000]}
    except Exception as error:
        result = {'url': url, 'status': None, 'error': str(error)}
    result['retrieved_at'] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result

def inspect(row):
    key = row['id']
    result = {**row, 'license_verified': False, 'files': [], 'requests': []}
    if key in ZENODO:
        response = request(f'https://zenodo.org/api/records/{ZENODO[key]}')
        result['requests'].append({k: v for k, v in response.items() if k != 'body'})
        if response['status'] == 200:
            data = response['body']
            result.update(record_id=str(ZENODO[key]), doi=data.get('doi'), title=data.get('metadata', {}).get('title'), license=data.get('metadata', {}).get('license'), publication_date=data.get('metadata', {}).get('publication_date'), description=data.get('metadata', {}).get('description'), files=[{'name': f['key'], 'bytes': f.get('size'), 'checksum': f.get('checksum'), 'url': f.get('links', {}).get('self')} for f in data.get('files', [])])
            result['license_verified'] = bool(result.get('license'))
    elif key in FIGSHARE:
        response = request(f'https://api.figshare.com/v2/articles/{FIGSHARE[key]}')
        result['requests'].append({k: v for k, v in response.items() if k != 'body'})
        if response['status'] == 200:
            data = response['body']
            result.update(record_id=str(FIGSHARE[key]), doi=data.get('doi'), title=data.get('title'), version=data.get('version'), license=data.get('license'), publication_date=data.get('published_date'), description=data.get('description'), files=[{'name': f['name'], 'bytes': f.get('size'), 'checksum': 'md5:' + f.get('computed_md5', ''), 'url': f.get('download_url')} for f in data.get('files', [])])
            result['license_verified'] = bool(result.get('license'))
    if key in GITHUB:
        repo = GITHUB[key]
        response = request(f'https://api.github.com/repos/{repo}')
        result['requests'].append({k: v for k, v in response.items() if k != 'body'})
        if response['status'] == 200:
            branch = response['body'].get('default_branch', 'main')
            revision = request(f'https://api.github.com/repos/{repo}/commits/{branch}')
            result['requests'].append({k: v for k, v in revision.items() if k != 'body'})
            commit = revision.get('body', {}).get('sha', branch)
            tree = request(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1')
            result['requests'].append({k: v for k, v in tree.items() if k != 'body'})
            if tree['status'] == 200:
                result['github'] = {'repository': repo, 'commit': commit, 'tree_sha': tree['body']['sha'], 'license': response['body'].get('license'), 'tree': tree['body'].get('tree', []), 'truncated': tree['body'].get('truncated')}
    if key == 'C25':
        response = request('https://huggingface.co/api/datasets/completeRXN-benchmark-26/completeRXN')
        result['requests'].append({k: v for k, v in response.items() if k != 'body'})
        if response['status'] == 200:
            result['huggingface'] = response['body']
    (OUT / 'sources').mkdir(parents=True, exist_ok=True)
    (OUT / 'sources' / (key + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result


def main(argv=None):
    """Refresh the stored expansion source metadata."""
    import argparse
    argparse.ArgumentParser(description='Refresh primary source metadata and its local audit cache.').parse_args(argv)
    rows = json.loads((OUT / 'candidates.json').read_text())['candidates']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(inspect, rows))
    (OUT / 'source_audit.json').write_text(json.dumps({'schema': 'synrxn.source-audit.v1', 'sources': results}, indent=2, ensure_ascii=False) + '\n')
    for r in results:
        print(r['id'], r['name'], 'license', r.get('license'), 'files', len(r['files']), flush=True)
        for f in r['files']:
            print(' ', f['name'], f.get('bytes'))
        for f in r.get('github', {}).get('tree', []):
            p = f['path']
            if f.get('type') == 'blob' and (re.search('\\.(csv|tsv|xlsx?|json|h5|hdf5|zip|gz|pkl|pck|parquet|sdf|jsonl)$', p, re.I) or re.search('licen[sc]e|readme', p, re.I)):
                print(' ', p, f.get('size'))


__configure_runtime(globals(), 'script/audit_expansion_sources.py')

if __name__ == "__main__":
    main()
