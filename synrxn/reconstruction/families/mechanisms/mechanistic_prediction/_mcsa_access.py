# Direct implementation; historical source identity is explicit below.
"""Prepare and verify separate M-CSA Parquet/query/API access without models."""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('script/prepare_mcsa_access.py')
import os
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import json
from synrxn.reconstruction.paths import HistoricalPath as Path
import sys
import time
import zipfile
ROOT = Path(__source_file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError('Choose a fresh derivative output directory')
    pointer = json.loads((ROOT / 'ncs/audit/mcsa-extension.json').read_text())
    artifact = ROOT / pointer['artifact']
    data = artifact / 'Data'
    manifest = artifact / 'manifest.json'
    from synrxn.data import DatasetCatalog
    from synrxn.access.query import convert_release, validate_parquet_release, QueryEngine
    from synrxn.access.service import create_app
    from fastapi.testclient import TestClient
    import duckdb
    import pyarrow
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
    catalog = DatasetCatalog(data / 'metadata.yaml')
    started = time.monotonic()
    artifacts = convert_release(data, output / 'parquet', catalog, manifest, partition='main')
    errors = validate_parquet_release(data, output / 'parquet', catalog)
    if errors:
        raise ValueError(errors)
    engine = QueryEngine(output / 'parquet', catalog)
    selected = {}
    for item in artifacts:
        name = item.dataset.split('/')[-1].removesuffix('.csv.gz')
        frame = engine.query('mechanism', name, columns=['sample_id', 'reaction_smiles', 'verification_level', 'initial_participant_context'], filters={'verification_level': 'EXACT'}, limit=2)
        if len(frame) != 2:
            raise ValueError('Bounded primary replay selection failed')
        selected[name] = frame.to_dict(orient='records')
    engine.close()
    with duckdb.connect() as connection:
        connection.execute('SET threads=2')
        query = 'SELECT verification_level, initial_participant_context, count(*) AS records FROM read_parquet(?) GROUP BY ALL ORDER BY ALL'
        result = connection.execute(query, [str(output / 'parquet/mechanism/mcsa_steps.parquet')])
        contingency = [dict(zip([column[0] for column in result.description], row)) for row in result.fetchall()]
    app = create_app(output / 'parquet', manifest, max_page_size=2, max_concurrent_queries=1, catalog=catalog)
    with TestClient(app) as client:
        health = client.get('/health').json()
        listing = client.get('/v1/datasets').json()
        if health['status'] != 'ready' or listing['count'] != 4:
            raise ValueError('Separate M-CSA API readiness failed')
        rows = client.get('/v1/datasets/mechanism/mcsa_steps/rows', params={'columns': 'sample_id,reaction_smiles,verification_level', 'limit': 2})
        if rows.status_code != 200 or len(rows.json()['rows']) != 2:
            raise ValueError('Bounded M-CSA API rows failed')
        over = client.get('/v1/datasets/mechanism/mcsa_steps/rows?limit=3')
        if over.status_code != 422:
            raise ValueError('M-CSA API did not reject oversized request')
    proof = {'schema': 'synrxn.mcsa-access.v1', 'status': 'pass', 'cpu_limit': 2, 'benchmark_execution': False, 'source_artifact': pointer['artifact'], 'source_manifest_sha256': sha(manifest), 'parquet_index_sha256': sha(output / 'parquet/index.json'), 'tables': len(artifacts), 'rows': sum((item.rows for item in artifacts)), 'full_typed_csv_parquet_comparison': 'pass', 'query_examples': selected, 'context_tier_query': query, 'context_tier_counts': contingency, 'api': {'health': health, 'catalog_count': listing['count'], 'bounded_rows': 2, 'over_limit_status': over.status_code}, 'elapsed_seconds': round(time.monotonic() - started, 3), 'scope': 'Separate four-view enzyme collection. Main 47-table and ShareAlike two-table APIs remain separate. No independence or predictor performance claim.'}
    (output / 'access-report.json').write_text(json.dumps(proof, indent=2) + '\n')
    archive = ROOT / 'releases/mcsa/SynRXN-MCSA-v5-parquet.zip'
    members = {p.relative_to(output).as_posix(): p for p in output.rglob('*') if p.is_file()}
    members['Data/metadata.yaml'] = data / 'metadata.yaml'
    members['Data/LICENSE'] = data / 'LICENSE'
    members['manifest.json'] = manifest
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for name, path in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 15, 0, 0, 0))
            info.external_attr = 33188 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, path.read_bytes())
    with zipfile.ZipFile(archive) as bundle:
        for name, path in members.items():
            with bundle.open(name) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != sha(path):
                    raise ValueError('Derivative ZIP member mismatch: ' + name)
    proof['archive'] = archive.relative_to(ROOT).as_posix()
    proof['archive_sha256'] = sha(archive)
    proof['archive_bytes'] = archive.stat().st_size
    proof['every_archived_member_verified'] = True
    (ROOT / 'ncs/audit/mcsa-access.json').write_text(json.dumps(proof, indent=2) + '\n')
    report = {'archive': proof['archive'], 'sha256': proof['archive_sha256'], 'bytes': proof['archive_bytes'], 'every_archived_member_verified': True, 'source_manifest_sha256': sha(manifest), 'cpu_limit': 2}
    (ROOT / 'releases/mcsa/derived-package-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: proof[key] for key in ['status', 'tables', 'rows', 'api', 'archive', 'archive_bytes']}, indent=2))
__configure_runtime(globals(), 'script/prepare_mcsa_access.py')
if __name__ == '__main__':
    main()
