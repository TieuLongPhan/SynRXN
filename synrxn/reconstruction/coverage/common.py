# Direct implementation; historical source identity is explicit below.
"""Small provenance and target-isolation helpers for new data acquisitions."""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/coverage/common.py')
from collections import Counter
import hashlib
import json
from synrxn.reconstruction.paths import HistoricalPath as Path
import pyarrow as pa
import pyarrow.parquet as pq
REPO = Path(__source_file__).resolve().parents[2]

def sha(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def encoded(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

def identity(value) -> str:
    return hashlib.sha256(encoded(value).encode()).hexdigest()

def split_group(group: str) -> str:
    bucket = int(identity([2027, group])[:8], 16) % 100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'

def asset(path: Path, url: str, *, repo: Path=REPO, expected_bytes=None, checksum=None) -> dict:
    path = path.resolve()
    if not path.is_relative_to(repo.resolve()):
        raise ValueError('Asset outside repository')
    size = path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise ValueError('Truncated or changed source asset: ' + str(path))
    if checksum:
        algorithm, expected = checksum.split(':', 1)
        with path.open('rb') as handle:
            actual = hashlib.file_digest(handle, algorithm).hexdigest()
        if actual.lower() != expected.lower():
            raise ValueError('Upstream checksum mismatch: ' + str(path))
    return {'path': path.relative_to(repo).as_posix(), 'url': url, 'bytes': size, 'sha256': sha(path), 'upstream_checksum': checksum, 'expected_bytes': expected_bytes, 'verification': 'verified'}

def record(source: str, table: str, native_id: str, group: str, *, inputs: dict, targets: dict, provenance: dict, context: dict | None=None, native_split: str | None=None) -> dict:
    if not targets:
        raise ValueError('A task row needs explicit reference targets')
    return {'sample_id': source + ':' + table + ':' + identity(native_id), 'source_id': source, 'source_record_id': str(native_id), 'parent_group': str(group), 'split': split_group(str(group)), 'native_split': native_split, 'inputs_json': encoded(inputs), 'targets_json': encoded(targets), 'context_json': encoded(context or {}), 'provenance_json': encoded(provenance)}

def emit(output: Path, name: str, rows: list[dict], *, source: dict, tasks: dict, unit: str, qualification: str, evidence: str, repo: Path=REPO, role: str='primary') -> dict:
    if not rows:
        raise ValueError('Do not emit an empty source-backed task table: ' + name)
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Repeated source row identities in ' + name)
    for row in rows:
        provenance = json.loads(row['provenance_json'])
        if not provenance.get('source_sha256') or not provenance.get('source_locator'):
            raise ValueError('Missing row-level provenance')
    output.mkdir(parents=True, exist_ok=True)
    path = output / (name + '.parquet')
    pq.write_table(pa.Table.from_pylist(rows), path, compression='zstd')
    columns = pq.ParquetFile(path).schema_arrow.names
    for contract in tasks.values():
        if set(contract['inputs']) & set(contract['targets']):
            raise ValueError('Target appears in input whitelist')
        if (set(contract['inputs']) | set(contract['targets']) | set(contract['group_fields'])) - set(columns):
            raise ValueError('Contract refers to missing storage columns')
    return {'id': source['id'] + '/' + name, 'path': path.relative_to(repo).as_posix(), 'format': 'parquet', 'source_id': source['id'], 'source_url': source['url'], 'source_sha256': source['assets'][0]['sha256'], 'license': source['license'], 'rights_status': source.get('rights_status', 'source_declared'), 'rows': len(rows), 'bytes': path.stat().st_size, 'sha256': sha(path), 'columns': columns, 'counting_unit': unit, 'parent_groups': len({row['parent_group'] for row in rows}), 'split_counts': dict(Counter((row['split'] for row in rows))), 'task_ids': sorted(tasks), 'task_fields': tasks, 'role': role, 'reference_evidence': evidence, 'qualification': qualification, 'source_assets': source['assets'], 'storage_layout': 'json_objects'}

def contract(inputs: list[str], targets: list[str], metrics: list[str]) -> dict:
    return {'inputs': ['inputs_json'], 'targets': ['targets_json'], 'input_keys': inputs, 'target_keys': targets, 'group_fields': ['parent_group'], 'metrics': metrics, 'split_policy': 'SHA-256 parent-group 80/10/10 seed 2027; native split retained separately'}
__configure_runtime(globals(), 'kdd/coverage/common.py')
