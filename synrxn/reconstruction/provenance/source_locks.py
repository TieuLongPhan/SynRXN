# Direct implementation; historical source identity is explicit below.
"""Generate and verify one source lock per SynRXN catalog dataset.

Every released row must be connectable to exactly one source lock recording the
pinned upstream file, version, checksum, license, and attribution.

The five task families released before 1.3.0 (``aam``, ``classification``,
``property``, ``rbl``, ``synthesis``) predate that requirement and shipped with
no locks at all. This tool retrofits a lock for each of them, filling in every
field that can be verified from repository evidence and marking the fields that
still require a human upstream audit as ``null`` rather than guessing them.

Locks that already exist are never rewritten; they are only verified.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('script/build_source_locks.py')
import argparse
from datetime import datetime, timezone
import json
import re
import sys
from synrxn.reconstruction.paths import HistoricalPath as Path
sys.path.insert(0, str(Path(__source_file__).resolve().parents[1]))
from synrxn.data.catalog import DatasetCatalog
ROOT = Path(__source_file__).resolve().parents[1]
SOURCES = ROOT / 'Data' / 'sources'
BIB = ROOT / 'doc' / 'refs.bib'
RETROFIT_TASKS = {'aam', 'classification', 'property', 'rbl', 'synthesis'}
UPSTREAM_FIELDS = ('source_url', 'source_file', 'source_commit_or_version', 'source_sha256', 'retrieved_at')
ARCHIVE_TYPES = {'misc', 'dataset'}

def source_pin_gaps(record: dict) -> list[str]:
    """Recognize single files and explicitly fingerprinted multi-file products."""
    if record.get('alias_of'):
        return ['canonical_alias_source_not_independently_pinned']
    checksums = record.get('source_files') or {}
    components = record.get('files') or []
    multi_hashes = isinstance(checksums, dict) and bool(checksums) and all((re.fullmatch('[a-f0-9]{64}', str(value)) for value in checksums.values()))
    component_hashes = bool(components) and all((isinstance(value, dict) and value.get('dataset_id') and re.fullmatch('[a-f0-9]{64}', str(value.get('sha256', ''))) for value in components))
    values = {'source_url': record.get('source_url') or record.get('source_repository'), 'source_file': record.get('source_file') or checksums or components, 'source_commit_or_version': record.get('source_commit_or_version') or record.get('source_commit') or record.get('source_version'), 'source_sha256': re.fullmatch('[a-f0-9]{64}', str(record.get('source_sha256', ''))) or multi_hashes or component_hashes, 'retrieved_at': record.get('retrieved_at')}
    return [field for field, value in values.items() if not value]

def parse_bibliography(path: Path) -> dict[str, dict[str, str]]:
    """Return ``{citation_key: {field: value}}`` including a ``_type`` field."""
    text = path.read_text(encoding='utf8') + '\n'
    entries: dict[str, dict[str, str]] = {}
    for match in re.finditer('@(\\w+)\\{([^,]+),(.*?)\\n\\}\\n', text, re.S):
        fields = {'_type': match.group(1).lower()}
        body = match.group(3) + '\n'
        for field in re.finditer('(\\w+)\\s*=\\s*\\{(.*?)\\}\\s*,?\\s*\\n', body, re.S):
            fields[field.group(1).lower()] = ' '.join(field.group(2).split())
        entries[match.group(2).strip()] = fields
    return entries

def index_existing_locks() -> dict[str, list[Path]]:
    """Keep every source-product lock, including declared train/test products."""
    index: dict[str, list[Path]] = {}
    for path in sorted(SOURCES.glob('*/*.json')):
        if path.name == 'build_report.json':
            continue
        try:
            record = json.loads(path.read_text(encoding='utf8'))
        except json.JSONDecodeError as exc:
            raise SystemExit(f'{path}: invalid JSON: {exc}') from exc
        key = record.get('dataset_key')
        if key:
            index.setdefault(key, []).append(path)
    for key, paths in index.items():
        if len(paths) > 1:
            records = [json.loads(path.read_text()) for path in paths]
            roles = [record.get('source_partition') for record in records]
            if not all(roles) or len(set(roles)) != len(roles):
                raise SystemExit(f'ambiguous source-product locks for {key}: {paths}')
    return index

def build_retrofit_lock(item, bib: dict[str, dict[str, str]]) -> dict[str, object]:
    """Build a lock containing only verifiable evidence, nulls elsewhere."""
    attribution: list[str] = []
    archive_dois: list[str] = []
    citations: list[dict[str, str]] = []
    for key in item.citations:
        entry = bib.get(key)
        if entry is None:
            raise SystemExit(f'{item.dataset_id.key}: citation {key!r} is not in refs.bib')
        reference = entry.get('doi') or entry.get('url') or ''
        citations.append({'citation_key': key, 'entry_type': entry['_type'], 'doi': entry.get('doi', ''), 'url': entry.get('url', ''), 'title': entry.get('title', '')})
        if reference:
            attribution.append(reference)
        if entry['_type'] in ARCHIVE_TYPES and entry.get('doi'):
            archive_dois.append(entry['doi'])
    lock: dict[str, object] = {'dataset_key': item.dataset_id.key, 'schema_version': '1.0', 'redistribution_partition': item.redistribution_partition, 'upstream_license': item.license, 'license_spdx': item.license_spdx, 'license_url': item.license_url or None, 'required_attribution': sorted(set(attribution)), 'citations': citations, 'source_doi': archive_dois[0] if len(archive_dois) == 1 else None, 'release_checksum_recorded_in': 'manifest.json'}
    for field in UPSTREAM_FIELDS:
        lock[field] = None
    outstanding = [f for f in UPSTREAM_FIELDS if lock[f] is None]
    if lock['source_doi'] is None:
        outstanding.append('source_doi')
    lock['lock_status'] = 'retrofit_pending_upstream_verification'
    lock['outstanding_fields'] = sorted(outstanding)
    lock['retrofit_note'] = 'Retrofitted for a task family released before source locks were required. License, partition, and attribution are verified against the catalog and bibliography. The listed outstanding fields must be filled by a human upstream audit before this dataset is cited as reproducibly pinned. Release-side size, checksum, and row count are in manifest.json.'
    return lock

def lock_partition_matches(record: dict, current_partition: str) -> bool:
    """Accept historical partitions when the current catalogue holds the data."""
    declared = record.get('redistribution_partition')
    return declared in {None, current_partition} or (
        current_partition == 'excluded'
        and record.get('lock_status') == 'pinned_published_synrxn_product'
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify lock coverage without writing files.')
    parser.add_argument('--report', type=Path, default=SOURCES / 'lock_coverage.json')
    args = parser.parse_args()
    catalog = DatasetCatalog()
    bib = parse_bibliography(BIB)
    existing = index_existing_locks()
    created, already, unlocked = ([], [], [])
    for item in catalog.list():
        key = item.dataset_id.key
        if key in existing:
            already.append(key)
            continue
        if item.task.value not in RETROFIT_TASKS:
            unlocked.append(key)
            continue
        lock = build_retrofit_lock(item, bib)
        target = SOURCES / item.task.value / f'{item.name}.json'
        if not args.check:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n', encoding='utf8')
        created.append(key)
    mismatched = []
    for key, paths in sorted(existing.items()):
        try:
            task, name = key.split('/', 1)
            expected = catalog.get(task, name).redistribution_partition
        except (KeyError, ValueError):
            continue
        for path in paths:
            record = json.loads(path.read_text(encoding='utf8'))
            declared = record.get('redistribution_partition')
            if not lock_partition_matches(record, expected):
                mismatched.append({'dataset_key': key, 'lock': str(path.relative_to(ROOT)), 'lock_partition': declared, 'catalog_partition': expected})
    pending = sorted((key for key, paths in existing.items() if any((str(json.loads(path.read_text()).get('lock_status', '')).startswith('retrofit') for path in paths))))
    records = {key: [json.loads(path.read_text()) for path in paths] for key, paths in existing.items()}
    pin_gaps = {key: sorted({gap for record in products for gap in source_pin_gaps(record)}) for key, products in records.items()}
    retrieval_pinned = sorted((key for key, gaps in pin_gaps.items() if not gaps))
    report = {'generated_at': datetime.now(timezone.utc).isoformat(), 'retrofit_pending_upstream_verification': pending, 'fully_pinned': sorted(set(retrieval_pinned) - set(pending)), 'retrieval_pinned': retrieval_pinned, 'retrieval_unpinned': sorted(set(existing) - set(retrieval_pinned)), 'retrieval_pin_gaps': {key: gaps for key, gaps in pin_gaps.items() if gaps}, 'pin_scope': 'Retrieval pins identify bytes. They do not independently certify ultimate source rights or reference validity.', 'catalog_datasets': len(catalog.list()), 'source_products': sum((len(paths) for paths in existing.values())), 'already_locked': sorted(already), 'retrofitted': sorted(created), 'still_unlocked': sorted(unlocked), 'partition_mismatches': mismatched, 'coverage': f'{len(already) + len(created)}/{len(catalog.list())}'}
    if not args.check:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k != 'already_locked'}, indent=2, sort_keys=True))
    if mismatched or unlocked:
        print('\nFAIL: incomplete lock coverage or partition disagreement.')
        return 1
    return 0
__configure_runtime(globals(), 'script/build_source_locks.py')
if __name__ == '__main__':
    raise SystemExit(main())
