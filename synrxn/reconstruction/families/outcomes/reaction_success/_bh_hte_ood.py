# Direct implementation; historical source identity is explicit below.
"""Recover NCS 2026 split metadata and native success scoring targets.

Offline by default. ``--fetch`` may download the exact, checksum-pinned author
CSV into Data/raw/yield. Split metadata and native binary scoring targets are
separate outputs; neither targets nor source/split metadata are model inputs.
Run from any directory with ``python kdd/prepare_ncs_yield_splits.py``.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/prepare_ncs_yield_splits.py')
import argparse
import ast
from collections import Counter
import csv
from decimal import Decimal
import gzip
import hashlib
import json
from synrxn.reconstruction.paths import HistoricalPath as Path
import urllib.request
ROOT = Path(__source_file__).resolve().parents[1]
SOURCE_COMMIT = 'a2770413fce269b49a35fb4b2e6471f25632d2ef'
SOURCE_NAME = 'BH_HTE_Curated_Dataset_v20260219.csv'
SOURCE_SHA256 = '247473787c33fb22ee48a4177593188f2a2c6ef140756566fe4610fee89ba333'
CANONICAL_SHA256 = '1cae5ff9010813e242f6de2dc6bd3e33342f759be79c5d88ac4f3f46dc4ec409'
SOURCE_URL = f'https://raw.githubusercontent.com/schwallergroup/bh-hte-ood/{SOURCE_COMMIT}/data/{SOURCE_NAME}'
CANONICAL_PATH = ROOT / 'Data/yield/buchwald_hartwig_ood.csv.gz'
SPLIT_COLUMNS = tuple((column for seed in range(10) for column in (f'iteration_{seed} cluster', f'iteration_{seed} train_test')))
OUTPUT_COLUMNS = ('sample_id', 'source_record_id', 'source_index', *SPLIT_COLUMNS)
SUCCESS_COLUMNS = ('sample_id', 'source_record_id', 'source_index', 'source', 'success')

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def verify_hash(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f'Checksum mismatch: {path}')

def find_source(explicit: Path | None=None, *, fetch: bool=False) -> Path:
    candidates = [explicit] if explicit is not None else [ROOT / 'Data/raw/yield' / SOURCE_NAME, ROOT / '.cache/ncs-primary-evidence/BH-HTE-OOD-official/data' / SOURCE_NAME]
    for candidate in candidates:
        if candidate.is_file():
            verify_hash(candidate, SOURCE_SHA256)
            return candidate
    if not fetch:
        raise FileNotFoundError('Pinned NCS author CSV is not cached. Supply --source or use --fetch.')
    destination = ROOT / 'Data/raw/yield' / SOURCE_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.download')
    try:
        request = urllib.request.Request(SOURCE_URL, headers={'User-Agent': 'SynRXN-reproducible-data-audit'})
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open('wb') as handle:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    handle.write(chunk)
        verify_hash(temporary, SOURCE_SHA256)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination

def load_native(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'', 'Source', 'Yield', 'labels', 'Aryl SMILES', 'Amine SMILES', 'Catalyst SMILES', 'Base SMILES', 'Solvent SMILES', 'Product SMILES', *SPLIT_COLUMNS}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f'Missing native columns: {sorted(missing)}')
        return list(reader)

def native_reaction(row: dict[str, str]) -> str:
    return '.'.join((row[column] for column in ('Aryl SMILES', 'Amine SMILES', 'Catalyst SMILES', 'Base SMILES', 'Solvent SMILES'))) + '>>' + row['Product SMILES']

def decode_assignments(value: str) -> tuple[str, ...]:
    try:
        labels = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError('Malformed native train/test assignment') from exc
    if not isinstance(labels, list) or len(labels) != 4 or labels.count('test') != 1 or (labels.count('train') != 3):
        raise ValueError('Native assignment must contain four cycles: one test, three train')
    return tuple(labels)

def join_native_splits(native_rows: list[dict[str, str]], canonical_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Check provenance identity before projecting unmodified split strings."""
    native_by_id: dict[str, tuple[int, dict[str, str]]] = {}
    for position, row in enumerate(native_rows):
        record_id = row['']
        if not record_id or record_id in native_by_id:
            raise ValueError('Missing or duplicate native source record ID')
        native_by_id[record_id] = (position, row)
    matched: set[str] = set()
    sample_ids: set[str] = set()
    output = []
    for row in canonical_rows:
        record_id = row['source_record_id']
        sample_id = row['sample_id']
        if not sample_id or sample_id in sample_ids or record_id in matched:
            raise ValueError('Missing or duplicate canonical sample/source record ID')
        if record_id not in native_by_id:
            raise ValueError(f'Canonical row has no native source record: {record_id}')
        position, original = native_by_id[record_id]
        if str(position) != row['source_index']:
            raise ValueError(f'Source index mismatch for {record_id}')
        if original['Source'] != row['source']:
            raise ValueError(f'Source label mismatch for {record_id}')
        if native_reaction(original) != row['raw_reaction_smiles']:
            raise ValueError(f'Reaction identity mismatch for {record_id}')
        if format(float(original['Yield']), '.12g') != row['yield_pct']:
            raise ValueError(f'Yield target mismatch for {record_id}')
        upstream_label = json.loads(row['source_metadata_json'])['upstream_labels']
        if original['labels'] != upstream_label:
            raise ValueError(f'Native success label mismatch for {record_id}')
        projected = {column: row[column] for column in OUTPUT_COLUMNS[:3]}
        for seed in range(10):
            cluster_column = f'iteration_{seed} cluster'
            cluster = original[cluster_column]
            if cluster not in {str(number) for number in range(12)}:
                raise ValueError(f'Invalid native cluster ID for {record_id}')
            split_column = f'iteration_{seed} train_test'
            decode_assignments(original[split_column])
            projected[cluster_column] = cluster
            projected[split_column] = original[split_column]
        output.append(projected)
        matched.add(record_id)
        sample_ids.add(sample_id)
    excluded = [row for row in native_rows if row[''] not in matched]
    return (output, excluded)

def seed_summary(rows: list[dict[str, str]], seed: int) -> dict:
    cluster_column = f'iteration_{seed} cluster'
    split_column = f'iteration_{seed} train_test'
    cluster_counts = Counter((row[cluster_column] for row in rows))
    assignment_counts = Counter((row[split_column] for row in rows))
    cluster_assignments: dict[str, tuple[str, ...]] = {}
    cycle_counts = [Counter() for _ in range(4)]
    for row in rows:
        cluster = row[cluster_column]
        assignment = decode_assignments(row[split_column])
        previous = cluster_assignments.setdefault(cluster, assignment)
        if previous != assignment:
            raise ValueError(f'Native cluster {cluster} crosses a fold in iteration {seed}')
        for cycle, label in enumerate(assignment):
            cycle_counts[cycle][label] += 1
    cycles = []
    for cycle, counts in enumerate(cycle_counts):
        held_out = sorted((int(cluster) for cluster, labels in cluster_assignments.items() if labels[cycle] == 'test'))
        if len(held_out) != 3:
            raise ValueError(f'Expected three held-out clusters in iteration {seed}, cycle {cycle}')
        cycles.append({'cycle': cycle, 'counts': dict(sorted(counts.items())), 'test_clusters': held_out})
    return {'iteration': seed, 'cluster_counts': dict(sorted(cluster_counts.items(), key=lambda pair: int(pair[0]))), 'native_assignment_string_counts': dict(sorted(assignment_counts.items())), 'cycles': cycles}

def success_target_rows(native_rows: list[dict[str, str]], verified_splits: list[dict[str, str]]) -> list[dict[str, str]]:
    """Project native labels after the chemistry/target/source identity join.

    ``verified_splits`` must come from :func:`join_native_splits`; this companion
    does not relabel observations from the yield, or create new observations.
    """
    native_by_id = {}
    for position, row in enumerate(native_rows):
        record_id = row['']
        if not record_id or record_id in native_by_id:
            raise ValueError('Missing or duplicate native source record ID')
        if row['labels'] not in {'0', '1'}:
            raise ValueError(f'Invalid native binary success label for {record_id}')
        native_by_id[record_id] = (position, row)
    output = []
    seen_samples, seen_records = (set(), set())
    for split_row in verified_splits:
        record_id, sample_id = (split_row['source_record_id'], split_row['sample_id'])
        if not sample_id or sample_id in seen_samples or record_id in seen_records:
            raise ValueError('Missing or duplicate success target sample/source ID')
        if record_id not in native_by_id:
            raise ValueError(f'Success target has no native source record: {record_id}')
        position, original = native_by_id[record_id]
        if split_row['source_index'] != str(position):
            raise ValueError(f'Success target source index mismatch for {record_id}')
        output.append({'sample_id': sample_id, 'source_record_id': record_id, 'source_index': split_row['source_index'], 'source': original['Source'], 'success': original['labels']})
        seen_samples.add(sample_id)
        seen_records.add(record_id)
    return output

def success_label_summary(native_rows: list[dict[str, str]]) -> dict:
    """Describe authored labels and audit, without imposing, yield thresholds."""
    counts_by_source: dict[str, Counter] = {}
    mismatches = {'yield_gt_10': Counter(), 'yield_ge_10': Counter()}
    for row in native_rows:
        label = row['labels']
        if label not in {'0', '1'}:
            raise ValueError(f"Invalid native binary success label for {row['']}")
        source, value = (row['Source'], Decimal(row['Yield']))
        counts_by_source.setdefault(source, Counter())[label] += 1
        for name, derived in (('yield_gt_10', value > 10), ('yield_ge_10', value >= 10)):
            if int(label) != int(derived):
                mismatches[name][source] += 1
    return {'rows': len(native_rows), 'native_label_counts': dict(sorted(Counter((row['labels'] for row in native_rows)).items())), 'native_label_counts_by_source': {source: dict(sorted(counts.items())) for source, counts in sorted(counts_by_source.items())}, 'threshold_comparison': {name: {'disagreement_rows': sum(by_source.values()), 'disagreement_rows_by_source': dict(sorted(by_source.items()))} for name, by_source in mismatches.items()}, 'label_policy': 'Preserve native labels. Yield-threshold comparisons are diagnostics, not corrections or independently qualified replacement endpoints.'}

def prepare(source_path: Path, output_directory: Path) -> dict:
    verify_hash(source_path, SOURCE_SHA256)
    verify_hash(CANONICAL_PATH, CANONICAL_SHA256)
    native_rows = load_native(source_path)
    with gzip.open(CANONICAL_PATH, 'rt', encoding='utf-8', newline='') as handle:
        canonical_rows = list(csv.DictReader(handle))
    output, excluded = join_native_splits(native_rows, canonical_rows)
    if len(native_rows) != 27497 or len(output) != 27438 or len(excluded) != 59:
        raise ValueError('Frozen NCS row-count contract changed')
    if any((Decimal(row['Yield']) <= 100 for row in excluded)):
        raise ValueError('Excluded native row is not an above-100 yield record')
    summaries = [seed_summary(output, seed) for seed in range(10)]
    success_rows = success_target_rows(native_rows, output)
    matched_ids = {row['source_record_id'] for row in output}
    canonical_native_rows = [row for row in native_rows if row[''] in matched_ids]
    output_directory.mkdir(parents=True, exist_ok=True)
    split_path = output_directory / 'native_splits.csv'
    with split_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(output)
    success_path = output_directory / 'success_targets.csv'
    with success_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=SUCCESS_COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerows(success_rows)
    report = {'schema_version': '1.1', 'status': 'passed', 'dataset_key': 'yield/buchwald_hartwig_ood', 'display_alias': 'NCS 2026 BH-HTE-OOD', 'independent_new_dataset': False, 'source': {'url': SOURCE_URL, 'commit': SOURCE_COMMIT, 'sha256': SOURCE_SHA256, 'rows': len(native_rows), 'source_cohort_counts': dict(sorted(Counter((row['Source'] for row in native_rows)).items())), 'paper': 'https://doi.org/10.1038/s43588-026-01017-6', 'data_record': 'https://doi.org/10.5281/zenodo.19636649', 'data_license': 'CC-BY-4.0', 'code_license': 'Apache-2.0'}, 'canonical': {'path': 'Data/yield/buchwald_hartwig_ood.csv.gz', 'sha256': CANONICAL_SHA256, 'rows': len(canonical_rows), 'source_cohort_counts': dict(sorted(Counter((row['source'] for row in canonical_rows)).items())), 'upstream_success_label_counts': dict(sorted(Counter((json.loads(row['source_metadata_json'])['upstream_labels'] for row in canonical_rows)).items()))}, 'excluded_from_canonical': {'rows': len(excluded), 'reason': 'All 59 native observations have yield >100%; existing canonical importer requires 0<=yield<=100. No extra exclusions were applied here.', 'source_record_ids': [row[''] for row in excluded], 'source_cohort_counts': dict(sorted(Counter((row['Source'] for row in excluded)).items())), 'minimum_yield_pct': str(min((Decimal(row['Yield']) for row in excluded))), 'maximum_yield_pct': str(max((Decimal(row['Yield']) for row in excluded)))}, 'join_checks': {'one_native_record_per_canonical_sample': True, 'all_canonical_rows_assigned': True, 'checks': ['unique sample_id and source_record_id', 'source_record_id matches native unnamed index column', 'source_index matches native zero-based record position', 'source cohort exact match', 'raw reaction exact match to native reactant/reagent/product strings', 'yield target matches original value at canonical 12-significant-digit precision', 'upstream success label exact match', 'ten native cluster and train/test columns copied verbatim', 'one held-out cycle per sample per seed and three held-out clusters per cycle']}, 'evaluation_scope': {'split_origin': 'Author-supplied full-reaction DRFP k-means cluster assignments, k=12, ten initializations; four cycles per initialization, nine train clusters and three test clusters.', 'author_primary_endpoint': 'Source-defined binary reaction success from the author-supplied labels. The paper describes a 10% yield success threshold, but the released labels do not equal one uniform >10 or >=10 rule; retain the original labels and report the source-specific discrepancies.', 'continuous_regression': "The same native cluster assignments can support a separately specified yield-regression experiment. This is not a reproduction of the paper's primary classification endpoint.", 'strict_substrate_novelty': "The author's per-source benchmark additionally excludes test reactions containing a training aryl or amine. This metadata recovery does not apply or claim that additional filter.", 'metadata_only': True, 'metadata_only_scope': 'native_splits.csv contains split metadata only; success_targets.csv is a separate scoring-target companion and is not an input feature table.', 'model_input_warning': 'Use cluster and train_test columns only to select splits. Never use outcomes, labels, held-out assignments, source outcome descriptors or split metadata as model features.', 'native_assignment_encoding': 'Each train_test cell is the original literal list of four cycle labels; strings are preserved verbatim.'}, 'iterations': summaries, 'success_label_audit': {'raw': success_label_summary(native_rows), 'canonical': success_label_summary(canonical_native_rows)}, 'success_targets': {'file': 'success_targets.csv', 'rows': len(success_rows), 'columns': list(SUCCESS_COLUMNS), 'sha256': sha256(success_path), 'target_column': 'success', 'target_source_column': 'labels', 'independent_new_dataset': False, 'native_labels_preserved': True, 'model_input_policy': 'Scoring targets and identifiers/source metadata only; exclude success, source, source_record_id, source_index and sample_id from chemical model features.'}, 'output': {'file': 'native_splits.csv', 'rows': len(output), 'columns': list(OUTPUT_COLUMNS), 'sha256': sha256(split_path)}, 'reproduction': {'command': 'python kdd/prepare_ncs_yield_splits.py', 'network_default': 'offline', 'fetch_command': 'python kdd/prepare_ncs_yield_splits.py --fetch'}}
    (output_directory / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Already-cached pinned author CSV')
    parser.add_argument('--fetch', action='store_true', help='Allow fetching the checksum-pinned source if absent')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'kdd/case_studies/ncs_yield')
    args = parser.parse_args()
    report = prepare(find_source(args.source, fetch=args.fetch), args.output_dir)
    print(json.dumps({'status': report['status'], 'matched_rows': report['output']['rows'], 'excluded_source_rows': report['excluded_from_canonical']['rows']}, sort_keys=True))
__configure_runtime(globals(), 'kdd/prepare_ncs_yield_splits.py')
if __name__ == '__main__':
    main()
