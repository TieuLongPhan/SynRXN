#!/usr/bin/env python
"""Trace import QC and test CanonRSMI on the current partial-AAM review inputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import csv
import gzip
from importlib.metadata import version
import inspect
import json
import logging
from pathlib import Path
import signal

from synrxn.reconstruction.paths import REPO as ROOT, repo_path
from synrxn.reconstruction.families.representation.atom_mapping._sources.audit_aam_overlap import bounded_map
from synrxn.reconstruction.families.representation.atom_mapping._sources.audit_metamdb_aam import csv_writer
from synrxn.reconstruction.families.representation.atom_mapping._sources import aam_overlap_chem as chem
from synrxn.reconstruction.families.representation.atom_mapping._sources import aam_mapping_constraints as constraints
from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import digest, write_json

PATTERNS = ('missing_labels_both', 'missing_labels_reactants', 'missing_labels_products',
            'unpaired_labels_reactants', 'unpaired_labels_products', 'unpaired_labels_both')
CASE_FIELDS = ['r_id', 'dataset', 'source_r_id', 'pattern', 'unmapped_reactant_atoms', 'unmapped_product_atoms',
               'reactant_only_labels', 'product_only_labels', 'reactant_heavy_atoms', 'product_heavy_atoms',
               'heavy_inventory_equal', 'ground_truth']
TRIAL_FIELDS = CASE_FIELDS + [
    'outcome', 'working_rsmi', 'canonical_rsmi', 'after_mapping_status', 'endpoints_preserved',
    'after_unmapped_heavy_atoms', 'known_constraints_preserved', 'error',
]


def csv_rows(path):
    with gzip.open(path, 'rt', newline='') as handle:
        yield from csv.DictReader(handle)


def diagnostic_rows(pool):
    """Read retained v5 annotations or a historical v4 review partition."""
    current = pool / 'diagnostics.csv.gz'
    return csv_rows(current if current.exists() else pool / 'review.csv.gz')


def case_order(row):
    return PATTERNS.index(row['pattern']), row['dataset'], row['source_r_id']


def shape(value):
    mols = chem.parse_reaction(value)
    atoms = [[a for a in m.GetAtoms() if a.GetAtomicNum() != 1] for m in mols]
    labels = [[a.GetAtomMapNum() for a in side] for side in atoms]
    zeros = [side.count(0) for side in labels]
    counts = [Counter(n for n in side if n) for side in labels]
    extra = [counts[0] - counts[1], counts[1] - counts[0]]
    if all(zeros):
        pattern = PATTERNS[0]
    elif zeros[0]:
        pattern = PATTERNS[1]
    elif zeros[1]:
        pattern = PATTERNS[2]
    elif extra[0] and not extra[1]:
        pattern = PATTERNS[3]
    elif extra[1] and not extra[0]:
        pattern = PATTERNS[4]
    else:
        pattern = PATTERNS[5]
    inventories = [Counter((a.GetAtomicNum(), a.GetIsotope()) for a in side) for side in atoms]
    return dict(pattern=pattern, unmapped_reactant_atoms=zeros[0], unmapped_product_atoms=zeros[1],
                reactant_only_labels=chem.encode(dict(extra[0])), product_only_labels=chem.encode(dict(extra[1])),
                reactant_heavy_atoms=len(atoms[0]), product_heavy_atoms=len(atoms[1]),
                heavy_inventory_equal=inventories[0] == inventories[1])


def quiet():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    logging.getLogger('synkit').setLevel(logging.CRITICAL)
    logging.disable(logging.CRITICAL)


def trial(task):
    from rdkit import Chem
    from synkit.Chem.Reaction.canon_rsmi import CanonRSMI
    row, timeout = task
    result = dict(row, outcome='', working_rsmi='', canonical_rsmi='', after_mapping_status='',
                  endpoints_preserved=False, after_unmapped_heavy_atoms='', known_constraints_preserved='', error='')

    def expired(*_):
        raise TimeoutError('CanonRSMI trial exceeded time limit')

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        before = chem.reference_info(row['ground_truth'])
        for mol in before['mols']:
            Chem.RemoveStereochemistry(mol)
        working = '>>'.join(Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) for m in before['mols'])
        result['working_rsmi'] = working
        target = CanonRSMI(backend='wl', wl_iterations=3).canonicalise(working).canonical_rsmi
        result['canonical_rsmi'] = target or ''
        after = chem.reference_info(target)
        preserved = before['target_rxn'] == after['target_rxn']
        result.update(after_mapping_status=after['mapping_status'], endpoints_preserved=preserved,
                      after_unmapped_heavy_atoms=after['unmapped_heavy_atoms'])
        if not preserved:
            result['outcome'] = 'endpoint_changed'
        elif after['mapping_status'] == 'partial':
            result['outcome'] = 'still_partial'
        elif after['mapping_status'] in {'ok', 'equivalent'}:
            valid = constraints.compare(chem.its_graph(after['mols']), chem.its_graph(before['mols']), refine=True)
            result.update(outcome='complete_constraints_preserved' if valid else 'complete_constraints_changed',
                          known_constraints_preserved=valid)
        else:
            result['outcome'] = 'invalid_mapping'
    except Exception as error:
        result.update(outcome='timeout' if isinstance(error, TimeoutError) else 'invalid_or_failed',
                      error=f'{type(error).__name__}: {error}'[:250])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    return result


def collect(pool):
    review = {r['r_id']: r for r in diagnostic_rows(pool)}
    selected, qc_keys = {}, {}
    for row in csv_rows(pool / 'members.csv.gz'):
        group = review.get(row['r_id'], {})
        key = row['dataset'], row['source_r_id']
        if group.get('status') == 'partial_mapping' and row['mapping_status'] == 'partial':
            selected[key] = row['r_id']
        if group.get('status') in {'source_qc_excluded', 'import_qc_flag'} and row['source_exclusion']:
            qc_keys[key] = row['r_id']
    cases, qc = {}, {}
    with gzip.open(pool / 'provenance.jsonl.gz', 'rt') as handle:
        for line in handle:
            record = json.loads(line)
            key = record['dataset'], record['source_r_id']
            source = record['source_record']
            if key in selected:
                row = dict(r_id=selected[key], dataset=key[0], source_r_id=key[1],
                           ground_truth=source['ground_truth'], **shape(source['ground_truth']))
                if row['r_id'] not in cases or case_order(row) < case_order(cases[row['r_id']]):
                    cases[row['r_id']] = row
            if key in qc_keys:
                group = qc.setdefault(qc_keys[key], {'flags': set(), 'statuses': set(), 'datasets': set(),
                                                     'records': 0, 'heavy_balanced': True})
                group['flags'].update(json.loads(source['exclusion_reasons']))
                group['statuses'].add(source['validation_status'])
                group['datasets'].add(record['dataset'])
                group['records'] += 1
                group['heavy_balanced'] &= source['heavy_element_balanced'] == 'true'
    assert len(cases) == sum(r['status'] == 'partial_mapping' for r in review.values())
    assert len(qc) == sum(r['status'] in {'source_qc_excluded', 'import_qc_flag'} for r in review.values())
    return [cases[k] for k in sorted(cases)], qc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, default=repo_path('Data/tables/development/aam_unique'))
    parser.add_argument('--output', type=Path, default=repo_path('Data/tables/development/aam_review_diagnostics'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--timeout', type=float, default=5)
    args = parser.parse_args(argv)
    if min(args.workers, args.batch_size, args.timeout) <= 0:
        parser.error('Workers, batch and timeout must be positive')
    manifest = json.loads((args.pool / 'manifest.json').read_text())
    for name, pin in manifest['files'].items():
        if digest(args.pool / name) != pin:
            raise ValueError(f'Pool checksum mismatch: {name}')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Use a fresh diagnostics directory')
    quiet()
    cases, qc = collect(args.pool)
    outcomes, patterns, by_pattern = Counter(), Counter(), defaultdict(Counter)
    with ExitStack() as stack:
        case_writer = csv_writer(stack, args.output / 'partial_cases.csv.gz', CASE_FIELDS)
        case_writer.writerows(cases)
        trial_writer = csv_writer(stack, args.output / 'canon_trials.csv.gz', TRIAL_FIELDS)
        executor = (stack.enter_context(ProcessPoolExecutor(max_workers=args.workers, initializer=quiet))
                    if args.workers > 1 else None)
        tasks = ((r, args.timeout) for r in cases)
        for i, result in enumerate(bounded_map(executor, trial, tasks, args.batch_size), 1):
            trial_writer.writerow(result)
            outcomes[result['outcome']] += 1
            patterns[result['pattern']] += 1
            by_pattern[result['pattern']][result['outcome']] += 1
            if i % 500 == 0:
                print(f'Tested {i:,}/{len(cases):,} partial inputs', flush=True)
        qc_writer = csv_writer(stack, args.output / 'import_qc.csv.gz',
                               ['r_id', 'datasets', 'import_flags', 'import_statuses', 'source_records',
                                'heavy_element_balanced'])
        qc_writer.writerows({'r_id': key, 'datasets': chem.encode(sorted(value['datasets'])),
                             'import_flags': chem.encode(sorted(value['flags'])),
                             'import_statuses': chem.encode(sorted(value['statuses'])),
                             'source_records': value['records'], 'heavy_element_balanced': value['heavy_balanced']}
                            for key, value in sorted(qc.items()))
    from synkit.Chem.Reaction.canon_rsmi import CanonRSMI
    summary = {'schema': 'synrxn.aam-review-diagnostics.v1', 'partial_inputs': len(cases),
               'partial_patterns': dict(patterns), 'canon_outcomes': dict(outcomes),
               'canon_outcomes_by_pattern': {k: dict(v) for k, v in by_pattern.items()},
               'import_qc_inputs': len(qc), 'qc_origin': 'local SIHUMIx importer, not author rejection labels',
               'qc_failed_inputs': sum('qc_failed' in v['statuses'] for v in qc.values()),
               'transport_only_inputs': sum(v['flags'] == {'transport_reaction'} for v in qc.values()),
               'all_qc_source_records_heavy_balanced': all(v['heavy_balanced'] for v in qc.values()),
               'qc_reason_sets': [{'flags': list(k), 'inputs': count} for k, count in
                                  Counter(tuple(sorted(v['flags'])) for v in qc.values()).items()],
               'pool_manifest_sha256': digest(args.pool / 'manifest.json'),
               'versions': {n: version(n) for n in ['synkit', 'rdkit', 'networkx']},
               'canon_source_sha256': digest(Path(inspect.getsourcefile(CanonRSMI))),
               'stereo_ignored_in_working_copy': True,
               'timeout_seconds': args.timeout, 'dataset_changed': False,
               'implementation_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [
                   Path(__file__).resolve(), ROOT / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/sihumix.py',
                   ROOT / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/sihumix_quality.py',
                   ROOT / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/aam_overlap_chem.py',
                   ROOT / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/aam_mapping_constraints.py']}}
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'manifest.json', {'files': {
        p.name: digest(p) for p in sorted(args.output.iterdir()) if p.name != 'manifest.json'}})
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
