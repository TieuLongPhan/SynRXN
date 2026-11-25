#!/usr/bin/env python
"""Label partial AAM references and certify endpoint-preserving SynKit completions.

The source pool and its benchmark are read-only. Labelled partial references and
computed two-sided completions are separate outputs, with the original string
and every retained import-QC reason recorded in the attempt ledger.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import signal

from synrxn.reconstruction.paths import REPO as ROOT, repo_path
from synrxn.reconstruction.families.representation.atom_mapping._sources.audit_aam_overlap import bounded_map
from synrxn.reconstruction.families.representation.atom_mapping._sources.audit_aam_partial_canon import collect, csv_rows, diagnostic_rows, quiet
from synrxn.reconstruction.families.representation.atom_mapping._sources.audit_metamdb_aam import csv_writer
from synrxn.reconstruction.families.representation.atom_mapping._sources.build_unique_aam import CORE
from synrxn.reconstruction.families.representation.atom_mapping._sources import aam_mapping_constraints as constraints
from synrxn.reconstruction.families.representation.atom_mapping._sources import aam_overlap_chem as chem
from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import digest, write_json

SCHEMA = 'synrxn.aam-completion.v1'
ATTEMPTS = ['r_id', 'dataset', 'source_r_id', 'sources', 'rxn', 'source_review_reasons', 'original_ground_truth',
            'labelled_ground_truth', 'new_heavy_labels', 'heavy_inventory_equal', 'outcome',
            'completed_ground_truth', 'completion_evidence', 'checked_source_references', 'error']
LABELLED = CORE + ['source_review_reasons', 'mapping_status', 'new_heavy_labels']
COMPLETED = CORE + ['source_review_reasons', 'checked_source_references']
OPTIONS = dict(preserve_older_map=True, require_constitution_preservation=True,
               fold_unmapped_explicit_hydrogens=True, ignore_stereochemistry=True,
               explicit_hydrogen=True, preserve_radical_state=True, constitutional_only=True)


class CompletionTimeout(BaseException):
    """Escape library handlers that wrap ordinary exceptions during reconstruction."""


def label_heavy_atoms(value):
    """Fill zero heavy-atom IDs uniquely; never infer a new correspondence."""
    from rdkit import Chem
    mols = chem.parse_reaction(value)
    atoms = [a for m in mols for a in m.GetAtoms()]
    next_label = max((a.GetAtomMapNum() for a in atoms), default=0) + 1
    added = 0
    for atom in atoms:
        if atom.GetAtomicNum() != 1 and not atom.GetAtomMapNum():
            atom.SetAtomMapNum(next_label)
            next_label += 1
            added += 1
    labelled = '>>'.join(Chem.MolToSmiles(m, isomericSmiles=True) for m in mols) if added else value
    if chem.reference_info(labelled)['target_rxn'] != chem.reference_info(value)['target_rxn']:
        raise ValueError('Filling labels changed a reaction endpoint')
    return labelled, added


def expansion_input(value):
    """Retain paired IDs; one-sided IDs provide no specified partner to preserve."""
    from rdkit import Chem
    mols = chem.parse_reaction(value)
    counts = [Counter(a.GetAtomMapNum() for a in m.GetAtoms() if a.GetAtomMapNum()) for m in mols]
    if any(count > 1 for side in counts for count in side.values()):
        raise ValueError('Repeated mapping classes must not be collapsed by the SynKit adapter')
    paired = {n for n in counts[0] if counts[0][n] == counts[1][n]}
    if not paired:
        raise ValueError('No specified correspondences anchor this completion')
    for mol in mols:
        Chem.RemoveStereochemistry(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() not in paired:
                atom.SetAtomMapNum(0)
    return '>>'.join(Chem.MolToSmiles(m, isomericSmiles=True) for m in mols)


def validate_completion(value, rxn, references):
    """Require a closed heavy map, unchanged endpoints and all source constraints."""
    info = chem.reference_info(value)
    if info['target_rxn'] != rxn or info['mapping_status'] != 'ok':
        return False
    candidate = chem.its_graph(info['mols'])
    return all(constraints.compare(candidate, chem.its_graph(chem.reference_info(ref)['mols']), refine=True)
               for ref in references)


def complete(task):
    from synkit.Graph.ITS.its_expand import ITSExpand
    row, references, timeout = task
    labelled, added = label_heavy_atoms(row['ground_truth'])
    result = dict(r_id=row['r_id'], dataset=row['dataset'], source_r_id=row['source_r_id'],
                  sources=row.get('sources', chem.encode([row['dataset']])),
                  rxn=row['rxn'], source_review_reasons=row['review_reasons'],
                  original_ground_truth=row['ground_truth'], labelled_ground_truth=labelled,
                  new_heavy_labels=added, heavy_inventory_equal=row['heavy_inventory_equal'],
                  outcome='', completed_ground_truth='', completion_evidence='[]',
                  checked_source_references=len(references), error='')
    if not row['heavy_inventory_equal']:
        result['outcome'] = 'unbalanced_heavy_inventory'
        return result

    def expired(*_):
        raise CompletionTimeout('Partial-map completion exceeded time limit')

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    candidates, evidence, errors = {}, [], []
    classes = chem.ITSClasses()
    try:
        for source in sorted(set(references)):
            try:
                working = expansion_input(source)
            except ValueError as error:
                errors.append(str(error))
                continue
            # Evaluate both directions: fallback alone could conceal disagreement.
            for use_reactants in (True, False):
                try:
                    report = ITSExpand.expand_aam_with_its_report(working, use_G=use_reactants, **OPTIONS)
                    value = report.rsmi
                    if not validate_completion(value, row['rxn'], references):
                        errors.append('Candidate failed endpoint, completeness or source-constraint checks')
                        continue
                    class_id = classes.assign(chem.its_graph(chem.reference_info(value)['mols']))
                    candidates.setdefault(class_id, value)
                    evidence.append(dict(source_ground_truth=source, working_rsmi=working, **report.to_dict()))
                except TimeoutError:
                    raise
                except Exception as error:
                    errors.append(f'{type(error).__name__}: {error}'[:400])
        if len(candidates) == 1:
            result.update(outcome='completed', completed_ground_truth=next(iter(candidates.values())))
        else:
            result['outcome'] = 'ambiguous_completion' if candidates else 'no_certified_completion'
    except CompletionTimeout as error:
        result['outcome'] = 'timeout'
        errors.append(str(error))
    except Exception as error:
        result['outcome'] = 'timeout' if isinstance(error, TimeoutError) else 'failed'
        errors.append(f'{type(error).__name__}: {error}'[:400])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    result['completion_evidence'] = chem.encode(evidence)
    result['error'] = chem.encode(sorted(set(errors)))
    return result


def tasks(pool, timeout):
    cases, _ = collect(pool)
    review = {r['r_id']: r for r in diagnostic_rows(pool)}
    references = {r['r_id']: [] for r in cases}
    import gzip
    with gzip.open(pool / 'provenance.jsonl.gz', 'rt') as handle:
        for line in handle:
            record = json.loads(line)
            if record['r_id'] in references:
                references[record['r_id']].append(record['source_record']['ground_truth'])
    for row in cases:
        group = review[row['r_id']]
        row.update(rxn=group['rxn'], review_reasons=group.get('quality_flags', group.get('review_reasons', '[]')),
                   sources=review[row['r_id']]['sources'])
        yield row, references[row['r_id']], timeout


def record(result, completed=False):
    return dict(r_id=result['r_id'],
                ground_truth=result['completed_ground_truth' if completed else 'labelled_ground_truth'],
                rxn=result['rxn'], original_id=result['dataset'] + ':' + result['source_r_id'],
                reference_id=result['r_id'] + (':computed_completion' if completed else ':labelled_partial'),
                representative_source=result['dataset'], representative_r_id=result['source_r_id'],
                sources=result['sources'], source_records=result['checked_source_references'],
                mapping_scope='heavy_atoms' if completed else 'partial_heavy_atoms',
                reference_tier='computational_completion' if completed else 'source_reference_labels_filled',
                validation_status='generated_reference_import_qc_retained' if completed else 'partial_reference',
                split_group_id=chem.identities(result['rxn'])['undirected_key'],
                source_review_reasons=result['source_review_reasons'])


def implementation_pins():
    import synkit
    local = [Path(__file__).resolve(), ROOT / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/audit_aam_partial_canon.py',
             ROOT / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/audit_aam_overlap.py', ROOT / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/build_unique_aam.py',
             ROOT / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/audit_metamdb_aam.py',
             ROOT / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/aam_overlap_chem.py',
             ROOT / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/aam_mapping_constraints.py',
             ROOT / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/aam_sources.py']
    package = Path(inspect.getfile(synkit)).parent
    return ({str(p.relative_to(ROOT)): digest(p) for p in local},
            {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob('*.py'))})


def checked_manifest(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, pin in manifest['files'].items():
        if digest(folder / name) != pin:
            raise ValueError(f'Checksum mismatch: {folder / name}')
    return digest(folder / 'manifest.json')


def verify(pool, output):
    checked_manifest(output)
    summary = json.loads((output / 'summary.json').read_text())
    if summary['schema'] != SCHEMA or checked_manifest(pool) != summary['pool_manifest_sha256']:
        raise ValueError('Completion source pool changed')
    local, package = implementation_pins()
    if local != summary['implementation_sha256'] or package != summary['synkit_source_sha256']:
        raise ValueError('Completion implementation changed')
    if summary['versions'] != {n: version(n) for n in summary['versions']}:
        raise ValueError('Completion dependencies changed')
    expected = {row['r_id']: (row, refs) for row, refs, _ in tasks(pool, summary['timeout_seconds'])}
    outcomes, seen, completed, labelled = Counter(), set(), {}, {}
    for row in csv_rows(output / 'attempts.csv.gz'):
        if row['r_id'] in seen or row['r_id'] not in expected:
            raise ValueError('Repeated or unexpected completion input')
        source, references = expected[row['r_id']]
        value, added = label_heavy_atoms(source['ground_truth'])
        if (row['original_ground_truth'] != source['ground_truth'] or row['labelled_ground_truth'] != value
                or int(row['new_heavy_labels']) != added or row['rxn'] != source['rxn']
                or row['source_review_reasons'] != source['review_reasons']):
            raise ValueError('Completion lost source identity, labels or QC annotations')
        labelled[row['r_id']] = value
        if row['outcome'] == 'completed':
            if not validate_completion(row['completed_ground_truth'], row['rxn'], references):
                raise ValueError('Completed reference failed verification')
            completed[row['r_id']] = row['completed_ground_truth']
        outcomes[row['outcome']] += 1
        seen.add(row['r_id'])
    for name, values in [('labelled.csv.gz', labelled), ('completed.csv.gz', completed)]:
        rows = list(csv_rows(output / name))
        if len(rows) != len(values) or {r['r_id']: r['ground_truth'] for r in rows} != values:
            raise ValueError(f'Completion table does not match ledger: {name}')
    if seen != set(expected) or dict(outcomes) != summary['outcomes']:
        raise ValueError('Completion coverage or summary mismatch')
    return dict(verified=True, partial_inputs=len(seen), completed=len(completed), outcomes=dict(outcomes))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, default=repo_path('Data/tables/development/aam_unique'))
    parser.add_argument('--output', type=Path, default=repo_path('Data/tables/development/aam_completion'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--timeout', type=float, default=5)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    quiet()
    if args.verify:
        print(json.dumps(verify(args.pool, args.output), indent=2))
        return 0
    if min(args.workers, args.batch_size, args.timeout) <= 0:
        parser.error('Workers, batch and timeout must be positive')
    pool_pin = checked_manifest(args.pool)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Use a fresh completion output directory')
    outcomes, label_cases, labels_added = Counter(), 0, 0
    with ExitStack() as stack:
        attempts = csv_writer(stack, args.output / 'attempts.csv.gz', ATTEMPTS)
        labelled = csv_writer(stack, args.output / 'labelled.csv.gz', LABELLED)
        completed = csv_writer(stack, args.output / 'completed.csv.gz', COMPLETED)
        executor = (stack.enter_context(ProcessPoolExecutor(max_workers=args.workers, initializer=quiet))
                    if args.workers > 1 else None)
        results = bounded_map(executor, complete, tasks(args.pool, args.timeout), args.batch_size)
        for index, result in enumerate(results, 1):
            attempts.writerow(result)
            info = chem.reference_info(result['labelled_ground_truth'])
            labelled.writerow(dict(record(result), mapping_status=info['mapping_status'],
                                   new_heavy_labels=result['new_heavy_labels']))
            if result['outcome'] == 'completed':
                completed.writerow(dict(record(result, completed=True),
                                        checked_source_references=result['checked_source_references']))
            outcomes[result['outcome']] += 1
            label_cases += result['new_heavy_labels'] > 0
            labels_added += result['new_heavy_labels']
            if index % 500 == 0:
                print(f'Processed {index:,} partial inputs', flush=True)
    if checked_manifest(args.pool) != pool_pin:
        raise ValueError('Source pool changed during completion')
    local, package = implementation_pins()
    summary = dict(schema=SCHEMA, partial_inputs=sum(outcomes.values()), outcomes=dict(outcomes),
                   inputs_with_labels_filled=label_cases, new_heavy_labels=labels_added,
                   pool_manifest_sha256=pool_pin, source_pool_changed=False, benchmark_changed=False,
                   scope='primary partial_mapping review inputs; no new atoms or endpoint changes',
                   method='SynKit ITSExpand, both directions, all source constraints checked',
                   documentation=('https://synkit.readthedocs.io/en/latest/api/graph.html'
                                  '#synkit.Graph.ITS.its_expand.ITSExpand'),
                   completion_options=OPTIONS, timeout_seconds=args.timeout,
                   qualification='Generated correspondences are computational references, '
                                 'not human-curated ground truth. '
                                 'Original import QC reasons remain attached; unbalanced references remain partial.',
                   implementation_sha256=local, synkit_source_sha256=package,
                   versions={n: version(n) for n in ['synkit', 'rdkit', 'networkx']})
    write_json(args.output / 'summary.json', summary)
    write_json(args.output / 'manifest.json', {'schema': SCHEMA, 'files': {
        p.name: digest(p) for p in sorted(args.output.iterdir()) if p.name != 'manifest.json'}})
    reported = ['partial_inputs', 'outcomes', 'inputs_with_labels_filled', 'new_heavy_labels']
    print(json.dumps({k: summary[k] for k in reported}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
