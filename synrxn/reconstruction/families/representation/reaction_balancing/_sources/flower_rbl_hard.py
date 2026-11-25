"""Restore native unchanged FlowER mixture context for harder balancing cases."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import io
from functools import lru_cache
from itertools import islice
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import tempfile

from rdkit import Chem, RDLogger

from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import digest, require, write_json
from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_center_groups import source_lines
from synrxn.reconstruction.families.representation.reaction_balancing._sources.flower_rbl import FIELDS as BASE_FIELDS, components, side_signature, side_text, signature

SCHEMA = 'synrxn.flower-rbl-hard.v1'
EXTRA_FIELDS = ('base_r_id', 'variant', 'has_context', 'context_count', 'input_context_count',
                'context_rxn', 'context_aam', 'input_context_rxn', 'context_hydrogens',
                'context_missing_from_products')
FIELDS = (*BASE_FIELDS, *EXTRA_FIELDS)


def mapped_fragments(side):
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    molecule = Chem.MolFromSmiles(side, parameters)
    require(molecule is not None, 'Invalid source endpoint')
    return list(Chem.GetMolFrags(molecule, asMols=True))


def maps(fragment):
    return {atom.GetAtomMapNum() for atom in fragment.GetAtoms()}


def mapped_component_counts(side):
    return Counter(Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
                   for fragment in mapped_fragments(side))


def mapped_pool(side):
    atoms = [atom for fragment in mapped_fragments(side) for atom in fragment.GetAtoms()]
    numbers = [atom.GetAtomMapNum() for atom in atoms]
    require(all(numbers) and len(numbers) == len(set(numbers)), 'Missing or duplicate expanded target map IDs')
    return {atom.GetAtomMapNum(): atom.GetAtomicNum() for atom in atoms}


def recover_context(task):
    """Recover only unchanged source components outside the retained parent atom pool."""
    parent, initial, final = task
    parent_sides = parent['ground_truth'].split('>>')
    retained = {number for side in parent_sides for fragment in mapped_fragments(side) for number in maps(fragment)}
    common = mapped_component_counts(initial) & mapped_component_counts(final)
    selected = []
    for text in sorted(common.elements()):
        fragment = mapped_fragments(text)[0]
        if not (maps(fragment) & retained):
            require(0 not in maps(fragment), 'Unmapped context atom')
            selected.append(text)
    context_aam = '.'.join(selected)
    records = components(context_aam) if context_aam else []
    heavy = [value for value in records if value['heavy_atoms'] > 0]
    hydrogen = [value['smiles'] for value in records if value['heavy_atoms'] == 0]
    expanded = [side + ('.' + context_aam if context_aam else '') for side in parent_sides]
    require(mapped_pool(expanded[0]) == mapped_pool(expanded[1]), 'Expanded target atom maps do not conserve elements')
    expanded_components = [components(side) for side in expanded]
    require(side_signature(expanded_components[0]) == side_signature(expanded_components[1]),
            'Expanded target is not element/charge balanced')
    return parent['r_id'], {
        'context_aam': context_aam, 'context_rxn': side_text(records),
        'input_context_rxn': side_text(heavy), 'context_hydrogens': sorted(hydrogen),
        'context_count': len(records), 'input_context_count': len(heavy),
        'gt_aam': '>>'.join(expanded), 'gt_rxn': '>>'.join(side_text(side) for side in expanded_components)}


def initialize_worker():
    RDLogger.DisableLog('rdApp.*')


def recover_all(parents, dump, workers):
    endpoints = []
    for parent in parents.values():
        lines = json.loads(parent['source_lines'])
        endpoints.append({**parent, 'source_lines': json.dumps(sorted({lines[0], lines[-1]}))})
    selected = source_lines(dump, endpoints)
    tasks = []
    for parent in parents.values():
        lines = json.loads(parent['source_lines'])
        first, first_group = selected[parent['source_member']][lines[0]]
        last, last_group = selected[parent['source_member']][lines[-1]]
        require(first_group == last_group == parent['source_group'], 'Source witness group changed')
        tasks.append((parent, first.split('>>')[0], last.split('>>')[1]))
    jobs, recovered = iter(tasks), {}
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                             mp_context=get_context('spawn')) as executor:
        while batch := list(islice(jobs, 64)):
            recovered.update(executor.map(recover_context, batch, chunksize=4))
            if len(recovered) % 1024 == 0 or len(recovered) == len(parents):
                print(f'Recovered native context for {len(recovered):,}/{len(parents):,} parents', flush=True)
    return recovered


def hard_case(base, context):
    """Keep original core removals; add context only to reactants and both complete targets."""
    left, right = base['rxn'].split('>>')
    if context['input_context_rxn']:
        left = '.'.join(sorted(left.split('.') + context['input_context_rxn'].split('.')))
    first, second = [input_signature(side) for side in (left, right)]
    element_balanced, charge_balanced = first[0] == second[0], first[1] == second[1]
    row = {**base, 'base_r_id': base['r_id'], 'r_id': base['r_id'] + '_hard', 'variant': 'hard',
           'rxn': left + '>>' + right, 'ground_truth': context['gt_rxn'],
           'gt_rxn': context['gt_rxn'], 'gt_aam': context['gt_aam'],
           'context_count': context['context_count'], 'input_context_count': context['input_context_count'],
           'has_context': bool(context['input_context_count']),
           'context_rxn': context['context_rxn'], 'context_aam': context['context_aam'],
           'input_context_rxn': context['input_context_rxn'],
           'context_hydrogens': json.dumps(context['context_hydrogens']),
           'context_missing_from_products': bool(context['input_context_count']),
           'input_element_balanced': element_balanced, 'input_charge_balanced': charge_balanced,
           'input_balanced': element_balanced and charge_balanced}
    for field in ('removed_hydrogen_reactants', 'removed_hydrogen_products'):
        row[field] = json.dumps(sorted(json.loads(base[field]) + context['context_hydrogens']))
    return row


@lru_cache(maxsize=16384)
def component_signature(text):
    """Cache only formula/charge, avoiding repeated sanitization of shared mixture context."""
    molecule = Chem.MolFromSmiles(text)
    require(molecule is not None, 'Invalid hard input component')
    counts, charge = signature(molecule)
    return tuple(sorted(counts.items())), charge


def input_signature(side):
    counts, charge = Counter(), 0
    for text in side.split('.'):
        atoms, value = component_signature(text)
        counts.update(dict(atoms))
        charge += value
    return +counts, charge


def build(source, base, dump, output, workers=2):
    """Build a separate hard table with one-to-one base links and native source evidence."""
    initialize_worker()
    with gzip.open(source / 'aam.csv.gz', 'rt') as handle:
        parent_pool = {row['r_id']: row for row in csv.DictReader(handle)}
    base_report = json.loads((base / 'source.json').read_text())
    require(base_report['source_sha256'] == digest(source / 'aam.csv.gz'), 'Base dataset parent source changed')
    with gzip.open(base / 'rbl.csv.gz', 'rt') as handle:
        needed = {row['parent_r_id'] for row in csv.DictReader(handle)}
    require(needed <= set(parent_pool), 'Missing parent AAM records')
    contexts = recover_all({key: row for key, row in parent_pool.items() if key in needed}, dump, workers)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        marker = output / 'manifest.json'
        require(marker.is_file() and json.loads(marker.read_text()).get('schema') == SCHEMA,
                'Refusing to replace unmanaged hard output')
    counts, roles, context_counts, balanced = Counter(), Counter(), Counter(), Counter()
    with tempfile.TemporaryDirectory(prefix='.flower-rbl-hard-', dir=output.parent) as temporary:
        folder = Path(temporary)
        with gzip.open(base / 'rbl.csv.gz', 'rt') as handle, (folder / 'rbl.csv.gz').open('wb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding='utf8', newline='') as text:
                    writer = csv.DictWriter(text, fieldnames=FIELDS, lineterminator='\n')
                    writer.writeheader()
                    for case in csv.DictReader(handle):
                        row = hard_case(case, contexts[case['parent_r_id']])
                        writer.writerow(row)
                        counts[row['case_type']] += 1
                        roles[row['role']] += 1
                        context_counts['augmented' if row['has_context'] else 'no_heavy_context'] += 1
                        if row['input_balanced']:
                            balanced[row['case_type']] += 1
        require(sum(counts.values()) == base_report['rows'], 'Hard/base case counts differ')
        shutil.copytree(base / 'licenses', folder / 'licenses')
        report = {'schema': SCHEMA, 'rows': sum(counts.values()), 'by_case_type': dict(counts), 'by_role': dict(roles),
                  'context_counts': dict(context_counts), 'remain_balanced': dict(balanced),
                  'base_file': str(base / 'rbl.csv.gz'), 'base_sha256': digest(base / 'rbl.csv.gz'),
                  'source_aam': str(source / 'aam.csv.gz'), 'source_aam_sha256': digest(source / 'aam.csv.gz'),
                  'source_dump': str(dump), 'source_dump_sha256': digest(dump),
                  'implementation_sha256': digest(Path(__file__)), 'license_expression': 'MIT',
                  'context_policy': 'Unchanged first/final source components outside the retained parent atom pool',
                  'input_policy': 'Add heavy context only to reactants; omit all free H-only species',
                  'target_policy': 'Carry all recovered context through both target sides with original atom maps',
                  'roles': 'Source mixture context has no verified reagent/solvent labels',
                  'case_type_scope': 'Core heavy omission; context is additionally missing from input products',
                  'no_context_policy': 'Keep one-to-one base rows; flag has_context=false; never invent molecules',
                  'source_provenance': base_report}
        write_json(folder / 'source.json', report)
        (folder / 'NOTICE.txt').write_text(
            'FlowER RBL hard variant. MIT. Native mixture context restored.\n'
            'Base cases and original splits unchanged; see base_r_id.\n'
            'Context is added to input reactants and carried on both target sides.\n'
            'Hydrogen-only context is omitted from inputs and retained in targets.\n')
        write_json(folder / 'manifest.json', {'schema': SCHEMA, 'rows': report['rows'],
                   'files': {p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob('*') if p.is_file()}})
        backup = output.with_name('.' + output.name + '.previous')
        require(not backup.exists(), 'Unresolved previous hard export')
        if output.exists():
            output.rename(backup)
        try:
            folder.rename(output)
        except BaseException:
            if backup.exists():
                backup.rename(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    return report
