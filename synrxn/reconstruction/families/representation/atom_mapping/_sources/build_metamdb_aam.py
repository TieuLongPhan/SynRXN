#!/usr/bin/env python
"""Build available paper AAM references; require the exact source for the 4,456 subset."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.model_selection import train_test_split

from synrxn.reconstruction.paths import REPO, physical_path
from synrxn.reconstruction.families.representation.atom_mapping._sources.audit_metamdb_aam import csv_writer, inspect_rxn, inspect_smiles
from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import checked_file, download, digest

COMMIT = 'b1d428f3ed2d1ef5a1311150cc335ba1a916c06a'
NAMES = ('metamdb', 'metambd_cyclic')
SCHEMA = 'synrxn.metamdb-first-reference.v1'
ARCHIVE = {
    'path': 'Data/raw/metamdb/atom_mappings.zip',
    'url': 'https://metamdb.tu-bs.de/img/atom_mappings.zip',
    'sha256': 'b460f3ec5f842f757e35ad9eb0df0e5edf3450c289c272b2bfdcfb88c3689616',
}
RING = {
    'path': 'Data/raw/metamdb/brmapper/code/02_data_preprocessing/input_data/ringreactions_gt.csv',
    'url': f'https://raw.githubusercontent.com/NoraBeier/BRMapper/{COMMIT}/'
           'code/02_data_preprocessing/input_data/ringreactions_gt.csv',
    'sha256': '54d8f4d6f4598bee6acd93df54d2136d2f412b5cc551c50271069c8c9c2cd20f',
}
CORE = ['r_id', 'ground_truth', 'rxn', 'original_id']
EXTRA = ['source', 'reference_tier', 'validation_status', 'mapping_scope',
         'hydrogen_map_mismatch', 'element_balanced', 'charge_balanced', 'spectator_pairs_removed']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n')


def first_reference(value):
    """Keep the first supplied reference; never replace it with a later valid one."""
    return value.split(',', 1)[0].strip()


def first_kekule(fragment):
    mol = Chem.MolFromSmiles(fragment)
    if mol is None:
        raise ValueError('Invalid source fragment')
    forms = Chem.ResonanceMolSupplier(mol, Chem.KEKULE_ALL)
    if not len(forms):
        raise ValueError('No Kekule form for source fragment')
    mol = forms[0]
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.AROMATIC and bond.GetIsAromatic():
            bond.SetIsAromatic(False)
            bond.GetBeginAtom().SetIsAromatic(False)
            bond.GetEndAtom().SetIsAromatic(False)
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def first_ring_reference(value):
    """Reproduce the first upstream tautomer/Kekule variant without a Cartesian expansion."""
    value = first_reference(value)
    left, right = value.split('>>')
    products = []
    for fragment in right.split('.'):
        mol = Chem.MolFromSmiles(fragment)
        if mol is None:
            raise ValueError('Invalid product in native ring reference')
        forms = rdMolStandardize.TautomerEnumerator().Enumerate(mol)
        first = min(
            (Chem.MolToSmiles(Chem.RemoveHs(form), canonical=True) for form in forms),
            default=fragment,
        )
        products.append(first)
    sides = (left.split('.'), products)
    return '>>'.join(
        '.'.join(first_kekule(fragment) for fragment in side) for side in sides
    )


def archive_cases(path):
    with zipfile.ZipFile(path) as source:
        for source_row, info in enumerate((x for x in source.infolist()
                                           if x.filename.lower().endswith('.rxn')), 1):
            yield {
                'original_id': Path(info.filename).stem, 'source_row': source_row,
                'source_member': info.filename, 'raw_rxn': source.read(info),
            }


def ring_cases(path):
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    for source_row, row in enumerate(rows, 1):
        row['_source_row'] = source_row
    train, test = train_test_split(rows, test_size=0.2, random_state=42,
                                   stratify=[row['cluster'] for row in rows])
    for role, selected in [('train', train), ('test', test)]:
        for role_row, row in enumerate(selected, 1):
            yield {
                'original_id': row['reaction_id'], 'source_row': row['_source_row'],
                'role': role, 'role_row': role_row,
                'source_record': {k: v for k, v in row.items() if k != '_source_row'},
                'selected_reference': first_ring_reference(row['reaction_smiles']),
            }


def process_case(name, case):
    source = {key: value for key, value in case.items() if key != 'raw_rxn'}
    if 'raw_rxn' in case:
        raw = case['raw_rxn']
        source['source_member_sha256'] = hashlib.sha256(raw).hexdigest()
        result, status, reason = inspect_rxn(raw)
    else:
        source['selected_reference'] = first_reference(case['selected_reference'])
        result, status, reason = inspect_smiles(source['selected_reference'])
    if result is None:
        return None, source, status, reason
    row = {
        'r_id': (f'{name}_{case["role"]}_{case["role_row"]}' if 'role' in case
                 else f'{name}_{case["original_id"]}'),
        'ground_truth': result['candidate_mapping'], 'rxn': result['rxn'],
        'original_id': case['original_id'],
        'source': 'MetAMDB/BRMapper' if 'role' in case else 'MetAMDB',
        'reference_tier': ('author_supplied_first_reference' if 'role' in case
                           else 'upstream_RDT_reference'),
        'validation_status': 'human_review_unverified', 'mapping_scope': 'heavy_atoms',
        'hydrogen_map_mismatch': result['hydrogen_maps_differ'],
        'element_balanced': result['element_balanced'],
        'charge_balanced': result['charge_balanced'],
        'spectator_pairs_removed': result['spectator_pairs_removed'],
    }
    if 'role' in case:
        row['role'] = case['role']
    return row, source, status, reason


def build(name, cases, output, source_pin):
    """Stream records into a managed bundle, retaining repeated chemical reactions."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        marker = output / 'manifest.json'
        legacy = output / 'native_report.json'
        managed = (marker.is_file() and json.loads(marker.read_text()).get('schema') == SCHEMA)
        managed |= (legacy.is_file() and
                    json.loads(legacy.read_text()).get('dataset') == name)
        if not managed:
            raise ValueError(f'Refusing to replace unmanaged directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = CORE + (['role'] if name == 'metambd_cyclic' else []) + EXTRA
    counts, roles = Counter(), Counter()
    identifiers = set()
    with tempfile.TemporaryDirectory(prefix='.metamdb-build-', dir=output.parent) as temp:
        folder = Path(temp)
        with ExitStack() as stack:
            table = csv_writer(stack, folder / 'aam.csv.gz', fields)
            rejects = csv_writer(stack, folder / 'rejected.csv.gz',
                                 ['original_id', 'source_row', 'status', 'reason'])
            binary = stack.enter_context((folder / 'native_provenance.jsonl.gz').open('wb'))
            provenance = stack.enter_context(
                gzip.GzipFile(filename='', mode='wb', fileobj=binary, mtime=0))
            for case in cases:
                counts['source_rows'] += 1
                row, source, status, reason = process_case(name, case)
                counts[status] += 1
                if row is None:
                    rejects.writerow({key: value for key, value in
                                      dict(original_id=case['original_id'], source_row=case['source_row'],
                                           status=status, reason=reason).items()})
                    continue
                if row['r_id'] in identifiers:
                    raise ValueError(f'Duplicate source record ID: {row["r_id"]}')
                identifiers.add(row['r_id'])
                table.writerow(row)
                roles[row.get('role', 'unsplit')] += 1
                source.update(r_id=row['r_id'], selected_reference_index=0)
                provenance.write((json.dumps(source, sort_keys=True, separators=(',', ':')) + '\n').encode())
                counts['hydrogen_map_mismatch_rows'] += row['hydrogen_map_mismatch']
                counts['spectator_component_pairs_removed'] += row['spectator_pairs_removed']
                if counts['source_rows'] % 10000 == 0:
                    print(f'{name}: inspected {counts["source_rows"]:,} records', flush=True)
        report = {
            'schema': SCHEMA, 'dataset': name, 'rows': len(identifiers), 'columns': fields,
            'counts': dict(counts), 'by_role': dict(roles), 'source': source_pin,
            'reference_policy': 'first supplied reference only; no tautomer canonicalization',
            'deduplication': 'none; all usable source memberships retained',
            'mapping_policy': 'retain source heavy-atom correspondences; no remapping or completion',
            'component_policy': 'remove exactly unchanged mapped spectator components',
            'validation_status': 'human_review_unverified',
            'split_policy': ('upstream stratified 80/20 split, random_state=42, cluster strata'
                             if name == 'metambd_cyclic' else 'unsplit; no artificial assignment'),
            'runtime_versions': {key: version(key) for key in ['rdkit', 'numpy', 'scikit-learn']},
            'implementation': [
                {'path': str(path.relative_to(REPO)), 'sha256': digest(path)}
                for path in [Path(__file__).resolve(), REPO / 'synrxn/reconstruction/families/representation/atom_mapping/_sources/audit_metamdb_aam.py']
            ],
            'qualification': 'Source reference mappings; no independent human validation claimed.',
        }
        write_json(folder / 'native_report.json', report)
        write_json(folder / 'native_source.json', {
            'dataset': name, 'source': source_pin, 'paper': 'https://ssrn.com/abstract=7404606',
            'upstream_repository': f'https://github.com/NoraBeier/BRMapper/tree/{COMMIT}',
            'qualification': report['qualification'], 'deduplication': report['deduplication'],
        })
        write_json(folder / 'manifest.json', {
            'schema': SCHEMA, 'dataset': name,
            'files': {path.name: digest(path) for path in sorted(folder.iterdir())},
        })
        previous = output.with_name('.' + output.name + '.previous')
        if previous.exists():
            raise ValueError(f'Unresolved previous build: {previous}')
        if output.exists():
            output.rename(previous)
        try:
            folder.rename(output)
        except BaseException:
            if previous.exists():
                previous.rename(output)
            raise
        if previous.exists():
            shutil.rmtree(previous)
    print(f'{name}: {report["rows"]:,} records; {dict(roles)}', flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=NAMES, default=['metambd_cyclic'])
    parser.add_argument('--output-root', type=Path, default=physical_path('Data/raw/aam'))
    parser.add_argument('--output', type=Path, help='Exact native output folder for one selected dataset')
    parser.add_argument('--fetch', action='store_true', help='Fetch missing checksum-pinned public inputs')
    args = parser.parse_args(argv)
    if args.output is not None and len(args.datasets) != 1:
        parser.error('--output requires exactly one --datasets selection')
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
    for name in args.datasets:
        if name == 'metamdb':
            raise ValueError(
                'The exact 4,456-reaction paper subset or its reaction-ID list is required. '
                'The public full archive is not that subset and is excluded from the selected AAM pool.'
            )
        pin = ARCHIVE if name == 'metamdb' else RING
        source = download(pin) if args.fetch else checked_file(pin)
        cases = archive_cases(source) if name == 'metamdb' else ring_cases(source)
        folder = args.output or args.output_root / name / 'native-product'
        build(name, cases, folder, pin)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
