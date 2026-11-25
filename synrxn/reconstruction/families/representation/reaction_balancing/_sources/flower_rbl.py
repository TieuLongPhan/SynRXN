"""Construct source-linked FlowER balancing cases with three heavy-removal types."""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import shutil
import tempfile

from rdkit import Chem, RDLogger

from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import digest, require, write_json
from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_center_groups import write_csv

CASE_TYPES = ('reactant', 'product', 'both')
TRACE_FIELDS = ('role', 'steps', 'center_cluster_id', 'center_type', 'source_group', 'source_parent_group',
                'source_lines', 'source_seed_id', 'source_member', 'source_split')
FIELDS = ('r_id', 'rxn', 'ground_truth', 'gt_rxn', 'gt_aam', 'original_id', 'parent_r_id', 'case_type', *TRACE_FIELDS,
          'removed_reactant', 'removed_product', 'removed_hydrogen_reactants', 'removed_hydrogen_products',
          'input_element_balanced', 'input_charge_balanced', 'input_balanced')


def signature(molecule):
    """Count all elements including bracket/implicit H, and retain total charge."""
    counts = Counter()
    charge = 0
    for atom in molecule.GetAtoms():
        counts[atom.GetAtomicNum()] += 1
        counts[1] += atom.GetTotalNumHs()
        charge += atom.GetFormalCharge()
    return +counts, charge


def side_signature(components):
    counts, charge = Counter(), 0
    for component in components:
        atoms, value = signature(component['molecule'])
        counts.update(atoms)
        charge += value
    return +counts, charge


def components(side):
    molecule = Chem.MolFromSmiles(side)
    require(molecule is not None, 'Invalid reference endpoint')
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    molecule = Chem.RemoveHs(molecule)
    records = []
    for index, fragment in enumerate(Chem.GetMolFrags(molecule, asMols=True)):
        heavy = sum(atom.GetAtomicNum() > 1 for atom in fragment.GetAtoms())
        require(not any(atom.GetAtomicNum() == 0 for atom in fragment.GetAtoms()), 'Generic reference atom')
        text = Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
        records.append({'molecule': fragment, 'smiles': text, 'heavy_atoms': heavy,
                        'size_key': (heavy, Chem.AddHs(fragment).GetNumAtoms(), text, index)})
    return records


def side_text(values):
    return '.'.join(sorted(value['smiles'] for value in values))


def make_cases(parent):
    """Strip free hydrogen species from both inputs, then one smallest heavy component."""
    left, right = parent['ground_truth'].split('>>')
    sides = [components(side) for side in (left, right)]
    require(side_signature(sides[0]) == side_signature(sides[1]), 'Reference is not element/charge balanced')
    target = '>>'.join(side_text(side) for side in sides)
    hydrogens = [[value['smiles'] for value in side if value['heavy_atoms'] == 0] for side in sides]
    heavy = [[value for value in side if value['heavy_atoms'] > 0] for side in sides]
    records, skipped = [], Counter()
    for kind in CASE_TYPES:
        remove = (kind in ('reactant', 'both'), kind in ('product', 'both'))
        if any(len(side) <= int(flag) for side, flag in zip(heavy, remove)):
            skipped[kind] += 1
            continue
        remaining, removed = [], []
        for side, flag in zip(heavy, remove):
            candidate = min(side, key=lambda value: value['size_key']) if flag else None
            remaining.append([value for value in side if value is not candidate])
            removed.append(candidate['smiles'] if candidate else '')
        first, second = [side_signature(side) for side in remaining]
        element_balanced, charge_balanced = first[0] == second[0], first[1] == second[1]
        records.append({'r_id': f"flower_rbl_{parent['r_id']}_{kind}",
                        'rxn': '>>'.join(side_text(side) for side in remaining), 'ground_truth': target,
                        'gt_rxn': target, 'gt_aam': parent['ground_truth'],
                        'original_id': parent['original_id'], 'parent_r_id': parent['r_id'], 'case_type': kind,
                        **{field: parent[field] for field in TRACE_FIELDS},
                        'removed_reactant': removed[0], 'removed_product': removed[1],
                        'removed_hydrogen_reactants': json.dumps(sorted(hydrogens[0])),
                        'removed_hydrogen_products': json.dumps(sorted(hydrogens[1])),
                        'input_element_balanced': element_balanced, 'input_charge_balanced': charge_balanced,
                        'input_balanced': element_balanced and charge_balanced})
    return records, dict(skipped)


def build(source, output):
    """Build one managed RBL dataset and preserve split, center and parent provenance."""
    RDLogger.DisableLog('rdApp.*')
    with gzip.open(source / 'aam.csv.gz', 'rt') as handle:
        parents = list(csv.DictReader(handle))
    require(bool(parents), 'No source AAM records')
    require(set(TRACE_FIELDS) <= set(parents[0]), 'Source AAM table needs global center annotations first')
    records, rejected, skipped = [], [], Counter()
    for parent in parents:
        try:
            cases, missing = make_cases(parent)
        except ValueError as error:
            rejected.append({'parent_r_id': parent['r_id'], 'reason': str(error)})
            continue
        records.extend(cases)
        skipped.update(missing)
    require(len({row['r_id'] for row in records}) == len(records), 'Duplicate RBL case identifiers')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        marker = output / 'manifest.json'
        require(marker.is_file() and json.loads(marker.read_text()).get('schema') in
                ('synrxn.flower-rbl.v1', 'synrxn.flower-rbl.v2'),
                'Refusing to replace unmanaged RBL output')
    with tempfile.TemporaryDirectory(prefix='.flower-rbl-', dir=output.parent) as temporary:
        folder = Path(temporary)
        write_csv(folder / 'rbl.csv.gz', records, FIELDS)
        if (source / 'licenses').is_dir():
            shutil.copytree(source / 'licenses', folder / 'licenses')
        source_info = json.loads((source / 'source.json').read_text())
        require(source_info['rights']['license_expression'] == 'MIT', 'Unexpected FlowER rights evidence')
        report = {'schema': 'synrxn.flower-rbl.v2', 'rows': len(records), 'source_parents': len(parents),
                  'target_fields': {'gt_rxn': 'Complete unmapped reference reaction',
                                    'gt_aam': 'Exact complete mapped reference from the parent FlowER AAM record',
                                    'ground_truth': 'Compatibility alias of gt_rxn used by existing RBL consumers'},
                  'by_case_type': dict(Counter(row['case_type'] for row in records)),
                  'by_role': dict(Counter(row['role'] for row in records)),
                  'by_role_and_case_type': {role: dict(Counter(
                      row['case_type'] for row in records if row['role'] == role))
                                            for role in ('train', 'val', 'test')},
                  'remain_element_balanced': dict(Counter(
                      row['case_type'] for row in records if row['input_element_balanced'])),
                  'remain_element_and_charge_balanced': dict(Counter(
                      row['case_type'] for row in records if row['input_balanced'])),
                  'skipped_empty_side_by_type': dict(skipped), 'rejected_parents': rejected,
                  'policy': 'Strip H-only components on both input sides, then remove the smallest heavy molecule',
                  'smallest': ['heavy atom count', 'total atom count including H', 'canonical SMILES', 'source index'],
                  'balanced_inputs': 'retained and flagged; deletion provenance is not uniquely inferable from balance',
                  'source_aam': str(source / 'aam.csv.gz'), 'source_sha256': digest(source / 'aam.csv.gz'),
                  'source_provenance': source_info, 'implementation_sha256': digest(Path(__file__)),
                  'grouping': 'Group siblings by parent_r_id; group centers by center_cluster_id for later splits',
                  'license_expression': 'MIT',
                  'qualification': 'Synthetic missing-component tasks from FlowER references'}
        write_json(folder / 'source.json', report)
        (folder / 'NOTICE.txt').write_text(
            'FlowER-derived missing-component balancing cases. MIT.\n'
            'Native roles and parent/center/source witnesses retained.\n'
            'gt_rxn is the complete unmapped target; gt_aam preserves the parent mapping.\n'
            'ground_truth remains an exact compatibility alias of gt_rxn.\n'
            'case_type identifies the heavy-component removal sides; hydrogen-only\n'
            'components are additionally omitted from both input sides.\n')
        write_json(folder / 'manifest.json', {'schema': report['schema'], 'rows': len(records),
                   'files': {p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob('*') if p.is_file()}})
        backup = output.with_name('.' + output.name + '.previous')
        require(not backup.exists(), 'Unresolved previous RBL export')
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
