#!/usr/bin/env python
"""Separate usable MetAMDB heavy-atom mappings from supported curated references.

Read the original ZIP one RXN file at a time. Curation requires external
MetAMDB metadata: the ZIP itself does not contain the website's Curated flag.
Passing mechanical QC never promotes a mapping to a human-curated reference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdChemReactions

from synrxn.reconstruction.paths import repo_path


FIELDS = [
    'r_id', 'original_id', 'source_member', 'rxn', 'candidate_mapping',
    'curation_status', 'ground_truth_status', 'archive_matches_reviewed_version',
    'hydrogen_maps_differ', 'element_balanced', 'charge_balanced',
    'spectator_pairs_removed', 'source_member_sha256',
]


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def csv_writer(stack, path, fields):
    binary = stack.enter_context(path.open('wb'))
    compressed = stack.enter_context(gzip.GzipFile(filename='', mode='wb', fileobj=binary, mtime=0))
    handle = stack.enter_context(io.TextIOWrapper(compressed, encoding='utf-8', newline=''))
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    return writer


def inventory(molecules):
    elements, charge = Counter(), 0
    for mol in molecules:
        for atom in mol.GetAtoms():
            elements[atom.GetAtomicNum()] += 1
            if atom.GetAtomicNum() != 1:
                elements[1] += atom.GetTotalNumHs(includeNeighbors=False)
            charge += atom.GetFormalCharge()
    return +elements, charge


def inspect_rxn(raw):
    try:
        reaction = rdChemReactions.ReactionFromRxnBlock(
            raw.decode('utf-8'), sanitize=False, removeHs=False, strictParsing=True)
        if reaction is None:
            raise ValueError('RXN parser returned no reaction')
        sides = [list(reaction.GetReactants()), list(reaction.GetProducts())]
        for side in sides:
            for mol in side:
                Chem.SanitizeMol(mol)
    except Exception as error:
        return None, 'parse_or_sanitization_failed', str(error)[:240]
    return inspect_molecules(sides)


def inspect_smiles(value):
    """Apply the same source-preserving QC directly to mapped reaction SMILES."""
    try:
        if value.count('>>') != 1:
            raise ValueError('Expected one reaction arrow')
        parameters = Chem.SmilesParserParams()
        parameters.removeHs = False
        sides = []
        for side in value.split('>>'):
            mol = Chem.MolFromSmiles(side, parameters)
            if mol is None:
                raise ValueError('Invalid or unsanitizable source SMILES')
            sides.append(list(Chem.GetMolFrags(mol, asMols=True)))
    except Exception as error:
        return None, 'parse_or_sanitization_failed', str(error)[:240]
    return inspect_molecules(sides)


def inspect_molecules(sides):
    """Check source molecules without recomputing their atom correspondences."""
    # RXN templates use QueryAtom even for a concrete element. A query wrapper
    # alone is not evidence of an unspecified chemical element.
    if any(atom.GetAtomicNum() == 0
           for side in sides for mol in side for atom in mol.GetAtoms()):
        return None, 'generic_or_dummy_atom', 'Unspecified element or dummy atom in source structure'
    maps, hydrogens = [], []
    for side in sides:
        atoms = [atom for mol in side for atom in mol.GetAtoms()]
        heavy = [atom for atom in atoms if atom.GetAtomicNum() != 1]
        labels = [atom.GetAtomMapNum() for atom in heavy]
        if not heavy or not all(labels) or len(labels) != len(set(labels)):
            return None, 'heavy_maps_not_conserved', 'Missing or duplicate heavy-atom map labels'
        maps.append({atom.GetAtomMapNum(): atom.GetAtomicNum() for atom in heavy})
        hydrogens.append(Counter(atom.GetAtomMapNum() for atom in atoms
                                 if atom.GetAtomicNum() == 1 and atom.GetAtomMapNum()))
    if maps[0] != maps[1]:
        return None, 'heavy_maps_not_conserved', 'Heavy-atom label/element inventories differ'
    hydrogen_maps_differ = hydrogens[0] != hydrogens[1]
    elements = [inventory(side) for side in sides]
    fragments = [Counter(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                         for mol in side) for side in sides]
    shared = fragments[0] & fragments[1]
    active = [side - shared for side in fragments]
    if not all(active):
        return None, 'unchanged_after_spectator_removal', 'No active reaction remains'
    unmapped, normalized = [], []
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    for side in active:
        molecule = Chem.MolFromSmiles('.'.join(sorted(side.elements())), parameters)
        if molecule is None:
            raise ValueError('Sanitized source could not be re-read as SMILES')
        # Heavy-atom task: keep proton participants, clear their correspondence.
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() == 1:
                atom.SetAtomMapNum(0)
        normalized.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        molecule = Chem.RemoveHs(molecule)
        unmapped.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    return {
        'rxn': '>>'.join(unmapped), 'candidate_mapping': '>>'.join(normalized),
        'hydrogen_maps_differ': hydrogen_maps_differ,
        'element_balanced': elements[0][0] == elements[1][0],
        'charge_balanced': elements[0][1] == elements[1][1],
        'spectator_pairs_removed': sum(shared.values()),
    }, 'usable_heavy_aam', ''


def load_metadata(path):
    if path is None:
        return {}
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        records = json.load(handle)
    if isinstance(records, dict):
        records = records['reactions']
    result = {}
    for record in records:
        if record is not None:
            key = str(record['id'])
            if key in result and result[key] != record:
                raise ValueError(f'Conflicting metadata for reaction {key}')
            result[key] = record
    return result


def reference_status(metadata, raw):
    if metadata is None or not isinstance(metadata.get('updated'), bool):
        return 'unknown', 'curation_not_established', ''
    if not metadata['updated']:
        return 'automatically_generated', 'computational_reference', ''
    current = metadata.get('rxnFile')
    if not isinstance(current, str):
        return 'manually_curated', 'curated_version_not_checked', ''
    matches = current.encode('utf-8') == raw
    return ('manually_curated',
            'upstream_human_curated_reference' if matches else 'curated_version_differs', matches)


def compare_brmapper_sources(root, archive_ids, output):
    """Check the published source tables without assuming a manual-review flag."""
    lock = json.loads((root / 'source.json').read_text())
    tables = {}
    full_ids = set()
    paths = ['data/aam_metamdb.csv', 'data/aam_ringreactions.csv',
             'code/02_data_preprocessing/input_data/ringreactions_gt.csv']
    for relative in paths:
        path = root / relative
        expected = next(x for x in lock['files'] if x['path'] == relative)
        checksum = digest(path.read_bytes())
        if checksum != expected['sha256']:
            raise ValueError(f'BRMapper source hash changed: {relative}')
        with path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = reader.fieldnames
        ids = {row['reaction_id'] for row in rows}
        tables[relative] = {
            'rows': len(rows), 'unique_reaction_ids': len(ids), 'columns': fields,
            'sha256': checksum,
            'has_curation_column': any('curat' in field.lower() or 'validat' in field.lower()
                                       for field in fields),
        }
        if relative == paths[0]:
            full_ids = ids
        if relative == paths[2]:
            with ExitStack() as stack:
                writer = csv_writer(stack, output / 'ring_author_ground_truth.csv.gz', fields)
                writer.writerows(rows)
    comparison = {
        'repository': lock['repository'], 'commit': lock['commit'], 'tables': tables,
        'full_csv_ids_equal_zip_ids': full_ids == archive_ids,
        'full_csv_ids_missing_from_zip': len(full_ids - archive_ids),
        'zip_ids_missing_from_full_csv': len(archive_ids - full_ids),
        'qualification': ('The full released CSV does not identify the paper-reported validated '
                          'subset. The ring ground-truth table is supplied by the authors; '
                          'its designation does not independently establish manual review. '
                          'Original source columns and mappings are copied without changes.'),
    }
    (output / 'brmapper-source-comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--metadata', type=Path,
                        help='Saved MetAMDB API reaction metadata, JSON or JSON.GZ')
    parser.add_argument('--brmapper-root', type=Path,
                        help='Pinned BRMapper source bundle with source.json and native data CSVs')
    parser.add_argument('--output', type=Path,
                        default=repo_path('Data/tables/development/metamdb/assessment'))
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.warning')
    RDLogger.DisableLog('rdApp.error')
    metadata = load_metadata(args.metadata)
    args.output.mkdir(parents=True, exist_ok=True)
    statuses, curation, reference, cross = Counter(), Counter(), Counter(), Counter()
    endpoints, mappings, covered = set(), set(), set()
    extras = Counter()
    archive_ids = set()
    with ExitStack() as stack, zipfile.ZipFile(args.archive) as source:
        usable = csv_writer(stack, args.output / 'usable_candidates.csv.gz', FIELDS)
        curated = csv_writer(stack, args.output / 'human_curated_references.csv.gz', FIELDS)
        rejected = csv_writer(stack, args.output / 'rejected.csv.gz',
                              ['original_id', 'source_member', 'status', 'reason'])
        for index, info in enumerate((x for x in source.infolist()
                                      if x.filename.lower().endswith('.rxn')), 1):
            raw = source.read(info)
            identifier = Path(info.filename).stem
            archive_ids.add(identifier)
            record, status, reason = inspect_rxn(raw)
            entry = metadata.get(identifier)
            if entry is not None:
                covered.add(identifier)
            cur_status, ref_status, matches = reference_status(entry, raw)
            statuses[status] += 1
            curation[cur_status] += 1
            cross[f'{status}|{cur_status}'] += 1
            if record is None:
                rejected.writerow({'original_id': identifier, 'source_member': info.filename,
                                   'status': status, 'reason': reason})
            else:
                row = dict(record, r_id=f'metamdb:{identifier}', original_id=identifier,
                           source_member=info.filename, curation_status=cur_status,
                           ground_truth_status=ref_status,
                           archive_matches_reviewed_version=matches,
                           source_member_sha256=digest(raw))
                usable.writerow(row)
                reference[ref_status] += 1
                if ref_status == 'upstream_human_curated_reference':
                    curated.writerow(row)
                endpoints.add(digest(record['rxn'].encode()))
                mappings.add(digest(record['candidate_mapping'].encode()))
                extras['hydrogen_map_mismatch_files'] += record['hydrogen_maps_differ']
                extras['spectator_component_pairs_removed'] += record['spectator_pairs_removed']
            if index % 5000 == 0:
                print(f'Inspected {index:,} RXN files', flush=True)
    comparison = (compare_brmapper_sources(args.brmapper_root, archive_ids, args.output)
                  if args.brmapper_root else None)
    report = {
        'schema': 'synrxn.metamdb-assessment.v1',
        'assessed_at': datetime.now(timezone.utc).isoformat(),
        'archive': str(args.archive.resolve()),
        'archive_sha256': digest(args.archive.read_bytes()),
        'rdkit_version': rdBase.rdkitVersion,
        'mapping_scope': 'heavy_atoms',
        'rxn_files': sum(statuses.values()), 'status_counts': dict(statuses),
        'distinct_directed_unmapped_reactions': len(endpoints),
        'distinct_normalized_mapped_strings': len(mappings),
        'curation_status_counts_all_files': dict(curation),
        'usable_reference_status_counts': dict(reference),
        'quality_by_curation_status': dict(cross),
        'metadata_records': len(metadata), 'metadata_archive_ids_covered': len(covered),
        'metadata_file': str(args.metadata.resolve()) if args.metadata else None,
        'metadata_file_sha256': digest(args.metadata.read_bytes()) if args.metadata else None,
        'curation_metadata_complete': (len(covered) == sum(statuses.values())
                                       and not curation['unknown']),
        'human_curation_count_in_archive': (curation['manually_curated']
                                            if len(covered) == sum(statuses.values())
                                            and not curation['unknown'] else None),
        'brmapper_source_comparison': comparison,
        'qualification': ('Mechanical heavy-atom QC is not chemical or human validation. '
                          'An empty curated export with absent metadata means unknown curation, '
                          'not zero curated reactions in MetAMDB. Curated references require '
                          'updated=true and an exact match to the reviewed RXN payload. '
                          'Distinct SMILES counts are not ITS-isomorphism deduplication.'),
        'processing_script': str(Path(__file__).resolve()),
        'processing_script_sha256': digest(Path(__file__).read_bytes()),
        **extras,
    }
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
