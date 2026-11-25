"""Test reaction identity and ITS disagreements independently of map labels."""
import csv
import gzip
import json
import subprocess
import sys

from synrxn.reconstruction.families.representation.atom_mapping._sources import audit_aam_overlap as audit
from synrxn.reconstruction.families.representation.atom_mapping._sources import report_aam_overlap as review
from synrxn.reconstruction.families.representation.atom_mapping._sources.audit_aam_overlap import index_record
from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import digest
from synrxn.reconstruction.families.representation.atom_mapping._sources import aam_overlap_chem as chem


REACTION = '[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]'
SWAPPED = '[CH3:1][CH2:2][OH:3]>>[CH3:2][CH:1]=[O:3]'


def graph(value, heavy=True):
    mols = chem.reference_info(value)['mols']
    return chem.its_graph(mols, heavy=heavy)


def test_same_endpoints_with_different_its_are_not_merged():
    assert chem.identities(REACTION)['exact_key'] == chem.identities(SWAPPED)['exact_key']
    classes = chem.ITSClasses()
    assert classes.assign(graph(REACTION)) != classes.assign(graph(SWAPPED))


def test_renumbering_and_symmetry_are_equivalent():
    renamed = REACTION.replace(':1]', ':101]').replace(':2]', ':102]').replace(':3]', ':103]')
    classes = chem.ITSClasses()
    assert classes.assign(graph(REACTION)) == classes.assign(graph(renamed))
    first = '[CH3:1][C:2](=[O:3])[CH3:4]>>[CH3:1][CH:2]([OH:3])[CH3:4]'
    second = '[CH3:1][C:2](=[O:3])[CH3:4]>>[CH3:4][CH:2]([OH:3])[CH3:1]'
    assert classes.assign(graph(first)) == classes.assign(graph(second))


def test_explicit_and_attached_hydrogens_have_same_heavy_its():
    one = '[CH3:1][OH:2]>>[CH2:1]=[O:2]'
    two = '[H:3][C:1]([H:4])([H:5])[O:2][H:6]>>[H:3][C:1]([H:4])=[O:2].[H:5][H:6]'
    # Additional H2 is a real product, so input identity remains different.
    classes = chem.ITSClasses()
    assert classes.assign(graph(one)) == classes.assign(graph(two))
    assert chem.identities(one)['exact_key'] != chem.identities(two)['exact_key']


def test_stereo_is_ignored_but_isotopes_and_tautomers_are_not_collapsed():
    chiral = '[CH3:1][C@H:2]([OH:3])[CH2:4][OH:5]>>[CH3:1][C@H:2]([OH:3])[CH:4]=[O:5]'
    opposite = chiral.replace('@H', '@@H')
    assert chem.identities(chiral)['exact_key'] == chem.identities(opposite)['exact_key']
    classes = chem.ITSClasses()
    assert classes.assign(graph(chiral)) == classes.assign(graph(opposite))
    isotope = REACTION.replace('[CH3:1]', '[13CH3:1]')
    assert classes.assign(graph(REACTION)) != classes.assign(graph(isotope))
    enol = REACTION.replace('[CH3:1][CH:2]=[O:3]', '[CH2:1]=[CH:2][OH:3]')
    assert chem.identities(REACTION)['exact_key'] != chem.identities(enol)['exact_key']


def test_double_bond_stereo_is_ignored_like_atom_stereo():
    value = '[F:1]/[CH:2]=[CH:3]/[Cl:4]>>[F:1][CH2:2][CH2:3][Cl:4]'
    renamed = value.replace(':1]', ':91]').replace(':2]', ':92]')
    opposite = value.replace('/[Cl:4]', '\\[Cl:4]')
    classes = chem.ITSClasses()
    assert classes.assign(graph(value)) == classes.assign(graph(renamed))
    assert classes.assign(graph(value)) == classes.assign(graph(opposite))


def test_spectators_and_reverse_direction_are_separate_audit_modes():
    extra = REACTION.replace('>>', '.[Na+:4]>>') + '.[Na+:4]'
    assert chem.identities(REACTION)['exact_key'] != chem.identities(extra)['exact_key']
    assert chem.identities(REACTION)['normalized_key'] == chem.identities(extra)['normalized_key']
    active = chem.reference_info(extra)['active']
    classes = chem.ITSClasses()
    assert classes.assign(graph(REACTION)) == classes.assign(chem.its_graph(active))
    reverse = '>>'.join(reversed(REACTION.split('>>')))
    assert chem.identities(reverse)['exact_key'] != chem.identities(REACTION)['exact_key']
    assert chem.identities(reverse)['undirected_key'] == chem.identities(REACTION)['undirected_key']
    assert classes.assign(chem.unpack(chem.pack(graph(reverse)), reverse=True)) == classes.assign(graph(REACTION))


def test_identity_reactions_do_not_collapse_to_empty_input():
    assert chem.identities('CO>>CO')['normalized_key'] != chem.identities('CC>>CC')['normalized_key']


def test_invalid_maps_and_mismatched_inputs_are_not_called_conflicts():
    value = '[CH3:1][NH2:1]>>[CH2:1]=[NH:1]'
    assert chem.reference_info(value)['mapping_status'] == 'mapped_atom_identity_mismatch'
    row = index_record({'dataset': 'example', 'r_id': '1', 'original_id': 'x', 'source_row': 1,
                        'role': 'test', 'steps': '', 'rxn': 'CN>>C=N', 'ground_truth': REACTION})
    assert row['mapping_status'] == 'ok' and not row['endpoint_match']
    other = index_record(dict(row, rxn='CCO>>CC=O', role='train'))
    assert other['endpoint_match']


def test_unbalanced_reactions_support_partial_its_without_inventing_atoms():
    first = '[CH3:1][Cl:2].[OH-:3]>>[CH3:1][OH:3]'
    renamed = '[CH3:11][Cl:12].[OH-:13]>>[CH3:11][OH:13]'
    assert chem.reference_info(first)['mapping_status'] == 'partial'
    classes = chem.ITSClasses()
    assert classes.assign(graph(first)) == classes.assign(graph(renamed))
    lost = json.loads(graph(first).nodes[2]['label'])
    assert lost[2] is not None and lost[3] is None
    wrong_element = '[CH3:1][Cl:2]>>[CH3:1][OH:2]'
    assert chem.reference_info(wrong_element)['mapping_status'] == 'mapped_atom_identity_mismatch'


def test_explicit_hydrogen_mapping_disagreement_is_separate_from_heavy_atoms():
    first = '[H:3][C:1]([H:4])([H:5])[O:2][H:6]>>[H:3][C:1]([H:4])=[O:2].[H:5][H:6]'
    second = '[H:3][C:1]([H:4])([H:5])[O:2][H:6]>>[H:3][C:1]([H:5])=[O:2].[H:4][H:6]'
    assert chem.identities(first)['exact_key'] == chem.identities(second)['exact_key']
    heavy = chem.ITSClasses()
    assert heavy.assign(graph(first)) == heavy.assign(graph(second))
    # The three methyl hydrogens are symmetric: exchanging them is equivalent.
    full = chem.ITSClasses()
    assert full.assign(graph(first, heavy=False)) == full.assign(graph(second, heavy=False))
    third = '[H:3][C:1]([H:4])([H:5])[O:2][H:6]>>[H:3][C:1]([H:6])=[O:2].[H:4][H:5]'
    assert heavy.assign(graph(first)) == heavy.assign(graph(third))
    assert full.assign(graph(first, heavy=False)) != full.assign(graph(third, heavy=False))


def test_complete_audit_keeps_roles_as_provenance_and_flags_bad_maps(tmp_path):
    reverse = '>>'.join(reversed(REACTION.split('>>')))
    invalid = '[CH3:1][CH2:1][OH:3]>>[CH3:1][CH:1]=[O:3]'
    spectator = REACTION.replace('>>', '.[Na+:4]>>') + '.[Na+:4]'
    partial = '[CH3:1][CH2:2][O:3][CH3:4]>>[CH3:4][OH:3]'
    partial_wrong = '[CH3:1][CH2:2][O:3][CH3:4]>>[CH3:1][OH:3]'
    values = {'first': [REACTION, REACTION, reverse, partial],
              'second': [SWAPPED, REACTION, invalid, spectator, partial, partial_wrong]}
    entries = []
    input_root = tmp_path / 'tables'
    for name, reactions in values.items():
        folder = input_root / name
        folder.mkdir(parents=True)
        table = folder / 'aam.csv.gz'
        with gzip.open(table, 'wt', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['r_id', 'original_id', 'rxn', 'ground_truth', 'role'])
            writer.writeheader()
            for i, reaction in enumerate(reactions):
                writer.writerow({'r_id': str(i), 'original_id': str(i),
                                 'rxn': chem.identities(reaction)['canonical_rxn'],
                                 'ground_truth': reaction, 'role': 'train' if i % 2 else 'test'})
        (folder / 'manifest.json').write_text(json.dumps({'files': {'aam.csv.gz': digest(table)}}))
        entries.append({'name': name, 'rows': len(reactions)})
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'schema': 'synrxn.aam-sources.v1', 'datasets': entries, 'expected_rows': 10}))
    output = tmp_path / 'audit'
    result = subprocess.run(
        [sys.executable, '-m', 'synrxn.reconstruction', 'builders', 'audit_aam_overlap',
         '--registry', str(registry), '--input-root', str(input_root),
         '--output', str(output), '--workers', '2', '--batch-size', '2'],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / 'summary.json').read_text())
    assert report['records'] == 10 and not report['roles_used_for_grouping']
    for mode, expected_count in [('exact', 8), ('normalized', 9), ('undirected', 10)]:
        stats = report['modes'][mode]['totals']
        assert stats['overlap_groups'] == stats['its_disagreement_groups'] == 2
        assert stats['overlap_rows'] == expected_count
        assert stats['groups_with_unverifiable_members'] == 0
        pair = report['modes'][mode]['pairs'][0]
        assert pair['mapping_disagreement_groups'] == pair['shared_mapping_groups'] == 2
        assert pair['no_shared_mapping_groups'] == 0
    assert review.main(['--audit', str(output), '--output', str(tmp_path / 'review'),
                        '--report', str(tmp_path / 'report.rst')]) == 0
    with gzip.open(tmp_path / 'review/conflicting_records.csv.gz', 'rt') as f:
        conflicting = list(csv.DictReader(f))
    assert len(conflicting) == 8
    assert {r['mapping_status'] for r in conflicting} >= {'ok', 'partial', 'equivalent'}
