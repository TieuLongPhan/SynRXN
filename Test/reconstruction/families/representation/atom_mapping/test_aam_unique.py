"""Exercise deduplication decisions, provenance, singleton QC and reproducibility."""
import csv
import gzip
import json

import pytest

from synrxn.reconstruction.families.representation.atom_mapping._sources import audit_aam_overlap as audit
from synrxn.reconstruction.families.representation.atom_mapping._sources import build_unique_aam as unique
from synrxn.reconstruction.families.representation.atom_mapping._sources import aam_overlap_chem as chem
from synrxn.reconstruction.families.representation.atom_mapping._sources.aam_sources import digest


ETHANOL = '[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]'
SWAPPED = '[CH3:1][CH2:2][OH:3]>>[CH3:2][CH:1]=[O:3]'
METHANOL = '[CH3:1][OH:2]>>[CH2:1]=[O:2]'
PARTIAL = '[CH3:1][Cl:2].[OH-:3]>>[CH3:1][OH:3]'
AMINE = '[CH3:1][NH2:2]>>[CH2:1]=[NH:2]'
BAD_AMINE = '[CH3:1][NH2:1]>>[CH2:1]=[NH:1]'
IDENTITY = '[CH3:1][CH3:2]>>[CH3:1][CH3:2]'
FLAGGED = '[CH3:1][CH2:2][CH2:3][OH:4]>>[CH3:1][CH2:2][CH:3]=[O:4]'
H_FIRST = '[H:3][C:1]([H:4])([H:5])[O:2][H:6]>>[H:3][C:1]([H:4])=[O:2].[H:5][H:6]'
H_OTHER = '[H:3][C:1]([H:4])([H:5])[O:2][H:6]>>[H:3][C:1]([H:6])=[O:2].[H:4][H:5]'


def renamed(value):
    return value.replace(':1]', ':101]').replace(':2]', ':102]').replace(':3]', ':103]')


@pytest.fixture
def sources(tmp_path):
    values = {'alpha': [ETHANOL, renamed(ETHANOL), METHANOL, renamed(METHANOL), PARTIAL,
                        BAD_AMINE, IDENTITY, FLAGGED, H_FIRST],
              'zeta': [SWAPPED, METHANOL, AMINE, H_OTHER]}
    entries = []
    input_root = tmp_path / 'sources'
    fields = ['r_id', 'original_id', 'rxn', 'ground_truth', 'role', 'mapping_curated', 'validation_status',
              'steps', 'center_cluster_id', 'source_parent_group']
    for name, reactions in values.items():
        folder = input_root / name
        folder.mkdir(parents=True)
        table = folder / 'aam.csv.gz'
        with gzip.open(table, 'wt', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for index, reaction in enumerate(reactions, 1):
                writer.writerow({'r_id': str(index), 'original_id': 'native-' + str(index),
                                 'rxn': chem.identities(reaction)['canonical_rxn'], 'ground_truth': reaction,
                                 'role': 'train' if name == 'alpha' else 'test',
                                 'mapping_curated': 'True' if name == 'zeta' else '',
                                 'validation_status': 'qc_failed' if reaction in [FLAGGED, IDENTITY] else '',
                                 'steps': '2', 'center_cluster_id': 'MC1', 'source_parent_group': 'mechanism1'})
        (folder / 'manifest.json').write_text(json.dumps({'files': {'aam.csv.gz': digest(table)}}))
        entries.append({'name': name, 'rows': len(reactions)})
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'schema': 'synrxn.aam-sources.v1', 'datasets': entries, 'expected_rows': 13}))
    audited = tmp_path / 'audit'
    assert audit.main(['--input-root', str(input_root), '--registry', str(registry),
                       '--output', str(audited), '--workers', '1']) == 0
    args = ['--input-root', str(input_root), '--registry', str(registry), '--audit', str(audited), '--workers', '1']
    return input_root, registry, audited, args


def test_unique_reactions_exclude_conflicts_and_choose_declared_curated_equivalent(sources, tmp_path):
    input_root, registry, audited, args = sources
    pins = {p: digest(p) for p in input_root.glob('*/aam.csv.gz')}
    output = tmp_path / 'unique'
    assert unique.main([*args, '--output', str(output)]) == 0
    summary = json.loads((output / 'summary.json').read_text())
    totals = summary['totals']
    assert (totals['source_records'], totals['unique_reactions'], totals['retained_reactions'],
            totals['diagnostic_reactions'], totals['reference_classes']) == (9, 5, 5, 3, 5)
    assert totals['input_source_records'] == 13 and totals['input_unique_reactions'] == 7
    assert totals['excluded_identity_reactions'] == totals['excluded_mapping_conflict_reactions'] == 1
    assert totals['excluded_reactions'] == 2 and totals['excluded_source_records'] == 4
    assert totals['mapping_conflict_reactions'] == 0 and totals['hydrogen_conflict_reactions'] == 1
    assert summary['additional_reference_graphs_checked'] == 2
    assert not summary['roles_used_for_grouping'] and not summary['new_splits_assigned']
    inventory = {r['rxn']: r for r in unique.csv_rows(output / 'reactions.csv.gz')}
    assert chem.identities(ETHANOL)['canonical_rxn'] not in inventory
    assert inventory[chem.identities(PARTIAL)['canonical_rxn']]['status'] == 'partial_mapping'
    assert inventory[chem.identities(AMINE)['canonical_rxn']]['status'] == 'unverifiable_reference'
    assert chem.identities(IDENTITY)['canonical_rxn'] not in inventory
    excluded = list(unique.csv_rows(output / 'excluded_reactions.csv.gz'))
    assert len(excluded) == 2
    identity = next(r for r in excluded if r['exclusion_reason'] == 'identical_canonical_endpoints')
    conflict = next(r for r in excluded if r['exclusion_reason'] == 'conflicting_heavy_atom_its')
    assert identity['rxn'] == chem.identities(IDENTITY)['canonical_rxn']
    assert conflict['rxn'] == chem.identities(ETHANOL)['canonical_rxn'] and conflict['heavy_its_classes'] == '2'
    removed_members = list(unique.csv_rows(output / 'excluded_members.csv.gz'))
    assert len(removed_members) == 4 and all(not r['reference_id'] for r in removed_members)
    assert next(r for r in removed_members if r['r_id'] == identity['r_id'])['source_exclusion'] == 'qc_failed'
    assert {r['heavy_class'] for r in removed_members if r['r_id'] == conflict['r_id']} == {'0', '1'}
    assert inventory[chem.identities(FLAGGED)['canonical_rxn']]['status'] == 'import_qc_flag'
    h_input = inventory[chem.identities(H_FIRST)['canonical_rxn']]
    assert h_input['status'] == 'comparable' and h_input['reference_comparable'] == 'True'
    assert h_input['hydrogen_disagreement'] == 'True' and json.loads(h_input['quality_flags']) == []
    assert 'reason_hydrogen_conflict' not in totals
    retained = {r['rxn']: r for r in unique.csv_rows(output / 'aam.csv.gz')}
    assert len(retained) == 5 and retained[chem.identities(H_FIRST)['canonical_rxn']]['mapping_scope'] == 'heavy_atoms'
    raw_partial = retained[chem.identities(PARTIAL)['canonical_rxn']]
    assert raw_partial['ground_truth'] == PARTIAL and raw_partial['mapping_scope'] == 'partial_heavy_atoms'
    raw_flagged = retained[chem.identities(FLAGGED)['canonical_rxn']]
    assert raw_flagged['ground_truth'] == FLAGGED and raw_flagged['source_validation_status'] == 'qc_failed'
    assert json.loads(raw_flagged['quality_flags']) == ['import_qc_flag']
    assert retained[chem.identities(AMINE)['canonical_rxn']]['ground_truth'] == AMINE
    methanol = retained[chem.identities(METHANOL)['canonical_rxn']]
    assert methanol['ground_truth'] == METHANOL
    assert methanol['representative_source'] == 'zeta'  # Smaller source wins on explicit curation evidence.
    assert methanol['source_records'] == '3' and 'role' not in methanol
    members = list(unique.csv_rows(output / 'members.csv.gz'))
    assert len(members) == 9 and {r['role'] for r in members} == {'train', 'test'}
    with gzip.open(output / 'provenance.jsonl.gz', 'rt') as handle:
        provenance = [json.loads(line) for line in handle]
    with gzip.open(output / 'excluded_provenance.jsonl.gz', 'rt') as handle:
        provenance += [json.loads(line) for line in handle]
    assert len(provenance) == 13
    assert all(r['source_record']['center_cluster_id'] == 'MC1' for r in provenance)
    assert all(r['source_record']['source_parent_group'] == 'mechanism1' for r in provenance)
    hydrogen = list(unique.csv_rows(output / 'hydrogen_conflicts.csv.gz'))
    assert len(hydrogen) == 2 and {r['ground_truth'] for r in hydrogen} == {H_FIRST, H_OTHER}
    assert {r['explicit_class'] for r in hydrogen} == {'0', '1'}
    assert {p: digest(p) for p in pins} == pins
    assert unique.main([*args, '--output', str(output), '--verify']) == 0
    second = tmp_path / 'unique-second'
    assert unique.main([*args, '--output', str(second)]) == 0
    for path in output.glob('*.gz'):
        assert digest(path) == digest(second / path.name), path.name
    with pytest.raises(ValueError, match='fresh output'):
        unique.main([*args, '--output', str(output)])


def test_singleton_graph_failure_retains_raw_target_with_unverified_scope(sources, tmp_path, monkeypatch):
    _, _, _, args = sources
    original = unique.prepare

    def failed(row):
        if row['ground_truth'] == FLAGGED:
            return row['ordinal'], None, None, 'graph_failed', 'Unsupported stereochemistry'
        return original(row)

    monkeypatch.setattr(unique, 'prepare', failed)
    output = tmp_path / 'failed'
    assert unique.main([*args, '--output', str(output)]) == 0
    row = next(r for r in unique.csv_rows(output / 'reactions.csv.gz')
               if r['rxn'] == chem.identities(FLAGGED)['canonical_rxn'])
    assert row['status'] == 'unverifiable_reference' and row['reference_comparable'] == 'False'
    member = next(r for r in unique.csv_rows(output / 'members.csv.gz') if r['comparison_status'] == 'graph_failed')
    assert member['comparison_error'] == 'Unsupported stereochemistry'
    target = next(r for r in unique.csv_rows(output / 'aam.csv.gz') if r['rxn'] == row['rxn'])
    assert target['ground_truth'] == FLAGGED and target['mapping_scope'] == 'unverified_heavy_atoms'
    assert target['comparison_status'] == 'graph_failed' and target['reference_id'].endswith(':raw')


def test_source_change_is_rejected_before_build(sources, tmp_path):
    input_root, _, _, args = sources
    with (input_root / 'alpha/aam.csv.gz').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='Source tables differ'):
        unique.main([*args, '--output', str(tmp_path / 'rejected')])


def test_verification_checks_complete_original_records(sources, tmp_path):
    input_root, registry, audited, args = sources
    output = tmp_path / 'unique'
    unique.main([*args, '--output', str(output)])
    path = output / 'provenance.jsonl.gz'
    with gzip.open(path, 'rt') as handle:
        records = [json.loads(line) for line in handle]
    records[0]['source_record']['source_parent_group'] = 'lost-original-mechanism'
    with gzip.open(path, 'wt') as handle:
        for row in records:
            handle.write(json.dumps(row) + '\n')
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest['files'][path.name] = digest(path)
    (output / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Original source record changed'):
        unique.verify(output, input_root, audited, registry)


def test_class_targets_and_raw_missing_labels_are_retained(tmp_path):
    from Test.reconstruction.families.representation.atom_mapping.test_aam_constraints import EQUIVALENT, SPECIFIC
    peroxide = '[O:1]=[O:1]>>[OH:1][OH:1]'
    partial = '[CH3:1]O>>[CH2:1]=O'
    values = {'classes': [EQUIVALENT, peroxide, partial], 'individual': [SPECIFIC]}
    input_root = tmp_path / 'source'
    entries = []
    for name, reactions in values.items():
        folder = input_root / name
        folder.mkdir(parents=True)
        table = folder / 'aam.csv.gz'
        with gzip.open(table, 'wt', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['r_id', 'rxn', 'ground_truth', 'original_id'])
            writer.writeheader()
            for i, value in enumerate(reactions):
                writer.writerow({'r_id': str(i), 'original_id': str(i),
                                 'rxn': chem.identities(value)['canonical_rxn'], 'ground_truth': value})
        (folder / 'manifest.json').write_text(json.dumps({'files': {'aam.csv.gz': digest(table)}}))
        entries.append({'name': name, 'rows': len(reactions)})
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'schema': 'synrxn.aam-sources.v1', 'datasets': entries, 'expected_rows': 4}))
    audited, output = tmp_path / 'audit', tmp_path / 'unique'
    common = ['--registry', str(registry), '--input-root', str(input_root), '--workers', '1']
    audit.main([*common, '--output', str(audited)])
    unique.main([*common, '--audit', str(audited), '--output', str(output)])
    retained = {r['rxn']: r for r in unique.csv_rows(output / 'aam.csv.gz')}
    assert len(retained) == 3
    assert retained[chem.identities(SPECIFIC)['canonical_rxn']]['ground_truth'] == SPECIFIC
    assert retained[chem.identities(peroxide)['canonical_rxn']]['mapping_scope'] == 'heavy_atom_equivalence_classes'
    missing = retained[chem.identities(partial)['canonical_rxn']]
    assert missing['mapping_scope'] == 'partial_heavy_atoms' and missing['ground_truth'] == partial
    assert chem.reference_info(missing['ground_truth'])['unmapped_heavy_atoms'] == 2
    diagnostic = list(unique.csv_rows(output / 'diagnostics.csv.gz'))
    assert len(diagnostic) == 1 and diagnostic[0]['status'] == 'partial_mapping'
    assert diagnostic[0]['unverifiable_records'] == '0'


def test_unverified_raw_targets_are_kept_and_valid_references_are_preferred(tmp_path):
    folder = tmp_path / 'sources' / 'example'
    folder.mkdir(parents=True)
    mismatched = '[NaH:1]>>[Na+:1]'
    records = [
        dict(r_id='1', rxn=chem.identities(ETHANOL)['canonical_rxn'], ground_truth=METHANOL, original_id='bad-pair'),
        dict(r_id='2', rxn=chem.identities(ETHANOL)['canonical_rxn'], ground_truth=ETHANOL, original_id='valid-pair'),
        dict(r_id='3', rxn=chem.identities(BAD_AMINE)['canonical_rxn'], ground_truth=BAD_AMINE, original_id='mixed'),
        dict(r_id='4', rxn='[Na]>>[Na+]', ground_truth=mismatched, original_id='mismatch'),
    ]
    with gzip.open(folder / 'aam.csv.gz', 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    table_pin = digest(folder / 'aam.csv.gz')
    (folder / 'manifest.json').write_text(json.dumps({'files': {'aam.csv.gz': table_pin}}))
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'schema': 'synrxn.aam-sources.v1',
                                    'datasets': [{'name': 'example', 'rows': 4}], 'expected_rows': 4}))
    audited, output = tmp_path / 'audit', tmp_path / 'unique'
    common = ['--registry', str(registry), '--input-root', str(folder.parent), '--workers', '1']
    audit.main([*common, '--output', str(audited)])
    unique.main([*common, '--audit', str(audited), '--output', str(output)])
    retained = {r['representative_r_id']: r for r in unique.csv_rows(output / 'aam.csv.gz')}
    assert set(retained) == {'2', '3', '4'}
    assert retained['2']['ground_truth'] == ETHANOL and retained['2']['comparison_status'] == 'ok'
    assert retained['3']['ground_truth'] == BAD_AMINE
    assert retained['3']['comparison_status'] == 'mapped_atom_identity_mismatch'
    assert retained['4']['ground_truth'] == mismatched and retained['4']['rxn'] == '[Na]>>[Na+]'
    assert retained['4']['comparison_status'] == 'input_target_endpoint_mismatch'
    for row in [retained['3'], retained['4']]:
        assert row['mapping_scope'] == 'unverified_heavy_atoms' and row['reference_id'].endswith(':raw')
    totals = json.loads((output / 'summary.json').read_text())['totals']
    assert totals['retained_reactions'] == totals['diagnostic_reactions'] == 3
    assert totals['reference_comparable_reactions'] == 1 and totals['raw_only_reference_reactions'] == 2
    assert totals['reference_classes'] == 1 and totals['reference_entries'] == 3
    assert digest(folder / 'aam.csv.gz') == table_pin
    assert unique.main([*common, '--audit', str(audited), '--output', str(output), '--verify']) == 0
