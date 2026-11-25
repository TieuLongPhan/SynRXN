"""Check hydrogen stripping, smallest-heavy removal and traceable RBL cases."""
import csv
import gzip
import json

import pytest

from synrxn.reconstruction.families.representation.reaction_balancing._sources import flower_rbl as rbl


def parent(reaction):
    row = {field: '' for field in rbl.TRACE_FIELDS}
    row.update(r_id='7', original_id='source-id', ground_truth=reaction,
               role='test', steps='2', center_cluster_id='center-3', center_type='MC',
               source_lines='[8,9]', source_group='12', source_member='inputs/test.txt.gz')
    return row


def test_all_hydrogen_only_species_are_removed_on_both_sides_then_one_heavy_molecule():
    reference = '[H+].[H-].O.CCO>>[H][H].O.CCO'
    cases, skipped = rbl.make_cases(parent(reference))
    assert not skipped and [case['case_type'] for case in cases] == ['reactant', 'product', 'both']
    for case in cases:
        assert json.loads(case['removed_hydrogen_reactants']) == ['[H+]', '[H-]']
        assert json.loads(case['removed_hydrogen_products']) == ['[H][H]']
        for side in case['rxn'].split('>>'):
            assert all(component['heavy_atoms'] > 0 for component in rbl.components(side))
        assert '[H+]' in case['ground_truth'] and '[H-]' in case['ground_truth']
        assert case['parent_r_id'] == '7' and case['role'] == 'test'
        assert case['center_cluster_id'] == 'center-3' and case['source_lines'] == '[8,9]'
    assert cases[0]['removed_reactant'] == 'O' and cases[0]['removed_product'] == ''
    assert cases[1]['removed_reactant'] == '' and cases[1]['removed_product'] == 'O'
    assert cases[2]['rxn'] == 'CCO>>CCO'


def test_cases_that_remain_balanced_are_kept_and_flagged():
    cases, _ = rbl.make_cases(parent('CCO.O>>COC.O'))
    both = next(case for case in cases if case['case_type'] == 'both')
    assert both['rxn'] == 'CCO>>COC'
    assert both['input_element_balanced'] and both['input_charge_balanced'] and both['input_balanced']
    assert all(not case['input_element_balanced'] for case in cases if case['case_type'] != 'both')


def test_explicit_ground_truth_fields_preserve_parent_mapping_and_legacy_alias():
    reference = '[CH3:1][CH2:2][OH:3].[OH2:4]>>[CH3:1][O:3][CH3:2].[OH2:4]'
    cases, _ = rbl.make_cases(parent(reference))
    expected = '>>'.join(rbl.side_text(rbl.components(side)) for side in reference.split('>>'))
    assert len(cases) == 3
    for case in cases:
        assert case['gt_aam'] == reference
        assert case['gt_rxn'] == case['ground_truth'] == expected
        assert ':' not in case['gt_rxn'] and ':' not in case['rxn']
        assert ':1]' in case['gt_aam'] and ':4]' in case['gt_aam']
        assert case['role'] == 'test' and case['center_cluster_id'] == 'center-3'


def test_only_one_occurrence_of_a_selected_heavy_component_is_removed():
    cases, _ = rbl.make_cases(parent('CCO.O.O>>CCO.O.O'))
    assert cases[0]['rxn'] == 'CCO.O>>CCO.O.O'
    assert cases[2]['rxn'] == 'CCO.O>>CCO.O'


def test_empty_sides_are_excluded_and_unbalanced_targets_are_rejected():
    cases, skipped = rbl.make_cases(parent('CCO>>CC=O.[H][H]'))
    assert not cases and skipped == {'reactant': 1, 'product': 1, 'both': 1}
    with pytest.raises(ValueError, match='not element/charge balanced'):
        rbl.make_cases(parent('CCO>>CC=O'))


def test_smallest_is_heavy_atom_count_then_total_atoms_and_is_deterministic():
    side = rbl.components('CCO.O.C')
    assert min(side, key=lambda value: value['size_key'])['smiles'] == 'O'
    first, _ = rbl.make_cases(parent('CCO.O.C>>CCO.O.C'))
    second, _ = rbl.make_cases(parent('C.O.CCO>>C.O.CCO'))
    assert [(r['rxn'], r['ground_truth']) for r in first] == [(r['rxn'], r['ground_truth']) for r in second]


def test_build_writes_one_case_type_dataset_with_licenses_and_repeatable_bytes(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    record = parent('[H+].[H-].O.CCO>>[H][H].O.CCO')
    with gzip.open(source / 'aam.csv.gz', 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    (source / 'source.json').write_text(json.dumps({'rights': {'license_expression': 'MIT'}}))
    (source / 'licenses').mkdir()
    (source / 'licenses' / 'LICENSE.txt').write_text('Fixture license evidence')
    report = rbl.build(source, output)
    assert report['rows'] == 3 and report['by_case_type'] == {'reactant': 1, 'product': 1, 'both': 1}
    assert report['rejected_parents'] == []
    with gzip.open(output / 'rbl.csv.gz', 'rt') as handle:
        rows = list(csv.DictReader(handle))
    assert {row['case_type'] for row in rows} == set(rbl.CASE_TYPES)
    assert {row['parent_r_id'] for row in rows} == {'7'}
    assert {row['role'] for row in rows} == {'test'}
    assert all(row['gt_rxn'] == row['ground_truth'] for row in rows)
    assert all(row['gt_aam'] == record['ground_truth'] for row in rows)
    assert report['schema'] == 'synrxn.flower-rbl.v2'
    assert (output / 'licenses' / 'LICENSE.txt').read_text() == 'Fixture license evidence'
    before = (output / 'rbl.csv.gz').read_bytes()
    rbl.build(source, output)
    assert (output / 'rbl.csv.gz').read_bytes() == before


def test_previous_rbl_schema_can_be_upgraded_without_losing_managed_output(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    output.mkdir()
    record = parent('CCO.O>>COC.O')
    with gzip.open(source / 'aam.csv.gz', 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    (source / 'source.json').write_text(json.dumps({'rights': {'license_expression': 'MIT'}}))
    (output / 'manifest.json').write_text(json.dumps({'schema': 'synrxn.flower-rbl.v1'}))
    report = rbl.build(source, output)
    assert report['rows'] == 3
    assert json.loads((output / 'manifest.json').read_text())['schema'] == 'synrxn.flower-rbl.v2'
