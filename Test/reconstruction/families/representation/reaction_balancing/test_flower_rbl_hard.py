"""Check native spectator recovery and reactant-only hard balancing context."""
import csv
import gzip
import json
import zipfile

from synrxn.reconstruction.families.representation.reaction_balancing._sources import flower_rbl_hard as hard
from synrxn.reconstruction.families.representation.reaction_balancing._sources.flower_rbl import make_cases


def fixture_parent():
    from synrxn.reconstruction.families.representation.reaction_balancing._sources.flower_rbl import TRACE_FIELDS
    row = {field: '' for field in TRACE_FIELDS}
    row.update(r_id='7', original_id='source', role='test', steps='1', center_type='RC',
               center_cluster_id='center-2', source_group='9', source_lines='[1]',
               source_member='inputs/test.txt.gz', source_split='test',
               ground_truth='[CH3:1][CH2:2][OH:3].[OH2:4]>>[CH3:1][O:3][CH3:2].[OH2:4]')
    return row


def test_native_context_is_unchanged_outside_parent_pool_and_carries_original_maps():
    parent = fixture_parent()
    left, right = parent['ground_truth'].split('>>')
    _, context = hard.recover_context((parent, left + '.[CH3:10][OH:11].[Na+:12].[H-:13]',
                                      right + '.[H-:13].[Na+:12].[CH3:10][OH:11]'))
    assert context['context_count'] == 3 and context['input_context_count'] == 2
    assert context['context_hydrogens'] == ['[H-]']
    assert context['input_context_rxn'] == 'CO.[Na+]'
    assert hard.mapped_pool(context['gt_aam'].split('>>')[0]) == hard.mapped_pool(context['gt_aam'].split('>>')[1])
    assert {10, 11, 12, 13} <= set(hard.mapped_pool(context['gt_aam'].split('>>')[0]))


def test_hard_input_adds_heavy_context_to_reactants_only_and_keeps_core_removals():
    parent = fixture_parent()
    left, right = parent['ground_truth'].split('>>')
    _, context = hard.recover_context((parent, left + '.[CH3:10][OH:11].[Na+:12].[H-:13]',
                                      right + '.[CH3:10][OH:11].[Na+:12].[H-:13]'))
    bases, _ = make_cases(parent)
    for base in bases:
        row = hard.hard_case(base, context)
        before_left, before_right = base['rxn'].split('>>')
        new_left, new_right = row['rxn'].split('>>')
        assert new_right == before_right
        assert set(before_left.split('.')) <= set(new_left.split('.'))
        assert 'CO' in new_left.split('.') and '[Na+]' in new_left.split('.')
        assert '[H-]' not in row['rxn']
        assert row['base_r_id'] == base['r_id'] and row['parent_r_id'] == base['parent_r_id']
        for field in ('role', 'case_type', 'center_cluster_id', 'source_lines', 'removed_reactant', 'removed_product'):
            assert row[field] == base[field]
        assert row['gt_rxn'] == row['ground_truth']
        assert '[H-]' in row['gt_rxn'].split('>>')[0] and '[H-]' in row['gt_rxn'].split('>>')[1]
        assert json.loads(row['removed_hydrogen_reactants']) == ['[H-]']
        assert row['has_context'] and row['context_missing_from_products']


def test_no_native_context_keeps_case_identity_without_inventing_distractors():
    parent = fixture_parent()
    left, right = parent['ground_truth'].split('>>')
    _, context = hard.recover_context((parent, left, right))
    base = make_cases(parent)[0][0]
    row = hard.hard_case(base, context)
    assert context['context_count'] == 0
    assert not row['has_context']
    assert row['rxn'] == base['rxn'] and row['gt_aam'] == base['gt_aam']


def test_changed_source_component_is_not_mislabeled_as_unchanged_context():
    parent = fixture_parent()
    left, right = parent['ground_truth'].split('>>')
    _, context = hard.recover_context((parent, left + '.[CH3:10][OH:11]', right + '.[CH:10]=[O:11]'))
    assert context['context_count'] == 0


def test_identical_parent_molecule_with_different_maps_is_kept_as_separate_context():
    parent = fixture_parent()
    left, right = parent['ground_truth'].split('>>')
    _, context = hard.recover_context((parent, left + '.[OH2:10]', right + '.[OH2:10]'))
    assert context['context_count'] == context['input_context_count'] == 1
    assert context['input_context_rxn'] == 'O'
    assert ':4]' in context['gt_aam'] and ':10]' in context['gt_aam']


def test_hard_bundle_preserves_base_table_and_rebuilds_identically(tmp_path):
    from synrxn.reconstruction.families.representation.reaction_balancing._sources.flower_rbl import build as build_base
    source, base, output = tmp_path / 'source', tmp_path / 'base', tmp_path / 'hard'
    source.mkdir()
    parent = fixture_parent()
    with gzip.open(source / 'aam.csv.gz', 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parent))
        writer.writeheader()
        writer.writerow(parent)
    (source / 'source.json').write_text(json.dumps({'rights': {'license_expression': 'MIT'}}))
    (source / 'licenses').mkdir()
    (source / 'licenses' / 'LICENSE.txt').write_text('MIT fixture evidence')
    build_base(source, base)
    before = (base / 'rbl.csv.gz').read_bytes()
    left, right = parent['ground_truth'].split('>>')
    raw = f'{left}.[CH3:10][OH:11]>>{right}.[CH3:10][OH:11]|9\n'
    dump = tmp_path / 'flower.zip'
    with zipfile.ZipFile(dump, 'w') as archive:
        archive.writestr('inputs/test.txt.gz', gzip.compress(raw.encode(), mtime=0))
    report = hard.build(source, base, dump, output, workers=1)
    assert report['rows'] == 3 and report['context_counts'] == {'augmented': 3}
    assert (base / 'rbl.csv.gz').read_bytes() == before
    with gzip.open(output / 'rbl.csv.gz', 'rt') as handle:
        rows = list(csv.DictReader(handle))
    assert all(row['has_context'] == 'True' and row['role'] == 'test' for row in rows)
    assert len({row['base_r_id'] for row in rows}) == 3
    first = (output / 'rbl.csv.gz').read_bytes()
    hard.build(source, base, dump, output, workers=1)
    assert (output / 'rbl.csv.gz').read_bytes() == first
