"""Verify first-reference selection, preserved memberships and reproducible exports."""
import csv
import gzip
import json

import pytest

from synrxn.reconstruction.families.representation.atom_mapping._sources import build_metamdb_aam as builder

KETO = '[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]'
ENOL = '[CH3:1][CH2:2][OH:3]>>[CH2:1]=[CH:2][OH:3]'


def case(identifier, references, role=None):
    row = {'original_id': str(identifier), 'source_row': identifier,
           'selected_reference': references}
    if role is not None:
        row.update(role=role, role_row=identifier)
    return row


def test_first_reference_is_used_for_train_and_test_without_fallback():
    for role in ['train', 'test']:
        row, source, _, _ = builder.process_case('metambd_cyclic', case(1, KETO + ',' + ENOL, role))
        expected, _, _, _ = builder.process_case('metambd_cyclic', case(1, KETO, role))
        assert row == expected
        assert source['selected_reference'] == KETO
        rejected, _, status, _ = builder.process_case('metambd_cyclic', case(1, 'bad,' + KETO, role))
        assert rejected is None and status == 'parse_or_sanitization_failed'


def test_generated_first_tautomer_is_not_replaced_by_canonical_tautomer():
    first = builder.first_ring_reference(KETO)
    record, _, _, _ = builder.process_case('metambd_cyclic', case(1, first, 'train'))
    expected, _, _, _ = builder.process_case('metambd_cyclic', case(1, ENOL, 'train'))
    assert record['ground_truth'] == expected['ground_truth']
    assert record['rxn'] == expected['rxn']


def test_identical_reactions_remain_separate_and_rebuild_bytes_are_stable(tmp_path):
    output = tmp_path / 'native-product'
    cases = [case(1, KETO), case(2, KETO), case(3, 'bad')]
    report = builder.build('metamdb', iter(cases), output, {'sha256': 'fixture'})
    with gzip.open(output / 'aam.csv.gz', 'rt') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert 'role' not in reader.fieldnames
        assert 'ground_truth_variants' not in reader.fieldnames
    assert report['rows'] == 2 and report['deduplication'].startswith('none')
    assert rows[0]['rxn'] == rows[1]['rxn']
    assert rows[0]['r_id'] != rows[1]['r_id']
    assert all(row['validation_status'] == 'human_review_unverified' for row in rows)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    builder.build('metamdb', iter(cases), output, {'sha256': 'fixture'})
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    manifest = json.loads((output / 'manifest.json').read_text())
    assert len(manifest['files']) == 5


def test_roles_and_source_records_are_retained(tmp_path):
    output = tmp_path / 'native-product'
    cases = [case(1, KETO + ',' + ENOL, 'train'), case(1, KETO + ',' + ENOL, 'test')]
    report = builder.build('metambd_cyclic', iter(cases), output, {'sha256': 'fixture'})
    with gzip.open(output / 'aam.csv.gz', 'rt') as f:
        rows = list(csv.DictReader(f))
    assert [row['role'] for row in rows] == ['train', 'test']
    assert report['by_role'] == {'train': 1, 'test': 1}
    assert len({row['ground_truth'] for row in rows}) == 1
    with gzip.open(output / 'native_provenance.jsonl.gz', 'rt') as f:
        source = [json.loads(line) for line in f]
    assert all(row['selected_reference'] == KETO for row in source)
    assert all(row['selected_reference_index'] == 0 for row in source)


def test_unmanaged_output_is_preserved(tmp_path):
    (tmp_path / 'important.txt').write_text('keep')
    with pytest.raises(ValueError, match='unmanaged'):
        builder.build('metamdb', iter([]), tmp_path, {})
    assert (tmp_path / 'important.txt').read_text() == 'keep'


def test_paper_subset_never_falls_back_to_full_archive(tmp_path):
    with pytest.raises(ValueError, match='exact 4,456-reaction paper subset'):
        builder.main(['--datasets', 'metamdb', '--output', str(tmp_path / 'metamdb')])
    assert not (tmp_path / 'metamdb').exists()
