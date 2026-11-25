"""Check the uniform cyclic AAM processor and full ITS candidate equivalence."""
import csv
import gzip
import io
import json
import zipfile

import pytest

from synrxn.reconstruction.families.representation.atom_mapping._sources import process_metambd_cyclic as PROCESSOR

KETO = '[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]'
ENOL = '[CH3:1][CH2:2][OH:3]>>[CH2:1]=[CH:2][OH:3]'
RENAMED = KETO.replace(':1]', ':101]').replace(':2]', ':102]').replace(':3]', ':103]')


def source_case(split, row, candidates):
    return {'split': split, 'source_row': row, 'archive_member': f'ringreactions/{split}_ringreactions.csv',
            'candidate_strings': candidates}


def test_distinct_tautomers_remain_distinct_targets():
    """First-candidate standardization must not rewrite a target's tautomer form."""
    keto, enol = [PROCESSOR.reaction_views(value) for value in (KETO, ENOL)]
    assert keto['rxn'] != enol['rxn']
    assert keto['ground_truth'] != enol['ground_truth']


def test_proton_participant_is_retained_without_invented_hydrogen_mapping():
    """Conserve heavy-atom references while projecting H-only species out of ITS."""
    result = PROCESSOR.reaction_views('[CH3:1][OH:2]>>[CH3:1][O-:2].[H+:3]')
    assert result['hydrogen_map_mismatch']
    assert '[H+]' in result['ground_truth'] and '[H+]' in result['rxn']
    assert ':3]' not in result['ground_truth']
    assert '[H+]' not in result['its_ground_truth']


def test_changed_heavy_atom_identity_is_rejected():
    with pytest.raises(ValueError, match='heavy_atom_mapping_not_conserved'):
        PROCESSOR.reaction_views('[CH3:1][OH:2]>>[CH3:1][NH2:2]')


def test_mapped_spectators_are_removed_from_inputs_and_references():
    result = PROCESSOR.reaction_views(KETO.replace('>>', '.[Na+:4]>>') + '.[Na+:4]')
    assert result['spectator_pairs_removed'] == 1
    assert result['rxn'] == PROCESSOR.reaction_views(KETO)['rxn']
    assert '[Na+' not in result['ground_truth']


def test_full_its_ignores_map_numbers_but_preserves_tautomers_and_context():
    """Full graph equality must not merge different endpoints or merely shared RCs."""
    classes = PROCESSOR.ITSClasses()
    longer = '[CH3:4][CH2:1][CH2:2][OH:3]>>[CH3:4][CH2:1][CH:2]=[O:3]'
    assert [classes.assign(PROCESSOR.full_its(value), ((), ()))
            for value in (KETO, RENAMED, ENOL, longer)] == [0, 0, 1, 2]


def test_full_its_preserves_stereochemistry():
    """Renumbering preserves class identity; opposite stereochemistry does not."""
    first = '[CH3:1][C@H:2]([OH:3])[CH2:4][OH:5]>>[CH3:1][C@H:2]([OH:3])[CH:4]=[O:5]'
    renamed = first
    for number in range(1, 6):
        renamed = renamed.replace(f':{number}]', f':{number + 100}]')
    classes = PROCESSOR.ITSClasses()
    assert [classes.assign(PROCESSOR.full_its(value), ((), ()))
            for value in (first, renamed, first.replace('@H', '@@H'))] == [0, 0, 1]


def test_full_its_preserves_isotopes_and_absolute_hydrogen_counts():
    classes = PROCESSOR.ITSClasses()
    isotope = KETO.replace('[CH3:1]', '[13CH3:1]')
    assert classes.assign(PROCESSOR.full_its(KETO), ((), ())) == 0
    assert classes.assign(PROCESSOR.full_its(isotope), ((), ())) == 1
    assert classes.assign(PROCESSOR.full_its(ENOL), ((), ())) == 2


def test_proton_counts_supplement_heavy_atom_graph_matching():
    classes = PROCESSOR.ITSClasses()
    graph = PROCESSOR.full_its(KETO)
    assert classes.assign(graph, PROCESSOR.hydrogen_only_key(KETO)) == 0
    assert classes.assign(graph, PROCESSOR.hydrogen_only_key(KETO + '.[H+]')) == 1


def test_one_pipeline_keeps_train_first_and_test_candidates_in_original_case():
    """Both splits share the record schema while preserving their selection rules."""
    candidates = [KETO, RENAMED, ENOL, 'truncated']
    train, train_source, train_stats, _ = PROCESSOR.process_case(source_case('train', 1, candidates))
    test, test_source, test_stats, failures = PROCESSOR.process_case(source_case('test', 1, candidates))
    assert set(train) == set(test) == set(PROCESSOR.FIELDS)
    assert train['rxn'] == test['rxn'] == PROCESSOR.reaction_views(KETO)['rxn']
    assert train['r_id'] != test['r_id']
    assert json.loads(train['ground_truth_variants']) == [train['ground_truth']]
    assert json.loads(test['ground_truth_variants']) == [
        PROCESSOR.reaction_views(v)['ground_truth'] for v in (KETO, ENOL)]
    assert train_source['candidate_strings'] == test_source['candidate_strings'] == candidates
    assert train_source['candidate_memberships'][1]['status'] == 'not_selected'
    assert [m['representative_source_column'] for m in test_source['candidate_memberships']] == [1, 1, 3, None]
    assert test_source['candidate_memberships'][-1]['status'] == 'invalid_source'
    assert train_stats['mapping_references'] == 1 and test_stats['mapping_references'] == 2
    assert failures[0]['source_column'] == 4
    repeated, _, _, _ = PROCESSOR.process_case(source_case('test', 2, candidates))
    assert repeated['rxn'] == test['rxn'] and repeated['r_id'] != test['r_id']


def test_invalid_first_candidate_is_never_replaced_by_a_later_candidate():
    for split in ('train', 'test'):
        with pytest.raises(ValueError, match='invalid first candidate'):
            PROCESSOR.process_case(source_case(split, 1, ['truncated', KETO]))


def test_archive_to_combined_table_preserves_headerless_rows_and_rebuild_bytes(tmp_path):
    """Exercise both splits, bounded workers, output schema and complete provenance."""
    archive = tmp_path / 'ringreactions.zip'
    with zipfile.ZipFile(archive, 'w') as target:
        for split, rows in [('train', [[KETO, ENOL], [KETO]]), ('test', [[KETO, RENAMED, ENOL, 'bad']])]:
            text = io.StringIO()
            csv.writer(text).writerows(rows)
            target.writestr(f'ringreactions/{split}_ringreactions.csv', text.getvalue())
    cases = list(PROCESSOR.archive_cases(archive))
    assert [(c['split'], c['source_row']) for c in cases] == [('train', 1), ('train', 2), ('test', 1)]
    assert cases[0]['candidate_strings'] == [KETO, ENOL]
    output, provenance, report = [tmp_path / name for name in ('table.csv.gz', 'candidates.jsonl.gz', 'report.json')]
    summary = PROCESSOR.build(archive, output, provenance, report, workers=1, batch_size=1)
    with gzip.open(output, 'rt') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == PROCESSOR.FIELDS
    assert len(rows) == summary['canonical_rows'] == 3
    assert [r['split'] for r in rows] == ['train', 'train', 'test']
    assert len({r['r_id'] for r in rows}) == 3
    assert len(json.loads(rows[2]['ground_truth_variants'])) == 2
    with gzip.open(provenance, 'rt') as handle:
        sources = [json.loads(line) for line in handle]
    assert sources[2]['candidate_strings'] == [KETO, RENAMED, ENOL, 'bad']
    assert summary['counts']['test']['source_cells'] == 4
    before = output.read_bytes(), provenance.read_bytes()
    PROCESSOR.build(archive, output, provenance, report, workers=2, batch_size=2)
    assert before == (output.read_bytes(), provenance.read_bytes())
