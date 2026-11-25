"""Check shared elementary/multistep center labels and source-linked annotations."""
import csv
import gzip

from synrxn.reconstruction.families.representation.atom_mapping._sources import flower_center_groups as groups
from synrxn.reconstruction.families.representation.atom_mapping._sources.flower_mechanistic_center import ExactClusters, mechanistic_center, pathway_mtg


def test_one_step_and_multistep_use_one_global_label_schema():
    """Independent local changes have identical MCs whether concerted or sequential."""
    first = '[CH3:1][CH2:2][OH:3].[CH3:4][CH2:5][OH:6]'
    middle = '[CH3:1][CH:2]=[O:3].[CH3:4][CH2:5][OH:6]'
    final = '[CH3:1][CH:2]=[O:3].[CH3:4][CH:5]=[O:6]'
    elementary = groups.single_step_center(first + '>>' + final)
    composite = mechanistic_center(pathway_mtg([first + '>>' + middle, middle + '>>' + final]))[0]
    classes = ExactClusters()
    assert classes.assign(elementary) == classes.assign(composite)


def test_one_step_labels_ignore_consistent_map_renumbering():
    first = '[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]'
    second = first.replace(':1]', ':101]').replace(':2]', ':102]').replace(':3]', ':103]')
    classes = ExactClusters()
    assert classes.assign(groups.single_step_center(first)) == classes.assign(groups.single_step_center(second))


def test_overlap_excludes_roles_and_step_counts_from_cluster_identity():
    records = [{'role': 'train', 'steps': 1, 'center_cluster_id': 'center-0'},
               {'role': 'test', 'steps': 2, 'center_cluster_id': 'center-0'},
               {'role': 'val', 'steps': 3, 'center_cluster_id': 'center-1'}]
    assert groups.overlap(records)['pairwise'] == {'train_val': 0, 'train_test': 1, 'val_test': 0}


def test_annotation_csv_round_trip_keeps_source_line_lists(tmp_path):
    row = {key: '' for key in groups.FIELDS}
    row.update(r_id='2', center_cluster_id='center-000001', source_lines='[1,2]', source_group='17::pd1')
    path = tmp_path / 'centers.csv.gz'
    groups.write_csv(path, [row], groups.FIELDS)
    with gzip.open(path, 'rt') as handle:
        assert list(csv.DictReader(handle)) == [row]
