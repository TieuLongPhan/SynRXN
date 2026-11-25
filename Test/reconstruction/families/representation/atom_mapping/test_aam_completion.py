"""Check that completion cannot silently alter endpoints or specified pairs."""
from types import SimpleNamespace
import signal

import pytest

from synrxn.reconstruction.families.representation.atom_mapping._sources import complete_aam_maps as completion
from synrxn.reconstruction.families.representation.atom_mapping._sources.audit_aam_partial_canon import shape
from synrxn.reconstruction.families.representation.atom_mapping._sources import aam_overlap_chem as chem


PARTIAL = 'C[CH2:2][OH:3]>>C[CH:2]=[O:3]'
ETHANOL = '[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]'
SWAPPED = '[CH3:1][CH2:2][OH:3]>>[CH3:2][CH:1]=[O:3]'


def task(value, reasons='["partial_mapping"]'):
    row = dict(r_id='aam_' + chem.identities(value)['exact_key'], dataset='example', source_r_id='1',
               rxn=chem.identities(value)['canonical_rxn'], ground_truth=value,
               review_reasons=reasons, **shape(value))
    return row, [value], 5


def test_fill_labels_preserves_existing_ids_without_inventing_pairs():
    value, count = completion.label_heavy_atoms(PARTIAL)
    assert count == 2
    info = chem.reference_info(value)
    assert info['mapping_status'] == 'partial' and info['unmapped_heavy_atoms'] == 0
    numbers = [[a.GetAtomMapNum() for a in m.GetAtoms()] for m in chem.parse_reaction(value)]
    assert set(numbers[0]) & set(numbers[1]) == {2, 3}
    assert sorted(numbers[0]) == [2, 3, 4] and sorted(numbers[1]) == [2, 3, 5]
    assert info['target_rxn'] == chem.reference_info(PARTIAL)['target_rxn']


def test_full_completion_preserves_known_mapping_and_source_qc():
    result = completion.complete(task(PARTIAL, '["partial_mapping","source_qc_excluded"]'))
    assert result['outcome'] == 'completed'
    assert completion.validate_completion(result['completed_ground_truth'], result['rxn'], [PARTIAL])
    assert result['checked_source_references'] == 1
    record = completion.record(result, completed=True)
    assert record['reference_tier'] == 'computational_completion'
    assert record['source_review_reasons'] == '["partial_mapping","source_qc_excluded"]'
    assert result['original_ground_truth'] == PARTIAL


def test_unbalanced_endpoints_are_not_completed_by_adding_atoms(monkeypatch):
    from synkit.Graph.ITS.its_expand import ITSExpand

    def unexpected(*args, **kwargs):
        raise AssertionError('Unbalanced endpoint inventory must be caught first')

    monkeypatch.setattr(ITSExpand, 'expand_aam_with_its_report', unexpected)
    source = '[CH3:1][Cl:2].[OH-:3]>>[CH3:1][OH:3]'
    result = completion.complete(task(source))
    assert result['outcome'] == 'unbalanced_heavy_inventory'
    assert result['completed_ground_truth'] == ''
    assert result['labelled_ground_truth'] == source


def test_validation_rejects_endpoint_changes_and_changed_known_pairs():
    rxn = chem.identities(ETHANOL)['canonical_rxn']
    assert not completion.validate_completion(SWAPPED, rxn, [ETHANOL])
    assert not completion.validate_completion(ETHANOL + '.[Cl-:4]', rxn, [ETHANOL])
    assert not completion.validate_completion(PARTIAL, rxn, [PARTIAL])


def test_disagreement_between_completion_directions_is_not_selected(monkeypatch):
    from synkit.Graph.ITS.its_expand import ITSExpand

    def direction(value, *, use_G, **kwargs):
        result = ETHANOL if use_G else SWAPPED
        return SimpleNamespace(rsmi=result, to_dict=lambda: dict(rsmi=result))

    monkeypatch.setattr(ITSExpand, 'expand_aam_with_its_report', direction)
    source = 'CC[OH:3]>>CC=[O:3]'
    result = completion.complete(task(source))
    assert result['outcome'] == 'ambiguous_completion'
    assert result['completed_ground_truth'] == ''


def test_class_valued_reference_is_not_collapsed_by_expansion_adapter():
    with pytest.raises(ValueError, match='Repeated mapping classes'):
        completion.expansion_input('[O:1]=[O:1]>>[OH:1][OH:1]')


def test_timeout_escapes_library_exception_wrapping(monkeypatch):
    from synkit.Graph.ITS.its_expand import ITSExpand

    def library_handler(*args, **kwargs):
        try:
            signal.raise_signal(signal.SIGALRM)
        except Exception:
            raise AssertionError('The timeout must escape a library Exception handler')

    monkeypatch.setattr(ITSExpand, 'expand_aam_with_its_report', library_handler)
    result = completion.complete(task(PARTIAL))
    assert result['outcome'] == 'timeout' and result['completed_ground_truth'] == ''


def test_single_sided_ids_may_be_relabelled_but_paired_ids_are_retained():
    source = '[CH3:8][CH2:2][OH:3]>>[CH3:9][CH:2]=[O:3]'
    work = completion.expansion_input(source)
    mols = chem.parse_reaction(work)
    assert [{a.GetAtomMapNum() for a in m.GetAtoms()} for m in mols] == [{0, 2, 3}, {0, 2, 3}]
    result = completion.complete(task(source))
    assert result['outcome'] == 'completed'
    assert completion.validate_completion(result['completed_ground_truth'], result['rxn'], [source])
