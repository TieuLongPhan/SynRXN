"""Check set-valued mappings, partial information and certified disagreement."""
import json

import pytest

from synrxn.reconstruction.families.representation.atom_mapping._sources import audit_aam_overlap as audit
from synrxn.reconstruction.families.representation.atom_mapping._sources import aam_overlap_chem as chem
from synrxn.reconstruction.families.representation.atom_mapping._sources import aam_mapping_constraints as constraints


SPECIFIC = '[O:1]=[C:2]=[O:3].[OH2:4]>>[O:1]=[C:2]([OH:3])[OH:4]'
EQUIVALENT = '[O:1]=[C:2]=[O:1].[OH2:1]>>[O:1]=[C:2]([OH:1])[OH:1]'
ETHANOL = '[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]'
SWAPPED = '[CH3:1][CH2:2][OH:3]>>[CH3:2][CH:1]=[O:3]'
COARSE = '[CH3:1][CH2:1][OH:3]>>[CH3:1][CH:1]=[O:3]'


def graph(value):
    return chem.its_graph(chem.reference_info(value)['mols'])


def classify(values):
    rows = []
    for i, value in enumerate(values):
        row = audit.index_record({'dataset': f'source{i}', 'r_id': str(i), 'original_id': str(i),
                                  'source_row': 1, 'role': 'train', 'steps': '',
                                  'rxn': chem.identities(value)['canonical_rxn'], 'ground_truth': value})
        row['ordinal'] = i
        prepared = audit.graph_record(row)
        row.update(dict(zip(['heavy_full', 'heavy_normalized', 'explicit_full', 'explicit_normalized',
                             'graph_status', 'graph_error'], prepared[1:])))
        rows.append(row)
    return audit.classify(rows, 'exact', 5)


def test_repeated_labels_are_classes_and_never_merge_endpoint_atoms():
    info = chem.reference_info(EQUIVALENT)
    assert info['mapping_status'] == 'equivalent' and info['mapping_encoding'] == 'equivalence_classes'
    value = graph(EQUIVALENT)
    assert value.graph['kind'] == 'classes'
    assert sum(json.loads(a['label'])[0] == 'atom' for _, a in value.nodes(data=True)) == 8
    assert len(constraints.Constraints(value).graph) == 8
    renamed = EQUIVALENT.replace(':1]', ':71]').replace(':2]', ':72]')
    classes = chem.ITSClasses()
    assert classes.assign(value) == classes.assign(graph(renamed))


def test_specific_mapping_refines_equivalence_class_and_is_chosen_as_anchor():
    assert constraints.compare(graph(SPECIFIC), graph(EQUIVALENT))
    assert constraints.compare(graph(SPECIFIC), graph(EQUIVALENT), refine=True)
    assert not constraints.compare(graph(EQUIVALENT), graph(SPECIFIC), refine=True)
    result = classify([EQUIVALENT, SPECIFIC])
    assert result['classes'] == 1 and not result['conflict'] and not result['constraints_unresolved']
    assert [m['is_anchor'] for m in result['members']] == [False, True]
    assert {m['heavy_class'] for m in result['members']} == {0}
    assert {m['constraint_class'] for m in result['members']} == {0, 1}


def test_a_broad_equivalence_class_cannot_mask_two_incompatible_complete_maps():
    result = classify([ETHANOL, SWAPPED, COARSE])
    assert result['conflict'] and result['cross_conflict']
    assert result['classes'] == 3
    assert not result['constraints_unresolved']


def test_missing_labels_are_partial_and_no_unmapped_atoms_are_collapsed():
    partial = '[CH3:1]O>>[CH2:1]=O'
    complete = '[CH3:1][OH:2]>>[CH2:1]=[O:2]'
    info = chem.reference_info(partial)
    assert info['mapping_status'] == 'partial' and info['unmapped_heavy_atoms'] == 2
    value = graph(partial)
    assert len(value) == 3
    assert not constraints.Constraints(value).complete
    assert constraints.compare(graph(complete), value, refine=True)
    result = classify([complete, partial])
    assert not result['conflict'] and result['classes'] == 1


def test_classes_mixing_elements_or_isotopes_remain_invalid():
    mixed = '[CH3:1][NH2:1]>>[CH2:1]=[NH:1]'
    assert chem.reference_info(mixed)['mapping_status'] == 'mapped_atom_identity_mismatch'
    with pytest.raises(ValueError, match='mixes elements'):
        graph(mixed)
    isotope = EQUIVALENT.replace('[O:1]=', '[18O:1]=', 1)
    assert chem.reference_info(isotope)['mapping_status'] == 'mapped_atom_identity_mismatch'


def test_stereo_ignored_without_removing_isotopes_charge_or_hydrogen_counts():
    one = '[F:1]/[CH:2]=[CH:3]/[Cl:4]>>[F:1][CH2:2][CH2:3][Cl:4]'
    two = one.replace('/[Cl:4]', '\\[Cl:4]')
    assert chem.identities(one)['exact_key'] == chem.identities(two)['exact_key']
    assert constraints.compare(graph(one), graph(two))
    assert chem.identities('[18OH2]>>O')['canonical_rxn'] != chem.identities('O>>O')['canonical_rxn']
    assert chem.identities('[NaH]>>[Na+]')['canonical_rxn'] != chem.identities('[Na]>>[Na+]')['canonical_rxn']


def test_constraint_timeout_stays_unresolved_instead_of_becoming_a_conflict(monkeypatch):
    def failed(*args, **kwargs):
        raise TimeoutError('test limit')
    monkeypatch.setattr(constraints, 'compare', failed)
    result = classify([SPECIFIC, EQUIVALENT])
    assert not result['conflict'] and result['constraints_unresolved']
    assert {m['status'] for m in result['members']} == {'mapping_constraints_unresolved'}


def test_equivalence_classes_reverse_without_changing_source_correspondences():
    forward = graph(EQUIVALENT)
    reverse = graph('>>'.join(reversed(EQUIVALENT.split('>>'))))
    restored = chem.unpack(chem.pack(reverse), reverse=True)
    classes = chem.ITSClasses()
    assert classes.assign(forward) == classes.assign(restored)


def test_dative_bonds_are_not_mistaken_for_reaction_separators():
    endpoint = '[NH3:1]->[Cu+:2]'
    mols = chem.parse_reaction(endpoint + '>>' + endpoint)
    assert [m.GetNumAtoms() for m in mols] == [2, 2]
    with_agents = chem.parse_reaction(endpoint + '>O>' + endpoint)
    assert chem.reaction_text(mols) == chem.reaction_text(with_agents)
