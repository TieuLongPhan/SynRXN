"""Source screening must preserve its limits and never turn unknowns into misses."""
from dataclasses import replace
import re

import pytest

from synrxn.reconstruction.provenance.source_history import load_tool

mechanism = pytest.importorskip('synkit.Mechanism')
audit = load_tool('audit_mechanism_source_relationships')

REACTION = '[NH3+:2][H:4].[O-:1][CH3:3]>>[NH3:2].[O:1]([CH3:3])[H:4]'
ARROWS = [['LP-/Sigma+', [1], [1, 4]], ['Sigma-/LP+', [2, 4], [2]]]


def test_map_labels_do_not_change_screen_or_necessary_shapes():
    a = mechanism.mechanism_from_legacy_epd(REACTION, ARROWS)
    reaction = re.sub(r':(\d+)\]', lambda m: ':' + str(int(m[1]) + 10) + ']', REACTION)
    arrows = [[kind, [i + 10 for i in source], [i + 10 for i in target]] for kind, source, target in ARROWS]
    b = mechanism.mechanism_from_legacy_epd(reaction, arrows)
    assert audit.endpoint_signature(REACTION) == audit.endpoint_signature(reaction)
    assert audit.event_shape(a) == audit.event_shape(b)
    assert audit.trajectory_shape(a) == audit.trajectory_shape(b)
    assert mechanism.mechanism_equivalent(a, b, level='trajectory')


def test_endpoint_screen_does_not_certify_supplied_atom_transport():
    a = mechanism.mechanism_from_legacy_epd('[CH3:1][OH:2]>>[CH2:1]=[O:2]', [])
    b = mechanism.mechanism_from_legacy_epd('[CH3:1][OH:2]>>[CH2:2]=[O:1]', [])
    assert audit.endpoint_signature(a.mapped_reaction) == audit.endpoint_signature(b.mapped_reaction)
    assert not mechanism.mechanism_equivalent(a, b, level='net')


def test_endpoint_screen_keeps_isotopes_and_explicit_hydrogen_inventory():
    assert audit.endpoint_signature('[13CH4:1]>>[13CH4:1]') != audit.endpoint_signature('[CH4:1]>>[CH4:1]')
    assert audit.endpoint_signature('[OH2:1]>>[OH2:1]') != audit.endpoint_signature('[O:1]([H:2])[H:3]>>[O:1]([H:2])[H:3]')


def test_native_group_boundary_changes_are_not_erased_by_invariant_gate():
    from synkit.Mechanism.adapters import group_from_legacy_epd
    a = mechanism.mechanism_from_legacy_epd(REACTION, ARROWS)
    b = replace(a, steps=(mechanism.MechanisticStep('s1', (group_from_legacy_epd(ARROWS),)),))
    assert audit.event_shape(a) != audit.event_shape(b)
    assert not mechanism.mechanism_equivalent(a, b, level='events')


def test_unresolved_pair_level_is_not_counted_as_event_disagreement():
    summary = audit.pair_summary([{'status': 'unresolved', 'both_primary': True, 'same_enzyme_entry': False,
                                  'equivalence': {'net': True, 'events': None, 'trajectory': None}}])
    assert summary['net_equivalent'] == 1
    assert summary['events_resolved'] == summary['net_equal_events_different'] == 0
    assert summary['primary_net_equal_events_different'] == 0
