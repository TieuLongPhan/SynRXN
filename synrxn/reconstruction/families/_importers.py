"""Run one source importer or organization tool without eager chemistry imports."""
from __future__ import annotations

import argparse
import json
import runpy
import sys

TARGETS = {
    'aam': 'families.representation.atom_mapping._sources.aam_benchmark',
    'audit_aam_overlap': 'families.representation.atom_mapping._sources.audit_aam_overlap',
    'audit_aam_partial_canon': 'families.representation.atom_mapping._sources.audit_aam_partial_canon',
    'audit_metamdb_aam': 'families.representation.atom_mapping._sources.audit_metamdb_aam',
    'build_metamdb_aam': 'families.representation.atom_mapping._sources.build_metamdb_aam',
    'build_unique_aam': 'families.representation.atom_mapping._sources.build_unique_aam',
    'complete_aam_maps': 'families.representation.atom_mapping._sources.complete_aam_maps',
    'export_uniform_aam': 'families.representation.atom_mapping._sources.export_uniform_aam',
    'process_metambd_cyclic': 'families.representation.atom_mapping._sources.process_metambd_cyclic',
    'report_aam_cleaning': 'families.representation.atom_mapping._sources.report_aam_cleaning',
    'report_aam_overlap': 'families.representation.atom_mapping._sources.report_aam_overlap',
    'rbl': 'families.representation.reaction_balancing._sources.rbl',
    'classification': 'families.representation.reaction_classification._classification',
    'synthesis': 'families.transformations._synthesis',
    'property': 'families.physics._property',
    'condition': 'families.conditions.condition_recovery._condition',
    'feasibility': 'families.outcomes.reaction_feasibility._feasibility',
    'impurity': 'families.selectivity.impurity_prediction._impurity',
    'selectivity': 'families.selectivity._experimental',
    'yield': 'families.outcomes.yield_conversion._yield',
    'brenda': 'families.representation.atom_mapping._sources.brenda_balanced',
    'flower': 'families.representation.atom_mapping._sources.flower_representatives',
    'build_flower_rbl': 'families.representation.reaction_balancing._sources.build_flower_rbl',
    'group_flower_centers': 'families.representation.atom_mapping._sources.flower_center_groups',
    'ehreact': 'families.representation.atom_mapping._sources.ehreact',
    'enzyme_map': 'families.representation.atom_mapping._sources.enzyme_map',
    'sihumix': 'families.representation.atom_mapping._sources.sihumix',
    'diels_alder': 'families.transformations.forward_prediction._diels_alder',
    'frea': 'families.outcomes.reaction_feasibility._frea',
    'task_views': 'families._task_views',
    'source_locks': 'provenance.source_locks',
    'overlap': 'families.representation.reaction_retrieval._overlap',
    'mcsa': 'families.mechanisms.mechanistic_prediction._mcsa',
    'mcsa_native': 'families.mechanisms.mechanistic_prediction._mcsa_native',
    'mcsa_access': 'families.mechanisms.mechanistic_prediction._mcsa_access',
    'synepd': 'families.mechanisms.mechanistic_prediction._synepd',
}


def module_for(target):
    """Return the maintained module for a source importer command."""
    return 'synrxn.reconstruction.' + TARGETS[target]


def main(argv=None):
    """List or invoke the selected source importer."""
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', choices=sorted(TARGETS))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    if not args or args[0] in {'-h', '--help'}:
        parser.print_help()
        return 0
    if args == ['--list']:
        print(json.dumps({key: module_for(key) for key in TARGETS}, indent=2))
        return 0
    target, rest = args[0], args[1:]
    if target not in TARGETS:
        parser.error('unknown target: ' + target)
    previous = sys.argv
    try:
        sys.argv = [module_for(target), *rest]
        runpy.run_module(module_for(target), run_name='__main__', alter_sys=True)
    finally:
        sys.argv = previous
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
