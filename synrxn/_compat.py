"""Lazy aliases for module paths published before the responsibility layout.

Implementations live at their canonical paths. The aliases keep existing user
imports and ``python -m`` commands usable without copies of those modules.
"""
from __future__ import annotations

from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import find_spec, spec_from_loader
import sys


MODULE_ALIASES = {
    'synrxn.condition_data': 'synrxn.reconstruction.families.conditions.condition_recovery._normalize',
    'synrxn.feasibility_data': 'synrxn.reconstruction.families.outcomes.reaction_feasibility._normalize',
    'synrxn.frea_data': 'synrxn.reconstruction.families.outcomes.reaction_feasibility._frea_source',
    'synrxn.impurity_data': 'synrxn.reconstruction.families.selectivity.impurity_prediction._normalize',
    'synrxn.selectivity_data': 'synrxn.reconstruction.families.selectivity._normalize',
    'synrxn.yield_data': 'synrxn.reconstruction.families.outcomes.yield_conversion._normalize',
    'synrxn.mapping_io': 'synrxn.benchmarks.mapping_io',
    'synrxn.reference': 'synrxn.benchmarks.contracts',
    'synrxn.reference_store': 'synrxn.benchmarks.reference_store',
    'synrxn.readiness': 'synrxn.benchmarks.readiness',
    'synrxn.task_views': 'synrxn.benchmarks.task_views',
    'synrxn.query': 'synrxn.access.query',
    'synrxn.query_benchmark': 'synrxn.access.benchmark',
    'synrxn.service': 'synrxn.access.service',
    'synrxn.build_catalog': 'synrxn.data.build_catalog',
    'synrxn.build_manifest': 'synrxn.data.build_manifest',
    'synrxn.validate_data': 'synrxn.data.validate',
    'synrxn.verify_manifest': 'synrxn.data.verify_manifest',
    'synrxn.evaluation': 'synrxn.metrics.reference',
    'synrxn.partitions': 'synrxn.split.dependence',
    'synrxn.expansion': 'synrxn.reconstruction.expansion',
    'synrxn.reconstruction.cohort': 'synrxn.reconstruction.families.outcomes.yield_conversion._hitea',
    'synrxn.reconstruction.scope': 'synrxn.reconstruction.families.__main__',
    'synrxn.reconstruction.coverage.audit': 'synrxn.reconstruction.coverage.verify_release',
    'synrxn.reconstruction.acquisition': 'synrxn.reconstruction.expansion',
    'synrxn.reconstruction.acquisition.__main__': 'synrxn.reconstruction.expansion.__main__',
    'synrxn.reconstruction.acquisition.expansion_fetch': 'synrxn.reconstruction.expansion.fetch',
    'synrxn.reconstruction.acquisition.expansion_sources': 'synrxn.reconstruction.expansion.audit_sources',
    'synrxn.reconstruction.sources': 'synrxn.reconstruction.builders.sources',
}

for _child in ('__main__', 'build', 'catalog', 'storage'):
    MODULE_ALIASES[f'synrxn.expansion.{_child}'] = f'synrxn.reconstruction.expansion.{_child}'


# The earlier reconstruction layout exposed these import paths.
_RECONSTRUCTION = 'synrxn.reconstruction.'
_BUILDER_ALIASES = {
    'mcsa': 'families.mechanisms.mechanistic_prediction._mcsa',
    'mcsa_native': 'families.mechanisms.mechanistic_prediction._mcsa_native',
    'mcsa_access': 'families.mechanisms.mechanistic_prediction._mcsa_access',
    'synepd': 'families.mechanisms.mechanistic_prediction._synepd',
    'classification': 'families.representation.reaction_classification._classification',
    'overlap': 'families.representation.reaction_retrieval._overlap',
    'synthesis': 'families.transformations._synthesis',
    'da': 'families.transformations.forward_prediction._diels_alder',
    'property': 'families.physics._property',
    'condition': 'families.conditions.condition_recovery._condition',
    'sources.condition': 'families.conditions.condition_recovery._normalize',
    'feasibility': 'families.outcomes.reaction_feasibility._feasibility',
    'sources.feasibility': 'families.outcomes.reaction_feasibility._normalize',
    'frea': 'families.outcomes.reaction_feasibility._frea',
    'sources.frea': 'families.outcomes.reaction_feasibility._frea_source',
    'yield_data': 'families.outcomes.yield_conversion._yield',
    'sources.yield_data': 'families.outcomes.yield_conversion._normalize',
    'impurity': 'families.selectivity.impurity_prediction._impurity',
    'sources.impurity': 'families.selectivity.impurity_prediction._normalize',
    'selectivity': 'families.selectivity._experimental',
    'sources.selectivity': 'families.selectivity._normalize',
    'task_views': 'families._task_views',
    'source_locks': 'provenance.source_locks',
    '__main__': 'families._importers',
}
_MAPPING_SOURCES = 'families.representation.atom_mapping._sources'
_BALANCING_SOURCES = 'families.representation.reaction_balancing._sources'
_BUILDER_ALIASES.update({name: f'{_MAPPING_SOURCES}.{name}' for name in (
    'aam_benchmark', 'aam_mapping_constraints', 'aam_overlap_chem', 'aam_sources',
    'audit_aam_overlap', 'audit_aam_partial_canon', 'audit_metamdb_aam', 'brenda_balanced',
    'build_metamdb_aam', 'build_unique_aam', 'complete_aam_maps', 'ehreact',
    'enzyme_map', 'export_uniform_aam', 'flower', 'flower_center_groups',
    'flower_dump', 'flower_mc_export', 'flower_mc_pipeline', 'flower_mc_store',
    'flower_mechanistic_center', 'flower_representatives', 'process_metambd_cyclic',
    'report_aam_cleaning', 'report_aam_overlap', 'sihumix', 'sihumix_quality',
)})
_BUILDER_ALIASES.update({name: f'{_BALANCING_SOURCES}.{name}' for name in (
    'build_flower_rbl', 'flower_rbl', 'flower_rbl_hard', 'flower_rbl_verify', 'rbl',
)})
_BUILDER_ALIASES.update({
    'process.aam': f'{_MAPPING_SOURCES}.conversion',
    'aam': f'{_MAPPING_SOURCES}.aam_benchmark',
    'group_flower_centers': f'{_MAPPING_SOURCES}.flower_center_groups',
    'process.rbl': f'{_BALANCING_SOURCES}.conversion',
})
MODULE_ALIASES.update({_RECONSTRUCTION + 'builders.' + old: _RECONSTRUCTION + new
                       for old, new in _BUILDER_ALIASES.items()})
MODULE_ALIASES.update({_RECONSTRUCTION + old: _RECONSTRUCTION + new for old, new in {
    'assay': 'families.outcomes.reaction_success._bh_hte_ood',
    'hitea': 'families.outcomes.yield_conversion._hitea',
    'suite': 'families.suite',
    'validate_plan': 'families._validate_study',
    'hierarchy': 'catalogue.hierarchy',
    'index': 'catalogue.index',
    'layout': 'catalogue.layout',
    'registry': 'catalogue.registry',
    'storage': 'catalogue.storage',
    'depictions': 'presentation.depictions',
    'documentation': 'presentation.documentation',
    'figures': 'presentation.figures',
    'manuscript': 'presentation.manuscript',
    'compat': 'provenance.compat',
    'literature': 'provenance.literature',
    'source_history': 'provenance.source_history',
}.items()})
MODULE_ALIASES.update({_RECONSTRUCTION + old: _RECONSTRUCTION + new for old, new in {
    'coverage.adapters.bio': 'families.biocatalysis._source_tables',
    'coverage.adapters.impurity': 'families.selectivity.impurity_prediction._source_tables',
    'coverage.adapters.knowledge': 'families.data._source_tables',
    'coverage.adapters.mcsa': 'families.mechanisms.mechanistic_prediction._mcsa_views',
    'coverage.adapters.physics': 'families.physics._source_tables',
    'coverage.adapters.process': 'families.process._source_tables',
    'coverage.adapters.quantumpioneer': 'families.physics.reaction_rates._quantumpioneer',
    'coverage.adapters.solution_barriers': 'families.physics.activation_barrier._solution_barriers',
    'coverage.adapters.solution_context': 'families.physics.activation_barrier._solution_context',
    'expansion.adapters': 'families._expansion_sources',
}.items()})
MODULE_ALIASES['synrxn.expansion.adapters'] = MODULE_ALIASES[_RECONSTRUCTION + 'expansion.adapters']
for _child in ('condition', 'feasibility', 'frea', 'impurity', 'selectivity', 'yield_data'):
    MODULE_ALIASES[f'synrxn.reconstruction.sources.{_child}'] = (
        _RECONSTRUCTION + _BUILDER_ALIASES['sources.' + _child])

_RETIRED_PACKAGES = {_RECONSTRUCTION + name for name in (
    'builders', 'builders.sources', 'builders.process', 'coverage.adapters',
)}


class _RetiredPackageLoader(Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if module.__name__ == _RECONSTRUCTION + 'builders':
            from .reconstruction.families._importers import TARGETS, module_for

            module.TARGETS = TARGETS
            module.module_for = module_for


class _AliasLoader(Loader):
    def __init__(self, target, package):
        self.target = target
        self.package = package

    def create_module(self, spec):
        # Both names return precisely the same module, including its globals.
        module = import_module(self.target)
        self.canonical_spec = module.__spec__
        return module

    def exec_module(self, module):
        module.__spec__ = self.canonical_spec

    def get_code(self, fullname):
        # runpy uses this route for a historical ``python -m`` command.
        spec = find_spec(self.target)
        return spec.loader.get_code(self.target)

    def is_package(self, fullname):
        return self.package


class _AliasFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        canonical = MODULE_ALIASES.get(fullname)
        if fullname in _RETIRED_PACKAGES:
            return spec_from_loader(fullname, _RetiredPackageLoader(), is_package=True)
        if canonical is None:
            return None
        canonical_spec = find_spec(canonical)
        package = canonical_spec.submodule_search_locations is not None
        spec = spec_from_loader(fullname, _AliasLoader(canonical, package),
                                origin=canonical_spec.origin)
        spec.has_location = True
        return spec


def install():
    """Install exact module aliases without importing their optional stacks."""
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
