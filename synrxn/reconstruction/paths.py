"""Repository locations and reversible aliases for historical provenance paths.

Catalogue path strings remain stable when repository storage is reorganized.
Active code uses explicit physical paths; source-backed modules can use
HistoricalPath to retain the logical identities recorded in source catalogues.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILIES = Path("Data/families")
RAW_CACHE = Path("archive/datasets/raw")
_CANONICAL = Path("Data/tables/canonical")
_TABLE_ALIASES = {"Data/condition/uspto_500.csv.gz": "Data/synthesis/uspto_500.csv.gz"}

# Match the most specific logical prefix first. These are storage aliases, not
# replacements for historical strings inside checksum-pinned source records.
_METADATA_SPELLINGS = ('kdd', 'Data/provenance/reconstruction',
                       'Data/metadata/reconstruction')
_STUDY_OUTPUTS = 'Data/metadata/reconstruction'
_STUDY_TABLES = {
    'preparation/memberships.jsonl.gz': 'preparation/memberships.jsonl.gz',
    'preparation/cohort.jsonl.gz': 'preparation/memberships.jsonl.gz',
    'case_studies/ncs_yield/success_targets.csv': 'case_studies/ncs_yield/success_targets.csv',
    'case_studies/classic_yield/bh_classic.csv.gz': 'case_studies/classic_yield/bh_classic.csv.gz',
    'case_studies/classic_yield/suzuki_native_reference.csv.gz':
        'case_studies/classic_yield/suzuki_native_reference.csv.gz',
}
RELOCATIONS = (
    ('Data/metadata.yaml', 'Data/metadata/catalog.yaml'),
    ('Data/LICENSE', 'Data/metadata/LICENSE'),
    ('Data/raw', RAW_CACHE.as_posix()),
    ('Data/aam/metambd_cyclic.csv.gz', 'archive/datasets/local-only/aam/metambd_cyclic.csv.gz'),
    ('Data/yield/suzuki_miyaura.csv.gz', 'archive/datasets/local-only/yield/suzuki_miyaura.csv.gz'),
    ('Data/coverage_expansion/bio/quarantine', 'archive/datasets/local-only/coverage/bio/quarantine'),
    ('Data/synthesis/da.csv.gz', 'archive/datasets/local-only/synthesis/da.csv.gz'),
    ('Data/tables/canonical/synthesis/da.csv.gz', 'archive/datasets/local-only/synthesis/da.csv.gz'),
    ('Data/archives', 'archive/datasets/data-archives'),
    ('Data/dumps/flower.dump.zip', 'Data/raw/flower/flower.dump.zip'),
    ('Data/tables/development', '.cache/synrxn/development'),
    ('.cache/ncs-primary-evidence/BH-HTE-OOD-official/data/'
     'BH_HTE_Curated_Dataset_v20260219.csv',
     'Data/raw/yield/BH_HTE_Curated_Dataset_v20260219.csv'),
    *((f'.cache/ncs-primary-evidence/FREA-{old}', f'Data/raw/frea/{new}')
      for old, new in (('corpus', 'corpus'), ('expert', 'expert'),
                       ('official', 'official'))),
    ('.cache/ncs-primary-evidence/property-followup', 'Data/raw/property'),
    ('.cache/classic-yield-reference', 'Data/raw/classic_yield'),
    ('.cache/brenda-iubmb-sparql.json.gz',
     'Data/raw/brenda/iubmb-sparql.json.gz'),
    ('.cache/synthesizability-primary-evidence',
     'Data/raw/literature/synthesizability'),
    *((f'.cache/ncs-primary-evidence/'
       f'SynTemp-Data_AAM_results_benchmark_{name}_{name}_aam_reactions.json.gz',
       f'Data/raw/rights/recheck/mapping/syntemp_native_{name}.txt')
      for name in ('golden', 'natcomm', 'ecoli', 'recon3d', 'uspto_3k')),
    ('.cache/ncs-primary-evidence/LocalMapper-comparison_Golden_mappings.csv',
     'Data/raw/rights/recheck/mapping/localmapper_2024_golden.txt'),
    ('.cache/ncs-primary-evidence', 'Data/raw/rights/research_inputs'),
    *((f'{prefix}/coverage/{old}', f'Data/raw/rights/{new}')
      for old, new in (('rights_recheck', 'recheck'), ('rights_legacy', 'legacy'))
      for prefix in _METADATA_SPELLINGS),
    *((f'{prefix}/literature/{old}', f'Data/raw/literature/{new}')
      for old, new in (('raw', 'downloads'), ('papers', 'papers'), ('text', 'text'))
      for prefix in ('kdd', 'Data/provenance', 'Data/metadata')),
    *((f'{prefix}/preparation/raw', 'Data/raw/hitea')
      for prefix in _METADATA_SPELLINGS),
    *((f'{prefix}/{relative}', f'{_STUDY_OUTPUTS}/{target}')
      for relative, target in _STUDY_TABLES.items()
      for prefix in _METADATA_SPELLINGS),
    *((f'Data/tables/studies/hitea/{name}.jsonl.gz', f'{_STUDY_OUTPUTS}/preparation/memberships.jsonl.gz')
      for name in ('memberships', 'cohort')),
    ('Data/tables/studies/bh_hte_ood/success_targets.csv',
     f'{_STUDY_OUTPUTS}/case_studies/ncs_yield/success_targets.csv'),
    *((f'Data/tables/studies/classic_yield/{name}.csv.gz', f'{_STUDY_OUTPUTS}/case_studies/classic_yield/{name}.csv.gz')
      for name in ('bh_classic', 'suzuki_native_reference')),
    ('ncs/sources/mcsa-v5', 'Data/raw/mcsa-v5'),
    ('Data/sources/previous/mcsa-v5', 'Data/raw/mcsa-v5'),
    ('Data/metadata/sources/previous/mcsa-v5', 'Data/raw/mcsa-v5'),
    ('Data/sources/property/snar-context.csv',
     'Data/raw/coverage/solution_context/snar.context.csv'),
    ('Data/metadata/sources/property/snar-context.csv',
     'Data/raw/coverage/solution_context/snar.context.csv'),
    *((f'Data/coverage_expansion/process/nmr_{name}_snapshots.jsonl.gz',
       f'Data/metadata/coverage/process/nmr_{name}_spectra.jsonl.gz')
      for name in ('saccharose_hydrolysis', 'pmg_284', 'pmg_287')),
    *((f'Data/tables/coverage/process/nmr_{name}_snapshots.jsonl.gz',
       f'Data/metadata/coverage/process/nmr_{name}_spectra.jsonl.gz')
      for name in ('saccharose_hydrolysis', 'pmg_284', 'pmg_287')),
    ('kdd/literature', 'Data/metadata/literature'),
    ('ncs/sources', 'Data/metadata/sources/previous'),
    ('Data/sources', 'Data/metadata/sources'),
    ('kdd', 'Data/metadata/reconstruction'),
    ('Data/provenance', 'Data/metadata'),
    ('ncs', 'archive/research/previous'),
    ('Data/condition_sharealike', 'Data/metadata/condition_sharealike'),
    ('Data/expansion', 'Data/metadata/expansion'),
    ('Data/coverage_expansion', 'Data/metadata/coverage'),
    ('Data/Benchmark', 'archive/benchmarks/features'),
    ('Data/results_benchmark', 'archive/benchmarks/results'),
)

_OLD_TABLE_ROOTS = (
    ('Data/coverage_expansion', 'Data/tables/coverage'),
    ('Data/expansion', 'Data/tables/expansion'),
    *((f'Data/{name}', f'Data/tables/canonical/{name}') for name in (
        'aam', 'classification', 'condition', 'condition_sharealike', 'feasibility',
        'mechanism', 'property', 'rbl', 'selectivity', 'synthesis', 'yield')),
)


# Superseded plans and generated summaries are historical read dependencies,
# never active output locations. physical_path deliberately ignores this list.
RETIRED_METADATA = frozenset({
    'contribution_ledger.csv', 'data_coverage.json', 'data_coverage.md',
    'family_coverage.csv', 'feasibility_structure.md', 'panel_review_rubric.md',
    'paper_blueprint.md', 'preparation_plan.md', 'repository_layout.md',
    'review_lessons.md', 'runbook.md', 'scope_coverage.md',
    'storage_architecture.md', 'task_dataset_counts.csv', 'task_structure.md',
    'work_packages.csv', 'work_plan.json',
    'reproduction/index.json', 'reproduction/index.md',
    'coverage/coverage_report.md', 'coverage/task_dataset_counts.csv',
    'case_studies/hitea_paper_blueprint.md',
    'case_studies/hitea_preparation_plan.md',
    'preparation/plan_validation.json',
})


def _plain(value):
    """Read the logical spelling without invoking a path-like I/O alias."""
    return Path(str(value)) if isinstance(value, Path) else Path(value)


def _root(repo):
    return _plain(REPO if repo is None else repo).resolve()


def _relative(value, root):
    path = _plain(value)
    if path.is_absolute():
        path = Path(os.path.normpath(str(path)))
        try:
            return path.relative_to(root)
        except ValueError:
            return None
    if '..' in path.parts:
        raise ValueError('Historical path escapes repository')
    return path


def _replace_prefix(relative, source, destination):
    try:
        tail = relative.relative_to(source)
    except ValueError:
        return None
    return Path(destination) / tail


@lru_cache(maxsize=16)
def _family_index(index, mtime_ns, inode):
    paths = {}
    for item in json.loads(index.read_text())["tables"]:
        stored = item.get("storage_path")
        if stored:
            destination = Path(stored)
            if destination.is_absolute() or '..' in destination.parts:
                raise ValueError(f'Invalid family payload path: {stored}')
            paths[item["source_path"]] = stored
    return paths


def family_storage(repo=None):
    """Map stable source paths to their single prepared file in the family tree."""
    index = _root(repo) / FAMILIES / 'index.json'
    try:
        info = index.stat()
    except FileNotFoundError:
        return {}
    return _family_index(index, info.st_mtime_ns, info.st_ino)


def physical_path(value, repo=None):
    """Resolve a logical repository location to its current storage location.

    Existing historical paths win in native rebuild and test workspaces. This
    permits exact historical directory layouts without root compatibility links.
    Absolute paths outside the repository remain ordinary external I/O paths.
    A missing relocated path still resolves to its new location for new writes.
    """
    root = _root(repo)
    relative = _relative(value, root)
    if relative is None:
        return _plain(value)
    historical = root / relative
    if historical.exists() or historical.is_symlink():
        return historical
    if relative.is_relative_to('Data') and len(relative.parts) >= 2:
        for old, previous in _OLD_TABLE_ROOTS:
            logical = _replace_prefix(relative, previous, old)
            if logical is not None:
                relative = logical
                break
        key = relative.as_posix()
        if key.startswith(('Data/coverage_expansion/process/nmr_', 'Data/tables/coverage/process/nmr_')):
            key = key.replace('_spectra.jsonl.gz', '_snapshots.jsonl.gz')
        stored = family_storage(root).get(_TABLE_ALIASES.get(key, key))
        if stored:
            return root / FAMILIES / stored
        for old, previous in _OLD_TABLE_ROOTS:
            mapped = _replace_prefix(relative, old, previous)
            if mapped is not None:
                if (root / mapped).exists():
                    return root / mapped
                renamed = root / str(mapped).replace('_snapshots.jsonl.gz', '_spectra.jsonl.gz')
                if renamed.exists():
                    return renamed
                if (root / previous).is_dir() and old not in {'Data/expansion', 'Data/coverage_expansion'}:
                    return root / mapped
    for source, destination in sorted(RELOCATIONS, key=lambda pair: len(Path(pair[0]).parts), reverse=True):
        mapped = _replace_prefix(relative, source, destination)
        if mapped is not None:
            if mapped.is_relative_to('Data/raw') and not (root / 'Data/raw').is_dir():
                mapped = RAW_CACHE / mapped.relative_to('Data/raw')
            return root / mapped
    return historical


def logical_relative(value, repo=None):
    """Return the stable POSIX provenance key for a current repository path."""
    root = _root(repo)
    relative = _relative(value, root)
    if relative is None:
        raise ValueError('Path is outside the repository: ' + str(value))
    if relative.is_relative_to(FAMILIES):
        stored = relative.relative_to(FAMILIES).as_posix()
        for source, target in family_storage(root).items():
            if stored == target:
                return source
    if relative.is_relative_to(RAW_CACHE):
        relative = Path('Data/raw') / relative.relative_to(RAW_CACHE)
    for source, destination in sorted(RELOCATIONS, key=lambda pair: len(Path(pair[1]).parts), reverse=True):
        mapped = _replace_prefix(relative, destination, source)
        if mapped is not None:
            return mapped.as_posix()
    for source, destination in _OLD_TABLE_ROOTS:
        mapped = _replace_prefix(relative, destination, source)
        if mapped is not None:
            return mapped.as_posix().replace('_spectra.jsonl.gz', '_snapshots.jsonl.gz')
    return relative.as_posix()


def repo_path(*parts, repo=None):
    return physical_path(Path(*parts), repo=repo)


def legacy_path(relative, repo=None):
    root = _root(repo)
    relative = _plain(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Historical source paths must be repository-relative')
    path = root / 'archive/source-code' / relative
    if not path.is_file():
        raise FileNotFoundError('Frozen historical source is unavailable: ' + str(relative))
    return path


def historical_file(relative, repo=None):
    """Return an explicit logical source identity, separate from __file__."""
    relative = _plain(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Historical source paths must be repository-relative')
    return str(_root(repo) / relative)


def source_path(relative, repo=None):
    """Resolve preserved original code first, then current metadata/data storage.

    Source hashes refer to frozen original kernels even when a physical package
    facade exists. Metadata aliases preserve original bytes and logical keys.
    """
    root = _root(repo)
    path = _plain(relative)
    logical = _relative(path, root)
    if logical is None:
        return path
    old_prefix = Path('synrxn/reconstruction/legacy')
    if logical.is_relative_to(old_prefix):
        return root / 'archive/source-code' / logical.relative_to(old_prefix)
    archived = root / 'archive/source-code' / logical
    if archived.is_file():
        return archived
    current = physical_path(path, repo=root)
    if current.is_file():
        return current
    for prefix in _METADATA_SPELLINGS:
        try:
            name = logical.relative_to(prefix).as_posix()
        except ValueError:
            continue
        if name in RETIRED_METADATA:
            retired = root / 'archive/retired-metadata/reconstruction' / name
            if retired.is_file():
                return retired
    return current


class HistoricalPath(type(Path())):
    """Source-backed I/O path with stable logical provenance display names.

    Pure path operations retain historical provenance. Filesystem consumers use
    os.fspath(), which supplies the relocated physical path. Converting this to
    an ordinary Path intentionally leaves this scoped compatibility behavior.
    """

    def __fspath__(self):
        absolute = _plain(self).absolute()
        return str(physical_path(absolute))

    def resolve(self, strict=False):
        resolved = physical_path(_plain(self).absolute()).resolve(strict=strict)
        try:
            return type(self)(REPO / logical_relative(resolved))
        except ValueError:
            return type(self)(resolved)

    def mkdir(self, mode=0o777, parents=False, exist_ok=False):
        # A relocated directory can have more physical ancestors than its
        # logical spelling. pathlib's logical parent recursion would skip them.
        physical_path(_plain(self).absolute()).mkdir(mode=mode, parents=parents,
                                                   exist_ok=exist_ok)
