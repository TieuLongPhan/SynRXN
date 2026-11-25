# Direct implementation; historical source identity is explicit below.
"""Measure cross-source reaction overlap between SynRXN corpora.

Release gate: duplicate leakage must be *measurable* before split manifests are
frozen. Corpora assembled from patents, HTE campaigns, and aggregated
repositories are not disjoint, so a pooled row count is not a reaction count.

Reaction identity is the canonical, atom-map-free ``reactants>>products``
string produced by :func:`synrxn.reconstruction.families.outcomes.yield_conversion._normalize.canonical_unmapped_reaction`, the
same function the release builders use. Corpora that already ship a
``canonical_reaction_smiles`` column reuse it directly; legacy corpora are
canonicalized on the fly with the identical routine so the two are comparable.

Identity is deliberately structure-only: reaction conditions, yields, and
measurement protocols are ignored. Two rows that share an identity are the same
transformation, not necessarily the same experiment.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('script/build_overlap_report.py')
import argparse
import gzip
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from synrxn.reconstruction.paths import HistoricalPath as Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__source_file__).resolve().parents[1]))
from synrxn.reconstruction.families.outcomes.yield_conversion._normalize import canonical_unmapped_reaction
ROOT = Path(__source_file__).resolve().parents[1]
DATA = ROOT / 'Data'
PRECOMPUTED = ('yield/buchwald_hartwig_ood', 'yield/hitea', 'yield/uspto_yield', 'condition/orderly_condition', 'condition_sharealike/openexp', 'condition_sharealike/ord_hte_condition')
DERIVED = {'condition/uspto_500': 'rxn', 'synthesis/uspto_500': 'rxn', 'synthesis/uspto_50k': None, 'synthesis/uspto_mit': None, 'classification/uspto_50k_b': 'rxn'}

def _hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode('utf8'), digest_size=8).digest(), 'big')

def _canonical_or_none(reaction: str) -> str | None:
    """Canonicalize one reaction, returning ``None`` when it cannot be parsed."""
    if not isinstance(reaction, str) or not reaction.strip():
        return None
    parts = reaction.split('>')
    if len(parts) == 3:
        reaction = f'{parts[0]}>>{parts[2]}'
    elif '>>' not in reaction:
        return None
    try:
        return canonical_unmapped_reaction(reaction)[0]
    except Exception:
        return None

def _reaction_column(path: Path, override: str | None) -> str:
    with gzip.open(path, 'rt', encoding='utf8') as handle:
        header = handle.readline().strip().split(',')
    if override:
        if override not in header:
            raise SystemExit(f'{path}: column {override!r} not found in {header}')
        return override
    for candidate in ('canonical_reaction_smiles', 'reaction_smiles', 'rxn', 'reactions', 'rsmi', 'reaction', 'ground_truth', 'aam'):
        if candidate in header:
            return candidate
    raise SystemExit(f'{path}: no reaction column among {header}')

def corpus_identities(key: str, column: str | None, workers: int, cache: Path) -> dict:
    """Return identity hashes for one corpus, caching the expensive pass."""
    cache_file = cache / f"{key.replace('/', '__')}.npz"
    path = DATA / f'{key}.csv.gz'
    if cache_file.exists():
        stored = np.load(cache_file)
        if int(stored['mtime']) == int(path.stat().st_mtime):
            return {'hashes': stored['hashes'], 'rows': int(stored['rows']), 'unparsed': int(stored['unparsed']), 'column': str(stored['column'])}
    name = _reaction_column(path, column)
    frame = pd.read_csv(path, usecols=[name], dtype=str)
    values = frame[name].tolist()
    rows = len(values)
    if name == 'canonical_reaction_smiles':
        canonical = [v if isinstance(v, str) and v.strip() else None for v in values]
    elif workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            canonical = list(pool.map(_canonical_or_none, values, chunksize=2000))
    else:
        canonical = [_canonical_or_none(v) for v in values]
    unparsed = sum((1 for c in canonical if c is None))
    hashes = np.fromiter((_hash(c) for c in canonical if c is not None), dtype=np.uint64, count=rows - unparsed)
    hashes = np.unique(hashes)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, hashes=hashes, rows=rows, unparsed=unparsed, column=name, mtime=int(path.stat().st_mtime))
    return {'hashes': hashes, 'rows': rows, 'unparsed': unparsed, 'column': name}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--cache', type=Path, default=ROOT / '.cache' / 'overlap')
    parser.add_argument('--out', type=Path, default=ROOT / 'Data' / 'sources' / 'overlap_report.json')
    parser.add_argument('--only', nargs='*', default=None, help='Restrict to these dataset keys.')
    args = parser.parse_args()
    corpora = {k: None for k in PRECOMPUTED} | DERIVED
    if args.only:
        corpora = {k: v for k, v in corpora.items() if k in set(args.only)}
    loaded: dict[str, dict] = {}
    for key, column in corpora.items():
        print(f'[identity] {key} ...', flush=True)
        loaded[key] = corpus_identities(key, column, args.workers, args.cache)
        info = loaded[key]
        print(f"           rows={info['rows']:,} unique={len(info['hashes']):,} unparsed={info['unparsed']:,} column={info['column']}", flush=True)
    keys = list(loaded)
    per_corpus = {}
    for key in keys:
        info = loaded[key]
        parsed = info['rows'] - info['unparsed']
        per_corpus[key] = {'rows': info['rows'], 'identity_column': info['column'], 'unparsed_rows': info['unparsed'], 'parsed_rows': parsed, 'unique_reactions': int(len(info['hashes'])), 'internal_duplicate_rows': parsed - int(len(info['hashes'])), 'internal_duplicate_fraction': round((parsed - len(info['hashes'])) / parsed, 6) if parsed else None}
    pairs = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ha, hb = (loaded[a]['hashes'], loaded[b]['hashes'])
            shared = int(np.intersect1d(ha, hb, assume_unique=True).size)
            union = len(ha) + len(hb) - shared
            pairs[f'{a} | {b}'] = {'shared_reactions': shared, 'jaccard': round(shared / union, 6) if union else 0.0, 'fraction_of_first': round(shared / len(ha), 6) if len(ha) else 0.0, 'fraction_of_second': round(shared / len(hb), 6) if len(hb) else 0.0}
    union = np.unique(np.concatenate([loaded[k]['hashes'] for k in keys]))
    total_rows = sum((per_corpus[k]['rows'] for k in keys))
    total_parsed = sum((per_corpus[k]['parsed_rows'] for k in keys))
    report = {'identity_definition': 'canonical atom-map-free reactants>>products, isomeric SMILES, components sorted; agents and conditions ignored', 'generated_at': '2026-08-22', 'corpora': per_corpus, 'pairwise_overlap': dict(sorted(pairs.items(), key=lambda kv: -kv[1]['shared_reactions'])), 'totals': {'corpora_compared': len(keys), 'pooled_rows': total_rows, 'pooled_parsed_rows': total_parsed, 'distinct_reactions_in_union': int(union.size), 'rows_beyond_distinct_reactions': total_parsed - int(union.size), 'redundancy_fraction': round((total_parsed - union.size) / total_parsed, 6) if total_parsed else None}, 'caveats': ['Structure-only identity. Shared identity means the same transformation, not the same experiment or measurement.', 'Unparsed rows are excluded from identity sets and are reported per corpus; they are not silently treated as unique.', 'Pooled row totals must never be reported as reaction counts.', 'High-throughput campaigns are redundant under this identity by design: one transformation is screened across many conditions. For HTE corpora the meaningful identity is (reaction, condition), so a low unique-reaction count is not a data-quality defect. It does bound how many independent transformations an HTE split can hold out.']}
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf8')
    print(json.dumps(report['totals'], indent=2, sort_keys=True))
    try:
        shown = args.out.relative_to(ROOT)
    except ValueError:
        shown = args.out
    print(f'\nwrote {shown}')
    return 0
__configure_runtime(globals(), 'script/build_overlap_report.py')
if __name__ == '__main__':
    raise SystemExit(main())
