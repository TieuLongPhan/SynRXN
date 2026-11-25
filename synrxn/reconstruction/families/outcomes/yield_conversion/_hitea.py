# Direct implementation; historical source identity is explicit below.
"""Audit the proposed HiTEA study and create label-independent memberships.

This prepares data only. It does not fit a predictor or certify experimental
independence. Run from any directory; outputs retain source row identities.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/prepare_cohort.py')
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
from synrxn.reconstruction.paths import HistoricalPath as Path, physical_path
import random
import urllib.request
ROOT = Path(__source_file__).resolve().parents[1]
RAW_URL = 'https://raw.githubusercontent.com/emmaking-smith/HiTEA/ccc8cce274e70429498be829dd3bc57754f62265/data/8_SEPT_APPROVED_full_dataset.csv'
RAW_SHA256 = '9d3b35b8606538df7fc8dd9104b6dd43188ddc8dc43e3d9d3972e10a621ef889'
RAW_CONTEXT = ['Solvent_1_Name', 'Reaction_T', 'Reaction_Time_hrs'] + [name for slot in (1, 2) for name in (f'Catalyst_{slot}_eq', f'catalyst_{slot}_ID_1_SMILES', f'catalyst_{slot}_ID_2_SMILES', f'Catalyst_{slot}_Short_Hand')] + [name for slot in (1, 2, 3) for name in (f'Reactant_{slot}_eq', 'Reactant_1_SMILES' if slot == 1 else f'reactant_{slot}_SMILES')] + [name for slot in (1, 2) for name in (f'Reagent_{slot}_eq', f'Reagent_{slot}_Short_Hand')]
FLAT_CONTEXT = ['reagents', 'catalyst', 'base', 'solvent', 'temperature_c', 'time_hours']
NUMERIC_CONTEXT = {'Reaction_T', 'Reaction_Time_hrs', 'temperature_c', 'time_hours'} | {name for name in RAW_CONTEXT if name.endswith('_eq')}
RELATIONS = ['screen_id', 'notebook_id', 'chemical_split_group', 'duplicate_group', 'reactants']

def sha(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def encode(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

def identity(prefix: str, value) -> str:
    return prefix + ':' + hashlib.sha256(encode(value).encode()).hexdigest()

def clean(value: str | None, numeric: bool=False) -> str:
    text = (value or '').strip()
    if text.lower() in {'', 'nan', 'none', 'null'}:
        return ''
    if numeric:
        try:
            number = Decimal(text)
            if not number.is_finite():
                raise ValueError('Nonfinite numeric context')
            return '0' if number == 0 else format(number.normalize(), 'f')
        except InvalidOperation:
            return text
    return text

def context(row: dict, fields: list[str]) -> dict:
    return {field: value for field in fields if (value := clean(row.get(field), field in NUMERIC_CONTEXT))}

def component_ids(rows: list[dict], relations: list[str]=RELATIONS) -> list[str]:
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    keys_by_row = []
    for row in rows:
        keys = [(field, row[field]) for field in relations if clean(row.get(field))]
        if not keys:
            raise ValueError('No grouping relation for ' + row['sample_id'])
        keys_by_row.append(keys)
        for key in keys[1:]:
            left, right = (find(keys[0]), find(key))
            if left != right:
                parent[right] = left
    members = defaultdict(set)
    for keys in keys_by_row:
        members[find(keys[0])].update(keys)
    names = {key: identity('component', sorted(values)) for key, values in members.items()}
    return [names[find(keys[0])] for keys in keys_by_row]

def assign_folds(group_ids: list[str], folds: int=5, seed: int=2027) -> dict[str, int]:
    """Balance row counts using only identities and group sizes, never targets."""
    if folds < 3 or len(set(group_ids)) < folds:
        raise ValueError('Need at least three folds and at least one group per fold')
    counts = Counter(group_ids)
    order = sorted(counts, key=lambda group: (-counts[group], identity(str(seed), group)))
    loads = [0] * folds
    result = {}
    for group in order:
        fold = min(range(folds), key=lambda index: (loads[index], index))
        result[group] = fold
        loads[fold] += counts[group]
    return result

def relation_overlap(rows: list[dict], folds: list[int], field: str) -> int:
    memberships = defaultdict(set)
    for row, fold in zip(rows, folds, strict=True):
        if clean(row.get(field)):
            memberships[row[field]].add(fold)
    return sum((len(assigned) > 1 for assigned in memberships.values()))

def prepare(data_dir: Path, raw_path: Path, output: Path, fetch: bool=False) -> dict:
    previous_audit, previous_cards = (output / 'audit.json', output / 'panel_cards.csv')
    if previous_audit.exists() and previous_cards.exists():
        expected_cards = json.loads(previous_audit.read_text()).get('initial_panel_cards_sha256')
        if expected_cards and sha(previous_cards) != expected_cards:
            raise ValueError('Panel cards have edits; preserve the reviewed version and use a fresh --output')
    if not raw_path.exists() and fetch:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(RAW_URL, timeout=45) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != RAW_SHA256:
            raise ValueError('Retrieved HiTEA source differs from its pinned SHA-256')
        raw_path.write_bytes(raw)
    if not raw_path.exists() or sha(raw_path) != RAW_SHA256:
        raise ValueError('Need the pinned HiTEA source; use --fetch to retrieve a missing file')
    source_path = data_dir / 'yield/hitea.csv.gz'
    source_hash = sha(source_path)
    csv.field_size_limit(16 * 1024 * 1024)
    with raw_path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = set(RAW_CONTEXT) - set(reader.fieldnames)
        if missing:
            raise ValueError('Missing source input fields: ' + repr(sorted(missing)))
        raw_rows = list(reader)
    original = {row['REACTION_ID']: row for row in raw_rows}
    if len(original) != len(raw_rows):
        raise ValueError('Original reaction IDs are not unique')
    with gzip.open(source_path, 'rt', encoding='utf8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Canonical sample IDs are not unique')
    canonical_rows = rows
    canonical_index = {row['sample_id']: index for index, row in enumerate(rows)}
    all_components = component_ids(rows)
    component_by_sample = dict(zip([row['sample_id'] for row in rows], all_components, strict=True))
    pairs, exclusions = ([], [])
    for row in rows:
        raw = original[row['source_record_id']]
        target, upstream_target = (Decimal(row['yield_pct']), Decimal(raw['Product_Yield_PCT_Area_UV']))
        if not target.is_finite() or not 0 <= target <= 100 or target != upstream_target:
            raise ValueError('Invalid or mismatched assay target: ' + row['sample_id'])
        if row['screen_id'] != raw['SCREEN_ID'] or row['notebook_id'] != raw['NOTEBOOK_ID']:
            raise ValueError('Mismatched source group identity: ' + row['sample_id'])
        missing = [field for field in ('reactants', 'products', 'screen_id') if not row[field]]
        if missing:
            exclusions.append({'sample_id': row['sample_id'], 'reason': 'missing_required_input', 'fields': missing, 'canonical_row': canonical_index[row['sample_id']]})
            continue
        pairs.append((row, raw))
    rows = [row for row, raw in pairs]
    components = [component_by_sample[row['sample_id']] for row in rows]
    folds_by_group = assign_folds(components)
    folds = [folds_by_group[group] for group in components]
    row_folds = assign_folds([row['sample_id'] for row in rows])
    overlaps = {field: relation_overlap(rows, folds, field) for field in RELATIONS}
    if any(overlaps.values()):
        raise ValueError('A declared relation crosses the proposed grouped folds')
    panels = defaultdict(set)
    panel_source = {}
    panel_rows = Counter()
    flat_to_retained, retained_to_raw = (defaultdict(set), defaultdict(set))
    field_counts = Counter()
    records = []
    for (row, raw), group, fold in zip(pairs, components, folds, strict=True):
        raw_context = context(raw, RAW_CONTEXT)
        field_counts.update(raw_context.keys())
        retained_context = context(json.loads(row['conditions_json']), list(json.loads(row['conditions_json'])))
        panel = identity('panel', [row['screen_id'], row['chemical_split_group'], row['yield_measurement_type'], clean(raw['Product_UV_Analysis_Wavelength_nm'], True)])
        candidate = identity('candidate', raw_context)
        cell = identity('cell', [panel, candidate])
        panels[panel].add(candidate)
        panel_rows[panel] += 1
        panel_source.setdefault(panel, {'screen_id': row['screen_id'], 'notebook_id': row['notebook_id'], 'chemical_split_group': row['chemical_split_group'], 'component_id': group, 'outer_fold': fold, 'reactants': row['reactants'], 'products': row['products'], 'measurement_type': row['yield_measurement_type'], 'uv_wavelength_nm': clean(raw['Product_UV_Analysis_Wavelength_nm'], True), 'example_source_record_id': row['source_record_id']})
        flat_key = (panel, encode(context(row, FLAT_CONTEXT)))
        retained_key = (panel, encode(retained_context))
        flat_to_retained[flat_key].add(encode(retained_context))
        retained_to_raw[retained_key].add(encode(raw_context))
        records.append({'sample_id': row['sample_id'], 'canonical_row': canonical_index[row['sample_id']], 'source_record_id': row['source_record_id'], 'component_id': group, 'outer_fold': fold, 'row_diagnostic_fold': row_folds[row['sample_id']], 'panel_id': panel, 'candidate_id': candidate, 'cell_id': cell, 'inputs': {'reactants': row['reactants'], 'products': row['products'], 'source_context': raw_context}})
    eligible_panels = {panel for panel, candidates in panels.items() if len(candidates) >= 5}
    cell_counts = Counter((record['cell_id'] for record in records))
    for record in records:
        record['ranking_eligible'] = record['panel_id'] in eligible_panels
        record['cell_row_count'] = cell_counts[record['cell_id']]
    bh_path = data_dir / 'yield/buchwald_hartwig_ood.csv.gz'
    bh_hash = sha(bh_path)
    with gzip.open(bh_path, 'rt', encoding='utf8', newline='') as handle:
        bh = list(csv.DictReader(handle))
    transformations = {row['chemical_split_group'] for row in rows}
    reactants = {row['reactants'] for row in rows}
    bh_shared_transform = sum((row['chemical_split_group'] in transformations for row in bh))
    bh_shared_reactants = sum((row['reactants'] in reactants for row in bh))
    bh_after_exclusions = [row for row in bh if row['source'] != 'HiTEA 2024' and row['chemical_split_group'] not in transformations and (row['reactants'] not in reactants)]
    output.mkdir(parents=True, exist_ok=True)
    membership = physical_path(output / 'memberships.jsonl.gz')
    membership.parent.mkdir(parents=True, exist_ok=True)
    with membership.open('wb') as handle:
        with gzip.GzipFile(fileobj=handle, mode='wb', filename='', mtime=0) as gz:
            for record in records:
                gz.write((encode(record) + '\n').encode())
    independent_sample = set(random.Random(2027).sample(sorted(panels), 200))
    flat_collapses = Counter((key[0] for key, values in flat_to_retained.items() if len(values) > 1))
    raw_collapses = Counter((key[0] for key, values in retained_to_raw.items() if len(values) > 1))
    challenge_sample = set(sorted(set(panels) - independent_sample, key=lambda panel: (-(flat_collapses[panel] + raw_collapses[panel]), panel))[:60])
    cards_path = output / 'panel_cards.csv'
    cards = []
    for panel in sorted(panels):
        cards.append({'panel_id': panel, **panel_source[panel], 'rows': panel_rows[panel], 'recorded_candidates': len(panels[panel]), 'ranking_candidate': panel in eligible_panels, 'flat_context_collapse_cells': flat_collapses[panel], 'retained_context_collapse_cells': raw_collapses[panel], 'independent_probability_sample': panel in independent_sample, 'separate_targeted_sample': panel in challenge_sample, 'primary_reviewer': '', 'primary_status': 'pending', 'independent_reviewer': '', 'independent_status': 'pending', 'adjudication': '', 'notes': ''})
    with cards_path.open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cards[0]))
        writer.writeheader()
        writer.writerows(cards)
    counts = Counter(folds)
    report = {'schema': 'synrxn.kdd-preparation.v1', 'status': 'planning_memberships; no predictive experiments', 'preparation_code_sha256': sha(Path(__source_file__)), 'target_meaning': 'Recorded Product_Yield_PCT_Area_UV, not independently calibrated isolated yield', 'source': {'canonical_file': 'Data/yield/hitea.csv.gz', 'canonical_sha256': source_hash, 'upstream_url': RAW_URL, 'upstream_sha256': RAW_SHA256, 'canonical_rows': len(canonical_rows), 'upstream_rows': len(raw_rows), 'exact_id_and_target_matches': len(canonical_rows), 'eligible_rows': len(rows), 'excluded_rows': len(exclusions), 'exclusion_reasons': dict(Counter((field for item in exclusions for field in item['fields'])))}, 'source_context_fields': RAW_CONTEXT, 'source_context_field_nonempty': dict(field_counts), 'observed_screens': len({row['screen_id'] for row in rows}), 'components_without_reactant_relation': len(set(component_ids(rows, RELATIONS[:-1]))), 'components_with_all_relations': len(set(components)), 'all_canonical_components_with_all_relations': len(set(all_components)), 'grouping_relations': RELATIONS, 'grouped_fold_relation_overlaps': overlaps, 'row_diagnostic_relation_overlaps': {field: relation_overlap(rows, [row_folds[r['sample_id']] for r in rows], field) for field in RELATIONS}, 'outer_fold_rows': {str(fold): counts[fold] for fold in range(5)}, 'fold_policy': 'For test fold f, validation=(f+1)%5; other three folds train; no target stratification', 'panels': len(panels), 'ranking_panels_at_least_5_candidates': len(eligible_panels), 'ranking_panel_rows': sum((panel_rows[panel] for panel in eligible_panels)), 'ranking_panels_by_fold': dict(Counter((str(record['outer_fold']) for record in {record['panel_id']: record for record in records if record['ranking_eligible']}.values()))), 'recorded_input_cells': len(cell_counts), 'cells_with_multiple_rows': sum((n > 1 for n in cell_counts.values())), 'flat_cells_collapsing_retained_contexts': sum((len(values) > 1 for values in flat_to_retained.values())), 'retained_cells_collapsing_raw_contexts': sum((len(values) > 1 for values in retained_to_raw.values())), 'bh_replication': {'canonical_sha256': bh_hash, 'rows': len(bh), 'rows_labeled_hitea_source': sum((row['source'] == 'HiTEA 2024' for row in bh)), 'rows_sharing_transform': bh_shared_transform, 'rows_sharing_reactants': bh_shared_reactants, 'rows_after_union_of_exclusions': len(bh_after_exclusions), 'remaining_sources': dict(Counter((row['source'] for row in bh_after_exclusions))), 'qualification': 'Exclusions use declared canonical keys and known source labels only'}, 'membership_sha256': sha(membership), 'initial_panel_cards_sha256': sha(cards_path), 'panel_review_sampling': {'population': len(panels), 'seed': 2027, 'independent_simple_random_sample': len(independent_sample), 'additional_targeted_context_collapse_sample': len(challenge_sample), 'samples_disjoint': not bool(independent_sample & challenge_sample), 'human_review_status': 'not performed; all cards pending'}, 'qualification': 'Recorded relation disjointness and input availability are verified; hidden experimental dependence, physical reference correctness and pretraining exposure are not certified. Memberships are proposed and must be versioned before fitting.'}
    if sha(source_path) != source_hash:
        raise ValueError('Canonical source changed during preparation')
    if sha(raw_path) != RAW_SHA256 or sha(bh_path) != bh_hash:
        raise ValueError('Original HiTEA or BH source changed during preparation')
    (output / 'exclusions.json').write_text(json.dumps(exclusions, indent=2) + '\n')
    (output / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'Data')
    parser.add_argument('--raw-source', type=Path, default=ROOT / 'kdd/preparation/raw/hitea.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'kdd/preparation')
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    report = prepare(args.data_dir, args.raw_source, args.output, args.fetch)
    print(json.dumps({key: report[key] for key in ['status', 'observed_screens', 'components_with_all_relations', 'panels', 'ranking_panels_at_least_5_candidates', 'outer_fold_rows']}))
__configure_runtime(globals(), 'kdd/prepare_cohort.py')
if __name__ == '__main__':
    main()
