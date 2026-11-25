"""Scientific reference and target-isolation checks on the acquired corpus."""
import json
import pyarrow.parquet as pq

from synrxn.reconstruction.paths import REPO, source_path
BASE = source_path(REPO / 'Data/coverage_expansion/knowledge')


def records(name):
    return pq.read_table(source_path(f'Data/coverage_expansion/knowledge/{name}.parquet')).to_pylist()


def test_extraction_offsets_and_relation_reference_closure():
    rows = records('documents')
    assert len(rows) == 1210
    for row in rows:
        inputs, targets = json.loads(row['inputs_json']), json.loads(row['targets_json'])
        assert set(inputs) == {'passages'}
        ids = {e['id'] for e in targets['entities']}
        for relation in targets['relations']:
            assert {n['refid'] for n in relation['nodes']} <= ids
        for entity in targets['entities']:
            for location in entity['locations']:
                matches = [p for p in inputs['passages'] if p['offset'] <= location['offset'] and
                           location['offset'] + location['length'] <= p['offset'] + len(p['text'])]
                assert matches
                p = matches[0]
                begin = location['offset'] - p['offset']
                assert p['text'][begin:begin + location['length']] == entity['text']


def test_curation_controls_keep_source_gold_and_document_split_together():
    gold = {r['source_record_id']: (json.loads(r['targets_json']), r['split']) for r in records('documents')}
    changed = 0
    for row in records('curation'):
        doc_id = row['source_record_id'].split(':')[0]
        target = json.loads(row['targets_json'])
        candidate = json.loads(row['inputs_json'])['candidate_record']
        assert target['corrected_record'] == gold[doc_id][0]
        assert row['split'] == gold[doc_id][1]
        assert target['needs_correction'] == (candidate != target['corrected_record'])
        if target['needs_correction']:
            assert len(candidate['relations']) + 1 == len(target['corrected_record']['relations'])
            assert candidate['entities'] == target['corrected_record']['entities']
            changed += 1
    assert changed == 1057


def test_retrieval_references_only_known_documents_and_gallery_hides_gold():
    known = {r['source_record_id'] for r in records('documents')}
    for row in records('retrieval'):
        inputs, targets = json.loads(row['inputs_json']), json.loads(row['targets_json'])
        assert len(inputs['chemical_identifiers']) == 2
        assert set(targets['relevant_document_ids']) <= known
        assert json.loads(row['context_json'])['gallery_inputs'] == ['document_id', 'passages']


def test_inverse_component_no_exact_catalyst_component_in_inputs():
    groups = {}
    for row in records('inverse_components'):
        inputs, targets = json.loads(row['inputs_json']), json.loads(row['targets_json'])
        catalyst = set(targets['catalyst_smiles'].split('.'))
        assert catalyst
        for key in ['reactants', 'products', 'reagents']:
            assert not (catalyst & set(inputs[key].split('.')))
        assert not any('catalyst' in key.lower() or 'yield' in key.lower() for key in inputs)
        lo, hi = inputs['desired_response_pct_interval']
        assert 0 <= lo <= targets['observed_response_pct'] <= hi <= 100
        assert row['parent_group'] not in groups or groups[row['parent_group']] == row['split']
        groups[row['parent_group']] = row['split']


def test_knowledge_data_rights_and_measured_failure_accounting():
    catalog = json.loads((BASE / 'catalog.json').read_text())
    assert len(catalog['tables']) == 5
    assert {s['license'] for s in catalog['sources']} == {'CC-BY-4.0', 'CC-BY-SA-4.0'}
    assert all(s['rights']['grant_snapshot_sha256'] for s in catalog['sources'])
    rows = records('scope_campaigns')
    assert len(rows) == 6865
    assert {json.loads(r['targets_json'])['native_hit_label'] for r in rows} == {0, 1}
    assert all('not globally novel' in json.loads(r['context_json'])['discovery_scope'] for r in rows)
