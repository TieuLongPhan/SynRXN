# Direct implementation; historical source identity is explicit below.
"""Normalize the acquired, pinned FREA sources under a two-core CPU limit."""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('script/integrate_frea.py')
import os
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '2'
from datetime import datetime, timezone
import json
from synrxn.reconstruction.paths import HistoricalPath as Path
import sys
ROOT = Path(__source_file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synrxn.reconstruction.families.outcomes.reaction_feasibility._frea_source import build_frea

def main():
    acquisition = json.loads((ROOT / 'ncs/source_acquisition.json').read_text())
    reports = []
    for name, dataset, expert in [('FREA-expert', 'frea_expert', True), ('FREA-corpus', 'frea_corpus', False)]:
        clone = next((c for c in acquisition['clones'] if c['name'] == name))
        sources = [{'path': ROOT / p['path'], 'sha256': p['sha256'], 'rows': p['rows'], 'split': Path(p['path']).stem} for p in clone['profiles']]
        output = ROOT / 'Data/feasibility' / (dataset + '.csv.gz')
        report = build_frea(sources, output, dataset=dataset, expert=expert)
        lock = {'schema_version': '1.0', 'dataset_key': output.relative_to(ROOT / 'Data').as_posix(), 'source_url': clone['url'], 'source_commit_or_version': clone['commit'], 'source_file': [s['source_file'] for s in report['source_files']], 'source_files': {s['source_file']: s['sha256'] for s in report['source_files']}, 'source_sha256': sources[0]['sha256'] if len(sources) == 1 else None, 'normalized_input_file': output.name, 'normalized_input_sha256': report['sha256'], 'upstream_license': 'CC-BY-4.0', 'license_url': clone['url'] + '/blob/' + clone['commit'] + '/README.md', 'terms_evidence': clone['terms_evidence'], 'document_sha256': clone['document_sha256'], 'required_attribution': ['FREA benchmark authors', 'https://arxiv.org/abs/2610.06614'], 'redistribution_partition': 'main', 'retrieved_at': '2026-10-14', 'recorded_at': datetime.now(timezone.utc).isoformat(), 'build_report': report, 'limits': [clone['limit'], 'Generated labels and expert judgments are separate evidence tiers.', 'No declaration that these labels are experimental success/failure measurements.']}
        lock_path = ROOT / 'Data/sources/feasibility' / (dataset + '.json')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
        reports.append(report)
        print(json.dumps({'dataset': dataset, 'rows': report['rows'], 'label_counts': report['label_counts']}), flush=True)
    (ROOT / 'ncs/audit/frea-integration.json').write_text(json.dumps({'cpu_limit': 2, 'datasets': reports}, indent=2) + '\n')
__configure_runtime(globals(), 'script/integrate_frea.py')
if __name__ == '__main__':
    main()
