# Direct implementation; historical source identity is explicit below.
"""Freeze full eight-task data views without fitting or invoking a predictor."""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('script/build_task_views.py')
import argparse
import hashlib
import json
import os
from synrxn.reconstruction.paths import HistoricalPath as Path
import sys
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
sys.path.insert(0, str(Path(__source_file__).resolve().parents[1]))
from synrxn.data.catalog import DatasetCatalog
from synrxn.data.release import selected_datasets
from synrxn.benchmarks.readiness import matches_selector
from synrxn.benchmarks.task_views import definitions, export_task_view

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('Data'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocols = definitions()
    catalog = DatasetCatalog(args.data_dir / 'metadata.yaml')
    summaries = []
    for task in protocols['tasks']:
        for partition in ('main', 'sharealike'):
            for dataset in selected_datasets(catalog, partition):
                key = dataset.dataset_id.key
                if not any((matches_selector({'key': key, 'targets': list(dataset.targets)}, selector) for selector in task['selectors'])):
                    continue
                target = args.output / task['id'] / (dataset.task.value + '--' + dataset.name)
                print('Exporting ' + task['id'] + ': ' + key, flush=True)
                report = export_task_view(task['id'], key, args.data_dir, target)
                summaries.append({'partition': partition, 'path': target.relative_to(args.output).as_posix(), 'report': report})
    (args.output / 'protocols.json').write_text(json.dumps(protocols, indent=2, ensure_ascii=False) + '\n')
    summary = {'schema': 'synrxn.full-task-views.v1', 'cpu_limit': 2, 'benchmark_execution': False, 'protocols': protocols, 'exports': summaries}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    files = {}
    for path in sorted(args.output.rglob('*')):
        if path.is_file():
            with path.open('rb') as handle:
                files[path.relative_to(args.output).as_posix()] = hashlib.file_digest(handle, 'sha256').hexdigest()
    (args.output / 'artifact-manifest.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
    print(f'Frozen {len(summaries)} task/dataset views', flush=True)
__configure_runtime(globals(), 'script/build_task_views.py')
if __name__ == '__main__':
    main()
