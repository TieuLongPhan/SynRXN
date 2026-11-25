# Direct implementation; historical source identity is explicit below.
"""Rebuild frozen M-CSA drawings and export SynRXN mechanism views (two CPUs).

The converter uses its own pinned SynKit 1.6.1 environment. The adapter uses
the current SynRXN mechanism backend; these environments are not mixed.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('script/rebuild_mcsa.py')
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from synrxn.reconstruction.paths import HistoricalPath as Path
import subprocess
import sys
import tarfile
ROOT = Path(__source_file__).resolve().parents[1]
BUNDLE = ROOT / 'ncs/sources/mcsa-v5'

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def extract(archive, output):
    """Accept regular files/directories only, with no escaping or duplicate paths."""
    output = Path(output).resolve()
    output.mkdir(parents=True)
    with tarfile.open(archive, 'r:gz') as bundle:
        seen = set()
        for member in bundle.getmembers():
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts or member.name in seen or (not (member.isfile() or member.isdir())) or (not (output / name).resolve().is_relative_to(output)):
                raise ValueError('Unsafe archive member: ' + member.name)
            seen.add(member.name)
        bundle.extractall(output, filter='data')

def main():
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--converter-python', required=True, type=Path, help='Python 3.12.13 with requirements-converter.txt installed')
    parser.add_argument('--snapshot-dir', type=Path, help='Optional already extracted, checksum-pinned native archive')
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--reuse-native-build', type=Path, help='Reuse a complete pinned native rebuild; adapter rechecks all native asset hashes')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError('Choose a new output directory; existing work is never overwritten')
    converter_python = args.converter_python.expanduser().absolute()
    if not converter_python.is_file():
        raise FileNotFoundError(converter_python)
    lock = json.loads((BUNDLE / 'bundle-lock.json').read_text())
    for name, pin in lock['files'].items():
        source = BUNDLE / name
        if source.stat().st_size != pin['bytes'] or digest(source) != pin['sha256']:
            raise ValueError('Frozen bundle differs: ' + name + '; run git lfs pull if needed')
    env = {**os.environ, 'PYTHONHASHSEED': '0', 'PYTHONUTF8': '1'}
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    subprocess.run([sys.executable, str(ROOT / 'script/prepare_mcsa_task.py'), '--check-backend'], check=True, cwd=ROOT, env=env)
    output.mkdir(parents=True)
    marker = output / 'build-report.json'
    report = {'schema': 'synrxn.mcsa-build.v1', 'status': 'incomplete', 'cpu_limit': 2, 'jobs': args.jobs, 'benchmark_execution': False, 'started_at': datetime.now(timezone.utc).isoformat(), 'bundle_lock_sha256': digest(BUNDLE / 'bundle-lock.json'), 'driver_sha256': digest(Path(__source_file__)), 'adapter_sha256': digest(ROOT / 'script/prepare_mcsa_task.py'), 'converter_python': str(converter_python), 'adapter_python': sys.executable}
    report['native_conversion_executed'] = args.reuse_native_build is None

    def save():
        marker.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    save()
    try:
        if args.reuse_native_build:
            native_build = args.reuse_native_build.resolve()
            report['reused_native_build'] = str(native_build)
        else:
            converter = output / 'work/converter'
            extract(BUNDLE / 'converter-v5.tar.gz', converter)
            for name, expected in lock['converter_files_sha256'].items():
                if digest(converter / name) != expected:
                    raise ValueError('Extracted converter differs: ' + name)
            if args.snapshot_dir:
                snapshot = args.snapshot_dir.resolve()
            else:
                snapshot = output / 'work/native-source'
                extract(BUNDLE / 'native-source-v5.tar.gz', snapshot)
            native_build = output / 'native-build'
            command = [str(converter_python), str(converter / 'scripts/rebuild_mcsa.py'), '--snapshot-dir', str(snapshot), '--output-dir', str(native_build), '--jobs', str(args.jobs)]
            with (output / 'native-build.log').open('w') as log:
                print('Rebuilding native drawings; progress: ' + str(output / 'native-build.log'), flush=True)
                subprocess.run(command, cwd=converter, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        report['adapter_sha256'] = digest(ROOT / 'script/prepare_mcsa_task.py')
        save()
        subprocess.run([sys.executable, str(ROOT / 'script/prepare_mcsa_task.py'), '--build-dir', str(native_build), '--output-dir', str(output / 'extension')], cwd=ROOT, env=env, check=True)
        report.update(status='complete', completed_at=datetime.now(timezone.utc).isoformat(), native_manifest_sha256=digest(native_build / 'manifest.json'), extension_manifest_sha256=digest(output / 'extension/artifact-manifest.json'))
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
    print(json.dumps(report, indent=2))
__configure_runtime(globals(), 'script/rebuild_mcsa.py')
if __name__ == '__main__':
    main()
