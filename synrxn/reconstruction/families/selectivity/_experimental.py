# Direct implementation; historical source identity is explicit below.
"""Build experimental SynRXN selectivity artifacts from pinned local inputs."""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('script/build_selectivity_dataset.py')
import argparse
import json
import sys
from synrxn.reconstruction.paths import HistoricalPath as Path
from typing import Iterable
REPO_ROOT = Path(__source_file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from synrxn.reconstruction.families.selectivity._normalize import build_hitea_chiral_resolution, build_regiosqm_eas

def parse_args(argv: Iterable[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hitea-input', type=Path)
    parser.add_argument('--regiosqm-dir', type=Path)
    parser.add_argument('--out-dir', type=Path, default=Path('Data/selectivity'))
    parser.add_argument('--report', type=Path, default=Path('Data/sources/selectivity/build_report.json'))
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(list(argv) if argv is not None else None)

def main(argv: Iterable[str] | None=None) -> int:
    args = parse_args(argv)
    if not args.hitea_input and (not args.regiosqm_dir):
        raise SystemExit('provide --hitea-input and/or --regiosqm-dir')
    if args.dry_run:
        if args.hitea_input:
            print(f'would build hitea_chiral_resolution from {args.hitea_input}')
        if args.regiosqm_dir:
            print(f'would build regiosqm_eas from {args.regiosqm_dir}')
        return 0
    reports = []
    if args.hitea_input:
        reports.append(build_hitea_chiral_resolution(args.hitea_input, args.out_dir / 'hitea_chiral_resolution.csv.gz'))
    if args.regiosqm_dir:
        reports.append(build_regiosqm_eas(args.regiosqm_dir, args.out_dir / 'regiosqm_eas.csv.gz'))
    previous = []
    if args.report.is_file():
        previous = json.loads(args.report.read_text(encoding='utf8')).get('datasets', [])
    merged = {report['dataset']: report for report in previous}
    merged.update({report['dataset']: report for report in reports})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({'schema_version': '1.0', 'datasets': list(merged.values())}, indent=2, sort_keys=True) + '\n', encoding='utf8')
    for report in reports:
        print(f"{report['dataset']}: {report['accepted_rows']} accepted, {report['quarantined_rows']} quarantined, {report['observed_labels']} observed labels")
    return 0
__configure_runtime(globals(), 'script/build_selectivity_dataset.py')
if __name__ == '__main__':
    raise SystemExit(main())
