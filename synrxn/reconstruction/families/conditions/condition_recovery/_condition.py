# Direct implementation; historical source identity is explicit below.
"""Build normalized SynRXN condition artifacts from pinned local sources."""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('script/build_condition_dataset.py')
import argparse
import json
import sys
from synrxn.reconstruction.paths import HistoricalPath as Path
from typing import Iterable
REPO_ROOT = Path(__source_file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from synrxn.reconstruction.families.conditions.condition_recovery._normalize import ORD_HTE_ALLOWLIST, build_openexp_condition, build_ord_hte_condition, build_orderly_condition

def parse_args(argv: Iterable[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orderly-train', type=Path)
    parser.add_argument('--orderly-test', type=Path)
    parser.add_argument('--ord-source-dir', type=Path, help='Directory containing the 16 pinned ORD parquet files.')
    parser.add_argument('--openexp', type=Path, help='Pinned OpenExp.tar from OSF file 3dv4k.')
    parser.add_argument('--output', type=Path, default=Path('Data/condition/orderly_condition.csv.gz'))
    parser.add_argument('--ord-output', type=Path, default=Path('Data/condition_sharealike/ord_hte_condition.csv.gz'))
    parser.add_argument('--openexp-output', type=Path, default=Path('Data/condition_sharealike/openexp.csv.gz'))
    parser.add_argument('--report', type=Path, default=Path('Data/sources/condition/build_report.json'))
    parser.add_argument('--combine-reports', type=Path, nargs='+', help='Combine previously generated report fragments without rebuilding data.')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(list(argv) if argv is not None else None)

def main(argv: Iterable[str] | None=None) -> int:
    args = parse_args(argv)
    if args.combine_reports:
        reports = []
        for report_path in args.combine_reports:
            reports.extend(json.loads(report_path.read_text(encoding='utf8'))['datasets'])
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({'schema_version': '1.1', 'datasets': reports}, indent=2, sort_keys=True) + '\n', encoding='utf8')
        print(f'combined {len(reports)} dataset reports into {args.report}')
        return 0
    if bool(args.orderly_train) != bool(args.orderly_test):
        raise SystemExit('--orderly-train and --orderly-test must be supplied together')
    if not any((args.orderly_train, args.ord_source_dir, args.openexp)):
        raise SystemExit('select ORDerly, ORD, and/or OpenExp source inputs')
    if args.dry_run:
        if args.orderly_train:
            print(f'would build orderly_condition from {args.orderly_train} and {args.orderly_test}')
        if args.ord_source_dir:
            print(f'would build ord_hte_condition from {args.ord_source_dir}')
        if args.openexp:
            print(f'would build openexp from {args.openexp}')
        return 0
    reports = []
    if args.orderly_train:
        reports.append(build_orderly_condition(args.orderly_train, args.orderly_test, args.output))
    if args.ord_source_dir:
        source_paths = [args.ord_source_dir / f'{dataset_id}.parquet' for dataset_id in ORD_HTE_ALLOWLIST]
        reports.append(build_ord_hte_condition(source_paths, args.ord_output))
    if args.openexp:
        reports.append(build_openexp_condition(args.openexp, args.openexp_output))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    payload = {'schema_version': '1.1', 'datasets': reports}
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf8')
    for report in reports:
        print(f"{report['dataset']}: {report['accepted_rows']} accepted, {report['quarantined_rows']} quarantined")
    return 0
__configure_runtime(globals(), 'script/build_condition_dataset.py')
if __name__ == '__main__':
    raise SystemExit(main())
