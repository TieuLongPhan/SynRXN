# Direct implementation; historical source identity is explicit below.
"""Build normalized SynRXN yield artifacts from pinned local source files."""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('script/build_yield_dataset.py')
import argparse
import json
import sys
from synrxn.reconstruction.paths import HistoricalPath as Path
from typing import Iterable
REPO_ROOT = Path(__source_file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from synrxn.reconstruction.families.outcomes.yield_conversion._normalize import build_buchwald_hartwig_ood, build_hitea_yield, build_suzuki_miyaura, build_uspto_yield

def parse_args(argv: Iterable[str] | None=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bh-input', type=Path)
    parser.add_argument('--uspto-archive', type=Path)
    parser.add_argument('--hitea-input', type=Path)
    parser.add_argument('--suzuki-input', type=Path)
    parser.add_argument('--out-dir', type=Path, default=Path('Data/yield'))
    parser.add_argument('--report', type=Path, default=Path('Data/sources/yield/build_report.json'))
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(list(argv) if argv is not None else None)

def main(argv: Iterable[str] | None=None) -> int:
    args = parse_args(argv)
    selected = [args.bh_input, args.uspto_archive, args.hitea_input, args.suzuki_input]
    if not any(selected):
        raise SystemExit('provide at least one of --bh-input, --uspto-archive, --hitea-input, or --suzuki-input')
    if args.dry_run:
        for label, source in (('buchwald_hartwig_ood', args.bh_input), ('uspto_yield', args.uspto_archive), ('hitea', args.hitea_input), ('suzuki_miyaura', args.suzuki_input)):
            if source:
                print(f'would build {label} from {source}')
        return 0
    reports = []
    if args.bh_input:
        reports.append(build_buchwald_hartwig_ood(args.bh_input, args.out_dir / 'buchwald_hartwig_ood.csv.gz'))
    if args.uspto_archive:
        reports.append(build_uspto_yield(args.uspto_archive, args.out_dir / 'uspto_yield.csv.gz'))
    if args.hitea_input:
        reports.append(build_hitea_yield(args.hitea_input, args.out_dir / 'hitea.csv.gz'))
    if args.suzuki_input:
        reports.append(build_suzuki_miyaura(args.suzuki_input, args.out_dir / 'suzuki_miyaura.csv.gz'))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    previous_reports = []
    if args.report.is_file():
        previous_payload = json.loads(args.report.read_text(encoding='utf8'))
        previous_reports = previous_payload.get('datasets', [])
    merged = {report['dataset']: report for report in previous_reports}
    merged.update({report['dataset']: report for report in reports})
    payload = {'schema_version': '1.0', 'datasets': list(merged.values())}
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf8')
    for report in reports:
        print(f"{report['dataset']}: {report['accepted_rows']} accepted, {report['quarantined_rows']} quarantined")
    return 0
__configure_runtime(globals(), 'script/build_yield_dataset.py')
if __name__ == '__main__':
    raise SystemExit(main())
