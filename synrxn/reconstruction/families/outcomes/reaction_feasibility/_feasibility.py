# Direct implementation; historical source identity is explicit below.
"""Build normalized SynRXN reaction-feasibility artifacts."""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('script/build_feasibility_dataset.py')
import argparse
import json
import sys
from synrxn.reconstruction.paths import HistoricalPath as Path
sys.path.insert(0, str(Path(__source_file__).resolve().parents[1]))
from synrxn.reconstruction.families.outcomes.reaction_feasibility._normalize import build_ursa_plausibility

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ursa', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path, default=Path('.cache/synrxn/runs/builders/feasibility_build_report.json'))
    parser.add_argument('--skip-checksum', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(f'would build ursa_plausibility from {args.ursa} -> {args.out}')
        return 0
    kwargs = {'expected_sha256': None} if args.skip_checksum else {}
    report = build_ursa_plausibility(args.ursa, args.out, **kwargs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({'datasets': [report]}, indent=2, sort_keys=True) + '\n', encoding='utf8')
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
__configure_runtime(globals(), 'script/build_feasibility_dataset.py')
if __name__ == '__main__':
    raise SystemExit(main())
