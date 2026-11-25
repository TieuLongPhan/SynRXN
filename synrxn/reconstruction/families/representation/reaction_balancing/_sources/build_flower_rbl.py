"""Reproduce and verify the FlowER base and hard reaction-balancing datasets."""
import argparse
import json
from pathlib import Path
import sys
import subprocess

from synrxn.reconstruction.paths import REPO as ROOT
from synrxn.reconstruction.paths import repo_path

from synrxn.reconstruction.families.representation.reaction_balancing._sources.flower_rbl import build
from synrxn.reconstruction.families.representation.reaction_balancing._sources.flower_rbl_verify import verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=repo_path('Data/tables/development/aam/flower'))
    parser.add_argument('--output', type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--hard', action='store_true', help='Build or verify only the hard dataset')
    modes.add_argument('--all', action='store_true', help='Build or verify both variants; --output is their root')
    parser.add_argument('--verify', action='store_true', help='Verify existing bundles without rebuilding')
    parser.add_argument('--prepare-source', action='store_true', help='Export pinned FlowER AAM source first')
    parser.add_argument('--base', type=Path, default=repo_path('Data/tables/development/rbl/flower'))
    parser.add_argument('--dump', type=Path, default=repo_path('Data/dumps/flower.dump.zip'))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('Workers must be positive')
    if args.verify and args.prepare_source:
        parser.error('--prepare-source cannot be combined with read-only --verify')
    if args.prepare_source:
        if args.source.name != 'flower':
            parser.error('--prepare-source requires a source directory named flower')
        subprocess.run([sys.executable, '-m', 'synrxn.reconstruction.families.representation.atom_mapping._sources.aam_benchmark', '--datasets', 'flower',
                        '--output', str(args.source.parent)], cwd=ROOT, check=True)
    root = args.output or repo_path('Data/tables/development/rbl')
    if args.all:
        selected = [('base', root / 'flower'), ('hard', root / 'flower-hard')]
    elif args.hard:
        selected = [('hard', args.output or root / 'flower-hard')]
    else:
        selected = [('base', args.output or root / 'flower')]
    base_path = selected[0][1] if args.all else args.base
    reports = []
    for kind, output in selected:
        if not args.verify and kind == 'base':
            build(args.source, output)
        elif not args.verify:
            build_hard_variant(args.source, base_path, args.dump, output, args.workers)
        reports.append(verify(output, kind))
    print(json.dumps(reports, indent=2))
    return reports


def build_hard_variant(source, base, dump, output, workers):
    """Import the raw-context processor only when rebuilding the hard variant."""
    from synrxn.reconstruction.families.representation.reaction_balancing._sources.flower_rbl_hard import build as build_hard
    return build_hard(source, base, dump, output, workers)


if __name__ == '__main__':
    main()
