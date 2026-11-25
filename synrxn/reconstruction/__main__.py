"""One entry point for reconstruction, manifests and local analytical storage."""
from __future__ import annotations

import argparse
from importlib import import_module
import runpy
import sys


COMMANDS = {
    'families': '.families.__main__',
    'expansion': '.expansion.__main__',
    'layout': '.catalogue.layout',
    'storage': '.catalogue.storage',
    'index': '.catalogue.index',
    'full': '.families.suite',
    'depictions': '.presentation.depictions',
    'manuscript': '.presentation.manuscript',
    'figures': '.presentation.figures',
}
COMMAND_ALIASES = {'builders': '.families._importers', 'acquire': COMMANDS['expansion']}
SCOPE_ALIASES = {
    'list': ['--list'],
    'plan': ['--plan'],
    'verify': ['--mode', 'verify'],
    'rebuild': ['--mode', 'rebuild'],
    'reproduce': [],
}
SCRIPT_COMMANDS = {
    'registry': 'catalogue.registry',
    'hierarchy': 'catalogue.hierarchy',
    'prepare-ncs': 'families.outcomes.reaction_success._bh_hte_ood',
    'prepare-hitea': 'families.outcomes.yield_conversion._hitea',
    'validate-plan': 'families._validate_study',
    'validate-literature': 'provenance.literature',
}


def main(argv=None):
    """Dispatch to the selected reconstruction command."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {'-h', '--help'}:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('command', choices=[*COMMANDS, 'registry', 'hierarchy'])
        parser.print_help()
        return 0

    command, rest = args[0], args[1:]
    if command in COMMANDS:
        return import_module(COMMANDS[command], __package__).main(rest)
    if command in COMMAND_ALIASES:
        return import_module(COMMAND_ALIASES[command], __package__).main(rest)
    if command in SCOPE_ALIASES:
        from .families.__main__ import _run_selection

        return _run_selection([*SCOPE_ALIASES[command], *rest])
    if command not in SCRIPT_COMMANDS:
        raise SystemExit('Unknown reconstruction command: ' + command)

    module = SCRIPT_COMMANDS[command]
    previous = sys.argv
    try:
        sys.argv = [command if command in {'registry', 'hierarchy'} else module, *rest]
        runpy.run_module(__package__ + '.' + module, run_name='__main__')
    finally:
        sys.argv = previous
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
