"""Explicit access to retired, guarded historical utilities.

Importing this module does not load any old workflow. Paper generators and
other unguarded scripts remain source archives and cannot be loaded here.
Active construction uses the family and builder interfaces instead.
"""
from __future__ import annotations

import ast
import re
import sys
from types import ModuleType

from synrxn.reconstruction.provenance.compat import _main_guard, load_legacy_module
from synrxn.reconstruction.paths import legacy_path


def load_tool(name: str) -> ModuleType:
    """Load definitions from one guarded script without executing its CLI.

    Names are single Python identifiers, never paths. The original logical
    ``__file__`` keeps historical source locators and digest pins stable.
    This deliberately provides no generic command-line execution interface.
    """
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
        raise ValueError('Use one historical tool name without a path or suffix')
    relative = 'script/' + name + '.py'
    resource = legacy_path(relative)
    if not any(_main_guard(node) for node in ast.parse(resource.read_bytes()).body):
        raise ValueError('Unguarded historical workflows are archived source only: ' + name)
    module_name = 'synrxn.reconstruction.provenance.source_history._' + name
    if module_name in sys.modules:
        return sys.modules[module_name]
    module = ModuleType(module_name)
    module.__package__ = 'synrxn.reconstruction'
    sys.modules[module_name] = module
    try:
        load_legacy_module(module.__dict__, relative)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module
