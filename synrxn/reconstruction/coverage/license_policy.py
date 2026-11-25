# Direct implementation; historical source identity is explicit below.
"""Supported redistribution grants, retaining their individual conditions.

User clarified that verified noncommercial redistribution is acceptable.
An allowed license never substitutes for a dataset-specific grant or clears
an imported component with contradictory permissions.
"""
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/coverage/license_policy.py')
import re
PNAS_EDUCATIONAL = 'LicenseRef-PNAS-NC-Educational'
ALLOWED_DATA_LICENSES = {'CC0-1.0', 'CC-BY-3.0', 'CC-BY-4.0', 'CC-BY-SA-3.0', 'CC-BY-SA-4.0', 'CC-BY-NC-3.0', 'CC-BY-NC-4.0', 'CC-BY-NC-SA-3.0', 'CC-BY-NC-SA-4.0', 'MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', 'ODbL-1.0', 'GPL-2.0', 'GPL-2.0-only', 'GPL-2.0-or-later', 'GPL-3.0', 'GPL-3.0-only', 'GPL-3.0-or-later', 'LGPL-2.1', 'LGPL-2.1-only', 'LGPL-2.1-or-later', 'LGPL-3.0', 'LGPL-3.0-only', 'LGPL-3.0-or-later', PNAS_EDUCATIONAL}

def components(expression):
    if not isinstance(expression, str) or not expression.strip():
        return set()
    return set(re.split('\\s+AND\\s+', expression.strip()))

def permitted(expression):
    values = components(expression)
    return bool(values) and values <= ALLOWED_DATA_LICENSES

def commercial_allowed(expression):
    return permitted(expression) and (not any(('-NC-' in item for item in components(expression))))

def partition_for(expression):
    values = components(expression)
    nc = any(('-NC-' in item for item in values))
    sa = any(('-SA-' in item for item in values))
    gpl = any((item.startswith('GPL-') for item in values))
    lgpl = any((item.startswith('LGPL-') for item in values))
    if PNAS_EDUCATIONAL in values:
        return 'noncommercial_educational'
    if nc:
        return 'noncommercial_sharealike' if sa else 'noncommercial'
    if gpl:
        return 'copyleft'
    if 'ODbL-1.0' in values:
        return 'database_sharealike'
    if sa:
        return 'sharealike'
    return 'lgpl' if lgpl else 'main'

def obligations(expression):
    result = ['Retain source attribution, applicable license notices and descriptions of changes.']
    if not commercial_allowed(expression):
        result.append('Use and redistribution are limited by the applicable NonCommercial terms; no commercial-use grant is asserted.')
    if PNAS_EDUCATIONAL in components(expression):
        result.append('The PNAS permission covers original figures/tables for noncommercial and educational use with the full journal reference. This compilation applies it only to the cited factual table extraction; it does not assert a Creative Commons grant for PNAS material or permission to redistribute the article or entire supplement.')
    if any(('-SA-' in item for item in components(expression))):
        result.append('Comply with the applicable ShareAlike requirements for adapted material.')
    if 'ODbL-1.0' in components(expression):
        result.append('Retain the Open Database License notices and comply with database ShareAlike/source-access conditions where applicable; do not treat database rights as unrestricted CC BY rights.')
    if any((item.startswith(('GPL-', 'LGPL-')) for item in components(expression))):
        result.append('Retain the applicable GNU license and provide corresponding source/preferred editable material where its terms require it.')
    return result
__configure_runtime(globals(), 'kdd/coverage/license_policy.py')
