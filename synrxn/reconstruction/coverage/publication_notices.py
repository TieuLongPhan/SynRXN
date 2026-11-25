# Direct implementation; historical source identity is explicit below.
"""Generate current per-table terms while preserving historical curation notices."""
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/coverage/publication_notices.py')
import hashlib
import json
from synrxn.reconstruction.paths import HistoricalPath as Path
import shutil
from synrxn.reconstruction.coverage.license_policy import PNAS_EDUCATIONAL, components

def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def build(repo, release_tables, legacy):
    output = repo / 'kdd/coverage/publication_licenses'
    output.mkdir(parents=True, exist_ok=True)
    files = []
    history = output / 'historical-curation-notice.txt'
    history.write_text('HISTORICAL CURATION NOTICE — NOT THE CURRENT PER-TABLE LICENSE LIST\n\nThe following is the preserved SynRXN dataset notice for an earlier release. Its blanket per-table CC BY descriptions do not override third-party NC, GNU or database terms. For the current compilation use CURRENT-DATA-TERMS.txt and release_manifest.json. The historical text is retained for curator attribution and an auditable correction.\n\n' + (repo / 'Data/LICENSE').read_text())
    files.append(history)
    text = ['CURRENT SYNRXN DATA REDISTRIBUTION TERMS', '', 'Each table retains the source grants below. This compilation has no blanket license that removes upstream conditions.', 'CC BY-NC tables may be shared and adapted for noncommercial purposes with attribution and change notices. No commercial-use grant is asserted for them.', 'LicenseRef-PNAS-NC-Educational is a custom publisher permission for the reviewed factual table extraction, limited to noncommercial educational use with the full source reference. It is not a Creative Commons license.', 'GPL/LGPL, ODbL and CC BY-SA conditions remain attached to their respective data or source components. Other tables do not inherit an NC condition merely by appearing in this collection.', 'Original source tables and prior release metadata are retained unchanged for provenance. This notice corrects the current publication labels; it does not manufacture a new upstream license.', '', 'TABLES']
    legacy_by_key = {(r['namespace'], r['key']): r for r in legacy['tables']}
    evidence = {e['id']: e for e in legacy['evidence']}
    used_evidence = set()
    for table in sorted(release_tables, key=lambda r: (r['namespace'], r['key'])):
        text += ['', table['namespace'] + ':' + table['key'], 'Path: ' + table['path'], 'License expression: ' + table['license_expression'], 'Partition: ' + table['partition'], 'Commercial use allowed by these grants: ' + str(table.get('commercial_use_allowed', True))]
        text += table.get('license_obligations', [])
        row = legacy_by_key.get((table['namespace'], table['key']))
        if row:
            text.append('Source/curation: ' + row['reason'])
            for name in row['evidence_ids']:
                used_evidence.add(name)
                entry = evidence.get(name, {})
                if entry.get('url'):
                    text.append('Evidence/source: ' + entry['url'])
            lock_path = row.get('source_lock_path')
            if lock_path and (repo / lock_path).is_file():
                lock = json.loads((repo / lock_path).read_text())
                for citation in lock.get('citations', []):
                    text.append('Attribute: ' + citation.get('title', '') + ' DOI: ' + citation.get('doi', ''))
        else:
            text.append('Source: ' + table.get('source_url', 'See branch catalog and its attribution notice.'))
    current = output / 'CURRENT-DATA-TERMS.txt'
    current.write_text('\n'.join(text) + '\n')
    files.append(current)
    if any((PNAS_EDUCATIONAL in components(table['license_expression']) for table in release_tables)):
        pnas = output / 'PNAS-NC-Educational.txt'
        pnas.write_text('LicenseRef-PNAS-NC-Educational\n\nThis identifier records the original-table permission in the Standard PNAS License Terms for volumes 106–118 (2009–2021).\nPublisher terms: https://www.pnas.org/author-center/publication-charges/standard-pnas-license-terms\nOriginal source: Huang et al. (2015), Panoramic view of a superfamily of phosphatases through substrate profiling, PNAS 112(16), E1974–E1983. DOI:10.1073/pnas.1423570112.\nCurated source: Esther Heid and William H. Green, Machine Learning of Reaction Properties via Learned Representations of the Condensed Graph of Reaction; https://github.com/hesther/reactiondatabase.\n\nPNAS permits reuse of original figures and tables for noncommercial and educational purposes with the full journal reference. Commercial table reuse requires permission from PNAS. No endorsement is implied.\nThe current SynRXN compilation applies this permission only to property/phosphatase, a factual tabular extraction with attributed reaction curation. It does not assert permission to redistribute the article or entire supporting-information package.\nThe separate CC-BY-4.0 curation notice does not remove these publisher conditions or assert a Creative Commons license for the original PNAS material. See CURRENT-DATA-TERMS.txt for the complete per-table expression and provenance.\n')
        files.append(pnas)
    for name in sorted(used_evidence):
        entry = evidence.get(name, {})
        path = repo / entry.get('path', '')
        if entry.get('status') != 'captured' or not path.is_file():
            continue
        if not any((word in (entry.get('url', '') + '/' + path.name).lower() for word in ['/license', '/copying', 'legalcode'])):
            continue
        destination = output / (hashlib.sha256(name.encode()).hexdigest()[:12] + '-' + path.name.replace('/', '_'))
        shutil.copyfile(path, destination)
        files.append(destination)
    nc_legalcode = repo / 'kdd/coverage/rights_recheck/policy/cc_by_nc_4_legalcode.txt'
    if nc_legalcode.is_file():
        destination = output / 'CC-BY-NC-4.0.txt'
        shutil.copyfile(nc_legalcode, destination)
        files.append(destination)
    return [{'path': path.relative_to(repo).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size} for path in files]
__configure_runtime(globals(), 'kdd/coverage/publication_notices.py')
