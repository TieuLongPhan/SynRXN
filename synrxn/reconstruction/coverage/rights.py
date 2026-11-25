# Direct implementation; historical source identity is explicit below.
"""Offline validation of source-specific redistribution grant evidence.

The assertions in a catalog are insufficient by themselves: the cited local
primary-source snapshot must exist, match its pin, contain the claimed grant,
and (for record APIs) identify the cited record. No network permission inference.
"""
from __future__ import annotations
from synrxn.reconstruction.paths import historical_file as __historical_file
from synrxn.reconstruction.provenance.compat import configure_runtime as __configure_runtime
__source_file__ = __historical_file('kdd/coverage/rights.py')
import hashlib
import json
from synrxn.reconstruction.paths import HistoricalPath as Path
import re
from urllib.parse import unquote, urlparse
from synrxn.reconstruction.coverage.license_policy import PNAS_EDUCATIONAL, components

def _path(repo: Path, value):
    if not isinstance(value, str):
        return None
    path = (repo / value).resolve()
    return path if path.is_relative_to(repo.resolve()) else None

def _snapshot(repo, reference, checksum=None):
    if isinstance(reference, dict):
        checksum = reference.get('sha256')
        reference = reference.get('path')
    path = _path(repo, reference)
    if path is None or not path.is_file() or (not isinstance(checksum, str)) or (not re.fullmatch('[0-9a-f]{64}', checksum)):
        return None
    data = path.read_bytes()
    return data if hashlib.sha256(data).hexdigest() == checksum else None

def _canonical_license(value):
    if isinstance(value, dict):
        value = value.get('id', value.get('name', value.get('url', '')))
    text = str(value or '').strip().lower()
    aliases = {'mit-license': 'MIT', 'mit': 'MIT', 'apache-2.0': 'Apache-2.0', 'cc0-1.0': 'CC0-1.0', 'cc-by-4.0': 'CC-BY-4.0', 'cc by 4.0': 'CC-BY-4.0', 'cc-by-3.0': 'CC-BY-3.0', 'cc by 3.0': 'CC-BY-3.0', 'cc-by-sa-4.0': 'CC-BY-SA-4.0', 'cc by-sa 4.0': 'CC-BY-SA-4.0'}
    for version in ['3.0', '4.0']:
        for variant in ['by-nc', 'by-nc-sa', 'by-sa']:
            spdx = 'CC-' + variant.upper() + '-' + version
            aliases['cc-' + variant + '-' + version] = spdx
            aliases['cc ' + variant + ' ' + version] = spdx
    if text in aliases:
        return aliases[text]
    for fragment, license_ in [('/licenses/by-nc-sa/4.0', 'CC-BY-NC-SA-4.0'), ('/licenses/by-nc/4.0', 'CC-BY-NC-4.0'), ('/licenses/by-nc-sa/3.0', 'CC-BY-NC-SA-3.0'), ('/licenses/by-nc/3.0', 'CC-BY-NC-3.0'), ('/licenses/by-sa/3.0', 'CC-BY-SA-3.0'), ('/licenses/by-sa/4.0', 'CC-BY-SA-4.0'), ('/licenses/by/4.0', 'CC-BY-4.0'), ('/licenses/by/3.0', 'CC-BY-3.0'), ('/publicdomain/zero/1.0', 'CC0-1.0')]:
        if fragment in text:
            return license_
    return str(value or '')

def _url_identity(url):
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme not in {'https', 'http'}:
        return None
    host, path = (parsed.netloc.lower(), parsed.path.rstrip('/'))
    if host == 'zenodo.org':
        match = re.fullmatch('/(?:api/)?records/(\\d+)', path)
        if match:
            return ('zenodo', match[1])
    if host in {'api.figshare.com', 'figshare.com'}:
        match = re.search('/(?:articles|article)/(\\d+)(?:/versions/(\\d+))?$', path)
        if match:
            return ('figshare', match[1])
    if host == 'api.crossref.org':
        match = re.fullmatch('/works/(10\\.\\d{4,9}/.+)', unquote(path), re.IGNORECASE)
        if match:
            return ('crossref', match[1].lower())
    return (host, path)

def _primary_url(url):
    identity = _url_identity(url)
    return bool(identity and identity[0] in {'zenodo', 'figshare', 'crossref', 'raw.githubusercontent.com', 'github.com', 'www.ebi.ac.uk', 'rest.uniprot.org', 'sparql.rhea-db.org', 'www.brenda-enzymes.org', 'creativecommons.org', 'www.gnu.org'})

def _grant_matches(data: bytes, url: str, expected: str):
    """Recognize the grant itself, not a citation or arbitrary source filename."""
    if expected == 'LicenseRef-PNAS-NC-Educational':
        if _url_identity(url) != ('www.pnas.org', '/author-center/publication-charges/standard-pnas-license-terms'):
            return False
        if data.startswith(b'%PDF-'):
            import subprocess
            try:
                data = subprocess.run(['pdftotext', '-', '-'], input=data, capture_output=True, check=True).stdout
            except (OSError, subprocess.SubprocessError):
                return False
        grant = ' '.join(data.decode('utf-8', errors='replace').lower().split())
        return all((fragment in grant for fragment in ['standard pnas license terms', '2009', '2021', 'anyone may, without requesting permission, use original figures or tables', 'noncommercial and educational use', 'full journal reference', 'commercial reuse of figures and tables', 'requires permission from pnas']))
    if not _primary_url(url):
        return False
    text = data.decode('utf-8', errors='replace')
    expected = _canonical_license(expected)
    try:
        record = json.loads(text)
    except (ValueError, TypeError):
        record = None
    identity = _url_identity(url)
    if identity[0] == 'crossref':
        if not isinstance(record, dict) or record.get('message-type') != 'work':
            return False
        message = record.get('message', {})
        if str(message.get('DOI', '')).lower() != identity[1]:
            return False
        licenses = message.get('license', [])
        return any((isinstance(item, dict) and item.get('content-version') == 'vor' and (item.get('delay-in-days') == 0) and (_canonical_license(item.get('URL')) == expected) for item in licenses))
    if isinstance(record, dict) and identity[0] in {'zenodo', 'figshare'}:
        if identity[0] == 'zenodo':
            record_id = record.get('record_id', record.get('id'))
            if str(record_id) != identity[1]:
                return False
            grant = record.get('metadata', {}).get('license', record.get('license'))
        else:
            identifiers = [str(record.get('id', '')), str(record.get('url', '')), str(record.get('figshare_url', '')), str(record.get('citation', ''))]
            if not any((re.search('(?<!\\d)' + re.escape(identity[1]) + '(?!\\d)', value) for value in identifiers)):
                return False
            grant = record.get('license')
        return _canonical_license(grant) == expected
    if isinstance(record, dict) and identity[0] == 'rest.uniprot.org':
        if record.get('id') != 'license':
            return False
        text = record.get('content', '')
    lower = ' '.join(text.lower().split())
    if expected.startswith('CC-') or expected == 'CC0-1.0':
        if any((marker in lower for marker in ['attribution-noderivatives', 'attribution-noncommercial-noderivatives', 'by-nd/', 'by-nc-nd/', 'no derivatives'])):
            return False
        if '-NC-' not in expected and any((marker in lower for marker in ['attribution-noncommercial', 'by-nc/', 'by-nc-'])):
            return False
        urls = re.findall('https?://creativecommons\\.org/[^\\s\\"\'<>\\\\)]+', text)
        if any((_canonical_license(url) == expected for url in urls)):
            return True
        if expected == 'CC-BY-SA-4.0':
            return 'attribution-sharealike 4.0 international' in lower and 'noncommercial' not in lower and ('no derivatives' not in lower) and ('reproduce and share' in lower) and ('adapted material' in lower)
        if expected in {'CC-BY-NC-4.0', 'CC-BY-NC-SA-4.0'}:
            heading = 'attribution-noncommercial' + ('-sharealike' if '-SA-' in expected else '') + ' 4.0 international'
            return heading in lower and 'reproduce and share' in lower and ('noncommercial purposes only' in lower)
        return False
    if expected == 'MIT':
        return 'permission is hereby granted, free of charge' in lower and 'to deal in the software' in lower
    if expected == 'Apache-2.0':
        return 'apache license' in lower and 'version 2.0' in lower
    if expected.startswith(('GPL-', 'LGPL-')):
        version = expected.split('-')[1].removesuffix('.0')
        heading = 'gnu lesser general public license' if expected.startswith('LGPL-') else 'gnu general public license'
        return heading in lower and bool(re.search('version\\s+' + re.escape(version), lower))
    if expected in {'BSD-2-Clause', 'BSD-3-Clause'}:
        return 'redistribution and use in source and binary forms' in lower
    return False

def _source_snapshot(source):
    rights = source.get('rights', {})
    grant = rights.get('grant_snapshot')
    if grant:
        return (grant, rights.get('grant_snapshot_sha256'), rights.get('grant_url'))
    grant = source.get('license_evidence')
    if grant:
        return (grant, source.get('license_evidence_sha256'), source.get('license_evidence_url'))
    return (None, None, None)

def source_has_evidence(source, catalog, repo):
    scope = str(source.get('rights', {}).get('scope', '')).lower()
    if any((marker in scope for marker in ['software_only', 'software only', 'code_only', 'code only'])):
        return False
    reference, checksum, url = _source_snapshot(source)
    identity = _url_identity(url)
    if identity and identity[0] == 'crossref':
        rights = source.get('rights', {})
        if rights.get('scope') != 'publisher_bundled_supplementary_data' or str(rights.get('publication_doi', '')).lower() != identity[1]:
            return False
    data = _snapshot(Path(repo), reference, checksum)
    expected = source.get('data_license_spdx', source.get('license', source.get('license_record')))
    if PNAS_EDUCATIONAL in components(expected):
        return False
    return bool(data is not None and _grant_matches(data, url, expected))

def _bio_locked_evidence(catalog, repo):
    data = _snapshot(repo, catalog.get('source_lock'), catalog.get('source_lock_sha256'))
    if data is None:
        return []
    lock = json.loads(data)
    evidence = []
    for asset in lock.get('assets', []):
        if asset.get('rights_evidence') or asset.get('metadata_snapshot'):
            evidence.append((asset.get('url'), asset))
    return evidence

def table_has_evidence(table, source, catalog, repo):
    repo = Path(repo)
    if not source_has_evidence(source, catalog, repo):
        return False
    chain = table.get('license_chain')
    if not chain:
        expression = table.get('license_expression', table.get('license', source.get('license')))
        return _canonical_license(expression) == _canonical_license(source.get('data_license_spdx', source.get('license', source.get('license_record'))))
    expression = table.get('license_expression', '')
    claimed = {_canonical_license(x.strip()) for x in expression.split(' AND ')}
    components = {_canonical_license(component.get('license')) for component in chain}
    if claimed != components:
        return False
    evidence = _bio_locked_evidence(catalog, repo)
    if not evidence:
        return False
    audit_data = _snapshot(repo, catalog.get('rights_audit_path'), catalog.get('rights_audit_sha256'))
    if audit_data is None:
        return False
    audit = json.loads(audit_data)
    if table.get('id') not in audit.get('release_table_ids', []) or table.get('id') in audit.get('quarantined_table_ids', []):
        return False
    source_ids = table.get('source_ids', [table.get('source_id')])
    sources = {s['id']: s for s in catalog.get('sources', [])}
    if not source_ids or any((sid not in sources or not source_has_evidence(sources[sid], catalog, repo) for sid in source_ids)):
        return False
    for component in chain:
        identity = _url_identity(component.get('evidence'))
        matched = False
        for url, reference in evidence:
            if identity != _url_identity(url):
                continue
            data = _snapshot(repo, reference)
            if data is not None and _grant_matches(data, url, component.get('license')):
                matched = True
                break
        if not matched:
            return False
    return True
__configure_runtime(globals(), 'kdd/coverage/rights.py')
