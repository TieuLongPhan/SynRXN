"""Noncommercial redistribution is allowed without erasing its conditions."""
import hashlib
import json

from synrxn.reconstruction.coverage import rights, verify_release
from synrxn.reconstruction.coverage.license_policy import permitted, partition_for, commercial_allowed, obligations, PNAS_EDUCATIONAL


def test_nc_and_nc_sharealike_have_real_separate_partitions():
    for expression, partition in [('CC-BY-NC-4.0', 'noncommercial'),
                                  ('CC-BY-4.0 AND MIT AND CC-BY-NC-3.0', 'noncommercial'),
                                  ('CC-BY-NC-SA-4.0', 'noncommercial_sharealike')]:
        assert permitted(expression)
        assert not commercial_allowed(expression)
        assert partition_for(expression) == partition


def test_nd_and_internal_use_are_not_redistribution_grants():
    for expression in ['CC-BY-NC-ND-4.0', 'CC-BY-ND-4.0', 'MIT; academic use only', 'internal noncommercial use only', '']:
        assert not permitted(expression)


def test_gnu_and_database_source_conditions_survive_compilation():
    expression = 'GPL-3.0-only AND LGPL-2.1-only AND MIT'
    assert permitted(expression) and commercial_allowed(expression)
    assert partition_for(expression) == 'copyleft'
    assert partition_for('CC-BY-4.0 AND MIT AND ODbL-1.0') == 'database_sharealike'


def test_actual_nc_grant_clears_noncommercial_use_but_not_a_ccby_relabel(tmp_path):
    path = tmp_path / 'grant.json'
    path.write_text(json.dumps({'id': 42, 'license': {'name': 'CC BY-NC 4.0'}}))
    source = {'id': 'fixture', 'license': 'CC-BY-NC-4.0', 'redistribution_allowed': True,
              'rights': {'grant_url': 'https://api.figshare.com/v2/articles/42',
                         'grant_snapshot': {'path': 'grant.json', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}}
    table = {'id': 'native', 'license': 'CC-BY-NC-4.0'}
    decision = verify_release.release_decision(table, source, {'sources': [source]}, tmp_path)
    assert decision['release_eligible']
    assert decision['partition'] == 'noncommercial' and not decision['commercial_use_allowed']
    assert not verify_release.release_decision({**table, 'license': 'CC-BY-4.0'}, source, {}, tmp_path)['release_eligible']


def test_nc_legal_code_cannot_clear_an_unrestricted_ccby_grant():
    from synrxn.reconstruction.paths import physical_path
    path = physical_path('kdd/coverage/rights_recheck/policy/cc_by_nc_4_legalcode.txt')
    data = path.read_bytes()
    assert rights._grant_matches(data, 'https://creativecommons.org/licenses/by-nc/4.0/', 'CC-BY-NC-4.0')
    assert not rights._grant_matches(data, 'https://creativecommons.org/licenses/by-nc/4.0/', 'CC-BY-4.0')


def test_pnas_table_permission_retains_its_purpose_and_is_not_a_cc_license():
    expression = 'CC-BY-4.0 AND ' + PNAS_EDUCATIONAL
    assert permitted(expression) and not commercial_allowed(expression)
    assert partition_for(expression) == 'noncommercial_educational'
    assert any('noncommercial and educational' in item for item in obligations(expression))
    grant = b'''Standard PNAS License Terms, 2009-2021.
    Anyone may, without requesting permission, use original figures or tables
    for noncommercial and educational use with the full journal reference.
    Commercial reuse of figures and tables requires permission from PNAS.'''
    url = 'https://www.pnas.org/author-center/publication-charges/standard-pnas-license-terms'
    assert rights._grant_matches(grant, url, PNAS_EDUCATIONAL)
    assert not rights._grant_matches(grant, url, 'CC-BY-4.0')
    assert not rights._grant_matches(grant, url, 'CC-BY-NC-4.0')
    assert not rights._grant_matches(grant, 'https://example.org/pnas', PNAS_EDUCATIONAL)
    assert not rights._grant_matches(b'PNAS article available to read.', url, PNAS_EDUCATIONAL)


def test_pnas_publisher_policy_cannot_authorize_an_unrelated_catalog_source(tmp_path):
    path = tmp_path / 'pnas-policy.txt'
    path.write_text('''Standard PNAS License Terms, 2009-2021.
    Anyone may, without requesting permission, use original figures or tables
    for noncommercial and educational use with the full journal reference.
    Commercial reuse of figures and tables requires permission from PNAS.''')
    source = {'id': 'unrelated', 'license': PNAS_EDUCATIONAL,
              'rights': {'grant_url': 'https://www.pnas.org/author-center/publication-charges/standard-pnas-license-terms',
                         'grant_snapshot': {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}}
    assert not rights.source_has_evidence(source, {'sources': [source]}, tmp_path)
    assert not verify_release.release_decision({'license': PNAS_EDUCATIONAL}, source, {}, tmp_path)['release_eligible']
