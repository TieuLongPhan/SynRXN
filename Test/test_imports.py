"""Public imports and command aliases retain their canonical identities."""
import gzip
import importlib
import pickle
import subprocess
import sys

import pytest

from synrxn._compat import MODULE_ALIASES
from synrxn.reconstruction.paths import REPO, source_path


def test_retired_module_imports_are_identical_to_canonical_modules():
    for old, canonical in MODULE_ALIASES.items():
        previous = importlib.import_module(old)
        current = importlib.import_module(canonical)
        assert previous is current
        assert current.__spec__.name == canonical
        suffix = '/__init__.py' if current.__spec__.submodule_search_locations is not None else '.py'
        historical = old.replace('.', '/') + suffix
        assert not (REPO / historical).exists()
        if not old.startswith('synrxn.expansion'):
            assert source_path(historical).is_file()


def test_historical_cli_commands_resolve_without_root_wrapper_files():
    for module in ('synrxn.query', 'synrxn.query_benchmark',
                   'synrxn.build_manifest', 'synrxn.validate_data',
                   'synrxn.verify_manifest'):
        result = subprocess.run([sys.executable, '-m', module, '--help'],
                                cwd=REPO, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert 'usage:' in result.stdout


def test_source_importer_command_survives_removal_of_the_builders_directory():
    current = subprocess.check_output([
        sys.executable, '-S', '-m', 'synrxn.reconstruction',
        'families', 'import-source', '--list',
    ], cwd=REPO)
    previous = subprocess.check_output([
        sys.executable, '-S', '-m', 'synrxn.reconstruction.builders', '--list',
    ], cwd=REPO)
    assert previous == current
    assert b'synrxn.reconstruction.families.representation.atom_mapping.' in current
    assert not (REPO / 'synrxn/reconstruction/builders').exists()


@pytest.mark.parametrize("command", [
    ['synrxn.expansion'],
    ['synrxn.reconstruction', 'expansion'],
])
def test_expansion_cli_routes_share_the_parser(command):
    result = subprocess.run([sys.executable, '-m', *command, '--help'],
                            cwd=REPO, capture_output=True, text=True, check=True)
    assert 'build,list,sources,verify,sample' in result.stdout


@pytest.mark.parametrize("command", [
    ['synrxn.reconstruction', 'expansion'],
    ['synrxn.reconstruction', 'acquire'],
    ['synrxn.reconstruction.acquisition'],
])
def test_expansion_acquisition_help_preserves_existing_commands(command):
    result = subprocess.run([sys.executable, '-m', *command, 'fetch', '--help'],
                            cwd=REPO, capture_output=True, text=True, check=True)
    assert '--source' in result.stdout and '--max-bytes' in result.stdout


def test_access_benchmark_spawn_uses_canonical_callable_identity(tmp_path):
    from synrxn import query_benchmark
    from synrxn.access import benchmark

    assert query_benchmark is benchmark
    assert pickle.loads(pickle.dumps(query_benchmark._run_case)) is benchmark._run_case
    data = tmp_path / 'Data'
    source = data / 'aam/ecoli.csv.gz'
    source.parent.mkdir(parents=True)
    source.write_bytes(gzip.compress(b'r_id,ground_truth\n1,C>>C\n2,O>>O\n', mtime=0))
    result = benchmark.benchmark_case('csv-projected', 'aam', 'ecoli',
                                      data, tmp_path / 'Parquet')
    assert result['rows'] == 2
    assert result['case'] == 'csv-projected'


def test_public_api_remains_available_without_eager_dataframe_import():
    program = (
        "import sys, synrxn; "
        "assert 'pandas' not in sys.modules; "
        "from synrxn import DataLoader, DatasetCatalog, DatasetMetadata, Task; "
        "assert all(x is not None for x in (DataLoader, DatasetCatalog, DatasetMetadata, Task))"
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
