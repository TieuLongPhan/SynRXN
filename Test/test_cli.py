from synrxn.__main__ import main, parse_args


def test_cli_exposes_public_maintenance_commands():
    assert parse_args(["verify-manifest"]).command == "verify-manifest"
    assert parse_args(["validate", "--quick"]).quick is True
    assert parse_args(["datasets", "list", "--task", "property"]).task == "property"
    assert parse_args(["parquet", "build", "--output-dir", "Parquet"]).parquet_command == "build"
    assert parse_args(["catalog-assets"]).command == "catalog-assets"


def test_cli_verify_manifest_delegates(tmp_path):
    missing = tmp_path / "missing.json"
    assert main(["verify-manifest", "--manifest", str(missing)]) == 2


def test_cli_catalog_list_and_describe(capsys):
    assert main(["datasets", "list", "--task", "rbl"]) == 0
    assert "rbl/complex" in capsys.readouterr().out
    assert main(["datasets", "describe", "property", "rgd1"]) == 0
    assert "RGD1 activation energies" in capsys.readouterr().out
