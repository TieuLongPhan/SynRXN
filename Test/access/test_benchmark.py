"""The packaged benchmark keeps its CLI and historical entry point usable."""
import json

from synrxn import query_benchmark


def test_benchmark_cli_preserves_cases_paths_and_report(tmp_path, monkeypatch):
    calls = []

    def benchmark(case, task, name, data_dir, parquet_dir):
        calls.append((case, task, name, data_dir, parquet_dir))
        return {"case": case, "seconds": 0.25, "peak_kib": 1024, "rows": 3}

    monkeypatch.setattr(query_benchmark, "benchmark_case", benchmark)
    monkeypatch.setattr(query_benchmark, "version", lambda name: "fixture-" + name)
    data = tmp_path / "data"
    parquet = tmp_path / "parquet"
    output = tmp_path / "benchmark.json"

    assert query_benchmark.main([
        "--data-dir", str(data), "--parquet-dir", str(parquet),
        "--datasets", "aam/ecoli", "--output", str(output),
    ]) == 0

    report = json.loads(output.read_text())
    assert [row["case"] for row in report["results"]] == [
        "csv-full", "csv-projected", "parquet-filtered", "parquet-first-page"
    ]
    assert all(row["dataset"] == "aam/ecoli" and row["rows"] == 3 for row in report["results"])
    assert report["environment"]["duckdb"] == "fixture-duckdb"
    assert all(call[1:] == ("aam", "ecoli", data.resolve(), parquet.resolve()) for call in calls)


def test_archived_query_launcher_reuses_the_package_and_no_root_script_tree():
    from synrxn.reconstruction.provenance.source_history import load_tool
    from synrxn.reconstruction.paths import REPO
    module = load_tool('benchmark_query_layer')
    assert module.main is query_benchmark.main
    assert module.benchmark_case is query_benchmark.benchmark_case
    assert module._run_case is query_benchmark._run_case
    assert module._run_case.__module__ == "synrxn.access.benchmark"
    assert not (REPO / 'script').exists()
