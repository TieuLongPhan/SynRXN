import json
from pathlib import Path

import pytest
import httpx

from synrxn.data import DatasetCatalog
from synrxn.query import convert_release
from synrxn.service import create_app

pytest.importorskip("fastapi")
pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")


@pytest.fixture()
def app(tmp_path):
    full = DatasetCatalog()
    item = full.get("aam", "ecoli")

    class OneDatasetCatalog:
        def list(self, task=None, has_split=None):
            records = [item]
            if task is not None and str(task) not in {"aam", "Task.AAM"}:
                return []
            if has_split is not None:
                records = [value for value in records if value.has_split is has_split]
            return records

        def get(self, task, name):
            return full.get(task, name)

    catalog = OneDatasetCatalog()
    parquet_dir = tmp_path / "Parquet"
    convert_release(Path("Data"), parquet_dir, catalog)
    manifest = {
        "schema_version": "1.0",
        "generated_at": "2026-07-16T00:00:00Z",
        "dataset": {
            "version": "1.0.0",
            "doi": "10.0000/fixture",
            "summary": {"datasets": 1},
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf8")
    return create_app(
        parquet_dir,
        manifest_path,
        max_page_size=10,
        catalog=catalog,
    )


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as test_client:
        yield test_client


@pytest.mark.anyio
async def test_service_readiness_catalog_and_openapi(client):
    assert (await client.get("/health")).json()["status"] == "ready"
    response = await client.get("/v1/datasets", params={"search": "coli"})
    assert response.status_code == 200
    assert response.json()["count"] == 1
    openapi = (await client.get("/openapi.json")).json()
    assert "/v1/datasets/{task}/{name}/rows" in openapi["paths"]


@pytest.mark.anyio
async def test_service_rows_are_bounded_deterministic_and_cacheable(client):
    first = await client.get(
        "/v1/datasets/aam/ecoli/rows",
        params={"columns": "r_id,original_id", "limit": 2},
    )
    second = await client.get(
        "/v1/datasets/aam/ecoli/rows",
        params={"columns": "r_id,original_id", "limit": 2},
    )
    assert first.status_code == 200
    assert first.json()["rows"] == second.json()["rows"]
    assert first.json()["provenance"]["ordering"] == "source_row"
    assert first.headers["etag"]
    assert "immutable" in first.headers["cache-control"]
    over_limit = await client.get("/v1/datasets/aam/ecoli/rows?limit=11")
    assert over_limit.status_code == 422


@pytest.mark.anyio
async def test_service_rejects_arbitrary_query_fields_and_exposes_release(client):
    invalid = await client.get(
        "/v1/datasets/aam/ecoli/rows",
        params={"filter": "r_id;select=*", "limit": 1},
    )
    assert invalid.status_code == 422
    missing = await client.get("/v1/datasets/aam/missing/rows?limit=1")
    assert missing.status_code == 404, missing.text
    release = await client.get("/v1/releases/1.0.0")
    assert release.status_code == 200
    assert release.json()["doi"] == "10.0000/fixture"
    assert (await client.get("/metrics")).json()["requests"] >= 3
