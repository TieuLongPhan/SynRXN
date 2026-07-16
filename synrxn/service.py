"""Optional read-only HTTP service over verified SynRXN Parquet artifacts."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any, Optional

from .data import DatasetCatalog
from .query import QueryEngine, sha256_path, validate_parquet_index

LOGGER = logging.getLogger("synrxn.service")


def create_app(  # noqa: C901
    parquet_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    max_page_size: int = 1_000,
    max_concurrent_queries: int = 4,
    catalog: Optional[DatasetCatalog] = None,
):
    """Create a stateless FastAPI application with bounded query endpoints."""
    try:
        from fastapi import FastAPI, HTTPException, Query, Request, Response
        from pydantic import BaseModel, ConfigDict
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "HTTP service requires: pip install synrxn[service]"
        ) from exc
    if max_page_size < 1 or max_concurrent_queries < 1:
        raise ValueError("service limits must be positive")

    class FlexiblePayload(BaseModel):
        model_config = ConfigDict(extra="allow")

    class HealthPayload(BaseModel):
        status: str
        datasets: int
        errors: list[str]

    class MetricsPayload(BaseModel):
        requests: int
        errors: int
        query_seconds: float

    class CatalogPayload(BaseModel):
        count: int
        items: list[dict[str, Any]]

    class RowsPayload(BaseModel):
        dataset: str
        limit: int
        offset: int
        rows: list[dict[str, Any]]
        provenance: dict[str, Any]

    class ReleasePayload(BaseModel):
        schema_version: Optional[str]
        version: Optional[str]
        doi: Optional[str]
        generated_at: Optional[str]
        summary: Optional[dict[str, Any]]

    # FastAPI resolves postponed annotations from module globals.
    globals().update({"Request": Request, "Response": Response})

    parquet_root = (
        Path(parquet_dir or os.environ.get("SYNRXN_PARQUET_DIR", "Parquet"))
        .expanduser()
        .resolve()
    )
    manifest_file = (
        Path(manifest_path or os.environ.get("SYNRXN_MANIFEST", "manifest.json"))
        .expanduser()
        .resolve()
    )
    catalog = catalog or DatasetCatalog()
    engine = QueryEngine(parquet_root, catalog)
    missing = [
        f"{item.task.value}/{item.name}"
        for item in catalog.list()
        if not (parquet_root / item.task.value / f"{item.name}.parquet").is_file()
    ]
    readiness_errors = validate_parquet_index(parquet_root)
    release = None
    if manifest_file.is_file():
        try:
            release = json.loads(manifest_file.read_text(encoding="utf8"))
        except (OSError, json.JSONDecodeError):
            readiness_errors.append("release manifest is invalid")
    else:
        readiness_errors.append("release manifest is missing")
    if release and release.get("schema_version") != "1.0":
        readiness_errors.append("release manifest schema is unsupported")
    indexed_release = (engine.index or {}).get("release")
    if indexed_release and release:
        if indexed_release.get("manifest_sha256") != sha256_path(manifest_file):
            readiness_errors.append(
                "parquet release was built from a different manifest"
            )
        if indexed_release.get("version") != release.get("dataset", {}).get("version"):
            readiness_errors.append("parquet and canonical release versions differ")
    expected_ids = {f"{item.task.value}/{item.name}" for item in catalog.list()}
    indexed_ids = {
        item.get("dataset") for item in (engine.index or {}).get("artifacts", [])
    }
    if expected_ids != indexed_ids:
        readiness_errors.append("parquet release index does not match the catalog")
    readiness_errors.extend(f"missing artifact: {value}" for value in missing)
    slots = BoundedSemaphore(max_concurrent_queries)

    app = FastAPI(
        title="SynRXN read-only API",
        version="1.1.1",
        description="Bounded catalog and record queries over immutable SynRXN releases.",
    )
    app.state.catalog = catalog
    app.state.engine = engine
    app.state.readiness_errors = readiness_errors
    app.state.metrics = {"requests": 0, "errors": 0, "query_seconds": 0.0}

    @app.middleware("http")
    async def request_observability(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        started = time.perf_counter()
        app.state.metrics["requests"] += 1
        try:
            response = await call_next(request)
        except Exception:
            app.state.metrics["errors"] += 1
            LOGGER.exception(
                "request failed id=%s path=%s", request_id, request.url.path
            )
            raise
        elapsed = time.perf_counter() - started
        response.headers["X-Request-ID"] = request_id
        response.headers["Server-Timing"] = f"app;dur={elapsed * 1000:.2f}"
        LOGGER.info(
            "request id=%s method=%s path=%s status=%s elapsed=%.4f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response

    @contextmanager
    def query_slot():
        if not slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="query capacity is exhausted")
        started = time.perf_counter()
        try:
            yield
        finally:
            app.state.metrics["query_seconds"] += time.perf_counter() - started
            slots.release()

    def dataset_or_404(task: str, name: str):
        try:
            return catalog.get(task, name)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc

    def artifact_path(dataset) -> Path:
        return parquet_root / dataset.task.value / f"{dataset.name}.parquet"

    def cache_headers(
        response: Response, path: Path, checksum: Optional[str] = None
    ) -> None:
        response.headers["ETag"] = f'"{checksum or sha256_path(path)}"'
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"

    @app.get("/health", tags=["operations"], response_model=HealthPayload)
    async def health():
        return {
            "status": "ready" if not readiness_errors else "not_ready",
            "datasets": len(catalog.list()),
            "errors": readiness_errors,
        }

    @app.get("/metrics", tags=["operations"], response_model=MetricsPayload)
    async def metrics():
        return dict(app.state.metrics)

    @app.get("/v1/datasets", tags=["catalog"], response_model=CatalogPayload)
    async def list_datasets(
        task: Optional[str] = None,
        has_split: Optional[bool] = None,
        search: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1_000),
    ):
        try:
            records = catalog.list(task=task, has_split=has_split)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if search:
            needle = search.lower()
            records = [
                item
                for item in records
                if needle
                in " ".join(
                    [
                        item.name,
                        item.title,
                        item.description,
                        item.benchmark_role,
                        *item.targets,
                        *item.citations,
                    ]
                ).lower()
            ]
        return {
            "count": len(records),
            "items": [asdict(item) for item in records[:limit]],
        }

    @app.get(
        "/v1/datasets/{task}/{name}",
        tags=["catalog"],
        response_model=FlexiblePayload,
    )
    async def describe_dataset(task: str, name: str, response: Response):
        dataset = dataset_or_404(task, name)
        path = artifact_path(dataset)
        if not path.is_file():
            raise HTTPException(
                status_code=503, detail="derived artifact is unavailable"
            )
        checksum = engine.artifact_checksum(dataset.task, dataset.name)
        cache_headers(response, path, checksum)
        return {
            **asdict(dataset),
            "artifact": {
                "format": "parquet",
                "bytes": path.stat().st_size,
                "sha256": checksum,
            },
        }

    @app.get(
        "/v1/datasets/{task}/{name}/rows",
        tags=["records"],
        response_model=RowsPayload,
    )
    async def dataset_rows(
        task: str,
        name: str,
        request: Request,
        response: Response,
        columns: Optional[str] = None,
        order_by: Optional[str] = None,
        descending: bool = False,
        limit: Optional[int] = Query(None, ge=1),
        offset: int = Query(0, ge=0),
    ):
        limit = limit or min(100, max_page_size)
        if limit > max_page_size:
            raise HTTPException(
                status_code=422, detail=f"limit exceeds service maximum {max_page_size}"
            )
        dataset = dataset_or_404(task, name)
        path = artifact_path(dataset)
        if not path.is_file():
            raise HTTPException(
                status_code=503, detail="derived artifact is unavailable"
            )
        filters = {}
        for raw in request.query_params.getlist("filter"):
            if "=" not in raw:
                raise HTTPException(
                    status_code=422, detail="filter must be column=value"
                )
            column, value = raw.split("=", 1)
            filters.setdefault(column, []).append(value)
        normalized_filters = {
            column: values[0] if len(values) == 1 else values
            for column, values in filters.items()
        }
        try:
            with query_slot():
                frame = engine.query(
                    dataset.task,
                    dataset.name,
                    columns=[value for value in (columns or "").split(",") if value]
                    or None,
                    filters=normalized_filters,
                    order_by=order_by,
                    descending=descending,
                    source_order=order_by is None,
                    limit=limit,
                    offset=offset,
                )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.warning("bounded query failed for %s/%s: %s", task, name, exc)
            raise HTTPException(
                status_code=503, detail="query could not be completed"
            ) from exc
        cache_headers(
            response, path, engine.artifact_checksum(dataset.task, dataset.name)
        )
        clean = frame.astype(object).where(frame.notna(), None)
        return {
            "dataset": f"{dataset.task.value}/{dataset.name}",
            "limit": limit,
            "offset": offset,
            "rows": clean.to_dict("records"),
            "provenance": frame.attrs.get("synrxn", {}),
        }

    @app.get(
        "/v1/datasets/{task}/{name}/stats",
        tags=["records"],
        response_model=FlexiblePayload,
    )
    async def dataset_stats(task: str, name: str, response: Response):
        dataset = dataset_or_404(task, name)
        path = artifact_path(dataset)
        if not path.is_file():
            raise HTTPException(
                status_code=503, detail="derived artifact is unavailable"
            )
        cache_headers(
            response, path, engine.artifact_checksum(dataset.task, dataset.name)
        )
        try:
            with query_slot():
                return engine.stats(dataset.task, dataset.name)
        except HTTPException:
            raise
        except Exception as exc:
            LOGGER.warning("statistics failed for %s/%s: %s", task, name, exc)
            raise HTTPException(
                status_code=503, detail="statistics could not be completed"
            ) from exc

    @app.get("/v1/releases/{version}", tags=["releases"], response_model=ReleasePayload)
    async def release_info(version: str, response: Response):
        if release is None or release.get("dataset", {}).get("version") != version:
            raise HTTPException(status_code=404, detail="release not found")
        dataset = release["dataset"]
        cache_headers(response, manifest_file)
        return {
            "schema_version": release.get("schema_version"),
            "version": dataset.get("version"),
            "doi": dataset.get("doi"),
            "generated_at": release.get("generated_at"),
            "summary": dataset.get("summary"),
        }

    return app


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "HTTP service requires: pip install synrxn[service]"
        ) from exc
    app = create_app()
    if app.state.readiness_errors:
        joined = "; ".join(app.state.readiness_errors)
        raise RuntimeError(f"service release validation failed: {joined}")
    host = os.environ.get("SYNRXN_HOST", "127.0.0.1")
    port = int(os.environ.get("SYNRXN_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
