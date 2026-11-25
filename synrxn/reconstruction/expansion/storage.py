"""Deterministic, bounded-memory Parquet output for heterogeneous observations."""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

from .catalog import group_split, sha256


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def encoded(value) -> str:
    return json.dumps(clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def observation(source: str, table: str, index, parent: str, *, inputs: dict,
                targets: dict, context: dict | None = None, native: dict | None = None,
                evidence: str = "reported_measurement", split: str | None = None) -> dict:
    import hashlib

    if not parent:
        raise ValueError("adapter must declare a parent rather than randomly split rows")
    record_id = f"{source}:{table}:{index}"
    group_id = source + ":" + hashlib.sha256(parent.encode()).hexdigest()[:24]
    return {"record_id": record_id, "source_id": source, "source_record_id": str(index),
            "group_id": group_id, "parent_id": parent, "split": split or group_split(group_id),
            "evidence": evidence, "inputs_json": encoded(inputs), "targets_json": encoded(targets),
            "context_json": encoded(context or {}), "native_json": encoded(native or {})}


def write_table(root: Path, source: dict, name: str, records, *, unit: str,
                task_fields: dict, assets: list[dict], qualification: str = "",
                role: str = "primary", units: dict | None = None) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    relative = Path(source["partition"]) / source["id"] / (name + ".parquet")
    output = root / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.part")
    fields = ["record_id", "source_id", "source_record_id", "group_id", "parent_id", "split",
              "evidence", "inputs_json", "targets_json", "context_json", "native_json"]
    schema = pa.schema([(key, pa.string()) for key in fields], metadata={
        b"synrxn.schema": b"expansion.v1", b"synrxn.source": source["id"].encode(),
        b"synrxn.license": source["license_spdx"].encode()})
    count, groups, splits, evidence = 0, set(), {}, {}
    # Source record IDs need not be unique (e.g. replicated experiments). The
    # physical row ordinal is the stable suffix when this occurs in an adapter.
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        buffer = []
        for row in records:
            buffer.append(row)
            count += 1
            groups.add(row["group_id"])
            splits[row["split"]] = splits.get(row["split"], 0) + 1
            evidence[row["evidence"]] = evidence.get(row["evidence"], 0) + 1
            if len(buffer) >= 4096:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    temporary.replace(output)
    return {"key": source["id"] + "/" + name, "source_id": source["id"], "path": relative.as_posix(),
            "rows": count, "parent_groups": len(groups), "counting_unit": unit, "role": role,
            "split_counts": splits, "evidence_counts": evidence,
            "split_protocol": "sha256-parent-80-10-10-seed-2027", "tasks": list(task_fields),
            "task_fields": task_fields, "target_units": units or {},
            "qualification": qualification, "source_assets": assets, "sha256": sha256(output),
            "bytes": output.stat().st_size}
