"""Checksum-pinned, read-only inspection of a small reference release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from synrxn.benchmarks.contracts import ReferenceRecord
from synrxn.reconstruction.paths import physical_path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ReferenceStore:
    """Inspect exact pilot revisions without a mutable database or hosted API."""

    def __init__(self, root: Path):
        self.root = physical_path(Path(root).expanduser().resolve())
        manifest_path = self.root / "artifact-manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema_version") != "1.0" or not manifest.get("files"):
            raise ValueError("invalid reference artifact manifest")
        self.release_id = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
        for name, expected in manifest["files"].items():
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("reference artifact escapes its release directory")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"reference release checksum mismatch: {name}")
        for name in ("observations.jsonl", "references.jsonl", "task-instances.jsonl"):
            if name not in manifest["files"]:
                raise ValueError(f"unmanifested reference asset: {name}")
        self.observations = read_jsonl(self.root / "observations.jsonl")
        self.references = read_jsonl(self.root / "references.jsonl")
        self.instances = read_jsonl(self.root / "task-instances.jsonl")
        self._by_id = {}
        for record in self.observations:
            key = record["observation_id"]
            if key in self._by_id:
                raise ValueError("duplicate observation accessions")
            self._by_id[key] = record
        for value in self.references:
            ReferenceRecord.from_dict(value)
        revisions = self._validate_reference_links()
        self._validate_task_links(revisions)
        self.review_evidence = (read_jsonl(self.root / "review-events.jsonl")
                                if "review-events.jsonl" in manifest["files"] else [])

    def _validate_reference_links(self) -> dict:
        revisions = {}
        for reference in self.references:
            key = (reference["reference_id"], reference["revision"])
            if key in revisions:
                raise ValueError("duplicate reference revisions")
            observation = self._by_id.get(reference["observation_id"])
            if observation is None or observation["source_record_id"] != reference["source_record_id"]:
                raise ValueError("reference observation/source identity mismatch")
            revisions[key] = reference
        for reference in self.references:
            previous = reference.get("previous_revision")
            if previous is not None:
                ancestor = revisions.get((reference["reference_id"], previous))
                if ancestor is None:
                    raise ValueError("reference correction history is incomplete")
                if (ancestor["observation_id"], ancestor["source_record_id"]) != (
                        reference["observation_id"], reference["source_record_id"]):
                    raise ValueError("reference revision changes source identity")
        return revisions

    def _validate_task_links(self, revisions: dict) -> None:
        instance_ids = set()
        for instance in self.instances:
            if instance["instance_id"] in instance_ids:
                raise ValueError("duplicate reference task instances")
            instance_ids.add(instance["instance_id"])
            reference = revisions.get((instance.get("reference_id"), instance.get("reference_revision")))
            if reference is None or reference["observation_id"] != instance["observation_id"]:
                raise ValueError("task instance/reference revision mismatch")

    def inspect(self, accession: str) -> dict:
        observation = self._by_id[accession]
        references = [r for r in self.references if r["observation_id"] == accession]
        latest = {}
        for reference in references:
            key = reference["reference_id"]
            if key not in latest or reference["revision"] > latest[key]["revision"]:
                latest[key] = reference
        return {"release_id": self.release_id, "observation": observation,
                "references": references, "latest_references": list(latest.values()),
                "review_evidence": [r for r in self.review_evidence if r["observation_id"] == accession],
                "task_instances": [i for i in self.instances if i["observation_id"] == accession]}

    def list(self, *, evidence_tier: str | None = None, offset: int = 0, limit: int = 20) -> dict:
        if offset < 0 or not 1 <= limit <= 1000:
            raise ValueError("reference pages require nonnegative offset and a 1–1000 limit")
        eligible = {r["observation_id"] for r in self.references
                    if evidence_tier is None or r["evidence_tier"] == evidence_tier}
        records = [r for r in self.observations if r["observation_id"] in eligible]
        return {"release_id": self.release_id, "total": len(records),
                "offset": offset, "limit": limit, "observations": records[offset:offset + limit]}
