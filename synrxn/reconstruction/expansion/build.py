"""Rebuild the expansion from pinned source assets; never modify legacy tables."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from synrxn.reconstruction.families._expansion_sources import ADAPTERS
from synrxn.reconstruction.paths import physical_path
from .catalog import sha256
from .storage import write_table

POLICY = {
    "C01": ("CC-BY-4.0", "main", "existing", "Already integrated as yield/buchwald_hartwig_ood; do not count twice."),
    "C02": ("CC-BY-4.0", "main", "admitted", "Zenodo data-record grant."),
    "C03": ("CC-BY-SA-4.0", "sharealike", "admitted", "ORD data license; Apache-2.0 applies separately to code."),
    "C04": ("CC-BY-4.0", "main", "existing", "Already integrated as condition/orderly; preserve upstream source attribution."),
    "C05": ("UNVERIFIED", "excluded", "terms_unresolved", "Accessible supplementary information alone does not establish a data redistribution grant."),
    "C06": ("MIT", "main", "admitted", "Author-provided Croissant dataset metadata explicitly declares MIT; README covers code and data."),
    "C07": ("CC-BY-4.0", "main", "admitted", "Zenodo data-record grant; preserve directions, solvents and resonance parent IDs."),
    "C08": ("MIT", "main", "admitted", "Figshare grant covers the native HDF5 asset."),
    "C09": ("CC-BY-4.0", "main", "admitted", "Zenodo data grant; software separately Apache-2.0."),
    "C10": ("CC-BY-4.0", "main", "admitted", "Zenodo grant and explicit data README; preserve source studies."),
    "C11": ("CC-BY-4.0", "main", "admitted", "Zenodo data-record grant."),
    "C12": ("CC-BY-4.0", "main", "admitted", "Figshare data-record grant."),
    "C13": ("CC-BY-4.0", "main", "admitted", "Zenodo compilation grant; retain source-paper and assay evidence."),
    "C14": ("MIT", "main", "admitted", "Primary Zenodo dataset 11494913 explicitly declares MIT; GitHub code is CC0. Retain UniProt/Rhea attribution and original terms."),
    "C15": ("MIT", "main", "admitted", "Zenodo data-record grant; retain original BRENDA/SABIO-RK attribution and measurement provenance."),
    "C16": ("MIT", "main", "admitted", "Official repository explicitly supplies code and datasets under MIT. Retain per-paper DOIs and endpoint definitions."),
    "C17": ("CC-BY-4.0", "main", "admitted", "Figshare data-record grant; patent-derived reactions are not quantitative selectivity experiments."),
    "C18": ("UNVERIFIED", "excluded", "terms_unresolved", "Existing Perera source remains excluded until a dataset-specific grant is verified."),
    "C19": ("UNVERIFIED", "excluded", "terms_unresolved", "Parrot code MIT does not resolve the underlying dataset redistribution terms."),
    "C20": ("CC-BY-4.0", "main", "admitted", "Official OC20NEB dataset page states CC BY 4.0. Preserve slab/cell/periodicity context."),
    "C21": ("CC-BY-4.0", "main", "admitted", "Zenodo data-record grant; preserve calculation methods and solvent groups."),
    "C22": ("UNVERIFIED", "excluded", "terms_unresolved", "No blanket data license verified in the pinned repository archive; individual rate libraries need source review."),
    "C23": ("UNVERIFIED", "excluded", "terms_unresolved", "MIT code does not establish all underlying active-site dataset terms. Preserve UniProt/M-CSA provenance and inferred reaction pairings."),
    "C24": ("CC-BY-4.0", "main", "admitted", "Official SIB/Zenodo data-record grant; manually annotated corpus is distinct from predicted text-mined expansion."),
    "C25": ("UNVERIFIED", "excluded", "terms_unresolved", "README limits MIT to benchmark code; underlying USPTO/FlowER data retain their original terms."),
    "C26": ("CC-BY-NC-ND", "excluded", "modified_redistribution_hold", "Official database pages specify NC/ND terms; no modified redistribution grant established."),
}


class BuildContext:
    def __init__(self, root: Path, source: dict, assets: list[dict], out: Path):
        self.root, self.source, self.out = root, source, out
        self.assets = {a["name"]: a for a in assets if a["source_id"] == source["id"]}
        self.tables, self.issues = [], []

    def has(self, name: str) -> bool:
        asset = self.assets.get(name)
        return bool(asset and asset.get("status") == "verified" and self.path(name).is_file())

    def path(self, name: str) -> Path:
        path = physical_path(self.assets[name]["path"], self.root).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("source asset escapes repository")
        return path

    def issue(self, asset: str, reason: str):
        self.issues.append({"asset": asset, "reason": reason})

    def emit(self, name: str, rows, *, unit: str, assets: list[str], tasks: dict, **kwargs) -> dict:
        used = []
        for name_ in assets:
            if not self.has(name_):
                raise ValueError("cannot build from an unverified source asset")
            asset = self.assets[name_]
            # Verify immediately before parsing: an old successful HTTP report
            # must not authorize a file which has subsequently changed.
            if sha256(self.path(name_)) != asset["sha256"]:
                raise ValueError(f"source bytes changed since acquisition: {name_}")
            used.append({k: asset[k] for k in ["name", "url", "sha256", "bytes_observed"]})
        result = write_table(self.out, self.source, name, rows, unit=unit, task_fields=tasks, assets=used, **kwargs)
        self.tables.append(result)
        print(self.source["id"], name, result["rows"], result["counting_unit"], flush=True)
        return result


def build(root: str | Path, output: str | Path | None = None, sources: list[str] | None = None) -> dict:
    root = Path(root).resolve()
    out = Path(output).resolve() if output else physical_path("Data/expansion", root)
    out.mkdir(parents=True, exist_ok=True)
    audit = json.loads(physical_path("kdd/integration/source_audit.json", root).read_text())
    acquisition = [json.loads(p.read_text()) for p in sorted(physical_path("kdd/integration/acquisition", root).glob("*.json"))]
    source_records, tables, issues = [], [], []
    if sources and (out / "catalog.json").exists():
        tables = [r for r in json.loads((out / "catalog.json").read_text())["tables"] if r["source_id"] not in sources]
    for candidate in audit["sources"]:
        key = candidate["id"]
        license_, partition, status, reason = POLICY[key]
        source = {k: candidate[k] for k in ["id", "name", "paper_urls", "data_urls", "project_urls"]}
        source.update(license_spdx=license_, partition=partition, admission=status, rights_evidence=reason)
        if "github" in candidate:
            source["revision"] = candidate["github"]["commit"]
        elif "record_id" in candidate:
            source["revision"] = candidate.get("doi", candidate["record_id"])
        if key == "C01":
            source["existing_artifact"] = "yield/buchwald_hartwig_ood.csv.gz"
        elif key == "C04":
            source["existing_artifact"] = "condition/orderly.csv.gz"
        ctx = BuildContext(root, source, acquisition, out)
        source["acquisition"] = {
            "verified_assets": sum(a.get("status") == "verified" for a in ctx.assets.values()),
            "failed_assets": sum(a.get("status") != "verified" for a in ctx.assets.values())}
        if status == "admitted" and (not sources or key in sources):
            if key in ADAPTERS:
                try:
                    ADAPTERS[key](ctx)
                except Exception as error:
                    ctx.issue("adapter", f"{type(error).__name__}: {error}")
            else:
                ctx.issue("adapter", "native adapter not yet available; source assets are not counted as usable tables")
            tables.extend(ctx.tables)
        source["integration_status"] = ("integrated" if any(t["source_id"] == key for t in tables)
                                        else "existing" if status == "existing"
                                        else "pending_acquisition_or_adapter" if status == "admitted" else status)
        source["issues"] = ctx.issues
        issues.extend({"source_id": key, **issue} for issue in ctx.issues)
        source_records.append(source)
        # Notices travel with each license partition; the project software's
        # license never replaces an upstream data license.
        if status == "admitted":
            directory = out / partition / key
            directory.mkdir(parents=True, exist_ok=True)
            notice = f"{key}: {source['name']}\nData license: {license_}\n{reason}\n\nPrimary source and attribution:\n"
            notice += "\n".join(source["paper_urls"] + source["data_urls"]) + "\n"
            (directory / "SOURCE-NOTICE.txt").write_text(notice)
    environment = {"python": platform.python_version()}
    for name in ["pandas", "pyarrow", "rdkit", "ord-schema", "openpyxl", "h5py"]:
        try:
            environment[name] = version(name)
        except PackageNotFoundError:
            environment[name] = "not installed"
    result = {"schema": "synrxn.expansion.v1", "built_at": datetime.now(timezone.utc).isoformat(),
              "survey_sha256": sha256(physical_path("kdd/integration/input_survey.md", root)),
              "source_audit_sha256": sha256(physical_path("kdd/integration/source_audit.json", root)),
              "legacy_manifest_sha256": sha256(root / "manifest.json"),
              "environment": environment, "sources": source_records,
              "tables": sorted(tables, key=lambda r: r["key"]), "issues": issues,
              "count_warning": "Physical records, trajectories, configurations, routes and derived variants must not be summed as independent reactions.",
              "legacy_scope": "The frozen 49-table candidate and exact SynRXN 1.0.0 baseline are separate from this companion."}
    temporary = out / "catalog.json.part"
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(out / "catalog.json")
    (root / "synrxn/data/expansion_sources.json").write_text(json.dumps({"schema": "synrxn.expansion-sources.v1", "sources": source_records}, indent=2, ensure_ascii=False) + "\n")
    return result
