#!/usr/bin/env python3
"""
build_citation_from_manifest.py

Generate a CITATION.cff file from a manifest.json produced by build_manifest.py.

Usage:
    python build_citation_from_manifest.py \
        --manifest manifest.json \
        --output CITATION.cff \
        --repo-url "https://github.com/TieuLongPhan/SynRXN/tree/v0.0.5" \
        --doi "10.5281/zenodo.17297723" \
        --verbose
"""
from __future__ import annotations
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import yaml  # PyYAML if installed

    HAVE_PYYAML = True
except Exception:
    HAVE_PYYAML = False


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf8") as fh:
        return json.load(fh)


def iso_to_date(iso: str) -> Optional[str]:
    if not iso:
        return None
    try:
        # allow either "2025-10-09T..." or "2025-10-09"
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.date().isoformat()
    except Exception:
        # last resort: extract leading YYYY-MM-DD
        m = re.match(r"(\d{4}-\d{2}-\d{2})", iso)
        return m.group(1) if m else None


def split_name(full: str) -> Dict[str, str]:
    """
    Best-effort split of a human name into given-names and family-names.
    Heuristics:
      - if comma in name: assume "Family, Given" or "Family, Given Middlename"
      - else: last token is family-name, rest are given-names
      - fallback: put entire name as family-names
    """
    s = full.strip()
    if not s:
        return {"given-names": "", "family-names": ""}
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        family = parts[0]
        given = parts[1]
        return {"given-names": given, "family-names": family}
    tokens = s.split()
    if len(tokens) == 1:
        return {"given-names": "", "family-names": tokens[0]}
    # handle simple "First Middle Last" -> given = "First Middle", family = "Last"
    return {"given-names": " ".join(tokens[:-1]), "family-names": tokens[-1]}


def normalize_license(lic: Optional[str]) -> Optional[str]:
    if not lic:
        return None
    # simple normalization: if license is "CC-BY-4.0" or "CC-BY 4.0", return SPDX-like "CC-BY-4.0"
    s = str(lic).strip()
    s = s.replace(" ", "-")
    return s


def build_cff_dict(
    manifest: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    ds = manifest.get("dataset", {})
    prov = manifest.get("provenance", {}) or {}
    cff: Dict[str, Any] = {}
    # cff version
    cff["cff-version"] = "1.2.0"

    # title: prefer override, then dataset.title, fallback to repo info
    title = (
        overrides.get("title")
        or ds.get("title")
        or f"Dataset: {ds.get('title', 'Unknown')}"
    )
    cff["title"] = str(title)

    # version: prefer override, then ds.version, else try manifest.generated_at date
    version = overrides.get("version") or ds.get("version")
    if not version:
        # try to coerce generated_at YYYY.MM.DD or use manifest generated_at
        gen = manifest.get("generated_at")
        if gen:
            try:
                dt = datetime.fromisoformat(gen.replace("Z", "+00:00"))
                version = f"v{dt.strftime('%Y.%m.%d')}"
            except Exception:
                version = None
    if version:
        cff["version"] = str(version)

    # type: dataset or software? use override or default to 'dataset'
    cff["type"] = overrides.get("type") or "dataset"

    # message
    cff["message"] = (
        overrides.get("message")
        or "If you use this dataset, please cite it using the metadata from this file."
    )

    # abstract: prefer override, then dataset.description
    abstract = overrides.get("abstract") or ds.get("description")
    if abstract:
        # strip HTML tags if any (basic)
        abstract = re.sub(r"<[^>]+>", "", str(abstract)).strip()
        cff["abstract"] = abstract

    # date-released: prefer override then dataset.generated_at or manifest.generated_at
    date = (
        overrides.get("date_released")
        or ds.get("date_released")
        or manifest.get("generated_at")
    )
    date_iso = iso_to_date(date) if date else None
    if date_iso:
        cff["date-released"] = date_iso

    # doi: prefer override then dataset.doi then manifest-level doi if present
    doi = overrides.get("doi") or ds.get("doi") or manifest.get("doi")
    if doi:
        # canonicalize: allow either 10.5281/... or https://doi.org/...
        doi_str = str(doi).strip()
        # if it's a full https URL, leave it in doi field but also add identifiers later
        cff["doi"] = doi_str

    # license
    lic = overrides.get("license") or ds.get("license")
    if lic:
        cff["license"] = normalize_license(lic)

    # repository-code: prefer override, then ds.source_url, then first git remote if available
    repo = overrides.get("repository_code") or ds.get("source_url")
    if not repo:
        remotes = prov.get("remotes")
        if remotes:
            # try to extract github url from remotes string like "origin\tgit@github.com:User/Repo.git (fetch)\n..."
            m = re.search(r"(https?://[^\s]+|git@[^)\s]+)", remotes)
            if m:
                repo = m.group(1)
    if repo:
        cff["repository-code"] = str(repo)

    # authors: try to convert manifest authors list to CFF authors
    manifest_authors = ds.get("authors") or []
    cff_authors: List[Dict[str, str]] = []
    if manifest_authors:
        for a in manifest_authors:
            # a might be {"name": "...", "email": "...", "affiliation": "..."} or already split
            if isinstance(a, dict):
                name = a.get("name") or ""
                em = a.get("email")
                aff = a.get("affiliation") or a.get("affiliaton")  # common typo
                # if manifest already has given/family, keep them
                if a.get("given-names") or a.get("family-names"):
                    entry = {}
                    if a.get("given-names"):
                        entry["given-names"] = a.get("given-names")
                    if a.get("family-names"):
                        entry["family-names"] = a.get("family-names")
                    if em:
                        entry["email"] = em
                    if aff:
                        entry["affiliation"] = aff
                else:
                    splitted = split_name(name)
                    entry = {}
                    if splitted.get("given-names"):
                        entry["given-names"] = splitted["given-names"]
                    entry["family-names"] = splitted.get("family-names") or name
                    if em:
                        entry["email"] = em
                    if aff:
                        entry["affiliation"] = aff
            else:
                # manifest author entry could be a string
                splitted = split_name(str(a))
                entry = {}
                if splitted.get("given-names"):
                    entry["given-names"] = splitted["given-names"]
                entry["family-names"] = splitted.get("family-names")
            cff_authors.append(entry)
    else:
        # fallback: if git user available in provenance, use it
        git_root = prov.get("git_root")
        if git_root:
            # don't auto-add fake authors; leave authors absent if not provided
            pass
    if cff_authors:
        cff["authors"] = cff_authors

    # optional identifiers block: include commit hash as identifier if present
    identifiers: List[Dict[str, str]] = []
    commit = prov.get("commit")
    if commit:
        identifiers.append({"type": "commit", "value": commit})
    if doi and doi.startswith("http"):
        identifiers.append({"type": "doi", "value": doi})
    elif doi and re.match(r"^\d+\.\d+\/", str(doi)):
        identifiers.append({"type": "doi", "value": str(doi)})

    if identifiers:
        cff["identifiers"] = identifiers

    return cff


def dump_cff_yaml(cff: Dict[str, Any], outpath: Path) -> None:
    if HAVE_PYYAML:
        # Use safe_dump with explicit string quoting where needed
        with outpath.open("w", encoding="utf8") as fh:
            yaml.safe_dump(cff, fh, sort_keys=False, allow_unicode=True)
        print(f"Wrote CITATION.cff using PyYAML to {outpath}")
        return

    # Manual conservative YAML emitter (safe for our simple types)
    def esc(s: Any) -> str:
        if s is None:
            return ""
        s = str(s)
        # Simple heuristic: quote if special chars present or leading/trailing spaces
        if re.search(r"[:\-\[\]\{\},#&*!|>\'\"%@`]", s) or s.strip() != s or "\n" in s:
            # double-quote and escape internal quotes/backslashes
            s = s.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{s}"'
        return s

    lines: List[str] = []
    # keep ordering reasonable
    order = [
        "cff-version",
        "title",
        "version",
        "type",
        "message",
        "abstract",
        "date-released",
        "doi",
        "license",
        "repository-code",
        "authors",
        "identifiers",
    ]
    for key in order:
        if key not in cff:
            continue
        val = cff[key]
        if key == "authors" and isinstance(val, list):
            lines.append("authors:")
            for a in val:
                lines.append("  -")
                for k2, v2 in a.items():
                    lines.append(f"    {k2}: {esc(v2)}")
            continue
        if key == "identifiers" and isinstance(val, list):
            lines.append("identifiers:")
            for ident in val:
                lines.append("  -")
                for k2, v2 in ident.items():
                    lines.append(f"    {k2}: {esc(v2)}")
            continue
        lines.append(f"{key}: {esc(val)}")
    outpath.write_text("\n".join(lines) + "\n", encoding="utf8")
    print(f"Wrote CITATION.cff (manual emitter) to {outpath}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest", "-m", default="manifest.json", help="Path to manifest.json"
    )
    p.add_argument(
        "--output", "-o", default="CITATION.cff", help="Output CITATION.cff path"
    )
    p.add_argument(
        "--doi",
        help="Override DOI (e.g. 10.5281/zenodo.17297723 or https://doi.org/...)",
    )
    p.add_argument("--title", help="Override title")
    p.add_argument("--version", help="Override version")
    p.add_argument("--repo-url", help="Override repository-code URL")
    p.add_argument("--abstract", help="Override abstract")
    p.add_argument("--date-released", help="Override release date (YYYY-MM-DD)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    manifest = load_manifest(manifest_path)

    overrides = {
        "doi": args.doi,
        "title": args.title,
        "version": args.version,
        "repository_code": args.repo_url,
        "abstract": args.abstract,
        "date_released": args.date_released,
    }

    cff = build_cff_dict(
        manifest, {k: v for k, v in overrides.items() if v is not None}
    )
    outpath = Path(args.output)
    dump_cff_yaml(cff, outpath)

    if args.verbose:
        print("Generated CITATION.cff content:")
        print(outpath.read_text(encoding="utf8"))


if __name__ == "__main__":
    main()
