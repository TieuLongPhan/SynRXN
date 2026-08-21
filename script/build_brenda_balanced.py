#!/usr/bin/env python3
"""Build a conservative, CC-BY-4.0 BRENDA atom-mapping dataset.

Data are read directly from BRENDA's official SPARQL endpoint. The default
subset requires:

* BRENDA's ``isIubmbReaction`` marker;
* at least one substrate and one product;
* an InChI structure for every reaction participant;
* exact atom and formal-charge balance after structure parsing; and
* a SynKit mapping with a complete, bijective explicit-atom map.

``isIubmbReaction`` is an authoritative-source/curation marker, not evidence of
independent experimental validation. Constant selection and mapping-QC facts
can optionally be exported as downstream provenance instead of repeated in
every row.

The BRENDA data are distributed under CC BY 4.0:
https://www.brenda-enzymes.org/license.php

Examples
--------
Smoke-test five reactions using the ``synrxn`` Conda environment::

    conda run -n synrxn python script/build_brenda_balanced.py \
        --max-reactions 5 --output /tmp/brenda-smoke.csv.gz \
        --provenance-output /tmp/brenda-smoke.provenance.json

Build the complete conservative subset::

    conda run -n synrxn python script/build_brenda_balanced.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ENDPOINT = "https://sparql.dsmz.de/api/brenda"
LICENSE_URL = "https://www.brenda-enzymes.org/license.php"
DOWNLOAD_URL = "https://www.brenda-enzymes.org/download.php"
DEFAULT_OUTPUT = Path("Data/aam/brenda.csv.gz")
DEFAULT_CACHE = Path(".cache/brenda-iubmb-sparql.json.gz")
SCHEMA_PREFIX = "https://purl.dsmz.de/schema/"

LOGGER = logging.getLogger("build_brenda_balanced")
_MAPPER: Any = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--provenance-output",
        type=Path,
        help="Optional downstream JSON provenance report.",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore and replace the source cache."
    )
    parser.add_argument("--page-size", type=int, default=10_000)
    parser.add_argument(
        "--max-source-pages",
        type=int,
        help="Fetch at most this many pages per query (smoke tests only).",
    )
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-reactions",
        type=int,
        help="Map at most this many structurally complete candidates (smoke tests).",
    )
    parser.add_argument(
        "--all-reactions",
        action="store_true",
        help="Do not require BRENDA's isIubmbReaction marker.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.page_size < 1:
        parser.error("--page-size must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_reactions is not None and args.max_reactions < 1:
        parser.error("--max-reactions must be positive")
    if args.max_source_pages is not None and args.max_source_pages < 1:
        parser.error("--max-source-pages must be positive")
    return args


def session_with_retries() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(
        {
            "Accept": "application/qlever-results+json",
            "User-Agent": "SynRXN-BRENDA-builder/1.0 (CC-BY-4.0 attribution build)",
        }
    )
    return session


def decode_rdf_term(value: str | None) -> str | None:
    """Decode QLever's compact RDF rendering into a plain Python string."""
    if value is None:
        return None
    if value.startswith("<") and value.endswith(">"):
        return value[1:-1]
    if value.startswith('"'):
        try:
            decoded, _ = json.JSONDecoder().raw_decode(value)
            return str(decoded)
        except json.JSONDecodeError:
            match = re.match(r'^"(.*?)"(?:\^\^<.*>|@[A-Za-z-]+)?$', value)
            if match:
                return match.group(1)
    return value


def execute_query(
    session: requests.Session, query: str, timeout: float
) -> list[dict[str, str | None]]:
    response = session.get(ENDPOINT, params={"query": query}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "OK":
        raise RuntimeError(f"BRENDA SPARQL query failed: {payload}")
    selected = [str(name).lstrip("?") for name in payload.get("selected", [])]
    return [
        {
            name: decode_rdf_term(value)
            for name, value in zip(selected, values, strict=True)
        }
        for values in payload.get("res", [])
    ]


def iubmb_clause(all_reactions: bool) -> str:
    if all_reactions:
        return ""
    # The prototype endpoint currently serializes this Boolean-like marker as
    # the plain literal strings "0" and "1", rather than xsd:boolean.
    return ' ; d3o:isIubmbReaction "1"'


def component_query(
    *, limit: int, offset: int, all_reactions: bool
) -> str:
    marker = iubmb_clause(all_reactions)
    return f"""
PREFIX d3o: <{SCHEMA_PREFIX}>
SELECT DISTINCT ?reagent ?compound ?inchi ?type ?reaction WHERE {{
  VALUES (?role ?type) {{
    (d3o:Substrate "substrate")
    (d3o:Product "product")
  }}
  ?reaction a d3o:Reaction{marker} ; d3o:catalyzedByClass ?ec .
  ?reagent a ?role ; d3o:partOf ?reaction ; d3o:refersToCompound ?compound .
  OPTIONAL {{
    ?compound d3o:hasStructure ?structure .
    ?structure d3o:hasInChI ?inchi .
  }}
}}
ORDER BY ?reaction ?reagent ?inchi
LIMIT {limit}
OFFSET {offset}
""".strip()


def reference_query(
    *, limit: int, offset: int, all_reactions: bool
) -> str:
    marker = iubmb_clause(all_reactions)
    return f"""
PREFIX d3o: <{SCHEMA_PREFIX}>
SELECT DISTINCT ?reaction ?reference ?pubmed WHERE {{
  ?reaction a d3o:Reaction{marker} .
  OPTIONAL {{
    ?reaction d3o:hasReference ?reference .
    OPTIONAL {{ ?reference d3o:hasPubMedID ?pubmed . }}
  }}
}}
ORDER BY ?reaction ?reference
LIMIT {limit}
OFFSET {offset}
""".strip()


def fetch_all_pages(
    session: requests.Session,
    query_factory: Any,
    *,
    page_size: int,
    timeout: float,
    all_reactions: bool,
    max_pages: int | None,
) -> list[dict[str, str | None]]:
    rows: list[dict[str, str | None]] = []
    offset = 0
    pages = 0
    while True:
        page = execute_query(
            session,
            query_factory(
                limit=page_size,
                offset=offset,
                all_reactions=all_reactions,
            ),
            timeout,
        )
        rows.extend(page)
        pages += 1
        LOGGER.info("Downloaded %d rows from BRENDA", len(rows))
        if len(page) < page_size or (max_pages is not None and pages >= max_pages):
            break
        offset += page_size
    return rows


def save_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def load_source(args: argparse.Namespace) -> dict[str, Any]:
    if args.cache.is_file() and not args.refresh:
        with gzip.open(args.cache, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if bool(payload.get("allReactions")) != bool(args.all_reactions):
            raise ValueError(
                "source cache selection differs from --all-reactions; use --refresh"
            )
        if not payload.get("complete", False) and args.max_source_pages is None:
            raise ValueError("source cache is a smoke-test subset; use --refresh")
        LOGGER.info("Loaded BRENDA source cache %s", args.cache)
        return payload

    session = session_with_retries()
    components = fetch_all_pages(
        session,
        component_query,
        page_size=args.page_size,
        timeout=args.timeout,
        all_reactions=args.all_reactions,
        max_pages=args.max_source_pages,
    )
    references = fetch_all_pages(
        session,
        reference_query,
        page_size=args.page_size,
        timeout=args.timeout,
        all_reactions=args.all_reactions,
        max_pages=args.max_source_pages,
    )
    payload = {
        "endpoint": ENDPOINT,
        "fetchedAt": datetime.now(timezone.utc).isoformat(),
        "allReactions": bool(args.all_reactions),
        "complete": args.max_source_pages is None,
        "components": components,
        "references": references,
    }
    save_cache(args.cache, payload)
    return payload


def uri_tail(uri: str | None, marker: str) -> str:
    if not uri:
        return ""
    if marker in uri:
        return uri.split(marker, 1)[1]
    return uri.rsplit("/", 1)[-1]


def group_source(payload: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in payload["components"]:
        reaction = str(row["reaction"])
        record = grouped.setdefault(
            reaction,
            {
                "reaction_uri": reaction,
                "original_id": uri_tail(reaction, "/reaction/"),
                "components": {},
                "reference_uris": set(),
                "pubmed_ids": set(),
            },
        )
        reagent = str(row["reagent"])
        component = record["components"].setdefault(
            reagent,
            {
                "reagent_uri": reagent,
                "compound_uri": row.get("compound"),
                "role": row.get("type"),
                "inchis": set(),
            },
        )
        if row.get("inchi"):
            component["inchis"].add(row["inchi"])

    for row in payload.get("references", []):
        reaction = row.get("reaction")
        if reaction not in grouped:
            continue
        if row.get("reference"):
            grouped[reaction]["reference_uris"].add(row["reference"])
        if row.get("pubmed"):
            grouped[reaction]["pubmed_ids"].add(str(row["pubmed"]))

    records = []
    for record in grouped.values():
        record["components"] = sorted(
            record["components"].values(), key=lambda item: item["reagent_uri"]
        )
        record["reference_uris"] = sorted(record["reference_uris"])
        record["pubmed_ids"] = sorted(record["pubmed_ids"])
        records.append(record)
    return sorted(records, key=lambda item: item["reaction_uri"])


def molecule_inventory(mol: Any) -> tuple[Counter[int], int]:
    atoms: Counter[int] = Counter()
    charge = 0
    for atom in mol.GetAtoms():
        atomic_number = int(atom.GetAtomicNum())
        atoms[atomic_number] += 1
        charge += int(atom.GetFormalCharge())
        if atomic_number != 1:
            atoms[1] += int(atom.GetTotalNumHs(includeNeighbors=False))
    return atoms, charge


def side_inventory(molecules: Iterable[Any]) -> tuple[Counter[int], int]:
    atoms: Counter[int] = Counter()
    charge = 0
    for mol in molecules:
        mol_atoms, mol_charge = molecule_inventory(mol)
        atoms.update(mol_atoms)
        charge += mol_charge
    return atoms, charge


def prepare_candidate(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    from rdkit import Chem

    substrates = []
    products = []
    for component in record["components"]:
        inchis = sorted(component["inchis"])
        if not inchis:
            return None, "missing_structure"
        if len(inchis) != 1:
            return None, "ambiguous_structure"
        mol = Chem.MolFromInchi(inchis[0], sanitize=True, removeHs=True)
        if mol is None:
            return None, "invalid_inchi"
        mol = Chem.RemoveHs(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        role = component["role"]
        if role == "substrate":
            substrates.append((smiles, mol))
        elif role == "product":
            products.append((smiles, mol))
        else:
            return None, "unknown_role"

    if not substrates or not products:
        return None, "missing_side"
    left_atoms, left_charge = side_inventory(mol for _, mol in substrates)
    right_atoms, right_charge = side_inventory(mol for _, mol in products)
    if left_atoms != right_atoms:
        return None, "atom_unbalanced"
    if left_charge != right_charge:
        return None, "charge_unbalanced"

    left = ".".join(sorted(smiles for smiles, _ in substrates))
    right = ".".join(sorted(smiles for smiles, _ in products))
    candidate = {
        "original_id": record["original_id"],
        "rxn": f"{left}>>{right}",
        "pubmed_ids": json.dumps(record["pubmed_ids"], separators=(",", ":")),
    }
    return candidate, "selected"


def map_candidate(candidate: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    global _MAPPER
    try:
        from synkit.Chem.Reaction.Mapper import AAMapper

        if _MAPPER is None:
            _MAPPER = AAMapper(deterministic_labels=True)
        else:
            _MAPPER.reset()
        _MAPPER.map_smiles(
            candidate["rxn"],
            # Map explicit graph atoms only. Hydrogen conservation has already
            # been checked from the InChI-derived molecular inventories; asking
            # the mapper to materialize implicit H can introduce duplicate map
            # labels for mobile-hydrogen/tautomer cases.
            add_Hs=False,
            break_sym="heavy",
            unique=True,
            certify=False,
            electron_balance=False,
        )
        acceptable = [
            result
            for result in _MAPPER.results
            if result.get("its_smiles")
            and mapping_is_bijective(result["its_smiles"])
        ]
        if not acceptable:
            return None, "mapping_qc_failed"
        best = min(acceptable, key=lambda result: float(result.get("cd", float("inf"))))
        output = dict(candidate)
        output["ground_truth"] = best["its_smiles"]
        output["chemical_distance"] = best.get("cd")
        return output, "mapped"
    except Exception as exc:  # individual bad reactions must not abort the build
        return None, f"mapping_error:{type(exc).__name__}"


def mapping_is_bijective(mapped_reaction: str) -> bool:
    """Require one positive, unique map number per explicit atom on both sides."""
    from rdkit import Chem

    try:
        left, right = mapped_reaction.split(">>", 1)
    except ValueError:
        return False

    def maps(side: str) -> list[int] | None:
        values: list[int] = []
        params = Chem.SmilesParserParams()
        params.removeHs = False
        for smiles in side.split("."):
            mol = Chem.MolFromSmiles(smiles, params)
            if mol is None:
                return None
            for atom in mol.GetAtoms():
                number = int(atom.GetAtomMapNum())
                if number <= 0:
                    return None
                values.append(number)
        return values

    left_maps = maps(left)
    right_maps = maps(right)
    if left_maps is None or right_maps is None:
        return False
    return (
        len(left_maps) == len(set(left_maps))
        and len(right_maps) == len(set(right_maps))
        and set(left_maps) == set(right_maps)
    )


def stable_deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    by_mapping: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        key = row["ground_truth"]
        if key not in by_mapping:
            by_mapping[key] = row
            continue
        duplicates += 1
        previous = by_mapping[key]
        previous["original_id"] = ";".join(
            sorted(set(previous["original_id"].split(";")) | {row["original_id"]})
        )
        previous["pubmed_ids"] = json.dumps(
            sorted(
                set(json.loads(previous["pubmed_ids"]))
                | set(json.loads(row["pubmed_ids"]))
            ),
            separators=(",", ":"),
        )
    return list(by_mapping.values()), duplicates


def save_output(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "r_id",
        "ground_truth",
        "rxn",
        "original_id",
        "pubmed_ids",
    ]
    normalized = []
    for index, row in enumerate(rows):
        normalized.append(
            {
                "r_id": index,
                "ground_truth": row["ground_truth"],
                "rxn": row["rxn"],
                "original_id": row["original_id"],
                "pubmed_ids": row["pubmed_ids"],
            }
        )
    frame = pd.DataFrame(normalized, columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    import synkit
    from rdkit import RDLogger

    args = parse_args(argv)
    # Invalid or mobile-H InChIs are counted and rejected explicitly below;
    # suppress RDKit's repeated parser diagnostics in long release builds.
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    source = load_source(args)
    grouped = group_source(source)
    counts: Counter[str] = Counter()
    candidates = []
    for record in grouped:
        candidate, status = prepare_candidate(record)
        counts[status] += 1
        if candidate is not None:
            candidates.append(candidate)
    if args.max_reactions is not None:
        candidates = candidates[: args.max_reactions]

    mapped: list[dict[str, Any]] = []
    if args.workers == 1:
        mapping_results = map(map_candidate, candidates)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        mapping_results = executor.map(map_candidate, candidates, chunksize=1)
    try:
        for index, (row, status) in enumerate(mapping_results, start=1):
            counts[status] += 1
            if row is not None:
                mapped.append(row)
            if index % 100 == 0:
                LOGGER.info("Mapped %d/%d candidates", index, len(candidates))
    finally:
        if args.workers != 1:
            executor.shutdown()

    mapped, duplicate_count = stable_deduplicate(mapped)
    counts["mapping_duplicates"] = duplicate_count
    save_output(mapped, args.output)
    provenance = {
        "builder": Path(__file__).name,
        "builtAt": datetime.now(timezone.utc).isoformat(),
        "dataset": "BRENDA IUBMB-curated balanced atom mappings",
        "source": "BRENDA",
        "endpoint": ENDPOINT,
        "downloadPage": DOWNLOAD_URL,
        "license": "CC-BY-4.0",
        "licenseUrl": LICENSE_URL,
        "mappingSoftware": "SynKit AAMapper",
        "mappingSoftwareVersion": synkit.__version__,
        "schemaColumns": [
            "r_id",
            "ground_truth",
            "rxn",
            "original_id",
            "pubmed_ids",
        ],
        "sourceFetchedAt": source.get("fetchedAt"),
        "selection": {
            "iubmbOnly": not args.all_reactions,
            "allParticipantsStructurallyResolved": True,
            "exactAtomBalanceIncludingHydrogen": True,
            "formalChargeBalance": True,
            "mappingBijectionQc": True,
            "independentlyValidated": False,
        },
        "notes": (
            "isIubmbReaction and BRENDA curation are source-curation markers, "
            "not independent experimental validation. Atom mappings were "
            "generated by SynKit and computationally checked."
        ),
        "counts": {
            "sourceReactions": len(grouped),
            "mappingCandidates": len(candidates),
            "outputRows": len(mapped),
            **dict(sorted(counts.items())),
        },
        "output": str(args.output),
        "outputSha256": sha256(args.output),
    }
    if args.provenance_output is not None:
        args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_output.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    LOGGER.info("Saved %d reactions to %s", len(mapped), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
