"""Cache CDK reaction diagrams for the documentation's admitted examples."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from synrxn.reconstruction.paths import REPO
from synrxn.reconstruction.presentation.documentation import CATALOGUES, RELEASE, canonical_catalog

BASE_URL = "https://www.simolecule.com/cdkdepict"
SETTINGS = {"zoom": "1.5", "abbr": "off", "hdisp": "bridgehead",
            "showtitle": "false", "annotate": "none"}
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def cdk_url(reaction, base_url=BASE_URL):
    """Return a CDK Depict URL preserving the complete reaction SMILES.

    :param reaction: Full reaction SMILES from the catalogue.
    :param base_url: CDK Depict service root.
    :return: Encoded SVG depiction request URL.
    """
    return base_url.rstrip("/") + "/depict/cow/svg?" + urlencode({"smi": reaction, **SETTINGS})


def cache_depiction(item, doc_root, base_url=BASE_URL, timeout=30, refresh=False):
    """Fetch an admitted example once and record its rendering inputs in the SVG.

    Existing RDKit diagrams and catalogue records are left unchanged. Failed
    requests raise an exception so callers can report the unavailable preview.

    :param item: Catalogue entry after applying the current release review.
    :param doc_root: Documentation source directory.
    :param base_url: CDK Depict service root.
    :param timeout: Request timeout in seconds.
    :param refresh: Replace an otherwise matching cached diagram.
    :return: ``rendered``, ``cached`` or ``skipped``.
    """
    if item.get("release_eligible") is False or not item.get("depiction") or not item.get("reaction_text"):
        return "skipped"
    root = Path(doc_root).resolve()
    original = (root / item["depiction"]).resolve()
    if not original.is_relative_to(root / "_static") or original.suffix != ".svg":
        raise ValueError("Depiction must be an SVG inside the documentation's _static directory")
    destination = original.with_suffix(".cdk.svg")
    record = {"engine": "CDK Depict", "endpoint": base_url.rstrip("/"), "parameters": SETTINGS,
              "reaction_sha256": hashlib.sha256(item["reaction_text"].encode("utf-8")).hexdigest()}
    if destination.exists():
        try:
            cached = ET.parse(destination).getroot().find(f"{{{SVG}}}metadata[@id='synrxn-rendering']")
            if not refresh and cached is not None and json.loads(cached.text) == record:
                return "cached"
        except (ET.ParseError, ValueError, TypeError):
            pass
        # A stale diagram must never be shown for a different reaction.
        destination.unlink()
    request = Request(cdk_url(item["reaction_text"], base_url),
                      headers={"User-Agent": "SynRXN documentation renderer", "Accept": "image/svg+xml"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read(8 * 1024 * 1024 + 1)
    if len(payload) > 8 * 1024 * 1024:
        raise ValueError("CDK response exceeds the preview size limit")
    diagram = ET.fromstring(payload)
    if diagram.tag != f"{{{SVG}}}svg":
        raise ValueError("CDK response is not an SVG diagram")
    if any(node.tag.rsplit("}", 1)[-1] in {"script", "foreignObject"} for node in diagram.iter()):
        raise ValueError("CDK response contains active content")
    metadata = ET.Element(f"{{{SVG}}}metadata", {"id": "synrxn-rendering"})
    metadata.text = json.dumps(record, sort_keys=True)
    diagram.insert(0, metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".svg.tmp", delete=False) as handle:
            temporary = Path(handle.name)
            ET.ElementTree(diagram).write(handle, encoding="utf-8", xml_declaration=True)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return "rendered"


def main(argv=None):
    """Render reviewed catalogue examples, retaining local RDKit fallbacks.

    :param argv: Optional command-line arguments.
    :return: Zero on success, one if any requested rendering failed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    release = json.loads((args.repo / RELEASE).read_text(encoding="utf-8"))
    items = []
    for name in CATALOGUES:
        catalogue = json.loads((args.repo / "doc/_static" / f"{name}.json").read_text(encoding="utf-8"))
        items.extend(canonical_catalog(catalogue, release)["datasets"])

    def render(item):
        try:
            return cache_depiction(item, args.repo / "doc", args.base_url, args.timeout, args.refresh), item["id"]
        except Exception as error:
            return "failed", f'{item["id"]}: {error}'

    counts = {key: 0 for key in ("rendered", "cached", "skipped", "failed")}
    with ThreadPoolExecutor(max_workers=2) as workers:
        for status, identity in workers.map(render, items):
            counts[status] += 1
            if status == "failed":
                print(identity)
    print(", ".join(f"{count} {status}" for status, count in counts.items()))
    return int(bool(counts["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
