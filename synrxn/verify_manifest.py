from __future__ import annotations
import argparse, json, hashlib, sys
from pathlib import Path
from typing import List, Dict, Any

CHUNK = 1024 * 1024


def sha256_of_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def try_roots(manifest_path: Path) -> List[Path]:
    roots = []
    mp = manifest_path.resolve()
    roots.append(mp.parent)  # manifest dir
    roots.append(mp.parent / "Data")  # common
    roots.append(Path.cwd())  # cwd
    roots.append(Path.cwd() / "Data")
    seen = set()
    out = []
    for r in roots:
        try:
            rp = r.resolve()
        except Exception:
            rp = r
        if str(rp) not in seen:
            seen.add(str(rp))
            out.append(rp)
    return out


def verify_with_root(
    manifest: Dict[str, Any], root: Path, quiet: bool = False
) -> Dict[str, Any]:
    files = manifest.get("files", [])
    missing = []
    size_mismatch = []
    checksum_mismatch = []
    for entry in files:
        rel = entry.get("path")
        expected_size = int(entry.get("size", 0))
        expected_checksum = entry.get("checksum")
        p = (root / rel).resolve()
        if not p.exists():
            missing.append(rel)
            continue
        actual_size = p.stat().st_size
        if actual_size != expected_size:
            size_mismatch.append((rel, expected_size, actual_size))
        if expected_checksum:
            actual_checksum = sha256_of_path(p)
            if actual_checksum != expected_checksum:
                checksum_mismatch.append((rel, expected_checksum, actual_checksum))
    return {
        "root": str(root),
        "missing": missing,
        "size_mismatch": size_mismatch,
        "checksum_mismatch": checksum_mismatch,
        "files_checked": len(files),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", "-m", default="manifest.json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print("manifest not found:", manifest_path)
        sys.exit(2)
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    roots = try_roots(manifest_path)
    for r in roots:
        print("Trying root:", r)
        res = verify_with_root(manifest, r, quiet=args.quiet)
        print(
            f" missing={len(res['missing'])}, size_mismatch={len(res['size_mismatch'])}, checksum_mismatch={len(res['checksum_mismatch'])}"
        )
        if (
            len(res["missing"]) == 0
            and len(res["size_mismatch"]) == 0
            and len(res["checksum_mismatch"]) == 0
        ):
            print("All good! verified against root:", r)
            sys.exit(0)
    # If we get here, none of the candidate roots matched perfectly — print details for the best one
    best = min(roots, key=lambda r: len(verify_with_root(manifest, r)["missing"]))
    best_res = verify_with_root(manifest, best)
    print("\nNo perfect match — best candidate root:", best)
    print("  missing:", len(best_res["missing"]))
    print("  sample missing (first 10):", best_res["missing"][:10])
    print("\nSuggested fixes:")
    print(
        f"  - If your data folder is '{best}/', run: python synrxn/verify_manifest.py --manifest {manifest_path} --root {best}"
    )
    print("  - Or regenerate manifest using the correct root, e.g.:")
    print(f"      python build_manifest.py --root {best} --output {manifest_path}")
    sys.exit(2)


if __name__ == "__main__":
    main()
