"""Zip src/ and tests/ into a verified source bundle plus SHA-256 manifest.

Usage: python scripts/colab/build_source_bundle.py [--out outputs/colab-expanded] [--name fjepa-expanded-source]
Writes <out>/<name>.zip and <out>/source_manifest.json (path -> sha256) in the same
layout the Colab prepare/reverify scripts expect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT
INCLUDE = ("src", "tests")
TEST_PREFIXES = ("test_pilot_", "test_expanded_", "test_wandb_", "test_tokenization_")


def files():
    for path in sorted((REPO / "src").iterdir()):
        if path.is_file() and path.suffix in {".py", ".txt", ".md", ".json"}:
            yield path
    for path in sorted((REPO / "tests").glob("test_*.py")):
        if path.name.startswith(TEST_PREFIXES):
            yield path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "outputs" / "colab-expanded"))
    parser.add_argument("--name", default="fjepa-expanded-source")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(out / f"{args.name}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files():
            rel = path.relative_to(REPO).as_posix()
            archive.write(path, rel)
            manifest[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"zip": str(out / f"{args.name}.zip"), "files": len(manifest),
                      "manifest": str(out / "source_manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
