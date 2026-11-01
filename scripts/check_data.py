"""Check the local AI-Hub acquisition manifest without modifying source data."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def check_manifest(data_dir: Path, manifest_path: Path) -> list[str]:
    """Return missing/size-mismatch messages using portable manifest paths.

    The manifest always stores POSIX-style separators. ``Path.__str__`` uses
    backslashes on Windows, so relative paths must be converted with
    ``as_posix()`` before they are compared.
    """

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = {
        normalize(path.relative_to(data_dir).as_posix()): path
        for path in data_dir.rglob("*.csv")
    }
    failures = []
    for record in manifest:
        relative = normalize(record["path"])
        path = actual.get(relative)
        if path is None:
            failures.append(f"{relative}: missing")
        elif path.stat().st_size != record["bytes"]:
            failures.append(f"{relative}: size mismatch")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the original local acquisition manifest (includes archive duplicates)."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/data_manifest.json")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = check_manifest(args.data_dir, args.manifest)
    print("Checked", len(manifest), "expected CSV files")
    for message in failures:
        print(message)
    if failures:
        sys.exit(1)
    print("PASS: paths and byte sizes match. This is not a content hash check.")


if __name__ == "__main__":
    main()
