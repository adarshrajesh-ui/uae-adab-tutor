#!/usr/bin/env python3
"""Build the exact public Hugging Face dataset upload bundle."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/uae_adab/v3/final_600/exact_silver_release_v1"
OUTPUT = ROOT / "submission/dataset/uae_adab_hf_dataset_upload.zip"
MANIFEST = ROOT / "submission/dataset/BUNDLE_MANIFEST.json"
FILES = {
    "README.md": ROOT / "submission/dataset/README.md",
    "train.jsonl": SOURCE / "complete_600_train.jsonl",
    "validation.jsonl": SOURCE / "complete_600_validation.jsonl",
    "release_manifest.json": SOURCE / "exact_silver_release_manifest.json",
    "release_limitation_ledger_600.jsonl": SOURCE / "release_limitation_ledger_600.jsonl",
}
EXPECTED = {
    "train.jsonl": "990cdc7ca494a4e12efa1cc7a739030ef1412e1dd1a350f91b1d453e49a756a0",
    "validation.jsonl": "cf10c2e316c13da2e2ebf1c1edee18ebfd3784085440a00b8a8c31e71fa0ec1c",
    "release_manifest.json": "ef8b907a497444b83410bd2fdcff37c4eb3922d90e485638303b8e5342ba662b",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    records = {}
    for name, path in FILES.items():
        raw = path.read_bytes()
        records[name] = {"bytes": len(raw), "sha256": sha256(raw)}
        if name in EXPECTED:
            assert records[name]["sha256"] == EXPECTED[name], (name, "hash mismatch")

    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in FILES.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 7, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())

    manifest = {
        "schema_version": "uae_adab_hf_dataset_upload_bundle.v1",
        "archive": OUTPUT.name,
        "archive_bytes": OUTPUT.stat().st_size,
        "archive_sha256": sha256(OUTPUT.read_bytes()),
        "files": records,
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(OUTPUT.relative_to(ROOT), manifest["archive_sha256"])


if __name__ == "__main__":
    main()
