#!/usr/bin/env python3
"""Build the deterministic, allowlisted reviewer release archive.

The archive is deliberately not a snapshot of the working tree.  It contains
only paths listed in ``submission/PUBLIC_RELEASE_FILES.txt`` and refuses raw
sources, internal review traces, credentials, model weights, symlinks, nested
archives, or path traversal.  Use ``--check`` in the final release gate.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "submission/PUBLIC_RELEASE_FILES.txt"
ARCHIVE = ROOT / "submission/uae_adab_reviewer_release.zip"
MANIFEST = ROOT / "submission/PUBLIC_RELEASE_MANIFEST.json"
CONTENT_MANIFEST_NAME = "PUBLIC_RELEASE_CONTENTS.json"
ZIP_TIMESTAMP = (2026, 7, 12, 0, 0, 0)

PUBLIC_DESTINATIONS = {
    "github": "https://github.com/adarshrajesh-ui/uae-adab-tutor",
    "model": "https://huggingface.co/adarshrajesh/uae-adab-tutor-qwen3-4b",
    "dataset": "https://huggingface.co/datasets/adarshrajesh/uae-adab-tutor-600",
    "temporary_colab": (
        "https://colab.research.google.com/github/adarshrajesh-ui/"
        "uae-adab-tutor/blob/main/submission/notebooks/"
        "03_demo_uae_adab_tutor.ipynb"
    ),
}

FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "archived",
    "credentials",
    "data",
    "outputs",
    "private",
    "scraped",
    "secrets",
    "zshots",
    "zeval",
}
FORBIDDEN_NAME_MARKERS = (
    "raw_transcript",
    "raw_caption",
    "rejected",
    "reviewer_judgment",
    "writer_events",
)
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".json3",
    ".key",
    ".m4a",
    ".mp3",
    ".mp4",
    ".p12",
    ".pem",
    ".pfx",
    ".pt",
    ".pth",
    ".safetensors",
    ".srt",
    ".vtt",
    ".wav",
    ".webm",
    ".zip",
}

SECRET_PATTERNS = {
    "OpenAI-style API key": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "Hugging Face token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    "TrueFoundry token": re.compile(rb"\btfy_[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(rb"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key": re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
}


class ReleaseError(ValueError):
    """Raised when a path or content fails the public-release contract."""


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_allowlist(path: Path = ALLOWLIST) -> list[str]:
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(entries) != len(set(entries)):
        raise ReleaseError("public release allowlist contains duplicate paths")
    return entries


def validate_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseError(f"unsafe release path: {value!r}")
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & FORBIDDEN_PARTS:
        raise ReleaseError(f"restricted directory in release path: {value}")
    lowered_name = path.name.lower()
    if lowered_name == ".env" or lowered_name.startswith(".env.") and lowered_name != ".env.example":
        raise ReleaseError(f"environment file is not public: {value}")
    if any(marker in lowered_name for marker in FORBIDDEN_NAME_MARKERS):
        raise ReleaseError(f"restricted generated artifact in release path: {value}")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ReleaseError(f"forbidden or nested binary artifact in release path: {value}")
    if lowered_name in {"brainlift.md", "modelbrainlift.md"}:
        raise ReleaseError(f"{path.name} is owner-controlled and not a release file")
    return path


def secret_findings(raw: bytes) -> list[str]:
    return [label for label, pattern in SECRET_PATTERNS.items() if pattern.search(raw)]


def load_public_files(entries: list[str]) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    root = ROOT.resolve()
    for value in entries:
        relative = validate_relative_path(value)
        path = (ROOT / Path(*relative.parts)).resolve()
        if not path.is_relative_to(root):
            raise ReleaseError(f"release path escapes repository: {value}")
        if not path.is_file() or path.is_symlink():
            raise ReleaseError(f"release path must be a regular file: {value}")
        raw = path.read_bytes()
        findings = secret_findings(raw)
        if findings:
            raise ReleaseError(f"possible secret in {value}: {', '.join(findings)}")
        files[value] = raw
    return files


def content_manifest_bytes(files: dict[str, bytes]) -> bytes:
    payload = {
        "schema_version": "uae_adab_reviewer_release_contents.v1",
        "selected_artifact": "exact-silver v1 Complete-600",
        "public_destinations": PUBLIC_DESTINATIONS,
        "allowlist_sha256": sha256(ALLOWLIST.read_bytes()),
        "files": {
            name: {"bytes": len(raw), "sha256": sha256(raw)}
            for name, raw in sorted(files.items())
        },
        "excluded_by_design": [
            "API keys, tokens, private credentials, and .env files",
            "raw transcripts, captions, scraped payloads, and permission records/attestations",
            "rejected candidates, writer events, reviewer traces, and local evaluation outputs",
            "model weights and nested archives",
            "the 400+ MB internal generation tree",
        ],
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_archive_bytes(files: dict[str, bytes], contents: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name, raw in sorted({**files, CONTENT_MANIFEST_NAME: contents}.items()):
            archive.writestr(zip_info(name), raw)
    return buffer.getvalue()


def external_manifest_bytes(
    files: dict[str, bytes], contents: bytes, archive: bytes
) -> bytes:
    payload = {
        "schema_version": "uae_adab_reviewer_release.v1",
        "archive": {
            "file": ARCHIVE.name,
            "bytes": len(archive),
            "sha256": sha256(archive),
        },
        "content_manifest": {
            "archive_member": CONTENT_MANIFEST_NAME,
            "bytes": len(contents),
            "sha256": sha256(contents),
        },
        "allowlist": {
            "file": str(ALLOWLIST.relative_to(ROOT)),
            "entries": len(files),
            "sha256": sha256(ALLOWLIST.read_bytes()),
        },
        "public_destinations": PUBLIC_DESTINATIONS,
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def expected_artifacts() -> tuple[bytes, bytes]:
    entries = read_allowlist()
    files = load_public_files(entries)
    contents = content_manifest_bytes(files)
    archive = build_archive_bytes(files, contents)
    manifest = external_manifest_bytes(files, contents, archive)
    return archive, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in archive or manifest is stale; write nothing",
    )
    args = parser.parse_args(argv)

    try:
        archive, manifest = expected_artifacts()
    except (OSError, ReleaseError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"PUBLIC RELEASE BUILD FAILED: {exc}", file=sys.stderr)
        return 1

    if args.check:
        errors = []
        if not ARCHIVE.is_file() or ARCHIVE.read_bytes() != archive:
            errors.append(f"stale or missing {ARCHIVE.relative_to(ROOT)}")
        if not MANIFEST.is_file() or MANIFEST.read_bytes() != manifest:
            errors.append(f"stale or missing {MANIFEST.relative_to(ROOT)}")
        if errors:
            print("PUBLIC RELEASE CHECK FAILED:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print(f"PUBLIC RELEASE CHECK PASSED: {sha256(archive)}")
        return 0

    ARCHIVE.write_bytes(archive)
    MANIFEST.write_bytes(manifest)
    print(
        json.dumps(
            {
                "archive": str(ARCHIVE.relative_to(ROOT)),
                "archive_bytes": len(archive),
                "archive_sha256": sha256(archive),
                "manifest": str(MANIFEST.relative_to(ROOT)),
                "allowlisted_files": len(read_allowlist()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
