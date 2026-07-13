#!/usr/bin/env python3
"""Static and optional network checks for the reviewer-facing release.

The default run is offline and non-mutating. It audits the candidate Git
surface, canonical notebooks, Markdown links, allowlisted release archive,
secret patterns, and restricted files. Pass ``--network`` only for the final
signed-out-style check of populated GitHub and Hugging Face destinations.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import stat
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
NOTEBOOKS = SUBMISSION / "notebooks"
LINKS = SUBMISSION / "LINKS.md"
MAX_PUBLIC_BYTES = 50 * 1024 * 1024
MAX_SINGLE_FILE_BYTES = 20 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 20 * 1024 * 1024

PUBLIC_URLS = {
    "github": "https://github.com/adarshrajesh-ui/uae-adab-tutor",
    "model": "https://huggingface.co/adarshrajesh/uae-adab-tutor-qwen3-4b",
    "dataset": "https://huggingface.co/datasets/adarshrajesh/uae-adab-tutor-600",
    "colab": (
        "https://colab.research.google.com/github/adarshrajesh-ui/"
        "uae-adab-tutor/blob/main/submission/notebooks/"
        "03_demo_uae_adab_tutor.ipynb"
    ),
}
RAW_DEMO_URL = (
    "https://raw.githubusercontent.com/adarshrajesh-ui/uae-adab-tutor/"
    "main/submission/notebooks/03_demo_uae_adab_tutor.ipynb"
)

RESTRICTED_PARTS = {
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
RESTRICTED_NAME_MARKERS = (
    "raw_caption",
    "raw_transcript",
    "rejected",
    "reviewer_judgment",
    "writer_events",
)
RESTRICTED_SUFFIXES = {
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
}

SECRET_PATTERNS = {
    "OpenAI-style API key": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "Hugging Face token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    "TrueFoundry token": re.compile(rb"\btfy_[A-Za-z0-9_-]{20,}\b"),
    "GitHub token": re.compile(rb"\bgh(?:p|o|u|s|r)_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key": re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
}

CANONICAL_NOTEBOOKS = {
    "01_train_exact_silver_v1.ipynb",
    "02_evaluate_exact_silver_v1.ipynb",
    "03_demo_uae_adab_tutor.ipynb",
}
REVIEWER_NOTEBOOKS = CANONICAL_NOTEBOOKS
PLACEHOLDER_RE = re.compile(
    r"\b(?:TODO|REPLACE_ME|YOUR_HF_|YOUR_USERNAME|YOUR_TOKEN)\b|<your[-_ ]",
    re.IGNORECASE,
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def secret_findings(raw: bytes) -> list[str]:
    return [label for label, pattern in SECRET_PATTERNS.items() if pattern.search(raw)]


def restricted_path_reason(path: PurePosixPath) -> str | None:
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & RESTRICTED_PARTS:
        return "restricted directory"
    name = path.name.lower()
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return "environment file"
    if name in {"brainlift.md", "modelbrainlift.md"}:
        return "owner-controlled writing excluded from this curated release"
    if any(marker in name for marker in RESTRICTED_NAME_MARKERS):
        return "internal generation/review artifact"
    if path.suffix.lower() in RESTRICTED_SUFFIXES:
        return "restricted binary or source payload"
    return None


def candidate_git_files() -> list[Path]:
    raw = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    return sorted(
        Path(item.decode("utf-8"))
        for item in raw.split(b"\0")
        if item and (ROOT / item.decode("utf-8")).is_file()
    )


def validate_archive(path: Path, *, expected_members: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)):
                errors.append(f"{path}: duplicate archive member")
            if expected_members is not None and set(names) != expected_members:
                errors.append(f"{path}: archive member set does not match manifest")
            for member in members:
                pure = PurePosixPath(member.filename)
                if (
                    pure.is_absolute()
                    or not pure.parts
                    or any(part in {"", ".", ".."} for part in pure.parts)
                ):
                    errors.append(f"{path}: unsafe member path {member.filename!r}")
                    continue
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    errors.append(f"{path}: symlink member {member.filename}")
                if member.flag_bits & 0x1:
                    errors.append(f"{path}: encrypted member {member.filename}")
                if member.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    errors.append(f"{path}: oversized member {member.filename}")
                    continue
                if pure.suffix.lower() == ".zip":
                    errors.append(f"{path}: nested archive {member.filename}")
                    continue
                reason = restricted_path_reason(pure)
                if reason:
                    errors.append(f"{path}: {member.filename}: {reason}")
                raw = archive.read(member)
                for finding in secret_findings(raw):
                    errors.append(f"{path}: {member.filename}: possible {finding}")
    except (OSError, zipfile.BadZipFile) as exc:
        errors.append(f"{path}: invalid ZIP: {exc}")
    return errors


def validate_git_surface() -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    files = candidate_git_files()
    total = 0
    for relative in files:
        pure = PurePosixPath(relative.as_posix())
        reason = restricted_path_reason(pure)
        if reason:
            errors.append(f"candidate Git file {relative}: {reason}")
            continue
        path = ROOT / relative
        size = path.stat().st_size
        total += size
        if size > MAX_SINGLE_FILE_BYTES:
            errors.append(f"candidate Git file {relative}: exceeds 20 MiB")
        if path.suffix.lower() == ".zip":
            errors.extend(validate_archive(path))
        else:
            raw = path.read_bytes()
            for finding in secret_findings(raw):
                errors.append(f"candidate Git file {relative}: possible {finding}")
    if total > MAX_PUBLIC_BYTES:
        errors.append(f"candidate Git surface is {total} bytes; maximum is {MAX_PUBLIC_BYTES}")
    return errors, {"files": len(files), "bytes": total}


def git_ignores(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", path],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def validate_gitignore_policy() -> list[str]:
    errors = []
    required_ignored = (
        ".env",
        "data/uae_adab/private.jsonl",
        "outputs/judge.json",
        "zeval/model_outputs.jsonl",
        "zshots/screenshot.png",
        "research/scraped/caption.txt",
        "private/permission.pdf",
        "adapter_model.safetensors",
        "brainlift.md",
        "modelBrainlift.md",
    )
    for path in required_ignored:
        if not git_ignores(path):
            errors.append(f".gitignore does not exclude {path}")
    if git_ignores(".env.example"):
        errors.append(".env.example must remain public")
    return errors


def markdown_targets(text: str) -> list[str]:
    return re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)


def validate_markdown_links() -> list[str]:
    errors: list[str] = []
    for path in SUBMISSION.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for target in markdown_targets(text):
            clean = target.strip().strip("<>")
            if clean.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = urllib.parse.unquote(clean.split("#", 1)[0].split("?", 1)[0])
            if relative and not (path.parent / relative).exists():
                errors.append(f"{path.relative_to(ROOT)}: broken link {target}")

    links_text = LINKS.read_text(encoding="utf-8")
    if "| Brainlift | OWNER TODO |" not in links_text:
        errors.append("submission/LINKS.md must retain the owner-controlled Brainlift TODO")
    unowned_todos = links_text.replace("OWNER TODO", "")
    if "TODO" in unowned_todos:
        errors.append("submission/LINKS.md contains an unresolved non-owner TODO")
    for label, url in PUBLIC_URLS.items():
        if url not in links_text:
            errors.append(f"submission/LINKS.md missing fixed {label} URL")
    return errors


def notebook_text(notebook: dict[str, Any]) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def validate_notebooks() -> list[str]:
    errors: list[str] = []
    manifest_path = NOTEBOOKS / "MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid notebook manifest: {exc}"]
    records = manifest.get("files", {})
    if set(records) != CANONICAL_NOTEBOOKS:
        errors.append(
            f"notebook manifest must contain {sorted(CANONICAL_NOTEBOOKS)}, got {sorted(records)}"
        )
    for filename in sorted(CANONICAL_NOTEBOOKS):
        path = NOTEBOOKS / filename
        if not path.is_file():
            errors.append(f"missing canonical notebook {path.relative_to(ROOT)}")
            continue
        raw = path.read_bytes()
        record = records.get(filename, {})
        if record.get("bytes") != len(raw) or record.get("sha256") != sha256(raw):
            errors.append(f"stale notebook manifest record for {filename}")
        try:
            notebook = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"{filename}: invalid JSON: {exc}")
            continue
        if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
            errors.append(f"{filename}: invalid notebook structure")
            continue
        if record.get("cells") != len(notebook["cells"]):
            errors.append(f"{filename}: stale cell count in manifest")
        text = notebook_text(notebook)
        if "/Users/" in text or re.search(r"[A-Za-z]:\\Users\\", text):
            errors.append(f"{filename}: personal local path embedded")
        for finding in secret_findings(raw):
            errors.append(f"{filename}: possible {finding}")
        if filename in REVIEWER_NOTEBOOKS and PLACEHOLDER_RE.search(text):
            errors.append(f"{filename}: unresolved reviewer-path placeholder")
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None or cell.get("outputs") not in ([], None):
                errors.append(f"{filename}:{index}: saved execution state/output")
            source = "".join(cell.get("source", []))
            if source.lstrip().startswith(("!", "%")):
                continue
            try:
                ast.parse(source, filename=f"{filename}:{index}")
            except SyntaxError as exc:
                errors.append(f"{filename}:{index}: Python syntax error: {exc}")
    return errors


def load_release_builder():
    path = ROOT / "scripts/build_public_submission_release.py"
    spec = importlib.util.spec_from_file_location("public_release_builder", path)
    if not spec or not spec.loader:
        raise RuntimeError("cannot load public release builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_release_package() -> list[str]:
    errors: list[str] = []
    try:
        builder = load_release_builder()
        expected_archive, expected_manifest = builder.expected_artifacts()
    except Exception as exc:  # report the preflight failure with the other checks
        return [f"cannot construct expected reviewer release: {exc}"]
    if not builder.ARCHIVE.is_file() or builder.ARCHIVE.read_bytes() != expected_archive:
        errors.append("reviewer release ZIP is missing or stale")
    if not builder.MANIFEST.is_file() or builder.MANIFEST.read_bytes() != expected_manifest:
        errors.append("reviewer release manifest is missing or stale")
    expected_members = set(builder.read_allowlist()) | {builder.CONTENT_MANIFEST_NAME}
    if builder.ARCHIVE.is_file():
        errors.extend(validate_archive(builder.ARCHIVE, expected_members=expected_members))
    return errors


def request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "uae-adab-release-audit/1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def request_status(url: str) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": "uae-adab-release-audit/1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.status


def validate_network_destinations() -> list[str]:
    errors: list[str] = []
    try:
        github = request_json("https://api.github.com/repos/adarshrajesh-ui/uae-adab-tutor")
        if github.get("private") is not False or github.get("default_branch") != "main":
            errors.append("GitHub repository is not public on main")
        if int(github.get("size") or 0) <= 0:
            errors.append("GitHub repository is still an empty shell")
    except Exception as exc:
        errors.append(f"GitHub repository check failed: {exc}")

    remote_specs = {
        "model": (
            "https://huggingface.co/api/models/adarshrajesh/uae-adab-tutor-qwen3-4b",
            {"README.md", "adapter_config.json", "adapter_model.safetensors"},
        ),
        "dataset": (
            "https://huggingface.co/api/datasets/adarshrajesh/uae-adab-tutor-600",
            {
                "README.md",
                "train.jsonl",
                "validation.jsonl",
                "release_manifest.json",
                "release_limitation_ledger_600.jsonl",
            },
        ),
    }
    for label, (url, required) in remote_specs.items():
        try:
            payload = request_json(url)
            if payload.get("private") is not False or payload.get("gated") not in (False, None):
                errors.append(f"Hugging Face {label} repository is not publicly accessible")
            siblings = {item.get("rfilename") for item in payload.get("siblings", [])}
            missing = sorted(required - siblings)
            if missing:
                errors.append(f"Hugging Face {label} repository missing: {', '.join(missing)}")
        except Exception as exc:
            errors.append(f"Hugging Face {label} check failed: {exc}")

    try:
        if request_status(RAW_DEMO_URL) != 200:
            errors.append("raw GitHub demo notebook did not return HTTP 200")
    except Exception as exc:
        errors.append(f"raw GitHub demo notebook check failed: {exc}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network",
        action="store_true",
        help="also require populated public GitHub/Hugging Face destinations",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    surface_errors, surface = validate_git_surface()
    errors.extend(surface_errors)
    errors.extend(validate_gitignore_policy())
    errors.extend(validate_markdown_links())
    errors.extend(validate_notebooks())
    errors.extend(validate_release_package())
    if args.network:
        errors.extend(validate_network_destinations())

    if errors:
        print("PUBLIC SUBMISSION VALIDATION FAILED:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        print(json.dumps({"candidate_git_surface": surface, "error_count": len(errors)}))
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "network_checked": args.network,
                "candidate_git_surface": surface,
                "canonical_notebooks": sorted(CANONICAL_NOTEBOOKS),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
