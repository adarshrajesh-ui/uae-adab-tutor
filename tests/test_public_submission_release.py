from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module(
    "build_public_submission_release",
    ROOT / "scripts/build_public_submission_release.py",
)
validator = load_module(
    "validate_public_submission",
    ROOT / "scripts/validate_public_submission.py",
)


def test_release_allowlist_is_explicit_and_restricted_material_is_absent() -> None:
    entries = builder.read_allowlist()
    assert "submission/notebooks/03_demo_uae_adab_tutor.ipynb" in entries
    assert "submission/model/README.md" in entries
    assert "deploy/reviewer_demo/app.py" in entries
    assert "brainlift.md" not in entries
    assert "modelBrainlift.md" not in entries
    assert not any(entry.startswith("data/") for entry in entries)
    assert not any("hf_space" in entry for entry in entries)
    assert not any("permission_attestation" in entry.lower() for entry in entries)
    for entry in entries:
        assert builder.validate_relative_path(entry) == PurePosixPath(entry)


def test_secret_scanner_detects_real_shapes_but_not_placeholders() -> None:
    fake_hf = b"hf_" + (b"A" * 24)
    fake_openai = b"sk-" + (b"B" * 24)
    assert "Hugging Face token" in validator.secret_findings(fake_hf)
    assert "OpenAI-style API key" in validator.secret_findings(fake_openai)
    assert validator.secret_findings(b"OPENAI_API_KEY=paste-your-key-here") == []


def test_restricted_path_gate_covers_private_surfaces() -> None:
    restricted = (
        "data/internal.jsonl",
        "outputs/judge.json",
        "research/scraped/caption.txt",
        "zshots/screenshot.png",
        "adapter_model.safetensors",
        "brainlift.md",
        "modelBrainlift.md",
        ".env",
    )
    for value in restricted:
        assert validator.restricted_path_reason(PurePosixPath(value)), value
    assert validator.restricted_path_reason(PurePosixPath(".env.example")) is None
    assert validator.restricted_path_reason(
        PurePosixPath("submission/model/README.md")
    ) is None


def test_zip_validator_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../secret.txt", "not a real secret")
    errors = validator.validate_archive(archive)
    assert any("unsafe member path" in error for error in errors)


def test_public_git_surface_and_ignore_policy_are_clean() -> None:
    surface_errors, surface = validator.validate_git_surface()
    assert surface_errors == []
    assert surface["files"] > 0
    assert surface["bytes"] < validator.MAX_PUBLIC_BYTES
    assert validator.validate_gitignore_policy() == []


def test_submission_links_and_canonical_notebooks_are_static_clean() -> None:
    assert validator.validate_markdown_links() == []
    assert validator.validate_notebooks() == []


def test_reviewer_release_is_reproducible_and_self_describing() -> None:
    expected_archive, expected_manifest = builder.expected_artifacts()
    assert builder.ARCHIVE.read_bytes() == expected_archive
    assert builder.MANIFEST.read_bytes() == expected_manifest
    external = json.loads(expected_manifest)
    assert external["archive"]["sha256"] == builder.sha256(expected_archive)

    with zipfile.ZipFile(io.BytesIO(expected_archive)) as archive:
        expected_members = set(builder.read_allowlist()) | {
            builder.CONTENT_MANIFEST_NAME
        }
        assert set(archive.namelist()) == expected_members
        contents = json.loads(archive.read(builder.CONTENT_MANIFEST_NAME))
        assert set(contents["files"]) == set(builder.read_allowlist())
        assert contents["public_destinations"] == builder.PUBLIC_DESTINATIONS


def test_complete_offline_submission_validation_passes() -> None:
    assert validator.main([]) == 0
