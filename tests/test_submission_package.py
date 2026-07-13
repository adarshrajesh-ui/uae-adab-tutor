from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
NOTEBOOKS = SUBMISSION / "notebooks"


def notebook_text(path: Path) -> tuple[dict, str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    return notebook, text


def assert_clean_parseable(notebook: dict, filename: str) -> None:
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
        source = "".join(cell.get("source", []))
        if source.lstrip().startswith(("!", "%")):
            continue
        ast.parse(source, filename=f"{filename}:{index}")


def test_submission_reading_path_is_complete() -> None:
    required = {
        "README.md",
        "PROJECT_SUMMARY.md",
        "EVALUATION.md",
        "LINKS.md",
        "SUBMISSION_CHECKLIST.md",
        "dataset/README.md",
        "dataset/PUBLISHING.md",
        "model/README.md",
        "notebooks/README.md",
        "notebooks/HISTORICAL_NOTEBOOK_INDEX.md",
        "video/DEMO_PROMPTS.md",
        "video/DEMO_SCRIPT.md",
        "video/FROZEN_STRONG_PROMPT.txt",
        "video/REHEARSAL_SCORECARD.md",
        "video/TUNED_REFERENCE_OUTPUT.md",
    }
    assert all((SUBMISSION / relative).is_file() for relative in required)
    readme = (SUBMISSION / "README.md").read_text(encoding="utf-8")
    assert "exact-silver v1 Complete-600" in readme
    assert "fixed control token only" in readme
    assert "Brainlift" in readme
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "submission/README.md" in root_readme
    assert "6.26 to 9.34" in root_readme


def test_submission_relative_markdown_links_resolve() -> None:
    for path in SUBMISSION.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#")):
                continue
            relative = target.split("#", 1)[0]
            assert (path.parent / relative).exists(), (path, target)


def test_public_dataset_card_is_exact_and_rights_bounded() -> None:
    card = (SUBMISSION / "dataset/README.md").read_text(encoding="utf-8")
    assert "license: other" in card
    assert "mixed-rights, noncommercial experimental release" in card
    assert "MathDial" in card and "CC BY-SA 4.0" in card
    assert "ConvoLearn" in card and "MIT" in card
    assert "Arabic YouTube" in card and "project-owner-attested permission" in card
    assert "540" in card and "60" in card and "120" in card and "480" in card
    assert "990cdc7ca494a4e12efa1cc7a739030ef1412e1dd1a350f91b1d453e49a756a0" in card
    assert "cf10c2e316c13da2e2ebf1c1edee18ebfd3784085440a00b8a8c31e71fa0ec1c" in card


def test_public_dataset_upload_bundle_matches_manifest() -> None:
    bundle = SUBMISSION / "dataset/uae_adab_hf_dataset_upload.zip"
    bundle_manifest = json.loads(
        (SUBMISSION / "dataset/BUNDLE_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == bundle_manifest[
        "archive_sha256"
    ]
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == set(bundle_manifest["files"])
        for name, record in bundle_manifest["files"].items():
            raw = archive.read(name)
            assert len(raw) == record["bytes"]
            assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        assert archive.read("README.md") == (
            SUBMISSION / "dataset/README.md"
        ).read_bytes()


def test_video_prompt_is_the_frozen_project_prompt() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "evals/uae_adab"))
    import run_benchmark  # type: ignore

    saved = (SUBMISSION / "video/FROZEN_STRONG_PROMPT.txt").read_text(
        encoding="utf-8"
    ).strip()
    assert saved == run_benchmark.STRONG_SYSTEM_PROMPT
    assert hashlib.sha256(saved.encode("utf-8")).hexdigest() == (
        "2788520ff0434f4ba2106eaa216edd2d2ad67999647006f68dcb69158b1bb077"
    )


def test_demo_turns_match_frozen_cross_domain_scenario() -> None:
    scenarios = json.loads(
        (ROOT / "evals/uae_adab/scenarios.json").read_text(encoding="utf-8")
    )
    expected = next(row for row in scenarios if row["id"] == "cross_domain_drift")
    prompt_text = (SUBMISSION / "video/DEMO_PROMPTS.md").read_text(encoding="utf-8")
    for turn in expected["turns"][:4]:
        assert turn["user"] in prompt_text


def test_canonical_training_notebook_is_embedded_and_locked() -> None:
    path = NOTEBOOKS / "01_train_exact_silver_v1.ipynb"
    notebook, text = notebook_text(path)
    assert_clean_parseable(notebook, path.name)
    assert 'SAVE_TO_DRIVE = False' in text
    assert 'RUN_EXPERIMENT = "complete600"' in text
    assert 'MAX_STEPS = 75' in text
    assert 'SAMPLED_CONVERSATIONS == 600' in text
    assert 'assert isinstance(freeze.get("release_authority"), bool)' in text
    assert 'assert freeze.get("release_authority") is False' not in text
    assert not re.search(r"hf_[A-Za-z0-9]{12,}", text)

    match = re.search(r'DATA_PAYLOAD_B64 = """(.*?)"""', text, re.DOTALL)
    hash_match = re.search(
        r'EXPECTED_DATA_PAYLOAD_SHA256 = "([0-9a-f]{64})"', text
    )
    assert match and hash_match
    payload = base64.b64decode("".join(match.group(1).split()))
    assert hashlib.sha256(payload).hexdigest() == hash_match.group(1)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {
            "exact_silver_release_manifest.json",
            "grounded_120_train.jsonl",
            "grounded_120_validation.jsonl",
            "complete_600_train.jsonl",
            "complete_600_validation.jsonl",
        }


def test_canonical_evaluation_notebook_has_only_final_conditions() -> None:
    path = NOTEBOOKS / "02_evaluate_exact_silver_v1.ipynb"
    notebook, text = notebook_text(path)
    assert_clean_parseable(notebook, path.name)
    assert 'USE_DRIVE = True' in text
    assert 'EXPECTED_CONDITIONS = {' in text
    assert '"base_no_prompt"' in text
    assert '"base_strong_prompt"' in text
    assert '"complete_600_exact_silver"' in text
    assert "grounded_120_exact_silver" not in text
    assert "EXPECTED_SCENARIOS_SHA256" in text
    assert "truncation=False" in text
    assert "do_sample=False" in text
    assert not re.search(r"hf_[A-Za-z0-9]{12,}", text)


def test_canonical_demo_notebook_is_clean_and_public() -> None:
    canonical = NOTEBOOKS / "03_demo_uae_adab_tutor.ipynb"
    notebook, text = notebook_text(canonical)
    assert_clean_parseable(notebook, canonical.name)
    assert "adarshrajesh/uae-adab-tutor-qwen3-4b" in text
    assert "c20d382f32810deaff2f691cdf78c0a3a4d9be59" in text
    assert "HF_TOKEN" not in text
    assert "notebook_login" not in text


def test_submission_notebook_manifest_matches_files() -> None:
    manifest = json.loads((NOTEBOOKS / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "uae_adab_submission_notebooks.v1"
    assert set(manifest["files"]) == {
        "01_train_exact_silver_v1.ipynb",
        "02_evaluate_exact_silver_v1.ipynb",
        "03_demo_uae_adab_tutor.ipynb",
    }
    for filename, record in manifest["files"].items():
        raw = (NOTEBOOKS / filename).read_bytes()
        assert record["bytes"] == len(raw)
        assert record["sha256"] == hashlib.sha256(raw).hexdigest()


def test_canonical_notebook_zip_is_complete() -> None:
    bundle = SUBMISSION / "uae_adab_canonical_colabs.zip"
    assert bundle.is_file()
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {
            "README.md",
            "HISTORICAL_NOTEBOOK_INDEX.md",
            "MANIFEST.json",
            "01_train_exact_silver_v1.ipynb",
            "02_evaluate_exact_silver_v1.ipynb",
            "03_demo_uae_adab_tutor.ipynb",
        }
        for filename in (
            "01_train_exact_silver_v1.ipynb",
            "02_evaluate_exact_silver_v1.ipynb",
            "03_demo_uae_adab_tutor.ipynb",
        ):
            assert archive.read(filename) == (NOTEBOOKS / filename).read_bytes()
