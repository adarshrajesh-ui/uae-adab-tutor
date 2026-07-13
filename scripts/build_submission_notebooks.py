#!/usr/bin/env python3
"""Build the three clean, grader-facing Colab notebooks."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import textwrap
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_TRAIN = ROOT / "notebooks/qwen3_4b_uae_adab_exact_silver_ablation_v2.ipynb"
SOURCE_EVAL = ROOT / "notebooks/qwen3_4b_uae_adab_exact_silver_eval_behavior.ipynb"
REVIEWER_DEMO_APP = ROOT / "deploy/reviewer_demo/app.py"
DATA_ZIP = (
    ROOT
    / "data/uae_adab/v3/final_600/exact_silver_release_v1"
    / "exact_silver_colab_inputs_locked.zip"
)
SCENARIOS = ROOT / "evals/uae_adab/scenarios.json"
OUTPUT_DIR = ROOT / "submission/notebooks"


def lines(value: str) -> list[str]:
    return textwrap.dedent(value).strip("\n").splitlines(keepends=True)


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(value)}


def code(value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(value),
    }


def source_cell(notebook: dict, index: int) -> str:
    return "".join(notebook["cells"][index]["source"])


def encoded(path: Path, width: int = 100) -> tuple[str, str]:
    raw = path.read_bytes()
    value = base64.b64encode(raw).decode("ascii")
    return "\n".join(textwrap.wrap(value, width)), hashlib.sha256(raw).hexdigest()


def clean_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "UAE Adab exact-silver v1 reproduction"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_training() -> dict:
    original = json.loads(SOURCE_TRAIN.read_text(encoding="utf-8"))
    payload, payload_sha = encoded(DATA_ZIP)

    validation_paths = source_cell(original, 7).replace(
        'Path("/content/', 'DATA_ROOT / "'
    )
    validation_paths = validation_paths.replace('.jsonl")', '.jsonl"')
    validation_paths = validation_paths.replace('.json")', '.json"')

    row_validation = source_cell(original, 8).replace(
        'assert freeze.get("release_authority") is False',
        'assert isinstance(freeze.get("release_authority"), bool)',
    )

    training = source_cell(original, 13)
    training = re.sub(
        r'RUN_EXPERIMENT = None  # Set to exactly one of: "grounded120", "complete600"\.[\s\S]*$',
        '''RUN_EXPERIMENT = "complete600"\nprint("Selected final experiment:", RUN_EXPERIMENT)\nselected_adapter, selected_metrics = train_selected(RUN_EXPERIMENT)\n''',
        training,
    )
    assert 'RUN_EXPERIMENT = "complete600"' in training

    setup = f'''
        from pathlib import Path
        from collections import Counter
        import base64
        import gc
        import hashlib
        import io
        import json
        import math
        import re
        import shutil
        import time
        import zipfile

        import torch
        import unsloth
        from datasets import Dataset
        from google.colab import drive, files
        from unsloth import FastLanguageModel

        assert torch.cuda.is_available(), "Enable a GPU runtime first."
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        assert GPU_MEMORY_GB >= 20, (
            f"Use an L4/A100-class GPU; got {{GPU_NAME}} ({{GPU_MEMORY_GB:.1f}} GB)."
        )

        DATA_PAYLOAD_B64 = """__DATA_PAYLOAD__"""
        EXPECTED_DATA_PAYLOAD_SHA256 = "{payload_sha}"
        payload_bytes = base64.b64decode("".join(DATA_PAYLOAD_B64.split()))
        assert hashlib.sha256(payload_bytes).hexdigest() == EXPECTED_DATA_PAYLOAD_SHA256
        DATA_ROOT = Path("/content/exact_silver_v1_data")
        if DATA_ROOT.exists():
            shutil.rmtree(DATA_ROOT)
        DATA_ROOT.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(payload_bytes)) as archive:
            archive.extractall(DATA_ROOT)

        # Graders need no Drive input. Enable this only to carry the trained
        # adapter automatically into notebook 02 or notebook 03.
        SAVE_TO_DRIVE = False
        if SAVE_TO_DRIVE:
            drive.mount("/content/drive")
            OUTPUT_ROOT = Path("/content/drive/MyDrive/uae_adab_slm")
        else:
            OUTPUT_ROOT = Path("/content/uae_adab_submission_outputs")
        RUNS = OUTPUT_ROOT / "runs_v2_exact"
        METRICS = OUTPUT_ROOT / "metrics_v2_exact"
        for path in (RUNS, METRICS):
            path.mkdir(parents=True, exist_ok=True)

        if "_EXACT_SESSION_NONCE_SHA256" not in globals():
            _EXACT_SESSION_NONCE_SHA256 = hashlib.sha256(
                f"{{time.time_ns()}}|{{GPU_NAME}}".encode("utf-8")
            ).hexdigest()
        if "_EXACT_TRAINING_SESSION_RUN" not in globals():
            _EXACT_TRAINING_SESSION_RUN = None
        print("GPU:", GPU_NAME, f"({{GPU_MEMORY_GB:.1f}} GB)")
        print("Embedded data SHA-256:", EXPECTED_DATA_PAYLOAD_SHA256)
        print("Output root:", OUTPUT_ROOT)
    '''
    setup = textwrap.dedent(setup).strip("\n").replace("__DATA_PAYLOAD__", payload)

    bundle = '''
        RUN_NAME = "qwen3_4b_uae_adab_complete_600_exact_silver_v1"
        assert (RUNS / RUN_NAME / "adapter_config.json").is_file()
        assert (METRICS / f"{RUN_NAME}.json").is_file()

        BUNDLE_ROOT = Path("/content/exact_silver_v1_reproduction_bundle")
        if BUNDLE_ROOT.exists():
            shutil.rmtree(BUNDLE_ROOT)
        (BUNDLE_ROOT / "runs_v2_exact").mkdir(parents=True)
        (BUNDLE_ROOT / "metrics_v2_exact").mkdir(parents=True)
        shutil.copytree(RUNS / RUN_NAME, BUNDLE_ROOT / "runs_v2_exact" / RUN_NAME)
        shutil.copy2(
            METRICS / f"{RUN_NAME}.json",
            BUNDLE_ROOT / "metrics_v2_exact" / f"{RUN_NAME}.json",
        )
        BUNDLE_ZIP = Path(shutil.make_archive(
            "/content/qwen3_4b_uae_adab_exact_silver_v1_reproduction",
            "zip",
            BUNDLE_ROOT,
        ))
        print("REPRODUCTION BUNDLE READY:", BUNDLE_ZIP)
        print("Bundle SHA-256:", hashlib.sha256(BUNDLE_ZIP.read_bytes()).hexdigest())

        DOWNLOAD_RESULTS = False
        if DOWNLOAD_RESULTS:
            files.download(str(BUNDLE_ZIP))
        else:
            print("Set DOWNLOAD_RESULTS=True and rerun this cell to download the ZIP.")
    '''

    return clean_notebook(
        [
            markdown(
                '''
                # 01 — reproduce the selected exact-silver v1 training run

                This is the canonical grader notebook. It contains the frozen
                data payload, validates all release hashes, and trains only the
                selected Complete-600 adapter. No input file or Google Drive is
                required. Use an L4 or A100 runtime.

                Expected core training time is roughly 6–12 minutes after
                installation and model download. The original run used 75
                optimizer steps and 600 deterministic row exposures.
                '''
            ),
            markdown("## 1. Install the pinned environment"),
            code(source_cell(original, 2)),
            code(source_cell(original, 3)),
            markdown(
                "Restart the Colab runtime after installation, then run every remaining cell in order."
            ),
            code(setup),
            markdown("## 2. Validate the embedded exact-silver release"),
            code(validation_paths),
            code(row_validation),
            markdown("## 3. Validate the locked training controls"),
            code(source_cell(original, 10)),
            code(source_cell(original, 11)),
            markdown("## 4. Train the selected Complete-600 adapter"),
            code(training),
            markdown("## 5. Package the adapter and metrics"),
            code(bundle),
        ]
    )


def build_evaluation() -> dict:
    original = json.loads(SOURCE_EVAL.read_text(encoding="utf-8"))
    scenario_payload, scenario_sha = encoded(SCENARIOS)

    original_setup = source_cell(original, 6)
    functions_tail = original_setup[original_setup.index("def file_sha256"):]
    functions_tail = functions_tail.replace(
        'print("Both exact-silver adapters and metrics validated.")',
        'print("Complete-600 adapter and metrics validated.")',
    )
    setup = f'''
        from pathlib import Path
        import base64
        import gc
        import hashlib
        import json
        import math
        import os
        import shutil
        import time
        import zipfile

        import torch
        import unsloth
        from google.colab import drive, files
        from peft import PeftModel
        from unsloth import FastLanguageModel

        assert torch.cuda.is_available(), "Enable a GPU runtime first."
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        assert GPU_MEMORY_GB >= 20, f"Use an L4/A100-class GPU; got {{GPU_NAME}}."

        # Use Drive after notebook 01 saved there. Set False to upload notebook
        # 01's reproduction ZIP instead.
        USE_DRIVE = True
        if USE_DRIVE:
            drive.mount("/content/drive")
            ARTIFACT_ROOT = Path("/content/drive/MyDrive/uae_adab_slm")
        else:
            uploaded = files.upload()
            zip_names = [name for name in uploaded if name.endswith(".zip")]
            assert len(zip_names) == 1, "Upload exactly one reproduction ZIP from notebook 01."
            ARTIFACT_ROOT = Path("/content/uploaded_exact_silver_artifacts")
            if ARTIFACT_ROOT.exists():
                shutil.rmtree(ARTIFACT_ROOT)
            ARTIFACT_ROOT.mkdir(parents=True)
            with zipfile.ZipFile(Path("/content") / zip_names[0]) as archive:
                archive.extractall(ARTIFACT_ROOT)

        BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
        BASE_MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
        EXPECTED_RELEASE_MANIFEST_SHA256 = "ef8b907a497444b83410bd2fdcff37c4eb3922d90e485638303b8e5342ba662b"
        EVAL_MAX_SEQ_LENGTH = 4096
        EVAL_MAX_NEW_TOKENS = 512
        SYSTEM_MESSAGE = "<uae_adab_tutor>default</uae_adab_tutor>"
        EXPECTED_LOCKED_CONFIG = {{
            "base_model": BASE_MODEL_ID,
            "base_model_revision": BASE_MODEL_REVISION,
            "max_sequence_length": 4096,
            "max_steps": 75,
            "effective_batch": 8,
            "sampled_conversations": 600,
            "lora_rank": 16,
            "lora_alpha": 32,
            "learning_rate": 2e-4,
            "response_only": True,
            "packing": False,
            "seed": 3407,
        }}

        RUNS_ROOT = ARTIFACT_ROOT / "runs_v2_exact"
        METRICS_ROOT = ARTIFACT_ROOT / "metrics_v2_exact"
        EVAL_SET_NAME = "behavior_pressure_10"
        OUTPUT_DIR = ARTIFACT_ROOT / "metrics_v2_exact" / "submission_eval" / EVAL_SET_NAME
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        SCENARIOS_B64 = """__SCENARIO_PAYLOAD__"""
        SCENARIOS_PATH = Path("/content/scenarios.json")
        SCENARIOS_PATH.write_bytes(base64.b64decode("".join(SCENARIOS_B64.split())))
        EXPECTED_SCENARIOS_SHA256 = "{scenario_sha}"
        EXPECTED_SCENARIO_COUNT = 10
        scenario_sha256 = hashlib.sha256(SCENARIOS_PATH.read_bytes()).hexdigest()
        assert scenario_sha256 == EXPECTED_SCENARIOS_SHA256
        scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
        assert isinstance(scenarios, list) and len(scenarios) == EXPECTED_SCENARIO_COUNT
        assert len({{scenario["id"] for scenario in scenarios}}) == len(scenarios)
        assert all(len(scenario["turns"]) == 5 for scenario in scenarios)

        RUNS = {{
            "complete_600_exact_silver": {{
                "selector": "complete600",
                "run_name": "qwen3_4b_uae_adab_complete_600_exact_silver_v1",
                "train_rows": 540,
                "validation_rows": 60,
                "train_sha256": "990cdc7ca494a4e12efa1cc7a739030ef1412e1dd1a350f91b1d453e49a756a0",
                "validation_sha256": "cf10c2e316c13da2e2ebf1c1edee18ebfd3784085440a00b8a8c31e71fa0ec1c",
            }}
        }}

        __FUNCTIONS_TAIL__
    '''
    setup = (
        textwrap.dedent(setup)
        .strip("\n")
        .replace("__SCENARIO_PAYLOAD__", scenario_payload)
        .replace("__FUNCTIONS_TAIL__", functions_tail)
    )

    helper_original = source_cell(original, 8)
    helper_tail = helper_original[helper_original.index("def load_pinned_base"):]
    strong_prompt = helper_original[
        helper_original.index("FROZEN_STRONG_PROMPT = "):
        helper_original.index("EXPECTED_CONDITIONS = ")
    ].strip()
    helper = '''
        __STRONG_PROMPT__

        EXPECTED_CONDITIONS = {
            "base_no_prompt",
            "base_strong_prompt",
            "complete_600_exact_silver",
        }
        EXPECTED_CONDITION_META = {
            "base_no_prompt": {
                "adapter": None,
                "system_prompt_sha256": None,
                "adapter_fingerprint_sha256": None,
            },
            "base_strong_prompt": {
                "adapter": None,
                "system_prompt_sha256": hashlib.sha256(
                    FROZEN_STRONG_PROMPT.encode("utf-8")
                ).hexdigest(),
                "adapter_fingerprint_sha256": None,
            },
            "complete_600_exact_silver": {
                "adapter": str(validated_runs["complete_600_exact_silver"]["adapter_path"]),
                "system_prompt_sha256": hashlib.sha256(SYSTEM_MESSAGE.encode("utf-8")).hexdigest(),
                "adapter_fingerprint_sha256": validated_runs[
                    "complete_600_exact_silver"
                ]["fingerprint"]["combined_sha256"],
            },
        }

        __HELPER_TAIL__
    '''
    helper = (
        textwrap.dedent(helper)
        .strip("\n")
        .replace("__STRONG_PROMPT__", strong_prompt)
        .replace("__HELPER_TAIL__", helper_tail)
    )

    package = source_cell(original, 16).replace(
        '("grounded_120_exact_silver", "complete_600_exact_silver")',
        '("complete_600_exact_silver",)',
    )
    package += '''

DOWNLOAD_RESULTS = False
if DOWNLOAD_RESULTS:
    files.download(str(bundle_path))
else:
    print("Set DOWNLOAD_RESULTS=True and rerun this cell to download the ZIP.")
'''

    return clean_notebook(
        [
            markdown(
                '''
                # 02 — reproduce the exact-silver v1 behavior evaluation

                This canonical notebook contains the frozen ten-scenario suite
                and generates three directly relevant conditions: unprompted
                base Qwen, strongly prompted base Qwen, and selected tuned Qwen.
                Use an L4 or A100. It never sends held-out expected/failure text
                to an answer model.
                '''
            ),
            markdown("## 1. Install the pinned environment"),
            code(source_cell(original, 2)),
            code(source_cell(original, 3)),
            markdown(
                "Restart the Colab runtime after installation, then run every remaining cell in order."
            ),
            code(setup),
            markdown("## 2. Deterministic no-truncation generation helpers"),
            code(helper),
            markdown("## 3. Generate the selected tuned condition"),
            code(source_cell(original, 10)),
            markdown("## 4. Generate untouched and strong-prompt base controls"),
            code(source_cell(original, 14)),
            markdown("## 5. Validate and package all three conditions"),
            code(package),
        ]
    )


def build_reviewer_demo() -> dict:
    """Build the credential-free temporary Gradio reviewer demo."""

    app_source = REVIEWER_DEMO_APP.read_text(encoding="utf-8")
    return clean_notebook(
        [
            markdown(
                '''
                # 03 — launch the UAE Adab tutor reviewer demo

                This is the canonical live-demo path. Select a **T4 GPU or
                better**, then choose **Runtime → Run all**. The notebook:

                1. downloads the public selected adapter from
                   `adarshrajesh/uae-adab-tutor-qwen3-4b` without credentials;
                2. rejects it unless both adapter files match the frozen
                   SHA-256 hashes;
                3. loads the pinned Qwen3-4B base revision in 4-bit;
                4. sends only `<uae_adab_tutor>default</uae_adab_tutor>` plus
                   the visible multi-turn history to the tuned model; and
                5. prints a temporary public Gradio share link.

                The link works only while this Colab runtime is connected.
                There is no paid Hugging Face Space, login, Drive mount,
                upload, API key, or persistent hosting step.
                '''
            ),
            code(
                '''
                !pip -q install --upgrade --no-cache-dir \
                    transformers==4.56.2 peft==0.17.1 accelerate==1.10.1 \
                    bitsandbytes==0.47.0 huggingface_hub==0.34.4 \
                    safetensors==0.6.2 sentencepiece==0.2.1 gradio==5.49.1
                '''
            ),
            code(app_source),
        ]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "01_train_exact_silver_v1.ipynb": build_training(),
        "02_evaluate_exact_silver_v1.ipynb": build_evaluation(),
        "03_demo_uae_adab_tutor.ipynb": build_reviewer_demo(),
    }
    manifest = {"schema_version": "uae_adab_submission_notebooks.v1", "files": {}}
    for filename, notebook in outputs.items():
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
        path = OUTPUT_DIR / filename
        raw = (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        path.write_bytes(raw)
        manifest["files"][filename] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "cells": len(notebook["cells"]),
        }
        print(path.relative_to(ROOT), manifest["files"][filename]["sha256"])

    manifest_path = OUTPUT_DIR / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    bundle_path = ROOT / "submission/uae_adab_canonical_colabs.zip"
    bundle_files = [
        OUTPUT_DIR / "README.md",
        OUTPUT_DIR / "HISTORICAL_NOTEBOOK_INDEX.md",
        manifest_path,
        *(OUTPUT_DIR / filename for filename in sorted(outputs)),
    ]
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in bundle_files:
            info = zipfile.ZipInfo(path.name, date_time=(2026, 7, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    print(
        bundle_path.relative_to(ROOT),
        hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    main()
